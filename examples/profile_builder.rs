//! Ad-hoc CPU profiler for the DLPack builder's construction paths.
//!
//! `pprof`'s signal-based sampling (`SIGPROF`/`setitimer`) doesn't touch the
//! `perf_event` subsystem, so it works even with `perf_event_paranoid`
//! locked down (unlike `perf`/`cargo-flamegraph`). This isn't wired through
//! criterion's own profiler hook because `pprof`'s "criterion" feature pins
//! its own (older) criterion version, which conflicts with the one already
//! used by `benches/builder.rs` — driving `ProfilerGuard` directly here
//! sidesteps that entirely.
//!
//! Run: `cargo run --release --example profile_builder`
//! Output: `target/flamegraph-<variant>.svg` per variant, viewable in a browser.

use dlpark::Builder;
use dlpark::ffi::DLManagedTensorVersioned;
use dlpark::metadata::{CopiedArray, CopiedSlice};
use dlpark::tensor::compact_strides_array;
use std::ptr::NonNull;

const N: usize = 64;
const ITERATIONS: u64 = 200_000_000;

fn context() -> NonNull<()> {
    static DUMMY: () = ();
    NonNull::from(&DUMMY)
}

fn profile<F: FnMut()>(name: &str, mut f: F) {
    let guard = pprof::ProfilerGuardBuilder::default()
        .frequency(2000)
        .blocklist(&["libc", "libgcc", "pthread", "vdso"])
        .build()
        .expect("failed to start profiler");

    let start = std::time::Instant::now();
    for _ in 0..ITERATIONS {
        f();
    }
    let elapsed = start.elapsed();
    println!(
        "{name}: {ITERATIONS} iterations in {elapsed:?} ({:.2} ns/iter)",
        elapsed.as_secs_f64() * 1e9 / ITERATIONS as f64
    );

    let report = guard.report().build().expect("failed to build report");

    // Flamegraph SVG, for visual inspection.
    let path = format!("target/flamegraph-{name}.svg");
    let file = std::fs::File::create(&path).expect("failed to create flamegraph file");
    report.flamegraph(file).expect("failed to write flamegraph");
    println!("wrote {path}");

    // Textual top-N leaf-function summary, since an SVG can't be read here.
    let mut leaf_counts: std::collections::HashMap<String, isize> =
        std::collections::HashMap::new();
    for (frames, count) in &report.data {
        if let Some(leaf) = frames.frames.first().and_then(|f| f.first()) {
            *leaf_counts.entry(leaf.name()).or_insert(0) += count;
        }
    }
    let mut sorted: Vec<_> = leaf_counts.into_iter().collect();
    sorted.sort_by_key(|&(_, count)| std::cmp::Reverse(count));
    let total: isize = sorted.iter().map(|&(_, c)| c).sum();
    println!("--- {name}: top leaf frames by sample count (total={total}) ---");
    for (symbol, count) in sorted.iter().take(10) {
        let pct = 100.0 * *count as f64 / total as f64;
        println!("  {count:>8} ({pct:5.1}%)  {symbol}");
    }
}

fn main() {
    let shape: [i64; N] = [1; N];
    let strides = compact_strides_array(shape).unwrap();

    // Order swapped vs the last run, to test whether allocator warm-up
    // ordering (not the approach itself) explains the alloc-time gap.
    profile("dynamic_layout_first", || {
        let dlpack = Builder::new(
            context(),
            CopiedSlice::new(
                std::hint::black_box(shape.as_slice()),
                std::hint::black_box(strides.as_slice()),
            ),
        )
        .try_build::<DLManagedTensorVersioned>()
        .unwrap();
        std::hint::black_box(dlpack);
    });

    profile("array_layout_second", || {
        let dlpack = Builder::new(
            context(),
            CopiedArray::new(std::hint::black_box(&shape), std::hint::black_box(&strides)),
        )
        .build::<DLManagedTensorVersioned>();
        std::hint::black_box(dlpack);
    });
}
