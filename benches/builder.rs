//! Compares the four metadata construction paths against each other,
//! across several tensor ranks (`ndim`), to see whether the gap between
//! variants shrinks, stays flat, or grows as shape/strides get bigger.
//!
//! All four use a `NonNull<()>` context (a no-op `OpaqueContext` impl — see
//! `src/context.rs`), so the only cost being measured is each variant's own
//! metadata allocation strategy, not context boxing.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use dlpark::Builder;
use dlpark::ffi::DLManagedTensor;
use dlpark::metadata::{BorrowedArray, BorrowedSlice, CopiedArray, CopiedSlice};
use dlpark::tensor::compact_strides_array;
use std::ptr::NonNull;

fn context() -> NonNull<()> {
    static DUMMY: () = ();
    NonNull::from(&DUMMY)
}

/// Registers all four variants at rank `N` under a `builder/ndim=N` group.
/// The array variants need `N` as a const generic, so a generic function is
/// all the per-size fixture needs.
fn bench_ndim<const N: usize>(c: &mut Criterion) {
    // Dims are all 1 regardless of N: the values are never read (this
    // benchmark never dereferences `data`), and it keeps the cumulative
    // product `compact_strides_array` computes from overflowing i64 at
    // high N (unlike e.g. `2, 3, 4, ...` which already overflows by N=64).
    let shape: [i64; N] = [1; N];
    let strides = compact_strides_array(shape).unwrap();
    let mut group = c.benchmark_group(format!("builder/ndim={N}"));

    group.bench_function(BenchmarkId::new("copied_array", N), |b| {
        b.iter(|| {
            let dlpack = Builder::new(
                context(),
                CopiedArray::new(std::hint::black_box(&shape), std::hint::black_box(&strides)),
            )
            .build::<DLManagedTensor>();
            std::hint::black_box(dlpack);
        });
    });

    group.bench_function(BenchmarkId::new("borrowed_array", N), |b| {
        b.iter(|| {
            let dlpack = unsafe {
                Builder::new(
                    context(),
                    BorrowedArray::new(
                        std::hint::black_box(&shape),
                        std::hint::black_box(&strides),
                    ),
                )
                .build::<DLManagedTensor>()
            };
            std::hint::black_box(dlpack);
        });
    });

    group.bench_function(BenchmarkId::new("copied_slice", N), |b| {
        b.iter(|| {
            let dlpack = Builder::new(
                context(),
                CopiedSlice::new(
                    std::hint::black_box(shape.as_slice()),
                    std::hint::black_box(strides.as_slice()),
                ),
            )
            .try_build::<DLManagedTensor>()
            .unwrap();
            std::hint::black_box(dlpack);
        });
    });

    group.bench_function(BenchmarkId::new("borrowed_slice", N), |b| {
        b.iter(|| {
            let dlpack = unsafe {
                Builder::new(
                    context(),
                    BorrowedSlice::new(
                        std::hint::black_box(shape.as_slice()),
                        std::hint::black_box(strides.as_slice()),
                    ),
                )
                .try_build::<DLManagedTensor>()
                .unwrap()
            };
            std::hint::black_box(dlpack);
        });
    });

    group.finish();
}

/// Isolates the allocator call itself (no shape/strides writes, no
/// `DLManagedTensor` construction) for the exact byte size/align that
/// copied array/slice metadata both request at `ndim=64`
/// (verified equal: 1088 bytes, align 8) — a clean baseline to compare
/// against those functions' full measured cost, since pprof's signal-based
/// sampling only had a few hundred samples to attribute time with, too few
/// to trust for a fine-grained "why" answer.
///
/// Two variants because the two call sites construct the `Layout` two
/// different ways (`Layout::new::<T>()` vs `Layout::from_size_align`) even
/// though both resolve to the same value — this checks whether the
/// construction method itself, not just the resulting value, affects
/// codegen/timing.
fn bench_alloc_dealloc_baseline(c: &mut Criterion) {
    #[repr(C)]
    struct Storage64 {
        _managed_tensor: DLManagedTensor,
        _shape: [i64; 64],
        _strides: [i64; 64],
    }

    let mut group = c.benchmark_group("alloc_dealloc_baseline/1088_bytes");

    group.bench_function("via_layout_new::<T>", |b| {
        b.iter(|| unsafe {
            let layout = std::alloc::Layout::new::<Storage64>();
            let ptr = std::alloc::alloc(std::hint::black_box(layout));
            std::hint::black_box(ptr);
            std::alloc::dealloc(ptr, layout);
        });
    });

    group.bench_function("via_from_size_align", |b| {
        b.iter(|| unsafe {
            let layout = std::alloc::Layout::from_size_align(1088, 8).unwrap();
            let ptr = std::alloc::alloc(std::hint::black_box(layout));
            std::hint::black_box(ptr);
            std::alloc::dealloc(ptr, layout);
        });
    });

    group.finish();
}

fn bench_all(c: &mut Criterion) {
    bench_ndim::<1>(c);
    bench_ndim::<2>(c);
    bench_ndim::<3>(c);
    bench_ndim::<4>(c);
    bench_ndim::<5>(c);
    bench_ndim::<16>(c);
    bench_ndim::<64>(c);
    bench_alloc_dealloc_baseline(c);
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
