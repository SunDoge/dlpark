//! Compares the four `DlpackBuilder` construction paths against each other,
//! across several tensor ranks (`ndim`), to see whether the gap between
//! variants shrinks, stays flat, or grows as shape/strides get bigger.
//!
//! All four use a `NonNull<()>` context (a no-op `OpaqueContext` impl — see
//! `src/context.rs`), so the only cost being measured is each variant's own
//! metadata allocation strategy, not context boxing.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use dlpark::DlpackBuilder;
use dlpark::ffi::DLManagedTensor;
use dlpark::tensor::compact_strides_array;
use std::ptr::NonNull;

fn context() -> NonNull<()> {
    static DUMMY: () = ();
    NonNull::from(&DUMMY)
}

/// Registers all four variants at rank `N` under a `builder/ndim=N` group.
/// `with_array_layout`/`with_slice_layout` need `N` as a const generic
/// anyway, so a generic function is all the per-size fixture needs — no
/// macro required.
fn bench_ndim<const N: usize>(c: &mut Criterion) {
    // Dims are all 1 regardless of N: the values are never read (this
    // benchmark never dereferences `data`), and it keeps the cumulative
    // product `compact_strides_array` computes from overflowing i64 at
    // high N (unlike e.g. `2, 3, 4, ...` which already overflows by N=64).
    let shape: [i64; N] = [1; N];
    let strides = compact_strides_array(shape).unwrap();
    let mut group = c.benchmark_group(format!("builder/ndim={N}"));

    group.bench_function(BenchmarkId::new("with_array_layout", N), |b| {
        b.iter(|| {
            let dlpack = DlpackBuilder::<DLManagedTensor, N>::with_array_layout(
                context(),
                std::hint::black_box(shape),
                std::hint::black_box(strides),
            )
            .build();
            std::hint::black_box(dlpack);
        });
    });

    group.bench_function(BenchmarkId::new("with_slice_layout", N), |b| {
        b.iter(|| {
            let dlpack = DlpackBuilder::<DLManagedTensor, N>::with_slice_layout(
                context(),
                std::hint::black_box(shape.as_slice()),
                std::hint::black_box(strides.as_slice()),
            )
            .build();
            std::hint::black_box(dlpack);
        });
    });

    group.bench_function(BenchmarkId::new("with_dynamic_layout", N), |b| {
        b.iter(|| {
            let dlpack = DlpackBuilder::<DLManagedTensor, 0>::with_dynamic_layout(
                context(),
                std::hint::black_box(shape.as_slice()),
                std::hint::black_box(strides.as_slice()),
            )
            .unwrap()
            .build();
            std::hint::black_box(dlpack);
        });
    });

    let mut shape_mut = shape;
    let mut strides_mut = strides;
    group.bench_function(BenchmarkId::new("with_pointer_layout", N), |b| {
        b.iter(|| {
            let dlpack = unsafe {
                DlpackBuilder::<DLManagedTensor, 0>::with_pointer_layout(
                    context(),
                    std::hint::black_box(shape_mut.as_mut_ptr()),
                    std::hint::black_box(strides_mut.as_mut_ptr()),
                    N as i32,
                )
                .unwrap()
                .build()
            };
            std::hint::black_box(dlpack);
        });
    });

    group.finish();
}

fn bench_all(c: &mut Criterion) {
    bench_ndim::<1>(c);
    bench_ndim::<4>(c);
    bench_ndim::<16>(c);
    bench_ndim::<64>(c);
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
