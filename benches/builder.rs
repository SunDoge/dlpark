//! Compares the four metadata construction paths against each other,
//! across several tensor ranks (`ndim`), to see whether the gap between
//! variants shrinks, stays flat, or grows as shape/strides get bigger.
//!
//! All four use a local no-op context, so the only cost being measured is each
//! variant's own metadata allocation strategy, not context boxing.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use dlpark::OpaqueContext;
use dlpark::ffi::DLManagedTensor;
use dlpark::metadata::{Borrowed, Copied, Dynamic, Fixed};
use dlpark::tensor::compact_strides_array;
use std::ffi::c_void;

struct NoopContext;

// SAFETY: drop_raw is a no-op and may be called on any thread.
unsafe impl OpaqueContext for NoopContext {
    fn into_raw(self) -> *mut c_void {
        std::ptr::null_mut()
    }

    unsafe fn drop_raw(_raw: *mut c_void) {}
}

fn context() -> NoopContext {
    NoopContext
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
            let prepared = Fixed::new(
                Copied(std::hint::black_box(&shape)),
                Copied(std::hint::black_box(&strides)),
            )
            .prepare::<DLManagedTensor>()
            .unwrap();
            std::hint::black_box(prepared.initialize(context()));
        });
    });

    group.bench_function(BenchmarkId::new("borrowed_array", N), |b| {
        b.iter(|| {
            let prepared = unsafe {
                Fixed::new(
                    Borrowed(std::hint::black_box(&shape)),
                    Borrowed(std::hint::black_box(&strides)),
                )
                .prepare_unchecked::<DLManagedTensor>()
                .unwrap()
            };
            std::hint::black_box(prepared.initialize(context()));
        });
    });

    group.bench_function(BenchmarkId::new("copied_slice", N), |b| {
        b.iter(|| {
            let prepared = Dynamic::new(
                Copied(std::hint::black_box(shape.as_slice())),
                Copied(std::hint::black_box(strides.as_slice())),
            )
            .prepare::<DLManagedTensor>()
            .unwrap();
            std::hint::black_box(prepared.initialize(context()).unwrap());
        });
    });

    group.bench_function(BenchmarkId::new("borrowed_slice", N), |b| {
        b.iter(|| {
            let prepared = unsafe {
                Dynamic::new(
                    Borrowed(std::hint::black_box(shape.as_slice())),
                    Borrowed(std::hint::black_box(strides.as_slice())),
                )
                .prepare_unchecked::<DLManagedTensor>()
                .unwrap()
            };
            std::hint::black_box(prepared.initialize(context()).unwrap());
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

fn bench_generic_metadata_copy(c: &mut Criterion) {
    const N: usize = 64;
    let source = [1u32; N];
    let mut destination = [0i64; N];
    let mut group = c.benchmark_group("generic_metadata_copy/len=64");

    group.bench_function("allocate_i64_then_copy_nonoverlapping", |b| {
        b.iter(|| {
            let converted: Vec<i64> = std::hint::black_box(&source)
                .iter()
                .copied()
                .map(Into::into)
                .collect();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    converted.as_ptr(),
                    std::hint::black_box(destination.as_mut_ptr()),
                    N,
                );
            }
            std::hint::black_box(&destination);
        });
    });

    group.bench_function("convert_directly_into_destination", |b| {
        b.iter(|| {
            for (destination, &source) in std::hint::black_box(&mut destination)
                .iter_mut()
                .zip(std::hint::black_box(&source))
            {
                *destination = source.into();
            }
            std::hint::black_box(&destination);
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
    bench_generic_metadata_copy(c);
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
