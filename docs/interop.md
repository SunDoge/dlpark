# Interop backends

All adapters are feature-gated, and no feature is enabled by default.

## image

Export wraps an owned `ImageBuffer<P, Vec<P::Subpixel>>` as a compact rank-3
HWC tensor. Import validates that layout and returns either a borrowed image or
an owning `ImageBuffer` whose storage retains the managed DLPack tensor.

## ndarray

Owned arrays export without copying. Imports produce `ArrayViewD<T>` or
`ArrayViewMutD<T>`. Mutable import rejects `READ_ONLY`, but remains unsafe
because the caller must establish that no aliases are concurrently accessed.

## candle

CPU `candle::Tensor` export is zero-copy because the boxed tensor keeps its
reference-counted storage alive. Import copies because candle has no borrowed
CPU tensor type. Compact input uses a bulk copy and arbitrary strides are
gathered. Candle storage is internally shared, so producers should set
`DlpackFlags::READ_ONLY` when they cannot guarantee external immutability.

## cudarc

`interop::cudarc::from_cuda_slice` exports higher-rank layouts and returns a
`CudaDlpackExport` containing both initialized metadata and the current CUDA
stream. Import uses `CudaImport::ready_on(stream)` and returns a
`ManagedCudaSlice` that retains the managed DLPack owner.

This adapter is for applications already using `cudarc`. Native runtime
loading and custom CUDA buffer policy belong in application code; see
`demos/cuda-python` for a minimal implementation.

## safetensors

`SafeTensorFile` parses owned bytes or opens a read-only mmap and exports named
tensors with versioned DLPack and `READ_ONLY`. Each tensor independently keeps
the file alive. Misaligned in-memory data is rejected, and zero-copy exchange
is rejected on big-endian targets.

`DlpackView::from_dlpack` borrows a compact CPU tensor for safetensors
serialization. It is unsafe because DLPack does not report allocation bounds.
All safetensors 0.8 dtypes, including packed F4/F6 and FP8 formats, have exact
DLPack mappings.

## Element types

`DlpackElement` covers Rust integer and floating-point primitives. The `half`
feature adds `half::f16` and `half::bf16`. Container adapters use this trait to
verify that Rust element types match DLPack descriptors.
