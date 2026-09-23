# Core API

## Ownership model

`Managed<M>` is an RAII owner for a DLPack managed tensor pointer. It invokes
the descriptor's deleter on drop. A null deleter means the producer retains
ownership, so dropping the handle does nothing. The ABI type is explicit:

- `legacy::Dlpack` is `Managed<DLManagedTensor>`.
- `versioned::Dlpack` is `Managed<DLManagedTensorVersioned>`.

`ManagedTensorBase` provides the common operations needed by generic allocation
and ownership code while preserving the concrete C layout.

Consumers call `Managed::validate()` to obtain a `TensorRef`. This validates
the descriptor metadata and provides safe access to shape, strides, dtype,
device, and computed size. Data access remains unsafe because DLPack does not
carry allocation bounds. Mutable access additionally rejects `READ_ONLY`, but
the caller must still establish exclusive access.

## Producing a tensor

Production has three stages: metadata preparation, initialization with an
owning context, and completion of scalar fields.

`metadata::Fixed` handles compile-time rank and `metadata::Dynamic` handles
runtime rank. Their standard constructors copy shape and stride values into the
managed allocation. The explicitly named `borrowed` constructors avoid that
copy but require `prepare_unchecked` because the referenced arrays must outlive
the exported tensor. Advanced callers can use `with_storage` with `Copied` and
`Borrowed` to select each part independently.

```rust
use dlpark::{
    ffi::{DLDataType, DLDevice, DLManagedTensorVersioned},
    metadata::Fixed,
    versioned,
};

let mut values = vec![0_f32; 6];
let data = values.as_mut_ptr().cast();
let mut initialized = Fixed::new([2, 3], [3, 1])
    .initialize::<DLManagedTensorVersioned>(Box::new(values))?;
initialized
    .set_data(data)
    .set_dtype(DLDataType::F32)
    .set_device(DLDevice::CPU);
let dlpack: versioned::Dlpack = unsafe { initialized.finish() };
# Ok::<(), Box<dyn std::error::Error>>(())
```

The context owns both the backing data and any borrowed metadata. `Box<T:
Send>` and `Arc<T: Send + Sync>` implement `OpaqueContext`; the bounds allow a
consumer to invoke the deleter on another thread.

For owned metadata, `initialize` fuses preparation and context installation
without changing the allocation strategy. Use `prepare` directly when those
steps need to remain separate. Borrowed metadata keeps the explicit unsafe
`prepare_unchecked` path so its lifetime obligation remains visible.

`finish` is unsafe because it asserts that all pointers, lifetimes, flags, and
layout fields satisfy the DLPack contract. `set_flags` prevents newly asserting
`IS_COPIED`; code that actually performed a copy may use
`set_flags_unchecked`.

## Layout validation

Legacy and versioned pre-1.2 imports accept null strides as compact row-major
layout. Versioned 1.2 and newer descriptors require explicit strides. Local
non-scalar exports always require explicit strides, regardless of ABI, and the
safe metadata builders provide them.

`TensorRef::cpu_slice` checks device, dtype, alignment, and compact layout.
`cpu_bytes` is dtype-agnostic and supports packed sub-byte types. Both remain
unsafe because the descriptor cannot prove the allocation's bounds.

## Versions and flags

The bundled headers currently define DLPack 1.3. `DLPackVersion` provides:

- `is_compatible_with` for major-version ABI compatibility.
- `ensure_compatible_with` for the checked form returning `VersionError`.
- `supports` for testing whether a feature-level version is available.

Versioned tensors carry `DlpackFlags`, including `READ_ONLY`, `IS_COPIED`, and
sub-byte padding. `Managed::flags_mut` is unsafe because clearing read-only or
asserting copied state changes the guarantees consumers may rely on.

## Import contexts

Backend conversions implement `TryFromDlpack<D, C>`. `D` is the owned or
borrowed DLPack representation; `C` is a consumer-defined import context. It
may be `()`, a stream, a backend client, a device mapping, or a policy object.
This keeps backend-specific runtime requirements outside the core crate.
