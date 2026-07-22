//! candle interop (CPU only).
//!
//! Provides conversions between [`candle_core::Tensor`] and DLPack managed
//! tensors.
//!
//! # Forward direction (`Tensor` → `legacy::Dlpack`/`versioned::Dlpack`)
//!
//! Zero-copy: the whole `candle_core::Tensor` is boxed as the DLPack `manager_ctx`,
//! keeping its `Arc`-refcounted storage alive for the DLPack tensor's
//! lifetime. Only CPU tensors are supported — candle's CUDA backend needs
//! separate investigation and a CUDA toolchain to verify against.
//!
//! Candle's storage sits behind an `RwLock`, so the exported pointer aliases
//! memory candle itself can still mutate; there is no way to enforce
//! read-only access at the type level. Versioned exports leave flags empty by
//! default; set [`crate::DlpackFlags::READ_ONLY`] on the returned initialized
//! allocation before finishing it when required.
//!
//! # Reverse direction (`&Local<M>` → `Tensor`)
//!
//! Always a copy: candle has no borrowed/strided CPU tensor view type, so a
//! fresh contiguous `Vec<T>` is always built. Compact-stride sources take a
//! fast `to_vec()` path; arbitrary strides fall back to a manual gather.
//!
//! # Sub-byte packed dtypes
//!
//! DLPack's 8-/6-/4-bit float codes (`F8E4M3`, `F8E8M0`, `F6E2M3`, `F6E3M2`,
//! `F4`) are supported as opaque packed byte blobs — candle treats the
//! sub-byte ones (all but `F8E4M3`) as `dummy` types where every *compute*
//! operation panics, but raw byte storage/round-tripping works fine, which is
//! all zero-copy DLPack interop needs. Only the compact (non-strided,
//! zero-offset) case is supported for these — see `dl_dtype_from_candle`'s
//! doc for why per-element addressing doesn't make sense for them.

use crate::{
    Local, ManagedTensorBase,
    allocation::dynamic,
    ffi::{DLDataType, DLDataTypeCode, DLDevice},
    metadata::{Copied, Dynamic},
};
use candle_core::{
    DType, Device, Storage, Tensor, backend::BackendStorage, cpu_backend::CpuStorage,
};
use snafu::Snafu;
use std::mem::size_of;
use std::os::raw::c_void;

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(transparent)]
    Metadata { source: crate::metadata::Error },

    #[snafu(display("candle tensor must be on CPU"))]
    UnsupportedDevice,

    #[snafu(display("unsupported candle dtype: {dtype:?}"))]
    UnsupportedCandleDType { dtype: DType },

    #[snafu(display("no candle dtype matches DLPack dtype: {dtype:?}"))]
    UnsupportedDlDataType { dtype: crate::ffi::DLDataType },

    #[snafu(display("shape/stride value does not fit the target integer type"))]
    DimensionOverflow,

    #[snafu(display(
        "sub-byte packed dtype {dtype:?} with a nonzero element offset is not supported"
    ))]
    SubByteOffsetUnsupported { dtype: DLDataType },

    #[snafu(display(
        "sub-byte packed dtype {dtype:?} only supports compact (non-strided) tensors"
    ))]
    SubByteStridesUnsupported { dtype: DLDataType },

    #[snafu(display("strided access spans outside the tensor data buffer"))]
    StridedSpanOverflow,

    #[snafu(transparent)]
    Tensor { source: crate::tensor::Error },

    #[snafu(transparent)]
    Candle { source: candle_core::Error },
}

// ---------------------------------------------------------------------------
// Forward: candle_core::Tensor → legacy::Dlpack / versioned::Dlpack
// ---------------------------------------------------------------------------

/// Maps a candle [`DType`] to the equivalent DLPack dtype, or `None` if
/// unsupported. Symmetric with [`candle_dtype_from_dl`] on the reverse
/// direction — a plain enum-to-enum lookup, `bits` computed from the Rust
/// type's own size where there is a real Rust type to size (bit widths for
/// `F8E4M3` and the sub-byte formats are hardcoded instead — they're fixed
/// by definition, and the dummy placeholder types candle uses for the
/// sub-byte ones have no meaningful `size_of`).
///
/// `F6E2M3`, `F6E3M2`, `F4`, and `F8E8M0` are candle's own `dummy` types
/// (`candle_core::dummy_dtype`) — every *compute* operation on them panics,
/// but `Tensor::from_raw_buffer`/`storage_and_layout` treat them as an opaque
/// packed byte blob, which is all zero-copy DLPack interop needs. Their
/// `bits` (8/6/6/4) are genuinely sub-byte-packed except `F8E8M0`, which is a
/// whole byte per element despite being grouped with the others in candle —
/// callers doing anything other than a compact round-trip on the packed ones
/// should not assume per-element addressability (see the `start_offset`/
/// stride guards in [`dlpack_layout_from_candle`] and
/// [`candle_tensor_from_dlpack`]).
///
/// `DType` is `#[non_exhaustive]`, so a wildcard arm is unavoidable, but every
/// currently-known variant is listed explicitly rather than hidden behind it.
fn dl_dtype_from_candle(dtype: DType) -> Option<DLDataType> {
    let (code, bits) = match dtype {
        DType::U8 => (DLDataTypeCode::UINT, u8::BITS as u8),
        DType::U32 => (DLDataTypeCode::UINT, u32::BITS as u8),
        DType::I16 => (DLDataTypeCode::INT, i16::BITS as u8),
        DType::I32 => (DLDataTypeCode::INT, i32::BITS as u8),
        DType::I64 => (DLDataTypeCode::INT, i64::BITS as u8),
        #[cfg(feature = "half")]
        DType::BF16 => (DLDataTypeCode::BFLOAT, (size_of::<half::bf16>() * 8) as u8),
        #[cfg(feature = "half")]
        DType::F16 => (DLDataTypeCode::FLOAT, (size_of::<half::f16>() * 8) as u8),
        DType::F32 => (DLDataTypeCode::FLOAT, (size_of::<f32>() * 8) as u8),
        DType::F64 => (DLDataTypeCode::FLOAT, (size_of::<f64>() * 8) as u8),
        DType::F8E4M3 => (DLDataTypeCode::FLOAT8_E4M3, 8),
        DType::F8E8M0 => (DLDataTypeCode::FLOAT8_E8M0FNU, 8),
        DType::F6E2M3 => (DLDataTypeCode::FLOAT6_E2M3FN, 6),
        DType::F6E3M2 => (DLDataTypeCode::FLOAT6_E3M2FN, 6),
        DType::F4 => (DLDataTypeCode::FLOAT4_E2M1FN, 4),
        _ => return None,
    };
    Some(DLDataType {
        code,
        bits,
        lanes: 1,
    })
}

struct CandleLayout {
    data_ptr: *mut c_void,
    dtype: DLDataType,
    dims: Vec<i64>,
    strides: Vec<i64>,
}

/// Extracts the raw data pointer, DLPack dtype, dims, and strides for a CPU
/// candle tensor.
///
/// The dtype comes from [`dl_dtype_from_candle`]; the pointer still needs a
/// match on [`CpuStorage`]'s own variants; since the dtype was already
/// validated above, every unhandled variant here is unreachable in practice
/// but is still handled as a defensive error rather than a panic. Drops the
/// internal storage read lock before returning, so the caller is free to box
/// `tensor` itself right after.
fn dlpack_layout_from_candle(tensor: &Tensor) -> Result<CandleLayout, Error> {
    let (storage, layout) = tensor.storage_and_layout();
    let Storage::Cpu(cpu_storage) = &*storage else {
        return Err(Error::UnsupportedDevice);
    };

    let dtype = dl_dtype_from_candle(cpu_storage.dtype()).ok_or(Error::UnsupportedCandleDType {
        dtype: cpu_storage.dtype(),
    })?;

    let base_ptr = match cpu_storage {
        CpuStorage::U8(v) => v.as_ptr() as *mut c_void,
        CpuStorage::U32(v) => v.as_ptr() as *mut c_void,
        CpuStorage::I16(v) => v.as_ptr() as *mut c_void,
        CpuStorage::I32(v) => v.as_ptr() as *mut c_void,
        CpuStorage::I64(v) => v.as_ptr() as *mut c_void,
        #[cfg(feature = "half")]
        CpuStorage::BF16(v) => v.as_ptr() as *mut c_void,
        #[cfg(feature = "half")]
        CpuStorage::F16(v) => v.as_ptr() as *mut c_void,
        CpuStorage::F32(v) => v.as_ptr() as *mut c_void,
        CpuStorage::F64(v) => v.as_ptr() as *mut c_void,
        CpuStorage::F8E4M3(v) => v.as_ptr() as *mut c_void,
        CpuStorage::F8E8M0(v) => v.as_ptr() as *mut c_void,
        CpuStorage::F6E2M3(v) => v.as_ptr() as *mut c_void,
        CpuStorage::F6E3M2(v) => v.as_ptr() as *mut c_void,
        CpuStorage::F4(v) => v.as_ptr() as *mut c_void,
        // Only reachable without the `half` feature (BF16/F16 have no arms
        // above then) — with `half` on, every `CpuStorage` variant is
        // already covered explicitly.
        #[cfg(not(feature = "half"))]
        other => {
            return Err(Error::UnsupportedCandleDType {
                dtype: other.dtype(),
            });
        }
    };
    // start_offset() is in elements. For byte-aligned dtypes (bits a
    // multiple of 8) that converts cleanly to a byte offset via
    // element_size(). For genuinely sub-byte packed dtypes (bits < 8) a
    // nonzero element offset would land mid-byte, which plain pointer
    // arithmetic can't express — reject rather than silently mis-offset.
    if dtype.bits < 8 && layout.start_offset() != 0 {
        return Err(Error::SubByteOffsetUnsupported { dtype });
    }
    let data_ptr =
        unsafe { (base_ptr as *mut u8).add(layout.start_offset() * dtype.element_size()) }
            as *mut c_void;

    let dims = layout
        .dims()
        .iter()
        .map(|&d| i64::try_from(d).map_err(|_| Error::DimensionOverflow))
        .collect::<Result<Vec<_>, _>>()?;
    let strides = layout
        .stride()
        .iter()
        .map(|&s| i64::try_from(s).map_err(|_| Error::DimensionOverflow))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(CandleLayout {
        data_ptr,
        dtype,
        dims,
        strides,
    })
}

/// Converts a boxed CPU [`Tensor`] into an initialized DLPack allocation.
///
/// # Errors
///
/// - [`Error::UnsupportedDevice`] if the tensor is not on CPU.
/// - [`Error::UnsupportedCandleDType`] if `dl_dtype_from_candle` has no
///   mapping for the tensor's dtype (currently the packed sub-byte float
///   formats, and `BF16`/`F16` when the `half` feature is disabled).
impl<M: ManagedTensorBase> TryFrom<Box<Tensor>> for dynamic::Initialized<M> {
    type Error = Error;

    fn try_from(tensor: Box<Tensor>) -> Result<Self, Self::Error> {
        if !tensor.device().is_cpu() {
            return Err(Error::UnsupportedDevice);
        }
        let CandleLayout {
            data_ptr,
            dtype,
            dims,
            strides,
        } = dlpack_layout_from_candle(&tensor)?;
        let prepared = Dynamic::new(Copied(dims), Copied(strides)).prepare::<M>()?;
        let mut initialized = prepared
            .initialize(tensor)
            .map_err(crate::metadata::Error::from)?;
        initialized.set_data(data_ptr);
        initialized.set_dtype(dtype);
        initialized.set_device(DLDevice::CPU);
        Ok(initialized)
    }
}

// ---------------------------------------------------------------------------
// Reverse: &Local<M> → candle_core::Tensor
// ---------------------------------------------------------------------------

/// Maps a DLPack dtype to the equivalent candle [`DType`], or `None` if
/// unsupported. `DLDataType` is a `(code, bits, lanes)` triple rather than a
/// Rust `enum`, but only a fixed, known set of combinations is actually
/// supported — matching on `(code, bits)` makes that enumeration explicit,
/// a plain enum-to-enum mapping with no generic dispatch involved.
fn candle_dtype_from_dl(dtype: DLDataType) -> Option<DType> {
    match (dtype.code, dtype.bits) {
        (DLDataTypeCode::UINT, 8) => Some(DType::U8),
        (DLDataTypeCode::UINT, 32) => Some(DType::U32),
        (DLDataTypeCode::INT, 16) => Some(DType::I16),
        (DLDataTypeCode::INT, 32) => Some(DType::I32),
        (DLDataTypeCode::INT, 64) => Some(DType::I64),
        #[cfg(feature = "half")]
        (DLDataTypeCode::BFLOAT, 16) => Some(DType::BF16),
        #[cfg(feature = "half")]
        (DLDataTypeCode::FLOAT, 16) => Some(DType::F16),
        (DLDataTypeCode::FLOAT, 32) => Some(DType::F32),
        (DLDataTypeCode::FLOAT, 64) => Some(DType::F64),
        (DLDataTypeCode::FLOAT8_E4M3, 8) => Some(DType::F8E4M3),
        (DLDataTypeCode::FLOAT8_E8M0FNU, 8) => Some(DType::F8E8M0),
        (DLDataTypeCode::FLOAT6_E2M3FN, 6) => Some(DType::F6E2M3),
        (DLDataTypeCode::FLOAT6_E3M2FN, 6) => Some(DType::F6E3M2),
        (DLDataTypeCode::FLOAT4_E2M1FN, 4) => Some(DType::F4),
        _ => None,
    }
}

/// Validates a strided index grid and returns its total element count.
///
/// `shape` must be non-negative and `strides` may be negative (DLPack permits
/// both). The total element count is computed with overflow checking. The
/// byte span covered by the index grid — `[min_elem, max_elem] × elem_size`,
/// taking per-axis min/max contributions so negative strides are handled —
/// must lie entirely within `[0, num_bytes)` of the base pointer, otherwise
/// [`Error::StridedSpanOverflow`] is returned. This is what makes the raw
/// pointer arithmetic in [`gather_strided_bytes`] sound.
fn validate_strided_span(
    shape: &[i64],
    strides: &[i64],
    elem_size: usize,
    num_bytes: usize,
) -> Result<usize, Error> {
    let mut total: usize = 1;
    let mut min_elem: i128 = 0;
    let mut max_elem: i128 = 0;
    for (&d, &s) in shape.iter().zip(strides) {
        if d < 0 {
            return Err(Error::Tensor {
                source: crate::tensor::Error::NegativeDimension { axis: 0, value: d },
            });
        }
        total = total.checked_mul(d as usize).ok_or(Error::Tensor {
            source: crate::tensor::Error::NumElementsOverflow,
        })?;
        if d == 0 {
            continue;
        }
        let last = (d - 1) as i128;
        let s = s as i128;
        let (lo, hi) = if s >= 0 { (0, last * s) } else { (last * s, 0) };
        min_elem += lo;
        max_elem += hi;
    }
    let min_byte = min_elem
        .checked_mul(elem_size as i128)
        .ok_or(Error::StridedSpanOverflow)?;
    let max_byte = max_elem
        .checked_add(1)
        .and_then(|m| m.checked_mul(elem_size as i128))
        .ok_or(Error::StridedSpanOverflow)?;
    if min_byte < 0 || max_byte > num_bytes as i128 {
        return Err(Error::StridedSpanOverflow);
    }
    Ok(total)
}

/// Gathers a strided tensor into a fresh contiguous (row-major) byte buffer.
///
/// Works purely in bytes — `elem_size` (from the DLPack dtype itself) stands
/// in for a generic element type, so this needs no `T` at all. `ptr` must
/// already be adjusted for `byte_offset`; `strides` are in elements, matching
/// DLPack's convention. `total` is the pre-validated element count from
/// [`validate_strided_span`], which also proved every accessed byte lies
/// within the source buffer, so the raw `ptr.offset` below stays in bounds
/// even for negative strides.
fn gather_strided_bytes(
    ptr: *const u8,
    elem_size: usize,
    shape: &[i64],
    strides: &[i64],
    total: usize,
) -> Vec<u8> {
    let ndim = shape.len();
    let mut out = Vec::with_capacity(total * elem_size);
    let mut idx = vec![0i64; ndim];

    for _ in 0..total {
        let elem_offset: i64 = idx.iter().zip(strides).map(|(&i, &s)| i * s).sum();
        let byte_offset = elem_offset as isize * elem_size as isize;
        unsafe {
            let src = ptr.offset(byte_offset);
            out.extend_from_slice(std::slice::from_raw_parts(src, elem_size));
        }

        for axis in (0..ndim).rev() {
            idx[axis] += 1;
            if idx[axis] < shape[axis] {
                break;
            }
            idx[axis] = 0;
        }
    }

    out
}

/// Converts a DLPack tensor into an owned, contiguous CPU [`Tensor`].
///
/// Always copies: candle has no borrowed/strided CPU tensor view type.
/// Compact-stride sources take a fast bulk-copy path; arbitrary strides are
/// gathered into a fresh contiguous buffer. Both paths build a plain `Vec<u8>`
/// and hand it to candle's own [`Tensor::from_raw_buffer`] (the same
/// byte+dtype-enum-driven constructor candle's safetensors loader uses
/// internally) — no generic element type appears anywhere in this function.
///
/// # Errors
///
/// - [`Error::UnsupportedDlDataType`] if the DLPack tensor's dtype has no
///   `candle_dtype_from_dl` mapping.
/// - Propagates [`crate::tensor::Error`] for device/null/offset issues (e.g.
///   the source tensor is not on CPU).
pub fn candle_tensor_from_dlpack<M: ManagedTensorBase>(dlpack: &Local<M>) -> Result<Tensor, Error> {
    let tensor = dlpack.tensor();
    let dl_dtype = tensor.dtype;
    let dtype =
        candle_dtype_from_dl(dl_dtype).ok_or(Error::UnsupportedDlDataType { dtype: dl_dtype })?;

    let shape = unsafe { tensor.shape()? };
    let strides = unsafe { tensor.strides()? };
    let ptr = unsafe { tensor.offset_bytes_ptr()? };

    let compact = match strides {
        None => true,
        Some(s) => crate::tensor::is_compact_strides(shape, Some(s))?,
    };
    let bytes: Vec<u8> = if compact {
        unsafe { std::slice::from_raw_parts(ptr, tensor.num_bytes()?) }.to_vec()
    } else if dl_dtype.bits < 8 {
        // gather_strided_bytes copies whole bytes per element; a sub-byte
        // packed dtype has no single-element byte to copy.
        return Err(Error::SubByteStridesUnsupported { dtype: dl_dtype });
    } else {
        let s = strides.unwrap();
        let num_bytes = unsafe { tensor.num_bytes()? };
        let total = validate_strided_span(shape, s, dl_dtype.element_size(), num_bytes)?;
        gather_strided_bytes(ptr, dl_dtype.element_size(), shape, s, total)
    };

    let dims = shape
        .iter()
        .map(|&d| usize::try_from(d).map_err(|_| Error::DimensionOverflow))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(Tensor::from_raw_buffer(&bytes, dtype, &dims, &Device::Cpu)?)
}

impl<'a, M> TryFrom<&'a Local<M>> for Tensor
where
    M: ManagedTensorBase,
{
    type Error = Error;

    fn try_from(dlpack: &'a Local<M>) -> Result<Self, Self::Error> {
        candle_tensor_from_dlpack(dlpack)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::{DLDataTypeCode, DLManagedTensor, DLManagedTensorVersioned};
    use crate::{DlpackElement, DlpackFlags, Local, legacy, test_support, versioned};

    fn managed_candle<M: ManagedTensorBase>(tensor: Tensor) -> Local<M> {
        let initialized: dynamic::Initialized<M> = Box::new(tensor).try_into().unwrap();
        test_support::managed(initialized)
    }

    fn managed_candle_with_flags<M: ManagedTensorBase>(
        tensor: Tensor,
        flags: DlpackFlags,
    ) -> Local<M> {
        let mut initialized: dynamic::Initialized<M> = Box::new(tensor).try_into().unwrap();
        initialized.set_flags(flags).unwrap();
        test_support::managed(initialized)
    }

    fn raw_tensor<T, const N: usize>(
        data: Vec<T>,
        dtype: DLDataType,
        shape: [i64; N],
        strides: [i64; N],
    ) -> Local<DLManagedTensor>
    where
        T: Send + 'static,
    {
        let data = Box::new(data);
        let data_ptr = data.as_ptr().cast_mut().cast();
        test_support::fixed_local(
            data,
            data_ptr,
            dtype,
            DLDevice::CPU,
            shape,
            strides,
            DlpackFlags::empty(),
        )
    }

    #[test]
    fn candle_tensor_to_legacy_dlpack_keeps_layout_and_data() {
        let tensor = Tensor::from_vec(vec![1i32, 2, 3, 4, 5, 6], (2, 3), &Device::Cpu).unwrap();
        let dlpack: legacy::Dlpack = managed_candle(tensor);

        assert_eq!(dlpack.shape().unwrap(), &[2, 3]);
        assert_eq!(dlpack.strides().unwrap().unwrap(), &[3, 1]);
        assert_eq!(
            unsafe { dlpack.tensor().cpu_slice::<i32>() }.unwrap(),
            &[1, 2, 3, 4, 5, 6]
        );
    }

    #[test]
    fn candle_builder_defaults_to_empty_flags() {
        let tensor = Tensor::from_vec(vec![1f32, 2., 3., 4.], (2, 2), &Device::Cpu).unwrap();
        let dlpack: versioned::Dlpack = managed_candle(tensor);

        assert_eq!(dlpack.flags(), DlpackFlags::empty());
        assert_eq!(
            unsafe { dlpack.tensor().cpu_slice::<f32>() }.unwrap(),
            &[1., 2., 3., 4.]
        );
    }

    #[test]
    fn candle_tensor_to_versioned_builder_allows_flags_before_build() {
        let tensor = Tensor::from_vec(vec![1f32, 2., 3., 4.], (2, 2), &Device::Cpu).unwrap();
        let dlpack: Local<DLManagedTensorVersioned> =
            managed_candle_with_flags(tensor, DlpackFlags::READ_ONLY);

        assert_eq!(dlpack.flags(), DlpackFlags::READ_ONLY);
        assert_eq!(
            unsafe { dlpack.tensor().cpu_slice::<f32>() }.unwrap(),
            &[1., 2., 3., 4.]
        );
    }

    #[test]
    fn candle_f8e4m3_tensor_roundtrips_dtype_and_shape() {
        // F8E4M3 is real (not a `dummy` candle type), but has no easy
        // Rust-side value comparisons without the separate `float8` crate —
        // so this only checks shape/dtype/byte-count, not element values.
        let tensor = Tensor::zeros((2, 3), DType::F8E4M3, &Device::Cpu).unwrap();
        let dlpack: legacy::Dlpack = managed_candle(tensor);

        assert_eq!(dlpack.shape().unwrap(), &[2, 3]);
        let dtype = dlpack.tensor().dtype;
        assert_eq!(dtype.code, DLDataTypeCode::FLOAT8_E4M3);
        assert_eq!(dtype.bits, 8);
        assert_eq!(dlpack.num_bytes().unwrap(), 6);
    }

    #[test]
    fn candle_f4_tensor_roundtrips_via_from_raw_buffer() {
        // F4 is one of candle's `dummy` sub-byte types: `Tensor::zeros` refuses
        // it (candle itself can't guess a packed byte count), but
        // `from_raw_buffer` accepts caller-supplied packed bytes directly.
        // 6 elements * 4 bits = 24 bits = 3 bytes exactly, no padding.
        let tensor =
            Tensor::from_raw_buffer(&[0xAB, 0xCD, 0xEF], DType::F4, &[6], &Device::Cpu).unwrap();
        let dlpack: legacy::Dlpack = managed_candle(tensor);

        assert_eq!(dlpack.shape().unwrap(), &[6]);
        let dtype = dlpack.tensor().dtype;
        assert_eq!(dtype.code, DLDataTypeCode::FLOAT4_E2M1FN);
        assert_eq!(dtype.bits, 4);
        // This is the whole point of the num_bytes() fix: 6 * 4 bits = 3
        // bytes, not 6 (which is what per-element rounding would have given).
        assert_eq!(dlpack.num_bytes().unwrap(), 3);
    }

    #[test]
    fn dlpack_f4_converts_to_candle_tensor_with_matching_dtype() {
        let dlpack = raw_tensor(
            vec![0xABu8, 0xCD, 0xEF],
            DLDataType {
                code: DLDataTypeCode::FLOAT4_E2M1FN,
                bits: 4,
                lanes: 1,
            },
            [6],
            [1],
        );

        let tensor = Tensor::try_from(&dlpack).unwrap();

        assert_eq!(tensor.dims(), &[6]);
        assert_eq!(tensor.dtype(), DType::F4);
    }

    #[test]
    fn non_compact_sub_byte_packed_dlpack_is_rejected() {
        // [3, 2] compact strides would be [2, 1]; [1, 3] is a (nonsensical
        // but sufficient) non-compact stride to exercise the rejection path.
        let dlpack = raw_tensor(
            vec![0xABu8, 0xCD, 0xEF],
            DLDataType {
                code: DLDataTypeCode::FLOAT4_E2M1FN,
                bits: 4,
                lanes: 1,
            },
            [3, 2],
            [1, 3],
        );

        let err = Tensor::try_from(&dlpack).unwrap_err();
        assert!(matches!(err, Error::SubByteStridesUnsupported { .. }));
    }

    #[test]
    fn compact_dlpack_to_candle_tensor_copies_values() {
        let dlpack = raw_tensor(
            vec![1i32, 2, 3, 4, 5, 6],
            <i32 as DlpackElement>::DTYPE,
            [2, 3],
            [3, 1],
        );

        let tensor = Tensor::try_from(&dlpack).unwrap();

        assert_eq!(tensor.dims(), &[2, 3]);
        assert_eq!(
            tensor.flatten_all().unwrap().to_vec1::<i32>().unwrap(),
            vec![1, 2, 3, 4, 5, 6]
        );
    }

    #[test]
    fn non_compact_strided_dlpack_to_candle_tensor_gathers_in_row_major_order() {
        // A 2x3 row-major buffer [1..6] viewed transposed as a 3x2 tensor via
        // strides — not compact for shape [3, 2].
        let dlpack = raw_tensor(
            vec![1i32, 2, 3, 4, 5, 6],
            <i32 as DlpackElement>::DTYPE,
            [3, 2],
            [1, 3],
        );

        let tensor = Tensor::try_from(&dlpack).unwrap();

        assert_eq!(tensor.dims(), &[3, 2]);
        assert_eq!(
            tensor.flatten_all().unwrap().to_vec1::<i32>().unwrap(),
            vec![1, 4, 2, 5, 3, 6]
        );
    }

    #[test]
    fn non_compact_dlpack_with_out_of_bounds_stride_is_rejected() {
        // Stride 10 on a 6-element buffer: index [1, 0] reaches element 10,
        // past the end. Must be rejected rather than read out of bounds.
        let dlpack = raw_tensor(
            vec![1i32, 2, 3, 4, 5, 6],
            <i32 as DlpackElement>::DTYPE,
            [2, 2],
            [10, 1],
        );

        let err = Tensor::try_from(&dlpack).unwrap_err();
        assert!(matches!(err, Error::StridedSpanOverflow));
    }

    #[test]
    fn non_compact_dlpack_with_negative_stride_is_rejected_as_out_of_bounds() {
        // With byte_offset 0 the data pointer is the allocation base, so any
        // negative stride indexes before the buffer. Rejected rather than read
        // out of bounds.
        let dlpack = raw_tensor(
            vec![1i32, 2, 3, 4, 5, 6],
            <i32 as DlpackElement>::DTYPE,
            [3, 2],
            [-1, -3],
        );

        let err = Tensor::try_from(&dlpack).unwrap_err();
        assert!(matches!(err, Error::StridedSpanOverflow));
    }

    #[test]
    fn dlpack_f8e4m3_converts_to_candle_tensor_with_matching_dtype() {
        let dlpack = raw_tensor(
            vec![0u8; 6],
            DLDataType {
                code: DLDataTypeCode::FLOAT8_E4M3,
                bits: 8,
                lanes: 1,
            },
            [2, 3],
            [3, 1],
        );

        let tensor = Tensor::try_from(&dlpack).unwrap();

        assert_eq!(tensor.dims(), &[2, 3]);
        assert_eq!(tensor.dtype(), DType::F8E4M3);
    }

    #[test]
    fn dlpack_with_unmatched_dtype_is_rejected() {
        let dlpack = raw_tensor(
            vec![0u8; 3],
            DLDataType {
                code: DLDataTypeCode(99),
                bits: 1,
                lanes: 1,
            },
            [3],
            [1],
        );

        let err = Tensor::try_from(&dlpack).unwrap_err();
        assert!(matches!(err, Error::UnsupportedDlDataType { .. }));
    }
}
