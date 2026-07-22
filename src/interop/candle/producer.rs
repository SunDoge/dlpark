use super::Error;
use crate::{
    ManagedTensorBase,
    allocation::dynamic,
    ffi::{DLDataType, DLDataTypeCode, DLDevice},
    metadata::{Copied, Dynamic},
};
use candle_core::{DType, Storage, Tensor, backend::BackendStorage, cpu_backend::CpuStorage};
use std::{mem::size_of, os::raw::c_void};

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
