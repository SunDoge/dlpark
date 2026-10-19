use super::Error;
use crate::{
    Managed, ManagedTensorBase, TryFromDlpack,
    ffi::{DLDataType, DLDataTypeCode},
};
use candle_core::{DType, Device, Tensor};

/// Maps a DLPack dtype to the equivalent candle [`DType`], or `None` if
/// unsupported. `DLDataType` is a `(code, bits, lanes)` triple rather than a
/// Rust `enum`, but only a fixed, known set of combinations is actually
/// supported — matching on `(code, bits)` makes that enumeration explicit,
/// a plain enum-to-enum mapping with no generic dispatch involved.
fn candle_dtype_from_dl(dtype: DLDataType) -> Option<DType> {
    if dtype.lanes != 1 {
        return None;
    }
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

/// Checks that every byte displacement and the complete strided span fit
/// Rust pointer-offset limits. Allocation bounds remain the caller's obligation:
/// DLPack reports a logical element count, not the backing allocation length.
fn validate_strided_span(shape: &[i64], strides: &[i64], elem_size: usize) -> Result<(), Error> {
    if shape.contains(&0) {
        return Ok(());
    }
    let mut min_elem = 0_i128;
    let mut max_elem = 0_i128;
    for (&dim, &stride) in shape.iter().zip(strides) {
        let last = i128::from(dim - 1) * i128::from(stride);
        min_elem = min_elem
            .checked_add(last.min(0))
            .ok_or(Error::StridedSpanOverflow)?;
        max_elem = max_elem
            .checked_add(last.max(0))
            .ok_or(Error::StridedSpanOverflow)?;
    }
    let min_byte = min_elem
        .checked_mul(elem_size as i128)
        .ok_or(Error::StridedSpanOverflow)?;
    let max_byte = max_elem
        .checked_mul(elem_size as i128)
        .ok_or(Error::StridedSpanOverflow)?;
    let span = max_byte
        .checked_sub(min_byte)
        .and_then(|span| span.checked_add(elem_size as i128))
        .ok_or(Error::StridedSpanOverflow)?;
    if min_byte < isize::MIN as i128 || max_byte > isize::MAX as i128 || span > isize::MAX as i128 {
        return Err(Error::StridedSpanOverflow);
    }
    Ok(())
}

/// Gathers a strided tensor into a fresh contiguous (row-major) byte buffer.
///
/// # Safety
///
/// The index grid must pass `validate_strided_span`, and every addressed element
/// must be readable within the same allocation as `ptr`.
unsafe fn gather_strided_bytes(
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
        let elem_offset: i128 = idx
            .iter()
            .zip(strides)
            .map(|(&i, &s)| i128::from(i) * i128::from(s))
            .sum();
        // The validated displacement range fits isize, including negative strides.
        let byte_offset = (elem_offset * elem_size as i128) as isize;
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
/// # Safety
///
/// The descriptor and all data and metadata pointers it references
/// must be valid and readable for the duration of this conversion. Every
/// element addressed by shape and strides, including negative displacements
/// from the byte-offset-adjusted pointer, must lie within its backing allocation.
pub unsafe fn candle_tensor_from_dlpack<M: ManagedTensorBase>(
    dlpack: &Managed<M>,
) -> Result<Tensor, Error> {
    let tensor = dlpack.validate()?;
    if tensor.device().device_type != crate::ffi::DLDeviceType::CPU {
        return Err(crate::tensor::Error::NotCpu {
            device_type: tensor.device().device_type,
        }
        .into());
    }
    let dl_dtype = tensor.dtype();
    let dtype =
        candle_dtype_from_dl(dl_dtype).ok_or(Error::UnsupportedDlDataType { dtype: dl_dtype })?;

    let shape = tensor.shape();
    let strides = tensor.strides();
    let ptr = unsafe { tensor.offset_bytes_ptr()? };

    let compact = match strides {
        None => true,
        Some(s) => crate::tensor::is_compact_strides(shape, Some(s))?,
    };
    let bytes: Vec<u8> = if compact {
        unsafe { std::slice::from_raw_parts(ptr, tensor.num_bytes()) }.to_vec()
    } else if dl_dtype.bits < 8 {
        // gather_strided_bytes copies whole bytes per element; a sub-byte
        // packed dtype has no single-element byte to copy.
        return Err(Error::SubByteStridesUnsupported { dtype: dl_dtype });
    } else {
        let s = strides.unwrap();
        validate_strided_span(shape, s, dl_dtype.element_size())?;
        unsafe {
            gather_strided_bytes(
                ptr,
                dl_dtype.element_size(),
                shape,
                s,
                tensor.num_elements(),
            )
        }
    };

    let dims = shape
        .iter()
        .map(|&d| usize::try_from(d).map_err(|_| Error::DimensionOverflow))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(Tensor::from_raw_buffer(&bytes, dtype, &dims, &Device::Cpu)?)
}

impl<'a, M> TryFromDlpack<&'a Managed<M>, ()> for Tensor
where
    M: ManagedTensorBase,
{
    type Error = Error;

    unsafe fn try_from_dlpack(dlpack: &'a Managed<M>, _context: ()) -> Result<Self, Self::Error> {
        unsafe { candle_tensor_from_dlpack(dlpack) }
    }
}
