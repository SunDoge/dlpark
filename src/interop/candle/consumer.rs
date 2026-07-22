use super::Error;
use crate::{
    Foreign, ManagedTensorBase, TryFromDlpack,
    ffi::{DLDataType, DLDataTypeCode},
};
use candle_core::{DType, Device, Tensor};

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
/// # Safety
///
/// The foreign descriptor and all data and metadata pointers it references
/// must be valid and readable for the duration of this conversion.
pub unsafe fn candle_tensor_from_dlpack<M: ManagedTensorBase>(
    dlpack: &Foreign<M>,
) -> Result<Tensor, Error> {
    let tensor = unsafe { dlpack.tensor() };
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

impl<'a, M> TryFromDlpack<&'a Foreign<M>> for Tensor
where
    M: ManagedTensorBase,
{
    type Error = Error;

    unsafe fn try_from_dlpack(dlpack: &'a Foreign<M>) -> Result<Self, Self::Error> {
        unsafe { candle_tensor_from_dlpack(dlpack) }
    }
}
