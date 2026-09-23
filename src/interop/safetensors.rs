//! Zero-copy interoperability with the safetensors file format.
//!
//! [`SafeTensorFile`](crate::interop::safetensors::SafeTensorFile) owns either in-memory file bytes or a read-only memory
//! map. Each exported DLPack tensor retains the shared file owner and carries
//! [`DlpackFlags::READ_ONLY`](crate::DlpackFlags::READ_ONLY).
//! [`DlpackView`](crate::interop::safetensors::DlpackView) provides the reverse direction:
//! a compact CPU DLPack tensor can be passed directly to
//! [`::safetensors::serialize`] or [`::safetensors::serialize_to_file`].

use crate::{
    DlpackFlags, Managed, ManagedTensorBase,
    ffi::{DLDataType, DLDevice, DLManagedTensorVersioned},
    metadata::Dynamic,
    tensor::TensorRef,
    versioned,
};
use memmap2::{Mmap, MmapOptions};
use safetensors::{Dtype, SafeTensorError, SafeTensors, View};
use snafu::Snafu;
use std::{
    borrow::Cow,
    fs::File,
    path::{Path, PathBuf},
    ptr::NonNull,
    sync::Arc,
};

/// Errors raised while converting between safetensors and DLPack.
#[derive(Debug, Snafu)]
pub enum Error {
    /// The safetensors file or byte buffer is invalid.
    #[snafu(transparent)]
    Safetensors {
        /// The underlying safetensors error.
        source: SafeTensorError,
    },

    /// DLPack shape or stride metadata could not be allocated or represented.
    #[snafu(transparent)]
    Metadata {
        /// The underlying metadata error.
        source: crate::metadata::Error,
    },

    /// The source DLPack tensor is invalid or is not a compact CPU tensor.
    #[snafu(transparent)]
    Tensor {
        /// The underlying DLPack validation error.
        source: crate::tensor::Error,
    },

    /// Opening or mapping a safetensors file failed.
    #[snafu(display("failed to open or map {}: {source}", path.display()))]
    File {
        /// File path being opened.
        path: PathBuf,
        /// The underlying I/O error.
        source: std::io::Error,
    },

    /// A safetensors dtype added by a newer release has no known DLPack mapping.
    #[snafu(display("safetensors dtype {dtype:?} has no known DLPack mapping"))]
    UnsupportedSafetensorsDtype {
        /// Unsupported safetensors dtype.
        dtype: Dtype,
    },

    /// The DLPack dtype cannot be represented by safetensors.
    #[snafu(display("DLPack dtype {dtype:?} cannot be represented by safetensors"))]
    UnsupportedDlpackDtype {
        /// Unsupported DLPack dtype.
        dtype: DLDataType,
    },

    /// A DLPack shape dimension does not fit the safetensors `usize` shape.
    #[snafu(display("DLPack shape value at axis {axis} does not fit usize: {value}"))]
    DimensionOverflow {
        /// Axis containing the invalid dimension.
        axis: usize,
        /// Dimension value that could not be converted.
        value: i64,
    },

    /// Safetensors stores tightly packed sub-byte values and cannot preserve padding.
    #[snafu(display("safetensors cannot represent per-element padding in a sub-byte tensor"))]
    PaddedSubByte,

    /// The backing byte allocation does not meet the tensor dtype's alignment.
    #[snafu(display(
        "safetensors tensor {name:?} starts at {ptr:#x}, which is not aligned to {align} bytes"
    ))]
    MisalignedData {
        /// Tensor name being exported.
        name: String,
        /// Address of the tensor data.
        ptr: usize,
        /// Required alignment in bytes.
        align: usize,
    },

    /// Safetensors bytes are little-endian while DLPack uses native endianness.
    #[snafu(display("zero-copy safetensors interop requires a little-endian target"))]
    BigEndian,
}

enum Storage {
    Bytes(Box<[u8]>),
    Mmap(Mmap),
}

impl Storage {
    fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Bytes(bytes) => bytes,
            Self::Mmap(mmap) => mmap,
        }
    }
}

struct Inner {
    storage: Storage,
    metadata: safetensors::tensor::Metadata,
    data_offset: usize,
}

/// An owned safetensors file that can export named tensors without copying.
///
/// Clones share the same bytes and parsed metadata. A DLPack tensor produced by
/// [`Self::tensor`] keeps this storage alive independently of the file handle.
#[derive(Clone)]
pub struct SafeTensorFile {
    inner: Arc<Inner>,
}

impl SafeTensorFile {
    /// Parses an owned safetensors byte buffer.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, Error> {
        Self::from_boxed_bytes(bytes.into_boxed_slice())
    }

    /// Parses owned safetensors bytes without copying their contents.
    pub fn from_boxed_bytes(bytes: Box<[u8]>) -> Result<Self, Error> {
        Self::from_storage(Storage::Bytes(bytes))
    }

    /// Opens a safetensors file through a read-only memory map.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|source| Error::File {
            path: path.to_owned(),
            source,
        })?;
        let mmap = unsafe { MmapOptions::new().map_copy_read_only(&file) }.map_err(|source| {
            Error::File {
                path: path.to_owned(),
                source,
            }
        })?;
        Self::from_storage(Storage::Mmap(mmap))
    }

    fn from_storage(storage: Storage) -> Result<Self, Error> {
        ensure_little_endian()?;
        let (header_len, metadata) = SafeTensors::read_metadata(storage.as_bytes())?;
        let data_offset = 8usize
            .checked_add(header_len)
            .ok_or(SafeTensorError::InvalidHeaderLength)?;
        Ok(Self {
            inner: Arc::new(Inner {
                storage,
                metadata,
                data_offset,
            }),
        })
    }

    /// Returns tensor names in file-offset order.
    pub fn names(&self) -> Vec<String> {
        self.inner.metadata.offset_keys()
    }

    /// Returns the parsed safetensors metadata.
    pub fn metadata(&self) -> &safetensors::tensor::Metadata {
        &self.inner.metadata
    }

    /// Exports a named tensor through the versioned DLPack ABI without copying.
    ///
    /// The returned tensor is read-only because its storage may be backed by a
    /// read-only memory map. Shape and compact row-major strides are copied
    /// into the managed allocation. The export fails if an in-memory byte
    /// allocation does not satisfy the dtype's natural alignment.
    pub fn tensor(&self, name: &str) -> Result<versioned::Dlpack, Error> {
        let info = self
            .inner
            .metadata
            .info(name)
            .ok_or_else(|| SafeTensorError::TensorNotFound(name.to_owned()))?;
        let dtype = dlpack_dtype(info.dtype)?;
        let start = self
            .inner
            .data_offset
            .checked_add(info.data_offsets.0)
            .ok_or(SafeTensorError::InvalidHeaderLength)?;
        let nbytes = info.data_offsets.1 - info.data_offsets.0;
        let data = if nbytes == 0 {
            NonNull::<u64>::dangling().as_ptr().cast()
        } else {
            let data = unsafe { self.inner.storage.as_bytes().as_ptr().add(start) };
            let align = dtype_alignment(info.dtype);
            if !(data as usize).is_multiple_of(align) {
                return Err(Error::MisalignedData {
                    name: name.to_owned(),
                    ptr: data as usize,
                    align,
                });
            }
            data.cast_mut().cast()
        };

        let prepared =
            Dynamic::compact(info.shape.as_slice()).prepare::<DLManagedTensorVersioned>()?;
        let mut initialized = prepared.initialize(Arc::clone(&self.inner));
        initialized
            .set_data(data)
            .set_device(DLDevice::CPU)
            .set_dtype(dtype)
            .set_flags(DlpackFlags::READ_ONLY)
            .expect("READ_ONLY never asserts copy ownership");
        Ok(unsafe { initialized.finish() })
    }
}

/// A borrowed safetensors serialization view over a compact CPU DLPack tensor.
///
/// Construct this with [`DlpackView::from_dlpack`], then pass it to
/// [`safetensors::serialize`] or [`safetensors::serialize_to_file`].
pub struct DlpackView<'a> {
    dtype: Dtype,
    shape: Vec<usize>,
    data: &'a [u8],
}

impl<'a> DlpackView<'a> {
    /// Borrows a compact CPU DLPack tensor as a safetensors view.
    ///
    /// # Safety
    ///
    /// The DLPack data pointer, after applying `byte_offset`, must cover all
    /// `num_bytes()` initialized bytes reported by the descriptor. The managed
    /// tensor's ownership contract must keep those bytes alive and immutable
    /// for the returned view's lifetime.
    pub unsafe fn from_dlpack<M>(dlpack: &'a Managed<M>) -> Result<Self, Error>
    where
        M: ManagedTensorBase,
    {
        ensure_little_endian()?;
        if dlpack.flags().contains(DlpackFlags::IS_SUBBYTE_TYPE_PADDED) {
            return Err(Error::PaddedSubByte);
        }
        let tensor = dlpack.validate()?;
        unsafe { Self::from_tensor_ref(tensor) }
    }

    unsafe fn from_tensor_ref(tensor: TensorRef<'a>) -> Result<Self, Error> {
        let dtype = safetensors_dtype(tensor.dtype())?;
        let shape = tensor
            .shape()
            .iter()
            .copied()
            .enumerate()
            .map(|(axis, value)| {
                usize::try_from(value).map_err(|_| Error::DimensionOverflow { axis, value })
            })
            .collect::<Result<_, _>>()?;
        let data = unsafe { tensor.cpu_bytes()? };
        Ok(Self { dtype, shape, data })
    }
}

impl View for DlpackView<'_> {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(self.data)
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

impl View for &DlpackView<'_> {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(self.data)
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

fn ensure_little_endian() -> Result<(), Error> {
    if cfg!(target_endian = "little") {
        Ok(())
    } else {
        Err(Error::BigEndian)
    }
}

fn dlpack_dtype(dtype: Dtype) -> Result<DLDataType, Error> {
    Ok(match dtype {
        Dtype::BOOL => DLDataType::BOOL,
        Dtype::F4 => DLDataType::F4E2M1FN,
        Dtype::F6_E2M3 => DLDataType::F6E2M3FN,
        Dtype::F6_E3M2 => DLDataType::F6E3M2FN,
        Dtype::U8 => DLDataType::U8,
        Dtype::I8 => DLDataType::I8,
        Dtype::F8_E5M2 => DLDataType::F8E5M2,
        // Safetensors' historical F8_E4M3 spelling denotes the finite-only
        // format exposed by PyTorch as float8_e4m3fn.
        Dtype::F8_E4M3 => DLDataType::F8E4M3FN,
        Dtype::F8_E8M0 => DLDataType::F8E8M0FNU,
        Dtype::F8_E4M3FNUZ => DLDataType::F8E4M3FNUZ,
        Dtype::F8_E5M2FNUZ => DLDataType::F8E5M2FNUZ,
        Dtype::I16 => DLDataType::I16,
        Dtype::U16 => DLDataType::U16,
        Dtype::F16 => DLDataType::F16,
        Dtype::BF16 => DLDataType::BF16,
        Dtype::I32 => DLDataType::I32,
        Dtype::U32 => DLDataType::U32,
        Dtype::F32 => DLDataType::F32,
        Dtype::C64 => DLDataType::C64,
        Dtype::F64 => DLDataType::F64,
        Dtype::I64 => DLDataType::I64,
        Dtype::U64 => DLDataType::U64,
        _ => return Err(Error::UnsupportedSafetensorsDtype { dtype }),
    })
}

fn dtype_alignment(dtype: Dtype) -> usize {
    match dtype {
        Dtype::I64 | Dtype::U64 | Dtype::F64 => 8,
        Dtype::I32 | Dtype::U32 | Dtype::F32 | Dtype::C64 => 4,
        Dtype::I16 | Dtype::U16 | Dtype::F16 | Dtype::BF16 => 2,
        _ => 1,
    }
}

fn safetensors_dtype(dtype: DLDataType) -> Result<Dtype, Error> {
    Ok(match dtype {
        DLDataType::BOOL => Dtype::BOOL,
        DLDataType::F4E2M1FN => Dtype::F4,
        DLDataType::F6E2M3FN => Dtype::F6_E2M3,
        DLDataType::F6E3M2FN => Dtype::F6_E3M2,
        DLDataType::U8 => Dtype::U8,
        DLDataType::I8 => Dtype::I8,
        DLDataType::F8E5M2 => Dtype::F8_E5M2,
        DLDataType::F8E4M3FN => Dtype::F8_E4M3,
        DLDataType::F8E8M0FNU => Dtype::F8_E8M0,
        DLDataType::F8E4M3FNUZ => Dtype::F8_E4M3FNUZ,
        DLDataType::F8E5M2FNUZ => Dtype::F8_E5M2FNUZ,
        DLDataType::I16 => Dtype::I16,
        DLDataType::U16 => Dtype::U16,
        DLDataType::F16 => Dtype::F16,
        DLDataType::BF16 => Dtype::BF16,
        DLDataType::I32 => Dtype::I32,
        DLDataType::U32 => Dtype::U32,
        DLDataType::F32 => Dtype::F32,
        DLDataType::C64 => Dtype::C64,
        DLDataType::F64 => Dtype::F64,
        DLDataType::I64 => Dtype::I64,
        DLDataType::U64 => Dtype::U64,
        _ => return Err(Error::UnsupportedDlpackDtype { dtype }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{allocation::dynamic, ffi::DLManagedTensor};
    use safetensors::tensor::TensorView;
    use std::collections::HashMap;

    fn serialized_u8() -> Vec<u8> {
        let bytes = [1_u8, 2, 3, 4];
        let view = TensorView::new(Dtype::U8, vec![2, 2], &bytes).unwrap();
        safetensors::serialize(HashMap::from([("weight", view)]), None).unwrap()
    }

    #[test]
    fn file_to_dlpack_is_zero_copy_read_only_and_retains_storage() {
        let file = SafeTensorFile::from_bytes(serialized_u8()).unwrap();
        assert_eq!(file.names(), ["weight"]);

        let mut dlpack = file.tensor("weight").unwrap();
        drop(file);
        assert_eq!(dlpack.flags(), DlpackFlags::READ_ONLY);
        let tensor = dlpack.validate().unwrap();
        assert!(tensor.dtype().matches(DLDataType::U8));
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.strides(), Some([2, 1].as_slice()));
        let values = unsafe { tensor.cpu_slice::<u8>() }.unwrap();
        assert_eq!(values, &[1, 2, 3, 4]);
        assert!(matches!(
            dlpack.validate_mut(),
            Err(crate::tensor::Error::ReadOnly)
        ));
    }

    #[test]
    #[cfg_attr(miri, ignore = "memory mapping is an operating-system facility")]
    fn opens_a_memory_mapped_file() {
        let path = std::env::temp_dir().join(format!(
            "dlpark-safetensors-{}-{:?}.safetensors",
            std::process::id(),
            std::thread::current().id()
        ));
        let values = [1.0_f32, 2.0, 3.0, 4.0];
        let bytes = values
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let view = TensorView::new(Dtype::F32, vec![2, 2], &bytes).unwrap();
        let serialized = safetensors::serialize(HashMap::from([("weight", view)]), None).unwrap();
        std::fs::write(&path, serialized).unwrap();

        let file = SafeTensorFile::open(&path).unwrap();
        let dlpack = file.tensor("weight").unwrap();
        assert_eq!(
            unsafe { dlpack.validate().unwrap().cpu_slice::<f32>() }.unwrap(),
            &[1.0, 2.0, 3.0, 4.0]
        );

        drop(dlpack);
        drop(file);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn empty_tensor_uses_an_aligned_non_null_pointer() {
        let view = TensorView::new(Dtype::F64, vec![0], &[]).unwrap();
        let bytes = safetensors::serialize(HashMap::from([("empty", view)]), None).unwrap();
        let file = SafeTensorFile::from_bytes(bytes).unwrap();
        let dlpack = file.tensor("empty").unwrap();
        let tensor = dlpack.validate().unwrap();

        assert_eq!(tensor.shape(), &[0]);
        assert!(!tensor.data_ptr().is_null());
        assert!(unsafe { tensor.cpu_slice::<f64>() }.unwrap().is_empty());
    }

    #[test]
    fn dlpack_view_serializes_without_copying_source_data() {
        let values = Box::new(vec![7_i32, 8, 9, 10]);
        let data = values.as_ptr().cast_mut().cast();
        let prepared = Dynamic::compact([2_i64, 2].as_slice())
            .prepare::<DLManagedTensor>()
            .unwrap();
        let mut initialized: dynamic::Initialized<_> = prepared.initialize(values);
        initialized
            .set_data(data)
            .set_device(DLDevice::CPU)
            .set_dtype(DLDataType::I32);
        let dlpack = unsafe { initialized.finish() };

        let view = unsafe { DlpackView::from_dlpack(&dlpack) }.unwrap();
        assert_eq!(view.data().as_ptr(), data.cast());
        let serialized = safetensors::serialize([("weight", &view)], None).unwrap();
        let parsed = SafeTensors::deserialize(&serialized).unwrap();
        let roundtrip = parsed.tensor("weight").unwrap();
        assert_eq!(roundtrip.dtype(), Dtype::I32);
        assert_eq!(roundtrip.shape(), &[2, 2]);
        assert_eq!(roundtrip.data(), view.data().as_ref());
    }

    #[test]
    fn dtype_mapping_covers_every_current_safetensors_dtype() {
        let dtypes = [
            Dtype::BOOL,
            Dtype::F4,
            Dtype::F6_E2M3,
            Dtype::F6_E3M2,
            Dtype::U8,
            Dtype::I8,
            Dtype::F8_E5M2,
            Dtype::F8_E4M3,
            Dtype::F8_E8M0,
            Dtype::F8_E4M3FNUZ,
            Dtype::F8_E5M2FNUZ,
            Dtype::I16,
            Dtype::U16,
            Dtype::F16,
            Dtype::BF16,
            Dtype::I32,
            Dtype::U32,
            Dtype::F32,
            Dtype::C64,
            Dtype::F64,
            Dtype::I64,
            Dtype::U64,
        ];
        for dtype in dtypes {
            let dlpack = dlpack_dtype(dtype).unwrap();
            assert_eq!(safetensors_dtype(dlpack).unwrap(), dtype);
        }
    }
}
