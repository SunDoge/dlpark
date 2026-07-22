//! candle interop (CPU only).
//!
//! Provides conversions between [`candle_core::Tensor`] and DLPack managed
//! tensors.
//!
//! # Forward direction (`Tensor` → `LegacyDlpack`/`VersionedDlpack`)
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
//! # Reverse direction (`&Foreign<M>` → `Tensor`)
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

use crate::ffi::DLDataType;
use candle_core::DType;
use snafu::Snafu;

mod consumer;
mod producer;

pub use consumer::candle_tensor_from_dlpack;

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

#[cfg(test)]
use crate::{Foreign, ManagedTensorBase, TryFromDlpack, allocation::dynamic, ffi::DLDevice};
#[cfg(test)]
use candle_core::{Device, Tensor};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::{DLDataTypeCode, DLManagedTensor, DLManagedTensorVersioned};
    use crate::{DlpackElement, DlpackFlags, Local, allocation::fixed::make_test_tensor};

    type LegacyDlpack = Local<DLManagedTensor>;
    type VersionedDlpack = Local<DLManagedTensorVersioned>;

    fn managed_candle<M: ManagedTensorBase>(tensor: Tensor) -> Local<M> {
        let initialized: dynamic::Initialized<M> = Box::new(tensor).try_into().unwrap();
        unsafe { initialized.finish() }
    }

    fn managed_candle_with_flags<M: ManagedTensorBase>(
        tensor: Tensor,
        flags: DlpackFlags,
    ) -> Local<M> {
        let mut initialized: dynamic::Initialized<M> = Box::new(tensor).try_into().unwrap();
        initialized.set_flags(flags).unwrap();
        unsafe { initialized.finish() }
    }

    fn raw_tensor<T, const N: usize>(
        data: Vec<T>,
        dtype: DLDataType,
        shape: [i64; N],
        strides: [i64; N],
    ) -> Foreign<DLManagedTensor>
    where
        T: Send + 'static,
    {
        let data = Box::new(data);
        let data_ptr = data.as_ptr().cast_mut().cast();
        make_test_tensor::<_, DLManagedTensor, N>(
            data,
            data_ptr,
            dtype,
            DLDevice::CPU,
            shape,
            strides,
            DlpackFlags::empty(),
        )
        .into_foreign()
    }

    #[test]
    fn candle_tensor_to_legacy_dlpack_keeps_layout_and_data() {
        let tensor = Tensor::from_vec(vec![1i32, 2, 3, 4, 5, 6], (2, 3), &Device::Cpu).unwrap();
        let dlpack: LegacyDlpack = managed_candle(tensor);

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
        let dlpack: VersionedDlpack = managed_candle(tensor);

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
        let dlpack: LegacyDlpack = managed_candle(tensor);

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
        let dlpack: LegacyDlpack = managed_candle(tensor);

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

        let tensor = unsafe { Tensor::try_from_dlpack(&dlpack) }.unwrap();

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

        let err = unsafe { Tensor::try_from_dlpack(&dlpack) }.unwrap_err();
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

        let tensor = unsafe { Tensor::try_from_dlpack(&dlpack) }.unwrap();

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

        let tensor = unsafe { Tensor::try_from_dlpack(&dlpack) }.unwrap();

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

        let err = unsafe { Tensor::try_from_dlpack(&dlpack) }.unwrap_err();
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

        let err = unsafe { Tensor::try_from_dlpack(&dlpack) }.unwrap_err();
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

        let tensor = unsafe { Tensor::try_from_dlpack(&dlpack) }.unwrap();

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

        let err = unsafe { Tensor::try_from_dlpack(&dlpack) }.unwrap_err();
        assert!(matches!(err, Error::UnsupportedDlDataType { .. }));
    }
}
