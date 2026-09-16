//! PyO3 conversions for owning DLPack capsules.

use std::ffi::CStr;

const DLTENSOR: &CStr = c"dltensor";
const USED_DLTENSOR: &CStr = c"used_dltensor";
const DLTENSOR_VERSIONED: &CStr = c"dltensor_versioned";
const USED_DLTENSOR_VERSIONED: &CStr = c"used_dltensor_versioned";

mod from_python;
mod into_python;

pub(crate) use from_python::{
    call_dlpack, consume_legacy_capsule, consume_versioned_capsule, is_legacy_capsule,
    is_versioned_capsule, validate_copy_result,
};

#[cfg(test)]
use crate::{
    Managed,
    ffi::{DLManagedTensor, DLManagedTensorVersioned},
    python::{DlpackStream, ImportedDlpack, from_dlpack},
};
#[cfg(test)]
mod tests {
    use super::*;
    use crate::DlpackFlags;
    use crate::{
        allocation::fixed::make_test_tensor,
        ffi::{DLDataType, DLDevice, DLDeviceType},
    };
    use pyo3::{
        conversion::IntoPyObject,
        exceptions::PyValueError,
        types::{PyAnyMethods, PyModule},
    };
    use std::ffi::c_void;

    struct TestStream;

    unsafe impl DlpackStream for TestStream {
        fn as_python_arg(
            &self,
            _py: pyo3::Python<'_>,
            device: DLDevice,
        ) -> pyo3::PyResult<crate::python::StreamArg> {
            assert_eq!(device.device_type, DLDeviceType::CUDA);
            assert_eq!(device.device_id, 3);
            Ok(crate::python::consumer::stream::cuda(
                std::ptr::without_provenance_mut(42),
            ))
        }
    }

    fn legacy_tensor() -> Managed<DLManagedTensor> {
        let data = Box::new(vec![1i32, 2, 3]);
        let data_ptr = data.as_ptr() as *mut c_void;
        make_test_tensor(
            data,
            data_ptr,
            DLDataType::of::<i32>(),
            DLDevice::CPU,
            [3],
            [1],
            DlpackFlags::empty(),
        )
    }

    fn versioned_tensor() -> Managed<DLManagedTensorVersioned> {
        versioned_tensor_with_flags(DlpackFlags::empty())
    }

    fn versioned_tensor_with_flags(flags: DlpackFlags) -> Managed<DLManagedTensorVersioned> {
        let data = Box::new(vec![4i32, 5, 6]);
        let data_ptr = data.as_ptr() as *mut c_void;
        make_test_tensor(
            data,
            data_ptr,
            DLDataType::of::<i32>(),
            DLDevice::CPU,
            [3],
            [1],
            flags,
        )
    }

    #[test]
    fn local_versioned_tensor_converts_to_capsule() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let data = Box::new(vec![4i32, 5, 6]);
            let data_ptr = data.as_ptr() as *mut c_void;
            let tensor: Managed<DLManagedTensorVersioned> = make_test_tensor(
                data,
                data_ptr,
                DLDataType::of::<i32>(),
                DLDevice::CPU,
                [3],
                [1],
                DlpackFlags::empty(),
            );
            let capsule = tensor.into_pyobject(py)?;
            let ImportedDlpack::Versioned(tensor) = from_dlpack(capsule.as_borrowed(), None, None)?
            else {
                panic!("versioned capsule was imported as legacy");
            };
            assert_eq!(
                unsafe { tensor.tensor().cpu_slice::<i32>() }.unwrap(),
                &[4, 5, 6]
            );
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn legacy_capsule_can_only_be_consumed_once() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let capsule = legacy_tensor().into_pyobject(py)?;

            let ImportedDlpack::Legacy(dlpack) = from_dlpack(capsule.as_borrowed(), None, None)?
            else {
                panic!("legacy capsule was imported as versioned");
            };
            assert_eq!(
                unsafe { dlpack.tensor().cpu_slice::<i32>() }.unwrap(),
                &[1, 2, 3]
            );

            let err = match from_dlpack(capsule.as_borrowed(), None, None) {
                Ok(_) => panic!("consuming the same DLPack capsule twice should fail"),
                Err(err) => err,
            };
            assert!(err.is_instance_of::<PyValueError>(py));

            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn versioned_capsule_can_only_be_consumed_once() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let capsule = versioned_tensor().into_pyobject(py)?;

            let ImportedDlpack::Versioned(dlpack) = from_dlpack(capsule.as_borrowed(), None, None)?
            else {
                panic!("versioned capsule was imported as legacy");
            };
            assert_eq!(
                unsafe { dlpack.tensor().cpu_slice::<i32>() }.unwrap(),
                &[4, 5, 6]
            );

            let err = match from_dlpack(capsule.as_borrowed(), None, None) {
                Ok(_) => panic!("consuming the same DLPack capsule twice should fail"),
                Err(err) => err,
            };
            assert!(err.is_instance_of::<PyValueError>(py));

            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn from_dlpack_rejects_copy_result_mismatch() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let module = PyModule::from_code(
                py,
                cr#"class Producer:
    def __init__(self, capsule):
        self.capsule = capsule

    def __dlpack_device__(self):
        return (1, 0)

    def __dlpack__(self, *, max_version=None, copy=None):
        return self.capsule
"#,
                c"copy_mismatch_producer.py",
                c"copy_mismatch_producer",
            )?;

            for (flags, requested) in [
                (DlpackFlags::empty(), true),
                (DlpackFlags::IS_COPIED, false),
            ] {
                let capsule = versioned_tensor_with_flags(flags).into_pyobject(py)?;
                let producer = module.getattr("Producer")?.call1((capsule,))?;
                let error = match from_dlpack(producer.as_borrowed(), None, Some(requested)) {
                    Ok(_) => panic!("copy result mismatch must be rejected"),
                    Err(error) => error,
                };
                assert!(error.is_instance_of::<pyo3::exceptions::PyBufferError>(py));
            }

            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn from_dlpack_negotiates_versioned_protocol() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let capsule = versioned_tensor().into_pyobject(py)?;
            let module = PyModule::from_code(
                py,
                cr#"class Producer:
    def __init__(self, capsule):
        self.capsule = capsule
        self.seen_max_version = None

    def __dlpack_device__(self):
        return (1, 0)

    def __dlpack__(self, *, max_version=None):
        self.seen_max_version = max_version
        return self.capsule
"#,
                c"versioned_producer.py",
                c"versioned_producer",
            )?;
            let producer = module.getattr("Producer")?.call1((capsule,))?;

            let ImportedDlpack::Versioned(dlpack) =
                from_dlpack(producer.as_borrowed(), None, None)?
            else {
                panic!("negotiated import returned the legacy ABI");
            };
            assert_eq!(
                unsafe { dlpack.tensor().cpu_slice::<i32>() }.unwrap(),
                &[4, 5, 6]
            );
            assert_eq!(
                producer
                    .getattr("seen_max_version")?
                    .extract::<(u32, u32)>()?,
                (
                    crate::ffi::DLPACK_MAJOR_VERSION,
                    crate::ffi::DLPACK_MINOR_VERSION
                )
            );

            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn from_dlpack_maps_device_and_passes_stream() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let capsule = versioned_tensor().into_pyobject(py)?;
            let module = PyModule::from_code(
                py,
                cr#"class Producer:
    def __init__(self, capsule):
        self.capsule = capsule
        self.seen_stream = None
        self.seen_max_version = None
        self.seen_copy = None

    def __dlpack_device__(self):
        return (2, 3)

    def __dlpack__(self, *, stream=None, max_version=None, copy=None):
        self.seen_stream = stream
        self.seen_max_version = max_version
        self.seen_copy = copy
        return self.capsule
"#,
                c"stream_producer.py",
                c"stream_producer",
            )?;
            let producer = module.getattr("Producer")?.call1((capsule,))?;

            let _dlpack = from_dlpack(producer.as_borrowed(), Some(&TestStream), Some(false))?;
            assert_eq!(producer.getattr("seen_stream")?.extract::<usize>()?, 42);
            assert!(!producer.getattr("seen_copy")?.extract::<bool>()?);
            assert_eq!(
                producer
                    .getattr("seen_max_version")?
                    .extract::<(u32, u32)>()?,
                (
                    crate::ffi::DLPACK_MAJOR_VERSION,
                    crate::ffi::DLPACK_MINOR_VERSION
                )
            );

            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn from_dlpack_passes_copy_without_stream() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let capsule = versioned_tensor_with_flags(DlpackFlags::IS_COPIED).into_pyobject(py)?;
            let module = PyModule::from_code(
                py,
                cr#"class Producer:
    def __init__(self, capsule):
        self.capsule = capsule
        self.seen_copy = None

    def __dlpack_device__(self):
        return (1, 0)

    def __dlpack__(self, *, max_version=None, copy=None):
        self.seen_copy = copy
        return self.capsule
"#,
                c"copy_producer.py",
                c"copy_producer",
            )?;
            let producer = module.getattr("Producer")?.call1((capsule,))?;

            let _dlpack = from_dlpack(producer.as_borrowed(), None, Some(true))?;
            assert!(producer.getattr("seen_copy")?.extract::<bool>()?);

            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn from_dlpack_does_not_retry_internal_type_error() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let module = PyModule::from_code(
                py,
                cr#"class Producer:
    def __init__(self):
        self.calls = 0

    def __dlpack__(self, *, max_version=None):
        self.calls += 1
        raise TypeError("producer failed")
"#,
                c"broken_versioned_producer.py",
                c"broken_versioned_producer",
            )?;
            let producer = module.getattr("Producer")?.call0()?;

            let err = match from_dlpack(producer.as_borrowed(), None, None) {
                Ok(_) => panic!("producer TypeError should propagate"),
                Err(err) => err,
            };

            assert!(err.is_instance_of::<pyo3::exceptions::PyTypeError>(py));
            assert_eq!(producer.getattr("calls")?.extract::<u32>()?, 1);
            Ok(())
        })
        .unwrap();
    }
}
