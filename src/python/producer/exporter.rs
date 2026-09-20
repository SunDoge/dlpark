//! Reusable Python DLPack export orchestration.

use super::ExportRequest;
use crate::{
    DlpackFlags, Managed,
    ffi::{DLDevice, DLManagedTensor, DLManagedTensorVersioned},
};
use pyo3::{Py, PyAny, PyResult, Python};

/// A reusable tensor holder that can produce fresh Python DLPack exports.
///
/// The implementor owns the actual buffer and metadata. Every export method
/// must create a fresh managed-tensor header so repeated `__dlpack__` calls are
/// independent; the returned capsule remains a one-shot ownership token.
///
/// [`prepare_export`](Self::prepare_export) is called only after dlpark has
/// validated zero-copy, destination-device, flag, and ABI constraints. Device
/// backends use it to make the tensor ready on the consumer stream encoded in
/// `request`.
pub trait DlpackExporter {
    /// Returns the tensor's source device.
    fn device(&self) -> DLDevice;

    /// Returns the flags carried by a versioned export.
    fn flags(&self) -> DlpackFlags;

    /// Performs backend-specific preparation for this export request.
    fn prepare_export(&self, py: Python<'_>, request: &ExportRequest<'_>) -> PyResult<()>;

    /// Creates a fresh legacy managed-tensor header.
    fn export_legacy(&self, py: Python<'_>) -> PyResult<Managed<DLManagedTensor>>;

    /// Creates a fresh versioned managed-tensor header.
    fn export_versioned(&self, py: Python<'_>) -> PyResult<Managed<DLManagedTensorVersioned>>;
}

/// Completes a reusable producer's standard Python `__dlpack__` request.
///
/// Validation runs before backend synchronization. The selected exporter
/// method is called exactly once and its fresh managed tensor is wrapped in a
/// one-shot Python capsule.
pub fn export_dlpack<'py, T>(
    exporter: &T,
    py: Python<'py>,
    request: ExportRequest<'py>,
) -> PyResult<Py<PyAny>>
where
    T: DlpackExporter + ?Sized,
{
    let device = exporter.device();
    let flags = exporter.flags();
    request.validate_zero_copy(device, flags)?;
    exporter.prepare_export(py, &request)?;
    request.export_zero_copy_validated(
        py,
        || exporter.export_legacy(py),
        || exporter.export_versioned(py),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        allocation::fixed::make_test_tensor,
        ffi::DLDataType,
        python::{ImportedDlpack, from_dlpack},
    };
    use std::{cell::Cell, ffi::c_void, sync::Arc};

    struct ReusableTensor {
        data: Arc<Vec<i32>>,
        prepares: Cell<usize>,
        legacy_exports: Cell<usize>,
        versioned_exports: Cell<usize>,
    }

    impl ReusableTensor {
        fn tensor<M: crate::ManagedTensorBase>(&self) -> Managed<M> {
            let data = Arc::clone(&self.data);
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
    }

    impl DlpackExporter for ReusableTensor {
        fn device(&self) -> DLDevice {
            DLDevice::CPU
        }

        fn flags(&self) -> DlpackFlags {
            DlpackFlags::empty()
        }

        fn prepare_export(&self, _py: Python<'_>, _request: &ExportRequest<'_>) -> PyResult<()> {
            self.prepares.set(self.prepares.get() + 1);
            Ok(())
        }

        fn export_legacy(&self, _py: Python<'_>) -> PyResult<Managed<DLManagedTensor>> {
            self.legacy_exports.set(self.legacy_exports.get() + 1);
            Ok(self.tensor())
        }

        fn export_versioned(&self, _py: Python<'_>) -> PyResult<Managed<DLManagedTensorVersioned>> {
            self.versioned_exports.set(self.versioned_exports.get() + 1);
            Ok(self.tensor())
        }
    }

    #[test]
    fn reusable_exporter_creates_fresh_capsules_and_selects_the_abi() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let exporter = ReusableTensor {
                data: Arc::new(vec![1, 2, 3]),
                prepares: Cell::new(0),
                legacy_exports: Cell::new(0),
                versioned_exports: Cell::new(0),
            };

            let legacy =
                export_dlpack(&exporter, py, ExportRequest::parse(None, None, None, None)?)?;
            let versioned = export_dlpack(
                &exporter,
                py,
                ExportRequest::parse(None, Some((1, 0)), None, None)?,
            )?;

            let ImportedDlpack::Legacy(legacy) =
                from_dlpack(legacy.bind(py).as_borrowed(), None, None)?
            else {
                panic!("legacy request selected the versioned ABI");
            };
            let ImportedDlpack::Versioned(versioned) =
                from_dlpack(versioned.bind(py).as_borrowed(), None, None)?
            else {
                panic!("versioned request selected the legacy ABI");
            };
            assert_eq!(
                legacy.validate().unwrap().data_ptr(),
                versioned.validate().unwrap().data_ptr()
            );
            assert_eq!(exporter.prepares.get(), 2);
            assert_eq!(exporter.legacy_exports.get(), 1);
            assert_eq!(exporter.versioned_exports.get(), 1);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn validation_happens_before_backend_preparation() {
        Python::initialize();
        Python::attach(|py| {
            let exporter = ReusableTensor {
                data: Arc::new(vec![1, 2, 3]),
                prepares: Cell::new(0),
                legacy_exports: Cell::new(0),
                versioned_exports: Cell::new(0),
            };
            let request = ExportRequest::parse(None, Some((1, 0)), None, Some(true)).unwrap();

            assert!(export_dlpack(&exporter, py, request).is_err());
            assert_eq!(exporter.prepares.get(), 0);
            assert_eq!(exporter.legacy_exports.get(), 0);
            assert_eq!(exporter.versioned_exports.get(), 0);
        });
    }
}
