use pyo3::{Bound, IntoPyObject, Py, PyAny, Python, exceptions::PyValueError};
use std::ffi::c_void;

use crate::ffi::DLDevice;

/// A value accepted by Python's `__dlpack__(stream=...)` protocol.
pub struct StreamArg(StreamArgInner);

enum StreamArgInner {
    None,
    Integer(isize),
    Pointer(usize),
    Python(Py<PyAny>),
}

impl StreamArg {
    pub(crate) fn into_python<'py>(
        self,
        py: Python<'py>,
    ) -> pyo3::PyResult<Option<Bound<'py, PyAny>>> {
        match self.0 {
            StreamArgInner::None => Ok(None),
            StreamArgInner::Integer(value) => Ok(Some(value.into_pyobject(py)?.into_any())),
            StreamArgInner::Pointer(value) => Ok(Some(value.into_pyobject(py)?.into_any())),
            StreamArgInner::Python(value) => Ok(Some(value.into_bound(py))),
        }
    }

    /// Wraps a backend-specific Python stream or queue object.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the producer understands this object for
    /// the relevant DLPack device and that any resources referenced by it
    /// remain alive for the duration of the `__dlpack__` call.
    pub unsafe fn from_python_object(value: Py<PyAny>) -> Self {
        Self(StreamArgInner::Python(value))
    }
}

/// Converts a backend stream into the value passed to Python's
/// `__dlpack__(stream=...)` protocol.
///
/// Returning `None` is appropriate for CPU and other devices without streams.
/// The Python value is borrowed by the `__dlpack__` call; dlpark never takes
/// ownership of the underlying backend stream.
///
/// # Safety
///
/// Implementors must ensure that the returned value identifies a live stream
/// compatible with `device`, follows that device type's Python DLPack stream
/// convention, and remains valid for the duration of the `__dlpack__` call.
pub unsafe trait DlpackStream {
    fn as_python_arg(&self, py: Python<'_>, device: DLDevice) -> pyo3::PyResult<StreamArg>;
}

/// Omits the Python stream argument.
pub fn none() -> StreamArg {
    StreamArg(StreamArgInner::None)
}

/// Requests that the producer perform no stream synchronization.
pub fn no_sync() -> StreamArg {
    StreamArg(StreamArgInner::Integer(-1))
}

/// Encodes a CUDA stream handle for Python's DLPack protocol.
///
/// A null handle maps to the legacy default stream sentinel `1`; CUDA's
/// per-thread default handle maps to `2`; all other handles are passed as
/// pointer-sized integers.
pub fn cuda(stream: *mut c_void) -> StreamArg {
    let raw = stream as usize;
    match raw {
        0 => StreamArg(StreamArgInner::Integer(1)),
        2 => StreamArg(StreamArgInner::Integer(2)),
        raw => StreamArg(StreamArgInner::Pointer(raw)),
    }
}

/// Encodes a ROCm stream handle for Python's DLPack protocol.
///
/// ROCm uses `0` for its default stream and does not support CUDA's `1` and
/// `2` default-stream sentinels.
pub fn rocm(stream: *mut c_void) -> pyo3::PyResult<StreamArg> {
    let raw = stream as usize;
    if matches!(raw, 1 | 2) {
        return Err(PyValueError::new_err(
            "ROCm DLPack streams do not support the CUDA 1 and 2 sentinels",
        ));
    }
    Ok(match raw {
        0 => StreamArg(StreamArgInner::Integer(0)),
        raw => StreamArg(StreamArgInner::Pointer(raw)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyAnyMethods;

    #[test]
    fn cuda_encodes_default_stream_sentinels() {
        Python::initialize();
        Python::attach(|py| -> pyo3::PyResult<()> {
            assert_eq!(
                cuda(std::ptr::null_mut())
                    .into_python(py)?
                    .unwrap()
                    .extract::<isize>()?,
                1
            );
            assert_eq!(
                cuda(std::ptr::without_provenance_mut(2))
                    .into_python(py)?
                    .unwrap()
                    .extract::<isize>()?,
                2
            );
            assert_eq!(
                cuda(std::ptr::without_provenance_mut(42))
                    .into_python(py)?
                    .unwrap()
                    .extract::<isize>()?,
                42
            );
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn rocm_rejects_cuda_default_stream_sentinels() {
        Python::initialize();
        Python::attach(|py| {
            assert!(rocm(std::ptr::without_provenance_mut(1)).is_err());
            assert!(rocm(std::ptr::without_provenance_mut(2)).is_err());
            assert_eq!(
                rocm(std::ptr::null_mut())
                    .unwrap()
                    .into_python(py)
                    .unwrap()
                    .unwrap()
                    .extract::<isize>()
                    .unwrap(),
                0
            );
        });
    }
}
