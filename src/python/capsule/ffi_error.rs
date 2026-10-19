//! Shared helpers for recovering Python errors from raw FFI calls.

use pyo3::{PyErr, Python};

/// Fetches the pending Python exception as a `PyErr`, or fabricates a
/// `SystemError` when none is pending.
///
/// Some CPython C APIs (notably `PyCapsule_GetPointer` on a capsule whose
/// stored pointer is NULL) return a failure value *without* setting an
/// exception. Blindly fetching would panic in debug builds and fabricate a
/// misleading `SystemError` in release, so callers that cannot prove an
/// exception is pending should check `PyErr_Occurred` first (see
/// [`checked_fetch`]).
///
/// # Safety
///
/// The current thread must be attached to the Python interpreter, which every
/// dlpark capsule call site satisfies by construction (they run from PyO3
/// conversion code).
pub(crate) unsafe fn fetch_python_error() -> PyErr {
    unsafe { PyErr::fetch(Python::assume_attached()) }
}

/// Fetches the pending exception if one is set; otherwise returns `fallback`.
///
/// Use this after C API calls that may fail without setting an exception.
///
/// # Safety
///
/// Same attachment requirement as [`fetch_python_error`].
pub(crate) unsafe fn checked_fetch_python_error(fallback: PyErr) -> PyErr {
    if unsafe { pyo3::ffi::PyErr_Occurred() }.is_null() {
        fallback
    } else {
        unsafe { fetch_python_error() }
    }
}
