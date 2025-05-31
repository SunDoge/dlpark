use std::ffi::CString;

// const DLTENSOR: &[u8] = b"dltensor\0";
// const USED_DLTENSOR: &[u8] = b"used_dltensor\0";
// const DLTENSOR_VERSIONED: &[u8] = b"dltensor_versioned\0";
// const USED_DLTENSOR_VERSIONED: &[u8] = b"used_dltensor_versioned\0";




unsafe extern "C" fn dlpack_capsule_deleter(capsule: *mut pyo3::ffi::PyObject) {
    unsafe { if pyo3::ffi::PyCapsule_IsValid(capsule, F.as_ptr() as *const i8) == 1 {} }
}
