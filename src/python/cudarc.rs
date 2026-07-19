use pyo3::exceptions::PyBufferError;

use crate::{
    ffi::{DLDevice, DLDeviceType},
    python::{DlpackStream, stream},
};

unsafe impl DlpackStream for cudarc::driver::CudaStream {
    fn as_python_arg(
        &self,
        _py: pyo3::Python<'_>,
        device: DLDevice,
    ) -> pyo3::PyResult<stream::StreamArg> {
        if device.device_type != DLDeviceType::CUDA
            && device.device_type != DLDeviceType::CUDAMANAGED
        {
            return Err(PyBufferError::new_err(format!(
                "a CUDA stream cannot consume DLPack device {:?}",
                device.device_type
            )));
        }

        let ordinal = i32::try_from(self.context().ordinal()).map_err(|_| {
            PyBufferError::new_err("CUDA stream device ordinal does not fit in i32")
        })?;
        if ordinal != device.device_id {
            return Err(PyBufferError::new_err(format!(
                "CUDA stream is on device {ordinal}, but the tensor is on device {}",
                device.device_id
            )));
        }

        Ok(stream::cuda(self.cu_stream().cast()))
    }
}

unsafe impl DlpackStream for std::sync::Arc<cudarc::driver::CudaStream> {
    fn as_python_arg(
        &self,
        py: pyo3::Python<'_>,
        device: DLDevice,
    ) -> pyo3::PyResult<stream::StreamArg> {
        self.as_ref().as_python_arg(py, device)
    }
}
