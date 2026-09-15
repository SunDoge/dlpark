//! Minimal CUDA Runtime loading and stream/event synchronization.
//!
//! The small macro-generated function table and attachment to an already
//! loaded `libcudart` follow the architecture used by safetensors' Python CUDA
//! loader. It contains only the calls needed to negotiate DLPack stream
//! ownership without linking a CUDA toolkit at build time.

#[cfg(feature = "pyo3")]
use crate::ffi::{DLDevice, DLDeviceType};
#[cfg(feature = "pyo3")]
use crate::python::{DlpackStream, StreamArg, stream};
use libloading::Library;
#[cfg(feature = "pyo3")]
use pyo3::{PyResult, Python, exceptions::PyValueError};
use std::{
    ffi::{CStr, c_char, c_int, c_uint, c_void},
    fmt,
    ptr::{self, NonNull},
    sync::OnceLock,
};

type RawStream = *mut c_void;
type RawEvent = *mut c_void;
type CudaError = c_int;

const CUDA_SUCCESS: CudaError = 0;
const CUDA_EVENT_DISABLE_TIMING: c_uint = 2;
const CUDA_STREAM_NON_BLOCKING: c_uint = 1;

/// An error returned by the dynamically loaded CUDA Runtime.
#[derive(Debug)]
pub struct Error(String);

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

macro_rules! cuda_fns {
    ($($field:ident => $symbol:literal: fn($($arg:ty),*) -> $result:ty;)+) => {
        struct CudaApi {
            _library: Library,
            $(
                $field: unsafe extern "C" fn($($arg),*) -> $result,
            )+
        }

        impl CudaApi {
            unsafe fn load() -> Result<Self, Error> {
                let library = unsafe { load_cudart()? };
                Ok(Self {
                    $($field: unsafe {
                        *library
                            .get::<unsafe extern "C" fn($($arg),*) -> $result>(concat!($symbol, "\0").as_bytes())
                            .map_err(|source| Error(format!("failed to load {}: {source}", $symbol)))?
                    },)+
                    _library: library,
                })
            }
        }
    };
}

cuda_fns! {
    get_error_string => "cudaGetErrorString": fn(CudaError) -> *const c_char;
    get_device => "cudaGetDevice": fn(*mut c_int) -> CudaError;
    set_device => "cudaSetDevice": fn(c_int) -> CudaError;
    stream_create_with_flags => "cudaStreamCreateWithFlags": fn(*mut RawStream, c_uint) -> CudaError;
    stream_destroy => "cudaStreamDestroy": fn(RawStream) -> CudaError;
    stream_synchronize => "cudaStreamSynchronize": fn(RawStream) -> CudaError;
    stream_wait_event => "cudaStreamWaitEvent": fn(RawStream, RawEvent, c_uint) -> CudaError;
    event_create_with_flags => "cudaEventCreateWithFlags": fn(*mut RawEvent, c_uint) -> CudaError;
    event_destroy => "cudaEventDestroy": fn(RawEvent) -> CudaError;
    event_record => "cudaEventRecord": fn(RawEvent, RawStream) -> CudaError;
}

static CUDA_API: OnceLock<CudaApi> = OnceLock::new();

fn api() -> Result<&'static CudaApi, Error> {
    if let Some(api) = CUDA_API.get() {
        return Ok(api);
    }

    // Do not cache failures: a framework may load libcudart after the first
    // probe. Concurrent successful probes are harmless; OnceLock keeps one.
    let loaded = unsafe { CudaApi::load()? };
    let _ = CUDA_API.set(loaded);
    Ok(CUDA_API.get().expect("CUDA API was just initialized"))
}

impl CudaApi {
    fn check(&self, operation: &str, result: CudaError) -> Result<(), Error> {
        if result == CUDA_SUCCESS {
            return Ok(());
        }
        let message = unsafe {
            let pointer = (self.get_error_string)(result);
            if pointer.is_null() {
                "unknown CUDA error".into()
            } else {
                CStr::from_ptr(pointer).to_string_lossy().into_owned()
            }
        };
        Err(Error(format!("{operation} failed ({result}): {message}")))
    }

    fn with_device<T>(
        &'static self,
        device: c_int,
        operation: impl FnOnce(&'static Self) -> Result<T, Error>,
    ) -> Result<T, Error> {
        let mut previous = 0;
        self.check("cudaGetDevice", unsafe { (self.get_device)(&mut previous) })?;
        if previous != device {
            self.check("cudaSetDevice", unsafe { (self.set_device)(device) })?;
        }
        let guard = DeviceGuard {
            api: self,
            previous,
            restore: previous != device,
        };
        let result = operation(self);
        drop(guard);
        result
    }
}

struct DeviceGuard {
    api: &'static CudaApi,
    previous: c_int,
    restore: bool,
}

impl Drop for DeviceGuard {
    fn drop(&mut self) {
        if self.restore {
            unsafe {
                (self.api.set_device)(self.previous);
            }
        }
    }
}

/// An owning non-blocking CUDA Runtime stream.
///
/// Creation and destruction temporarily select the stream's CUDA device and
/// restore the calling thread's previous device afterward.
pub struct CudaStream {
    api: &'static CudaApi,
    raw: NonNull<c_void>,
    device: c_int,
}

impl CudaStream {
    /// Creates a non-blocking stream on `device`.
    ///
    /// On Linux, dlpark first attaches to a `libcudart` already loaded by a
    /// framework, then tries common CUDA Runtime sonames. On Windows it tries
    /// the CUDA 11–13 runtime DLL names. A failed lookup is not cached, so
    /// construction may be retried after a framework initializes CUDA.
    pub fn new(device: c_int) -> Result<Self, Error> {
        let api = api()?;
        let raw = api.with_device(device, |api| {
            let mut stream = ptr::null_mut();
            api.check("cudaStreamCreateWithFlags", unsafe {
                (api.stream_create_with_flags)(&mut stream, CUDA_STREAM_NON_BLOCKING)
            })?;
            NonNull::new(stream).ok_or_else(|| {
                Error("cudaStreamCreateWithFlags succeeded but returned a null stream".into())
            })
        })?;
        Ok(Self { api, raw, device })
    }

    /// Returns the borrowed native `cudaStream_t` handle.
    pub fn as_raw(&self) -> *mut c_void {
        self.raw.as_ptr()
    }

    /// Blocks the host until all previously queued work completes.
    pub fn synchronize(&self) -> Result<(), Error> {
        self.api.with_device(self.device, |api| {
            api.check("cudaStreamSynchronize", unsafe {
                (api.stream_synchronize)(self.raw.as_ptr())
            })
        })
    }

    /// Orders `consumer` after everything currently queued on this stream.
    ///
    /// The handoff records a timing-disabled event on this stream and queues
    /// a wait for that event on `consumer`; it does not block the host.
    ///
    /// # Safety
    ///
    /// `consumer` must be a live `cudaStream_t` on the same CUDA device. CUDA's
    /// null legacy-default-stream handle and per-thread sentinel are valid.
    pub unsafe fn hand_off_to_raw(&self, consumer: *mut c_void) -> Result<(), Error> {
        self.api.with_device(self.device, |api| {
            let mut event = ptr::null_mut();
            api.check("cudaEventCreateWithFlags", unsafe {
                (api.event_create_with_flags)(&mut event, CUDA_EVENT_DISABLE_TIMING)
            })?;
            let event = Event {
                api,
                raw: NonNull::new(event).ok_or_else(|| {
                    Error("cudaEventCreateWithFlags succeeded but returned a null event".into())
                })?,
            };
            api.check("cudaEventRecord", unsafe {
                (api.event_record)(event.raw.as_ptr(), self.raw.as_ptr())
            })?;
            api.check("cudaStreamWaitEvent", unsafe {
                (api.stream_wait_event)(consumer, event.raw.as_ptr(), 0)
            })
        })
    }

    /// Orders this stream after the work already queued on `producer`.
    pub fn wait_for(&self, producer: &Self) -> Result<(), Error> {
        if self.device != producer.device {
            return Err(Error(format!(
                "cannot synchronize CUDA streams on devices {} and {}",
                self.device, producer.device
            )));
        }
        unsafe { producer.hand_off_to_raw(self.raw.as_ptr()) }
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        let _ = self.api.with_device(self.device, |api| {
            api.check("cudaStreamDestroy", unsafe {
                (api.stream_destroy)(self.raw.as_ptr())
            })
        });
    }
}

#[cfg(feature = "pyo3")]
unsafe impl DlpackStream for CudaStream {
    fn as_python_arg(&self, _py: Python<'_>, device: DLDevice) -> PyResult<StreamArg> {
        if device.device_type != DLDeviceType::CUDA || device.device_id != self.device {
            return Err(PyValueError::new_err(format!(
                "CUDA stream belongs to device {}, tensor reports {:?}:{}",
                self.device, device.device_type, device.device_id
            )));
        }
        Ok(stream::cuda(self.raw.as_ptr()))
    }
}

struct Event {
    api: &'static CudaApi,
    raw: NonNull<c_void>,
}

impl Drop for Event {
    fn drop(&mut self) {
        unsafe {
            (self.api.event_destroy)(self.raw.as_ptr());
        }
    }
}

#[cfg(target_os = "linux")]
unsafe fn loaded_cudart() -> Option<Library> {
    unsafe extern "C" fn find(
        info: *mut libc::dl_phdr_info,
        _size: usize,
        output: *mut c_void,
    ) -> c_int {
        let name = unsafe { (*info).dlpi_name };
        if name.is_null() {
            return 0;
        }
        let bytes = unsafe { CStr::from_ptr(name) }.to_bytes();
        if !bytes
            .windows(b"libcudart.so".len())
            .any(|part| part == b"libcudart.so")
        {
            return 0;
        }
        let handle =
            unsafe { libc::dlopen(name, libc::RTLD_NOW | libc::RTLD_LOCAL | libc::RTLD_NOLOAD) };
        if handle.is_null() {
            return 0;
        }
        unsafe { output.cast::<*mut c_void>().write(handle) };
        1
    }

    let mut handle: *mut c_void = ptr::null_mut();
    unsafe { libc::dl_iterate_phdr(Some(find), (&mut handle as *mut *mut c_void).cast()) };
    (!handle.is_null()).then(|| {
        let library = unsafe { libloading::os::unix::Library::from_raw(handle) };
        library.into()
    })
}

unsafe fn load_cudart() -> Result<Library, Error> {
    #[cfg(target_os = "linux")]
    if let Some(library) = unsafe { loaded_cudart() } {
        return Ok(library);
    }

    #[cfg(target_os = "linux")]
    const NAMES: &[&str] = &["libcudart.so", "libcudart.so.13", "libcudart.so.12"];
    #[cfg(target_os = "windows")]
    const NAMES: &[&str] = &["cudart64_13.dll", "cudart64_12.dll", "cudart64_110.dll"];

    let mut errors = Vec::new();
    for name in NAMES {
        match unsafe { Library::new(name) } {
            Ok(library) => return Ok(library),
            Err(error) => errors.push(format!("{name}: {error}")),
        }
    }
    Err(Error(format!(
        "failed to load CUDA Runtime; tried {}",
        errors.join(", ")
    )))
}
