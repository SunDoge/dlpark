//! Minimal CUDA Runtime loading and stream/event synchronization.
//!
//! The small macro-generated function table and attachment to an already
//! loaded `libcudart` follow the architecture used by safetensors' Python CUDA
//! loader. It contains only the calls needed to negotiate DLPack stream
//! ownership without linking a CUDA toolkit at build time.

#[cfg(feature = "pyo3")]
use crate::ffi::{DLDevice, DLDeviceType};
#[cfg(feature = "pyo3")]
use crate::python::{DlpackStream, StreamArg, consumer::stream};
use libloading::Library;
#[cfg(feature = "pyo3")]
use pyo3::{PyResult, Python, exceptions::PyValueError};
use snafu::{ResultExt, Snafu};
use std::{
    ffi::{CStr, c_char, c_int, c_uint, c_void},
    path::PathBuf,
    ptr::{self, NonNull},
    sync::OnceLock,
};

type RawStream = *mut c_void;
type RawEvent = *mut c_void;
type CudaError = c_int;

const CUDA_SUCCESS: CudaError = 0;
const CUDA_EVENT_DISABLE_TIMING: c_uint = 2;
const CUDA_STREAM_NON_BLOCKING: c_uint = 1;
const CUDART_PATH_ENV: &str = "DLPARK_CUDART_PATH";

/// An error returned by the dynamically loaded CUDA Runtime.
#[derive(Debug, Snafu)]
pub enum Error {
    /// No CUDA Runtime has been loaded by the producer framework.
    #[snafu(display(
        "CUDA Runtime is not loaded; initialize CUDA in the producer framework first or set DLPARK_CUDART_PATH"
    ))]
    RuntimeNotLoaded,

    /// The runtime selected through `DLPARK_CUDART_PATH` could not be loaded.
    #[snafu(display("failed to load CUDA Runtime from DLPARK_CUDART_PATH={path:?}: {source}"))]
    LoadCudart {
        /// Requested runtime library path.
        path: PathBuf,
        /// Dynamic-loader error.
        source: libloading::Error,
    },

    /// A required CUDA Runtime symbol could not be resolved.
    #[snafu(display("failed to load CUDA Runtime symbol {symbol}: {source}"))]
    LoadSymbol {
        /// Required symbol name.
        symbol: &'static str,
        /// Dynamic-loader error.
        source: libloading::Error,
    },

    /// A CUDA Runtime operation returned an error code.
    #[snafu(display("{operation} failed ({code}): {message}"))]
    CudaCall {
        /// CUDA operation name.
        operation: &'static str,
        /// CUDA Runtime error code.
        code: CudaError,
        /// Message returned by `cudaGetErrorString`.
        message: String,
    },

    /// A successful CUDA operation returned a null resource handle.
    #[snafu(display("{operation} succeeded but returned a null handle"))]
    NullHandle {
        /// CUDA operation name.
        operation: &'static str,
    },

    /// Stream synchronization was requested across CUDA devices.
    #[snafu(display(
        "cannot synchronize CUDA streams on devices {consumer_device} and {producer_device}"
    ))]
    StreamDeviceMismatch {
        /// Consumer stream device.
        consumer_device: c_int,
        /// Producer stream device.
        producer_device: c_int,
    },
}

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
                let library = unsafe { open_cudart_library()? };
                Ok(Self {
                    $($field: unsafe {
                        *library
                            .get::<unsafe extern "C" fn($($arg),*) -> $result>(concat!($symbol, "\0").as_bytes())
                            .context(LoadSymbolSnafu { symbol: $symbol })?
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
    fn check(&self, operation: &'static str, result: CudaError) -> Result<(), Error> {
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
        Err(Error::CudaCall {
            operation,
            code: result,
            message,
        })
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
    /// By default, dlpark attaches to the CUDA Runtime already loaded by the
    /// producer framework so both sides use the same runtime instance. Set
    /// `DLPARK_CUDART_PATH` to force a specific runtime library. A failed
    /// lookup is not cached, so construction may be retried after the
    /// framework initializes CUDA.
    pub fn new(device: c_int) -> Result<Self, Error> {
        let api = api()?;
        let raw = api.with_device(device, |api| {
            let mut stream = ptr::null_mut();
            api.check("cudaStreamCreateWithFlags", unsafe {
                (api.stream_create_with_flags)(&mut stream, CUDA_STREAM_NON_BLOCKING)
            })?;
            NonNull::new(stream).ok_or(Error::NullHandle {
                operation: "cudaStreamCreateWithFlags",
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
        unsafe { self.order_raw_streams(self.raw.as_ptr(), consumer) }
    }

    /// Orders this stream after work already queued on `producer`.
    ///
    /// # Safety
    ///
    /// `producer` must be a live `cudaStream_t` on the same CUDA device. CUDA's
    /// null legacy-default-stream handle and per-thread sentinel are valid.
    pub unsafe fn wait_for_raw(&self, producer: *mut c_void) -> Result<(), Error> {
        unsafe { self.order_raw_streams(producer, self.raw.as_ptr()) }
    }

    unsafe fn order_raw_streams(
        &self,
        producer: *mut c_void,
        consumer: *mut c_void,
    ) -> Result<(), Error> {
        self.api.with_device(self.device, |api| {
            let mut event = ptr::null_mut();
            api.check("cudaEventCreateWithFlags", unsafe {
                (api.event_create_with_flags)(&mut event, CUDA_EVENT_DISABLE_TIMING)
            })?;
            let event = Event {
                api,
                raw: NonNull::new(event).ok_or(Error::NullHandle {
                    operation: "cudaEventCreateWithFlags",
                })?,
            };
            api.check("cudaEventRecord", unsafe {
                (api.event_record)(event.raw.as_ptr(), producer)
            })?;
            api.check("cudaStreamWaitEvent", unsafe {
                (api.stream_wait_event)(consumer, event.raw.as_ptr(), 0)
            })
        })
    }

    /// Orders this stream after the work already queued on `producer`.
    pub fn wait_for(&self, producer: &Self) -> Result<(), Error> {
        if self.device != producer.device {
            return Err(Error::StreamDeviceMismatch {
                consumer_device: self.device,
                producer_device: producer.device,
            });
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

    unsafe fn wait_for_producer(&self, device: DLDevice, producer: *mut c_void) -> PyResult<bool> {
        if device.device_type != DLDeviceType::CUDA || device.device_id != self.device {
            return Err(PyValueError::new_err(format!(
                "CUDA stream belongs to device {}, tensor reports {:?}:{}",
                self.device, device.device_type, device.device_id
            )));
        }
        unsafe { self.wait_for_raw(producer) }
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(true)
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
unsafe fn attach_to_loaded_cudart() -> Option<Library> {
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

#[cfg(target_os = "windows")]
unsafe fn attach_to_loaded_cudart() -> Option<Library> {
    const CANDIDATE_NAMES: &[&str] = &["cudart64_13.dll", "cudart64_12.dll", "cudart64_110.dll"];

    for name in CANDIDATE_NAMES {
        if let Ok(library) = libloading::os::windows::Library::open_already_loaded(name) {
            return Some(library.into());
        }
    }
    None
}

unsafe fn open_cudart_library() -> Result<Library, Error> {
    if let Some(path) = std::env::var_os(CUDART_PATH_ENV) {
        let path = PathBuf::from(path);
        return unsafe { Library::new(&path) }.context(LoadCudartSnafu { path });
    }

    unsafe { attach_to_loaded_cudart() }.ok_or(Error::RuntimeNotLoaded)
}
