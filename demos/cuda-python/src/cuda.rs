//! Minimal CUDA Runtime loading and stream/event synchronization.
//!
//! The small macro-generated function table and attachment to an already
//! loaded `libcudart` follow the architecture used by safetensors' Python CUDA
//! loader. It contains only the calls needed to negotiate DLPack stream
//! ownership without linking a CUDA toolkit at build time.

use dlpark::python::{DlpackStream, StreamArg, consumer::stream};
use dlpark::{
    AllocationDeleter,
    ffi::{DLDevice, DLDeviceType},
};
use libloading::Library;
use pyo3::{PyResult, Python, exceptions::PyValueError};
use snafu::{ResultExt, Snafu};
use std::{
    collections::HashMap,
    ffi::{CStr, c_char, c_int, c_uint, c_void},
    path::PathBuf,
    ptr::{self, NonNull},
    sync::{Arc, Mutex, OnceLock},
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
    malloc => "cudaMalloc": fn(*mut *mut c_void, usize) -> CudaError;
    free => "cudaFree": fn(*mut c_void) -> CudaError;
    stream_create_with_flags => "cudaStreamCreateWithFlags": fn(*mut RawStream, c_uint) -> CudaError;
    stream_destroy => "cudaStreamDestroy": fn(RawStream) -> CudaError;
    stream_synchronize => "cudaStreamSynchronize": fn(RawStream) -> CudaError;
    stream_wait_event => "cudaStreamWaitEvent": fn(RawStream, RawEvent, c_uint) -> CudaError;
    event_create_with_flags => "cudaEventCreateWithFlags": fn(*mut RawEvent, c_uint) -> CudaError;
    event_destroy => "cudaEventDestroy": fn(RawEvent) -> CudaError;
    event_record => "cudaEventRecord": fn(RawEvent, RawStream) -> CudaError;
}

static CUDA_API: OnceLock<CudaApi> = OnceLock::new();
static CUDA_STREAMS: OnceLock<Mutex<HashMap<c_int, Arc<CudaStream>>>> = OnceLock::new();

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

// CUDA stream handles may be used from different host threads. Every operation
// first selects the owning device on the calling thread through `with_device`.
unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

/// A zero-copy CUDA allocation with a custom deleter.
///
/// The buffer records the byte-offset-adjusted device address. It does not
/// assume how the memory was allocated; dropping the last owner invokes the
/// supplied deleter exactly once.
pub struct CudaBuffer {
    address: Option<NonNull<c_void>>,
    byte_len: usize,
    device: c_int,
    _deleter: AllocationDeleter,
}

// CUDA device pointers are opaque handles. Their allocation lifetime is owned
// by the Send + Sync AllocationDeleter and CUDA permits passing them between
// host threads.
unsafe impl Send for CudaBuffer {}
unsafe impl Sync for CudaBuffer {}

impl CudaBuffer {
    /// Allocates `byte_len` bytes on `device` with `cudaMalloc`.
    ///
    /// The returned buffer owns the allocation and releases it with
    /// `cudaFree`. An empty allocation uses a null address and needs no CUDA
    /// allocation or deallocation call.
    pub fn allocate(byte_len: usize, device: c_int) -> Result<Self, Error> {
        if byte_len == 0 {
            let deleter = AllocationDeleter::new(|| {});
            return Ok(unsafe { Self::from_external(0, 0, device, deleter) });
        }

        let api = api()?;
        let raw = api.with_device(device, |api| {
            let mut raw = ptr::null_mut();
            api.check("cudaMalloc", unsafe { (api.malloc)(&mut raw, byte_len) })?;
            NonNull::new(raw).ok_or(Error::NullHandle {
                operation: "cudaMalloc",
            })
        })?;
        let address = raw.as_ptr() as usize;
        let deleter = AllocationDeleter::new(move || {
            // SAFETY: `address` came from cudaMalloc on this runtime and this
            // closure owns the only release operation for it.
            unsafe {
                let _ = api.with_device(device, |api| {
                    api.check("cudaFree", (api.free)(address as *mut c_void))
                });
            }
        });
        Ok(unsafe { Self::from_external(address, byte_len, device, deleter) })
    }

    /// Creates a zero-copy view over an externally owned CUDA allocation.
    ///
    /// # Safety
    ///
    /// `address` must remain valid for `byte_len` bytes on `device` until the
    /// deleter runs. The deleter must release that ownership without unwinding.
    /// For a non-empty buffer, `address` must be nonzero.
    pub unsafe fn from_external(
        address: usize,
        byte_len: usize,
        device: c_int,
        deleter: AllocationDeleter,
    ) -> Self {
        let address = NonNull::new(ptr::without_provenance_mut(address));
        assert!(
            byte_len == 0 || address.is_some(),
            "a non-empty CUDA buffer must have a non-null address"
        );
        Self {
            address,
            byte_len,
            device,
            _deleter: deleter,
        }
    }

    /// Returns the byte-offset-adjusted CUDA device pointer.
    pub fn as_raw(&self) -> *mut c_void {
        self.address.map_or(ptr::null_mut(), NonNull::as_ptr)
    }

    /// Returns the allocation view length in bytes.
    pub fn byte_len(&self) -> usize {
        self.byte_len
    }

    /// Returns the CUDA device ordinal.
    pub fn device(&self) -> c_int {
        self.device
    }
}

impl CudaStream {
    /// Returns the process-wide relay stream for `device`.
    ///
    /// By default, dlpark attaches to the CUDA Runtime already loaded by the
    /// producer framework so both sides use the same runtime instance. Set
    /// `DLPARK_CUDART_PATH` to force a specific runtime library. A failed
    /// lookup is not cached, so construction may be retried after the
    /// framework initializes CUDA.
    pub fn for_device(device: c_int) -> Result<Arc<Self>, Error> {
        let streams = CUDA_STREAMS.get_or_init(|| Mutex::new(HashMap::new()));
        let mut streams = streams.lock().unwrap_or_else(|error| error.into_inner());
        if let Some(stream) = streams.get(&device) {
            return Ok(Arc::clone(stream));
        }

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
        let stream = Arc::new(Self { api, raw, device });
        streams.insert(device, Arc::clone(&stream));
        Ok(stream)
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
