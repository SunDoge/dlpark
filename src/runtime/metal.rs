//! Shared Metal allocations for zero-copy DLPack export on Apple silicon.
//!
//! Storage-mode-shared buffers are visible to both the CPU and GPU on Apple
//! silicon. DLPack consumers receive the Objective-C `id<MTLBuffer>` in
//! `DLTensor.data`; [`MetalBuffer::contents_ptr`](crate::runtime::metal::MetalBuffer::contents_ptr)
//! is only the CPU mapping.

use crate::{allocation::device::DeviceAllocation, ffi::DLDevice};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{
    MTLBuffer as RawMTLBuffer, MTLCreateSystemDefaultDevice, MTLDevice, MTLResourceOptions,
};
use snafu::Snafu;
use std::{ffi::c_void, ptr::NonNull, sync::OnceLock};

/// An error returned while obtaining a Metal device or allocating a buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Snafu)]
pub enum Error {
    /// No system-default Metal device is available.
    #[snafu(display("no system-default Metal device"))]
    DeviceUnavailable,
    /// Metal returned `nil` for an allocation request.
    #[snafu(display("Metal buffer allocation failed for {nbytes} bytes"))]
    Allocation {
        /// Requested logical byte length.
        nbytes: usize,
    },
}

static DEVICE: OnceLock<Option<DeviceHandle>> = OnceLock::new();

struct DeviceHandle(Retained<ProtocolObject<dyn MTLDevice>>);

// Apple documents MTLDevice as thread-safe.
unsafe impl Send for DeviceHandle {}
unsafe impl Sync for DeviceHandle {}

fn device() -> Result<&'static ProtocolObject<dyn MTLDevice>, Error> {
    DEVICE
        .get_or_init(|| MTLCreateSystemDefaultDevice().map(DeviceHandle))
        .as_ref()
        .map(|handle| &*handle.0)
        .ok_or(Error::DeviceUnavailable)
}

/// An owning shared-mode `MTLBuffer` allocation.
pub struct MetalBuffer {
    buffer: Retained<ProtocolObject<dyn RawMTLBuffer>>,
    nbytes: usize,
    contents: NonNull<c_void>,
}

// MTLBuffer is thread-safe. Mutable host access still requires `&mut self`.
unsafe impl Send for MetalBuffer {}
unsafe impl Sync for MetalBuffer {}

impl MetalBuffer {
    /// Allocates shared CPU/GPU storage on the system-default Metal device.
    ///
    /// A zero-byte logical allocation requests one physical byte because Metal
    /// must still return a valid buffer object for DLPack consumers.
    pub fn allocate(nbytes: usize) -> Result<Self, Error> {
        let buffer = device()?
            .newBufferWithLength_options(nbytes.max(1), MTLResourceOptions::StorageModeShared)
            .ok_or(Error::Allocation { nbytes })?;
        let contents = buffer.contents();
        Ok(Self {
            buffer,
            nbytes,
            contents,
        })
    }

    /// Returns the logical allocation size in bytes.
    pub fn len(&self) -> usize {
        self.nbytes
    }

    /// Returns whether the logical allocation is empty.
    pub fn is_empty(&self) -> bool {
        self.nbytes == 0
    }

    /// Returns the CPU-visible shared-storage address.
    pub fn contents_ptr(&self) -> *mut c_void {
        self.contents.as_ptr()
    }

    /// Returns mutable access to the CPU-visible shared storage.
    pub fn as_mut_bytes(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.contents.cast().as_ptr(), self.nbytes) }
    }

    /// Returns the Objective-C `id<MTLBuffer>` used by DLPack Metal tensors.
    pub fn as_metal_id(&self) -> *mut c_void {
        (&*self.buffer as *const ProtocolObject<dyn RawMTLBuffer>)
            .cast_mut()
            .cast()
    }
}

unsafe impl DeviceAllocation for MetalBuffer {
    fn dlpack_data(&self) -> *mut c_void {
        self.as_metal_id()
    }

    fn device(&self) -> DLDevice {
        DLDevice::metal(0)
    }
}
