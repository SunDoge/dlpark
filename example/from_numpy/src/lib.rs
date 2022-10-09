use dlpak::dlpack;

static mut GIVEN: *mut dlpack::ManagedTensor = std::ptr::null_mut();

fn display(managed_tensor: &dlpack::ManagedTensor) {
    println!("On Rust side:");

    let ndim = managed_tensor.dl_tensor.ndim as usize;

    println!("data = {:?}", managed_tensor.dl_tensor.data);
    println!("device = {:?}", managed_tensor.dl_tensor.device);
    println!("dtype = {:?}", managed_tensor.dl_tensor.dtype);
    println!("ndim = {}", managed_tensor.dl_tensor.ndim);
    println!("shape = {:?}", unsafe {
        std::slice::from_raw_parts(managed_tensor.dl_tensor.shape, ndim)
    });
    println!("strides = {:?}", unsafe {
        std::slice::from_raw_parts(managed_tensor.dl_tensor.strides, ndim)
    });
}

#[no_mangle]
unsafe extern "C" fn finalize() {
    println!("call drop");
    if let Some(deleter) = (*GIVEN).deleter {
        deleter(GIVEN);
    }
}

#[no_mangle]
unsafe extern "C" fn give(managed_tensor: dlpack::ManagedTensor) {
    display(&managed_tensor);
    GIVEN = Box::into_raw(Box::new(managed_tensor));
}

#[no_mangle]
unsafe extern "C" fn free_handle() {
    GIVEN.drop_in_place();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {}
}
