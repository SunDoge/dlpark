# dlpark

![Crates.io](https://img.shields.io/crates/v/dlpark?style=for-the-badge)

A pure Rust implementation of [dmlc/dlpack](https://github.com/dmlc/dlpack).

Check [example/with_pyo3](./example/with_pyo3) for usage.

This implementation focuses on transferring tensor from Rust to Python and vice versa.

It can also be used without `pyo3` as a Rust library with `default-features = false`, check [example/from_numpy](./example/from_numpy).

## Quick Start

We provide a simple example of how to transfer `image::RgbImage` to Python and `torch.Tensor` to Rust.

[Full code is here](https://github.com/SunDoge/dlparkimg).

### Rust $\rightarrow$ Python

We have to implement some traits for a struct to be able to converted to `PyObject`

```rust
use std::ffi::c_void;
use dlpark::prelude::*;

struct PyRgbImage(image::RgbImage);

impl HasData for PyRgbImage {
    fn data(&self) -> *mut std::ffi::c_void {
        self.0.as_ptr() as *const c_void as *mut c_void
    }
}

impl HasDevice for PyRgbImage {
    fn device(&self) -> Device {
        Device::CPU
    }
}

impl HasDtype for PyRgbImage {
    fn dtype(&self) -> DataType {
        DataType::U8
    }
}

impl HasShape for PyRgbImage {
    fn shape(&self) -> Shape {
        Shape::Owned(
            [self.0.height(), self.0.width(), 3]
                .map(|x| x as i64)
                .to_vec(),
        )
    }
}

// Strides can be infered from Shape since it's compact and row-majored.
impl HasStrides for PyRgbImage {}
impl HasByteOffset for PyRgbImage {}
```

Then we can return a `ManagerCtx<PyRgbImage>`

```rust
#[pyfunction]
fn read_image(filename: &str) -> ManagerCtx<PyRgbImage> {
    let img = image::open(filename).unwrap();
    let rgb_img = img.to_rgb8();
    PyRgbImage(rgb_img).into()
}
```

You can acess it in Python

```python
import dlparkimg
from torch.utils.dlpack import to_dlpack, from_dlpack
import matplotlib.pyplot as plt

tensor = from_dlpack(dlparkimg.read_image("candy.jpg"))

print(tensor.shape)
plt.imshow(tensor.numpy())
plt.show()
```

If you want to convert it to `numpy.ndarray` , you can make a simple wrapper like this

```python
import numpy as np
import dlparkimg

class FakeTensor:

    def __init__(self, x):
        self.x = x

    def __dlpack__(self):
        return self.x

arr = np.from_dlpack(FakeTensor(dlparkimg.read_image("candy.jpg")))
```

### Python $\rightarrow$ Rust

`ManagedTensor` holds the memory of tensor and provide methods to access the tensor's attributes.

```rust
#[pyfunction]
fn write_image(filename: &str, tensor: ManagedTensor) {
    let buf = tensor.as_slice::<u8>();

    let rgb_img = image::ImageBuffer::<Rgb<u8>, _>::from_raw(
        tensor.shape()[1] as u32,
        tensor.shape()[0] as u32,
        buf,
    )
    .unwrap();

    rgb_img.save(filename).unwrap();
}
```

And you can call it in Python

```python
import dlparkimg
from torch.utils.dlpack import to_dlpack, from_dlpack

bgr_img = tensor[..., [2, 1, 0]] # [H, W, C=3]
dlparkimg.write_image('bgr.jpg', to_dlpack(bgr_img))
```
