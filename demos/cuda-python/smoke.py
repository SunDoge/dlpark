import cupy as cp
import torch
from torch.utils.dlpack import from_dlpack

import dlpark_cuda


print("=== CuPy → Rust CUDA FFI → Torch ===", flush=True)
cupy_source = cp.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=cp.float32)
producer = dlpark_cuda.CudaTensor(cupy_source)
torch_target = from_dlpack(producer)
torch_target_again = from_dlpack(producer)

assert producer.device_id == cupy_source.device.id == torch_target.device.index
assert cupy_source.data.ptr == producer.device_pointer == torch_target.data_ptr()
assert torch_target_again.data_ptr() == producer.device_pointer
assert producer.length == cupy_source.size == torch_target.numel()
assert list(torch_target.shape) == [2, 3]
assert torch_target.cpu().tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
print("passed", flush=True)

print("\n=== Torch → Rust CUDA FFI → CuPy ===", flush=True)
torch_source = torch.tensor(
    [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
    dtype=torch.float32,
    device="cuda",
)
producer = dlpark_cuda.CudaTensor(torch_source)
cupy_target = cp.from_dlpack(producer)
cupy_target_again = cp.from_dlpack(producer)

assert producer.device_id == torch_source.device.index == cupy_target.device.id
assert torch_source.data_ptr() == producer.device_pointer == cupy_target.data.ptr
assert cupy_target_again.data.ptr == producer.device_pointer
assert producer.length == torch_source.numel() == cupy_target.size
assert list(cupy_target.shape) == [2, 3]
assert cupy_target.get().tolist() == [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]
print("passed", flush=True)
