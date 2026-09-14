# CUDA Python interop

This local smoke test passes contiguous `float32` CUDA buffers through DLPack in both
directions: `CuPy → cudarc → Torch` and `Torch → cudarc → CuPy`. It verifies that all
three stages in each direction observe the same CUDA device pointer.

Prerequisites:

- a CUDA-capable GPU and working NVIDIA driver
- a CUDA toolkit discoverable by `cudarc`
- Python 3.12 and `uv`

Run it with the dependency group matching the installed CUDA major version:

```bash
uv run --reinstall-package dlpark-cuda-python \
  --group cuda12 \
  cupy_torch.py
```

For CUDA 13, replace `cuda12` with `cuda13`. All Python dependencies are declared in
`pyproject.toml` and locked by `uv.lock`. This example requires GPU hardware and is
intentionally excluded from GitHub Actions.

`CudarcTensorF32` owns the intermediate cudarc view and implements Python's DLPack
producer protocol for the destination framework. Its Rust implementation logs the
negotiated version and flags, device, dtype, shape, strides, compactness, byte offset,
allocation size, stream, and cudarc device pointer. The Python assertions then verify
both directions preserve the allocation.
