# CUDA Python interop

This local smoke test passes contiguous `float32` CUDA buffers through Rust and DLPack
in both directions: `CuPy → Rust CUDA FFI → Torch` and
`Torch → Rust CUDA FFI → CuPy`. It verifies that all three stages in each
direction observe the same CUDA device pointer.

Prerequisites:

- Linux and a CUDA-capable GPU with a working NVIDIA driver
- a CUDA Runtime (`libcudart`) already loaded by a framework such as Torch,
  or `DLPARK_CUDART_PATH` set to its library path
- Python 3.12 and `uv`

Run it with the dependency group matching the installed CUDA major version:

```bash
uv run --reinstall-package dlpark-cuda-python \
  --group cuda12 \
  smoke.py
```

For CUDA 13, replace `cuda12` with `cuda13`. All Python dependencies are declared in
`pyproject.toml` and locked by `uv.lock`. This example requires GPU hardware and is
intentionally excluded from GitHub Actions.

`CudaTensor` is an ordinary PyO3 class that owns the incoming managed DLPack tensor
and a `dlpark::runtime::cuda::CudaStream`. Every `__dlpack__` call constructs a fresh
legacy or versioned managed tensor according to the consumer's `max_version`.
The extension does not link a CUDA SDK or construct a fake owning buffer around foreign
memory. dlpark uses a macro to generate its small CUDA Runtime function table and
resolves it from the `libcudart` already loaded by CuPy or Torch. This loading design follows
[safetensors' minimal CUDA engine](https://github.com/huggingface/safetensors/blob/b7c0f38b6ae072c3cc6208933df0c81fbd2ef837/bindings/python/src/engine/cuda.rs).

On import, the Python producer is asked to make its data ready on the Rust stream. On
each export, Rust records an event there and queues `cudaStreamWaitEvent` on the destination
stream. A destination that omits its stream gets the protocol's conservative host
synchronization path. The logs show the negotiated version and flags, device, dtype,
layout, allocation pointer, and stream handoffs. The Python assertions export each
producer twice and verify both directions remain zero-copy.
