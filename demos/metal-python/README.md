# Metal Python interop

This Apple-silicon-only smoke test allocates a shared `MTLBuffer` in Rust,
fills it through its CPU-visible mapping, and transfers the same Metal buffer
to MLX through DLPack. Passing `copy=False` makes MLX reject the input rather
than silently copy if the buffer cannot be adopted.

`MetalTensor.from_values(...)` and `MetalTensor.empty(...)` allocate shared
storage through `MetalBuffer::allocate`. `MetalTensor.from_dlpack(obj)` adopts an
external `id<MTLBuffer>` without copying and stores the original DLPack release
operation as an `AllocationDeleter`. Tensor metadata and `byte_offset` are owned
independently by `MetalTensor`; every export creates a fresh managed header.

Run it on an Apple silicon Mac with Python 3.12 and `uv`:

```bash
uv run --reinstall-package dlpark-metal-python smoke.py
```

The Rust log prints both relevant addresses. `metal_buffer` is the Objective-C
`id<MTLBuffer>` stored in `DLTensor.data`; `contents_pointer` is only its shared
CPU mapping. They are intentionally different values.

The PyO3 class uses `ExportRequest` for Python `__dlpack__` negotiation and
implements `DlpackExchangeProducer`, so DLPack 1.3 consumers can discover its
`__dlpack_c_exchange_api__` type attribute. Its current-work-stream callback
returns null because the demo has no outstanding Metal command queue work.
