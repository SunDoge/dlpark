# Metal Python interop

This Apple-silicon-only smoke test allocates a shared `MTLBuffer` in Rust,
fills it through its CPU-visible mapping, and transfers the same Metal buffer
to MLX through DLPack. Passing `copy=False` makes MLX reject the input rather
than silently copy if the buffer cannot be adopted.

Run it on an Apple silicon Mac with Python 3.12 and `uv`:

```bash
uv run --reinstall-package dlpark-metal-python mlx_smoke.py
```

The Rust log prints both relevant addresses. `metal_buffer` is the Objective-C
`id<MTLBuffer>` stored in `DLTensor.data`; `contents_pointer` is only its shared
CPU mapping. They are intentionally different values.
