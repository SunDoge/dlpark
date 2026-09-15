# ndarray and candle interop

This demo round-trips a CPU tensor through `ndarray → DLPack → candle → DLPack → ndarray`.
The ndarray/DLPack boundaries are zero-copy; importing into candle copies because candle has no
borrowed CPU tensor type.

Run it from the repository root:

```bash
cargo run --manifest-path demos/ndarray-candle/Cargo.toml
```
