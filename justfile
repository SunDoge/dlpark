bindgen:
    cargo run -r -p dlpark-bindgen

miri:
    cargo miri test

doc:
    cargo doc --open --no-default-features --features candle,half,image,ndarray
