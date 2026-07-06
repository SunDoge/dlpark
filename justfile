

bindgen:
    cargo run -r -p dlpark-bindgen


miri:
    cargo miri test
