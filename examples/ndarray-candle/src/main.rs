use candle_core::Tensor;
use snafu::{ResultExt, Whatever};

fn main() -> Result<(), Whatever> {
    let tensor = Tensor::new(&[1, 2, 3], &candle_core::Device::Cpu)
        .whatever_context("fail to create candle tensor")?;
    println!("Hello, world!");

    Ok(())
}
