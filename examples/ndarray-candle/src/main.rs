use candle_core::Tensor;
use dlpark::{Local, allocation::dynamic, ffi::DLManagedTensorVersioned};
use ndarray::{Array2, ArrayViewD};
use snafu::{ResultExt, Whatever};

fn main() -> Result<(), Whatever> {
    let array = Array2::<f32>::from_shape_vec((2, 3), vec![1., 2., 3., 4., 5., 6.])
        .whatever_context("failed to build ndarray array")?;
    println!("ndarray array:\n{array}");

    // ndarray -> DLPack: zero-copy, the DLPack tensor borrows the array's own buffer.
    let initialized: dynamic::Initialized<DLManagedTensorVersioned> = Box::new(array)
        .try_into()
        .whatever_context("ndarray -> DLPack failed")?;
    let dlpack: Local<DLManagedTensorVersioned> = unsafe { initialized.finish() };

    // DLPack -> candle::Tensor: a copy, candle has no borrowed CPU tensor type.
    let tensor = Tensor::try_from(&dlpack).whatever_context("DLPack -> candle::Tensor failed")?;
    println!("candle tensor shape: {:?}", tensor.dims());

    let sum = tensor
        .sum_all()
        .whatever_context("candle sum_all failed")?
        .to_scalar::<f32>()
        .whatever_context("candle to_scalar failed")?;
    println!("sum computed by candle: {sum}");
    assert_eq!(sum, 21.0);

    // candle::Tensor -> DLPack: zero-copy, boxes the whole Tensor as the DLPack manager_ctx.
    let initialized: dynamic::Initialized<DLManagedTensorVersioned> =
        Box::new(tensor)
            .try_into()
            .whatever_context("candle::Tensor -> DLPack failed")?;
    let dlpack_back: Local<DLManagedTensorVersioned> = unsafe { initialized.finish() };

    // DLPack -> ndarray view: zero-copy.
    let view = ArrayViewD::<f32>::try_from(&dlpack_back)
        .whatever_context("DLPack -> ndarray view failed")?;
    println!("round-tripped ndarray view:\n{view}");
    assert_eq!(
        view.iter().copied().collect::<Vec<_>>(),
        vec![1., 2., 3., 4., 5., 6.]
    );

    println!("round trip OK");
    Ok(())
}
