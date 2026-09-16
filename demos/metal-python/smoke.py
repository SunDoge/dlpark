import mlx.core as mx

import dlpark_metal


source = dlpark_metal.MetalTensor.from_values(
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    2,
    3,
)
exchange_source = dlpark_metal.MetalTensor.from_dlpack(source)
target = mx.from_dlpack(source, copy=False)
target_again = mx.from_dlpack(source, copy=False)
exchange_target = mx.from_dlpack(exchange_source, copy=False)
mx.eval(target)
mx.eval(target_again)
mx.eval(exchange_target)

assert list(target.shape) == [2, 3]
assert target.tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
assert target_again.tolist() == target.tolist()
assert exchange_source.metal_buffer == source.metal_buffer
assert exchange_target.tolist() == target.tolist()
print("Metal MTLBuffer → Rust DLPack → MLX: passed")
