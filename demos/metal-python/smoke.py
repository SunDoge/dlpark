import mlx.core as mx

import dlpark_metal


source = dlpark_metal.MetalTensor(
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    2,
    3,
)
target = mx.from_dlpack(source, copy=False)
target_again = mx.from_dlpack(source, copy=False)
mx.eval(target)
mx.eval(target_again)

assert list(target.shape) == [2, 3]
assert target.tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
assert target_again.tolist() == target.tolist()
print("Metal MTLBuffer → Rust DLPack → MLX: passed")
