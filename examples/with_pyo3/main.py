import sys
import os

if os.path.exists('target/debug/mylib.so'):
    os.unlink('target/debug/mylib.so')

os.link('target/debug/libmylib.so', 'target/debug/mylib.so')
sys.path.append("target/debug")

import mylib
from torch.utils.dlpack import from_dlpack, to_dlpack
import numpy as np
import torch


class SimpleTensor:
    def __init__(self, x) -> None:
        self.x = x

    def __dlpack__(self):
        return self.x


print(mylib.add(1, 2))

x1 = np.from_dlpack(SimpleTensor(mylib.arange(10000)))
print(x1.shape, x1[:10], x1[-10:])

x2 = np.from_dlpack(SimpleTensor(mylib.arange(100)))
print(x2.shape, x2[:10], x2[-10:])

print(from_dlpack(mylib.arange(11)) + 1)

dic = mylib.tensordict()
print({k: from_dlpack(v) for k, v in dic.items()})

x3 = torch.rand(2, 3)
print('x3', x3)
mylib.print_tensor(to_dlpack(x3))
# mylib.print_tensor(
#     to_dlpack(
#         torch.rand(21000, 3)
#         # torch.rand(21000, 3).cuda()
#     )
# )
