
from torch.utils.dlpack import from_dlpack
import numpy as np
import sys
import os

if not os.path.exists('target/debug/mylib.so'):
    os.link('target/debug/libmylib.so', 'target/debug/mylib.so')
sys.path.append("target/debug")
import mylib


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
