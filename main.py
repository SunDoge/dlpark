import mylib
import numpy as np
from torch.utils.dlpack import from_dlpack


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
