import mylib
import numpy as np


class SimpleTensor:
    def __init__(self, x) -> None:
        self.x = x

    def __dlpack__(self):
        return self.x

print(mylib.add(1, 2))

x = mylib.arange(10)
print(x)
y = np.from_dlpack(SimpleTensor(x))
print(y)
print(y.shape)

x2 = mylib.arange2(11)
y2 = np.from_dlpack(x2)
print(y)
y2 = np.from_dlpack(x2)
