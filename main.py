import mylib
import numpy as np


class SimpleTensor:
    def __init__(self, x) -> None:
        self.x = x

    def __dlpack__(self):
        return self.x

print(mylib.add(1, 2))

x2 = mylib.arange(100)
print(x2)
y = np.from_dlpack(SimpleTensor(x2))
print(y)
print(y.shape)

x = mylib.arange(10)
print(x)
y = np.from_dlpack(SimpleTensor(x))
print(y)
print(y.shape)
del x


