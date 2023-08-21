import dlparkimg
from torch.utils.dlpack import to_dlpack, from_dlpack
# import matplotlib.pyplot as plt

tensor = from_dlpack(dlparkimg.read_image("candy.jpg"))

print(tensor.shape)
# plt.imshow(tensor.numpy())
# plt.show()

bgr_img = tensor[..., [2, 1, 0]]
dlparkimg.write_image('bgr.jpg', to_dlpack(bgr_img))
