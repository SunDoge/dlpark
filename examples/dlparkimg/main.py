import dlparkimg
from torch.utils.dlpack import to_dlpack, from_dlpack
from loguru import logger
# import matplotlib.pyplot as plt

image_path = "candy.jpg"
logger.info("Reading image: {}", image_path)
tensor = from_dlpack(dlparkimg.read_image(image_path))
logger.info("Tensor shape: {}", tensor.shape)

logger.info("Converting to BGR")
bgr_img = tensor[..., [2, 1, 0]]
output_path = "bgr.jpg"
logger.info("Writing image: {}", output_path)
dlparkimg.write_image(output_path, to_dlpack(bgr_img))
logger.info("Done")
