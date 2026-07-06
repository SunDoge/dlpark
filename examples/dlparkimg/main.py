from loguru import logger
from torch.utils.dlpack import from_dlpack

import dlparkimg
# import matplotlib.pyplot as plt

image_path = "candy.jpg"
logger.info("Reading image: {}", image_path)
tensor = from_dlpack(dlparkimg.read_image(image_path))
logger.info("Tensor shape: {}", tensor.shape)

logger.info("Converting to BGR")
bgr_img = tensor[..., [2, 1, 0]]
output_path = "bgr.jpg"
logger.info("Writing image: {}", output_path)
dlparkimg.write_image(output_path, bgr_img)
logger.info("Done")
