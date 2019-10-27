import PIL.Image
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)

image_path = '/Users/ryo/Desktop/test_annotations/test_data_mask/index_color_image/23823333-189975791565259-1963930219081367552-n_all_objects.png'
try:
    image = PIL.Image.open(image_path, 'r')
    print(image.mode)
    img = np.array(image)
    print(img[np.where(img > 0)])
    palette = image.getpalette()
    palette = np.array(palette).reshape(-1, 3)
    print(palette)
except IOError:
    print("IOError", image_path)
