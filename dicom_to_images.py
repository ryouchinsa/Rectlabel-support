import numpy as np
import png, os, pydicom
import PIL.Image
import sys

save_as_jpg = True

dcm_file = sys.argv[1]
ds = pydicom.dcmread(dcm_file)
print(ds)

dcm_filename = os.path.splitext(dcm_file)[0]
for idx, pixel in enumerate(ds.pixel_array):
    shape = pixel.shape
	image_2d = pixel.astype(float)
	image_2d = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0
	image_2d = np.uint8(image_2d)
	if save_as_jpg:
		path_jpg = dcm_filename + '_' + str(idx) + '.jpg'
		im = PIL.Image.fromarray(image_2d)
		im.save(path_jpg)
	else:
		path_png = dcm_filename + '_' + str(idx) + '.png'
		with open(path_png, 'wb') as file:
		    w = png.Writer(shape[1], shape[0], greyscale=True)
		    w.write(file, image_2d)

