import numpy as np
import png, os, pydicom
import PIL.Image
import sys

save_as_jpg = True

dcm_file = sys.argv[1]
output_folder = sys.argv[2]

def from2dArray(dcm_filename, idx, pixel):
    shape = pixel.shape
    image_array = pixel.astype(float)
    image_array = (np.maximum(image_array, 0) / image_array.max()) * 255.0
    image_array = np.uint8(image_array)
    if save_as_jpg:
        path_jpg = output_folder + '/' + dcm_filename + '_' + str(idx) + '.jpg'
        im = PIL.Image.fromarray(image_array)
        im.save(path_jpg)
    else:
        path_png = output_folder + '/' + dcm_filename + '_' + str(idx) + '.png'
        with open(path_png, 'wb') as file:
            w = png.Writer(shape[1], shape[0], greyscale=True)
            w.write(file, image_array)

ds = pydicom.dcmread(dcm_file)
dcm_filename = os.path.splitext(os.path.basename(dcm_file))[0]
print(ds)
print(ds.pixel_array.shape)

if ds.pixel_array.ndim == 2:
    idx = 0
    from2dArray(dcm_filename, idx, ds.pixel_array)
else:
    for idx, pixel in enumerate(ds.pixel_array):
        from2dArray(dcm_filename, idx, pixel)


