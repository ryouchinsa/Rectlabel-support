import PIL.Image
import numpy as np

if __name__ == '__main__':
    image_path = '/Users/ryo/Desktop/test_annotations/mask/tuce-311434-unsplash_all_objects.png';
    try:
        image = PIL.Image.open(image_path, 'r')
        print(image.mode)
        palette = image.getpalette()
        palette = np.array(palette).reshape(-1, 3)
        print(palette)
    except IOError:
        print("IOError", image_path)

        