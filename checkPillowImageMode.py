import PIL.Image


if __name__ == '__main__':
    # mask_path = '/Users/ryo/Downloads/attachments/12108_left_object0.png';
    mask_path = '/Users/ryo/Downloads/attachments/12108_left_object_class0.png';
    # mask_path = '/Users/ryo/Downloads/attachments/12108_left_all_objects.png';
    try:
        image = PIL.Image.open(mask_path, 'r')
        print(image.mode)
    except IOError:
        print("IOError", mask_path)
        