import os
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET 

masks_path = '/Users/ryo/Desktop/test_annotations/Custom_Mask_RCNN/annotations/masks'
xmls_path = '/Users/ryo/Desktop/test_annotations/Custom_Mask_RCNN/annotations/xmls_from_pngs';
template_xml_path = '/Users/ryo/rcam/waysify/downloader/pixels_template.xml';
fill_margin = 1

def makedir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def fillBlack(image_array, rangew, rangeh):
    for j in rangeh:
        for i in rangew:
            image_array[j][i] = 0

def convertMask(image):
    image_array = np.array(image)
    fillBlack(image_array, range(0, w), range(0, fill_margin))
    fillBlack(image_array, range(0, w), range(h - fill_margin, h))
    fillBlack(image_array, range(0, fill_margin), range(0, h))
    fillBlack(image_array, range(w - fill_margin, w), range(0, h))
    image_array[image_array[:] > 0] = 255
    image_from_array = Image.fromarray(image_array).convert('LA')
    return image_from_array

tree = ET.parse(template_xml_path)
root = tree.getroot()
makedir(xmls_path)
mask_files = os.listdir(masks_path)
for img_file in mask_files:
    print(img_file)
    img_filename = os.path.splitext(os.path.basename(img_file))[0]
    image_path = os.path.join(masks_path, img_file)
    try:
        image = Image.open(image_path, 'r')
        w, h = image.size
        image = convertMask(image)
        image.save(xmls_path + '/' + img_filename + '_pixels0.png')

        folder = xmls_path.split('/')[-1]
        root.find('folder').text = folder
        root.find('filename').text = img_filename + '.jpg'
        root.find('size').find('width').text = str(w)
        root.find('size').find('height').text = str(h)
        for obj in root.iter('object'):
            obj.find('name').text = 'toy'
        tree.write(xmls_path + '/' + img_filename + '.xml', encoding='UTF-8')
    except IOError:
        print("IOError", image_path)

