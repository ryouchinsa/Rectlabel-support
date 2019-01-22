import json
import numpy as np
import PIL.Image
from pycocotools import mask


if __name__ == '__main__':
    json_path = '/Users/ryo/Desktop/annotations.json';
    with open(json_path,'r') as fid:
        groundtruth_data = json.load(fid)
        annotations_index = {}
        if 'annotations' in groundtruth_data:
            for annotation in groundtruth_data['annotations']:
                image_id = annotation['image_id']
                if image_id not in annotations_index:
                    annotations_index[image_id] = []
                annotations_index[image_id].append(annotation)
        images = groundtruth_data['images']
        for idx, image in enumerate(images):
            annotations_list = annotations_index[image['id']]
            for idx, object_annotations in enumerate(annotations_list):
                binary_mask = mask.decode(object_annotations['segmentation'])
                mask_shape = binary_mask.shape
                binary_mask = np.ravel(binary_mask, order='F')
                binary_mask = binary_mask.reshape(mask_shape, order='C')
                binary_mask[binary_mask > 0] = 255
                pil_image = PIL.Image.fromarray(binary_mask)
                pil_image.save(str(image_id) + "_"  + str(idx)+ ".png")