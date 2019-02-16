r"""
python object_detection/dataset_tools/rectlabel_create_pascal_tf_record.py \
    --images_dir="${IMAGES_DIR}" \
    --annotations_dir="${ANNOTATIONS_DIR}" \
    --image_list_path="${IMAGE_LIST_PATH}" \
    --label_map_path="${LABEL_MAP_PATH}" \
    --output_path="${OUTPUT_PATH}" \
    --include_masks
"""
import hashlib
import io
import os
import glob
import random
from pprint import pprint

from lxml import etree
import numpy as np
np.set_printoptions(threshold=np.nan)
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('images_dir', '', 'Full path to images directory.')
flags.DEFINE_string('annotations_dir', 'annotations', 'Relative path to annotations directory from images_dir.')
flags.DEFINE_string('image_list_path', 'train.txt', 'Full path to image list file which is one of the output files from "Export train, val, and test.txt".')
flags.DEFINE_string('label_map_path', 'label_map.pbtxt', 'Full path to label map file.')
flags.DEFINE_string('output_path', '', 'Full path to output TFRecord file.')
flags.DEFINE_boolean('include_masks', False, 'If you train Mask-RCNN, add --include_masks otherwise you can remove it. Mask images are expected to be png files and in the annotations folder.')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore difficult instances.')
FLAGS = flags.FLAGS


def getClassId(name, label_map_dict):
    class_id = -1
    for item_name, item_id in label_map_dict.items():
        if name in item_name:
            class_id = item_id
            break
    return class_id

def dict_to_tf_example(data, image_file, annotations_dir, label_map_dict, include_masks, ignore_difficult_instances):
    with tf.gfile.GFile(image_file, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()
    width = int(data['size']['width'])
    height = int(data['size']['height'])
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    masks = []
    if 'object' in data:
        for idx, obj in enumerate(data['object']):
            difficult = bool(int(obj['difficult']))
            if ignore_difficult_instances and difficult:
                continue
            class_id = getClassId(obj['name'], label_map_dict)
            if class_id < 0:
                continue
            difficult_obj.append(int(difficult))
            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(obj['name'].encode('utf8'))
            classes.append(class_id)
            truncated.append(int(obj['truncated']))
            poses.append(obj['pose'].encode('utf8'))

            if include_masks:
                mask_path = os.path.join(annotations_dir, os.path.splitext(data['filename'])[0] + '_object' + str(idx) + '.png')
                with tf.gfile.GFile(mask_path, 'rb') as fid:
                    encoded_mask_png = fid.read()
                encoded_png_io = io.BytesIO(encoded_mask_png)
                mask = PIL.Image.open(encoded_png_io)
                if mask.format != 'PNG':
                    raise ValueError('Mask format not PNG')
                mask_np = np.asarray(mask)
                mask_remapped = (mask_np == 255).astype(np.uint8)
                masks.append(mask_remapped)
    feature_dict = {
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }

    if include_masks:
        encoded_mask_png_list = []
        for mask in masks:
            img = PIL.Image.fromarray(mask)
            output = io.BytesIO()
            img.save(output, format='PNG')
            encoded_mask_png_list.append(output.getvalue())
        feature_dict['image/object/mask'] = (dataset_util.bytes_list_feature(encoded_mask_png_list))
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def main(_):
    images_dir = FLAGS.images_dir
    image_files = dataset_util.read_examples_list(FLAGS.image_list_path)
    annotations_dir = os.path.join(images_dir, FLAGS.annotations_dir)
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    for idx, image_file in enumerate(image_files):
        print(idx, image_file)
        image_file_split = image_file.split('/')
        annotation_path = os.path.join(annotations_dir, os.path.splitext(image_file_split[-1])[0] + '.xml')
        with tf.gfile.GFile(annotation_path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
        tf_example = dict_to_tf_example(data, image_file, annotations_dir, label_map_dict, FLAGS.include_masks, FLAGS.ignore_difficult_instances)
        writer.write(tf_example.SerializeToString())
    writer.close()

        
if __name__ == '__main__':
    tf.app.run()
