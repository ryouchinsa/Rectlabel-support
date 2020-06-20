import hashlib
import io
import os
import glob
import random
from pprint import pprint

from lxml import etree
import numpy as np
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('images_dir', '', 'Full path to images folder.')
flags.DEFINE_string('train_txt_path', '', 'Full path to train.txt.')
flags.DEFINE_string('val_txt_path', '', 'Full path to val.txt.')
flags.DEFINE_string('annotations_dir', '', 'Full path to annotations directory.')
flags.DEFINE_string('masks_dir', '', 'Full path to exported mask images folder.')
flags.DEFINE_string('label_map_path', 'label_map.pbtxt', 'Full path to label map file.')
flags.DEFINE_string('output_dir', '', 'Full path to output directory.')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore difficult instances')
FLAGS = flags.FLAGS

def getClassId(name, label_map_dict):
    if name in label_map_dict:
        return label_map_dict[name]
    name_split = name.split('-')
    name = name_split[0]
    for object_name, object_id in label_map_dict.items():
        object_name_escape = object_name.replace('-', '_')
        if object_name_escape == name:
            return object_id
    return -1

def getDifficult(obj):
    key = 'difficult'
    if key not in obj:
        return False
    difficult = bool(int(obj[key]))
    return difficult

def dict_to_tf_example(data, image_path, masks_dir, label_map_dict, ignore_difficult_instances=False):
    with tf.gfile.GFile(image_path, 'rb') as fid:
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
    masks = []
    difficult_obj = []
    if 'object' in data:
        for idx, obj in enumerate(data['object']):
            name = obj['name']
            if name is None:
                name = ''
            class_id = getClassId(name, label_map_dict)
            if class_id < 0:
                print(name + ' is not in the label map.')
                continue
            difficult = getDifficult(obj)
            if ignore_difficult_instances and difficult:
                print(name + ' is difficult so that skipped.')
                continue
            difficult_obj.append(int(difficult))
            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(name.encode('utf8'))
            classes.append(class_id)
            if masks_dir:
                mask_path = os.path.join(masks_dir, os.path.splitext(data['filename'])[0] + '_object' + str(idx) + '.png')
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
        'image/filename': dataset_util.bytes_feature(data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(data['filename'].encode('utf8')),
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
    }
    if masks_dir:
        encoded_mask_png_list = []
        for mask in masks:
            img = PIL.Image.fromarray(mask)
            output = io.BytesIO()
            img.save(output, format='PNG')
            encoded_mask_png_list.append(output.getvalue())
        feature_dict['image/object/mask'] = (dataset_util.bytes_list_feature(encoded_mask_png_list))
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example

def process_images(image_files, output_path):
    print('# Started ' + output_path)
    annotations_dir = FLAGS.annotations_dir
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    writer = tf.python_io.TFRecordWriter(output_path)
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(FLAGS.images_dir, image_file)
        print(idx, image_path)
        annotation_path = os.path.join(annotations_dir, os.path.splitext(image_file)[0] + '.xml')
        if not os.path.exists(annotation_path):
            print(annotation_path + ' not exists')
            continue;
        with tf.gfile.GFile(annotation_path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
        tf_example = dict_to_tf_example(data, image_path, FLAGS.masks_dir, label_map_dict, FLAGS.ignore_difficult_instances)
        writer.write(tf_example.SerializeToString())
    writer.close()

def main(_):
    train_images = dataset_util.read_examples_list(FLAGS.train_txt_path)
    val_images = dataset_util.read_examples_list(FLAGS.val_txt_path)
    train_output_path = os.path.join(FLAGS.output_dir, 'train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'val.record')
    process_images(train_images, train_output_path)
    process_images(val_images, val_output_path)
    print('# Finished.')

if __name__ == '__main__':
    tf.app.run()
