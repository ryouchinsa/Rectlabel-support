import hashlib
import io
import json
import os
import contextlib2
import numpy as np
import PIL.Image

from pycocotools import mask
import tensorflow as tf

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('images_dir', '', 'Full path to images folder.')
flags.DEFINE_string('train_txt_path', '', 'Full path to train.txt.')
flags.DEFINE_string('val_txt_path', '', 'Full path to val.txt.')
flags.DEFINE_string('annotations_file', '', 'Full path to annotations JSON file.')
flags.DEFINE_string('output_dir', '', 'Full path to output directory.')
flags.DEFINE_boolean('include_masks', False, 'To train Mask-RCNN, add --include_masks.')
FLAGS = flags.FLAGS

DUMP_MASK_IMAGES = False

tf.logging.set_verbosity(tf.logging.INFO)

def create_tf_example(image_path,
                      image,
                      annotations_list,
                      category_index,
                      include_masks=False):
    image_height = image['height']
    image_width = image['width']
    filename = image['file_name']
    image_id = image['id']
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    key = hashlib.sha256(encoded_jpg).hexdigest()
    xmin = []
    xmax = []
    ymin = []
    ymax = []
    is_crowd = []
    category_names = []
    category_ids = []
    area = []
    encoded_mask_png = []
    num_annotations_skipped = 0
    for idx, object_annotations in enumerate(annotations_list):
        (x, y, width, height) = tuple(object_annotations['bbox'])
        if width <= 0 or height <= 0:
            num_annotations_skipped += 1
            continue
        if x + width > image_width or y + height > image_height:
            num_annotations_skipped += 1
            continue
        xmin.append(float(x) / image_width)
        xmax.append(float(x + width) / image_width)
        ymin.append(float(y) / image_height)
        ymax.append(float(y + height) / image_height)
        is_crowd.append(object_annotations['iscrowd'])
        category_id = int(object_annotations['category_id'])
        category_ids.append(category_id)
        category_names.append(category_index[category_id]['name'].encode('utf8'))
        area.append(object_annotations['area'])
        if include_masks:
            segm = object_annotations['segmentation']
            if isinstance(segm, list):
                rles = mask.frPyObjects(segm, image_height, image_width)
                rle = mask.merge(rles)
                m = mask.decode(rle)
            else:
                m = mask.decode(segm)     
                mask_shape = m.shape
                m = np.ravel(m, order='F')
                m = m.reshape(mask_shape, order='C')  
            pil_image = PIL.Image.fromarray(m)
            output_io = io.BytesIO()
            pil_image.save(output_io, format='PNG')
            encoded_mask_png.append(output_io.getvalue())            
            if DUMP_MASK_IMAGES:
                m[m > 0] = 255
                pil_image = PIL.Image.fromarray(m)
                save_path = filename.split('.')[0] + "_"  + str(idx)+ ".png"
                save_path = FLAGS.output_dir + '/' + filename.split('.')[0] + '_mask_' + str(idx)+ '.png'
                pil_image.save(save_path)
    feature_dict = {
        'image/height': dataset_util.int64_feature(image_height),
        'image/width': dataset_util.int64_feature(image_width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(str(image_id).encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/label': dataset_util.int64_list_feature(category_ids),
        'image/object/is_crowd': dataset_util.int64_list_feature(is_crowd),
        'image/object/area': dataset_util.float_list_feature(area),
    }
    if include_masks:
        feature_dict['image/object/mask'] = (dataset_util.bytes_list_feature(encoded_mask_png))
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example

def _create_tf_record_from_coco_annotations(tf_record_close_stack, category_index, images, images_id_list, image_files, annotations_index, output_path, include_masks, num_shards):
    print('# Started ' + output_path)
    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(tf_record_close_stack, output_path, num_shards)
    for idx, image_id in enumerate(images_id_list):
        image_file = image_files[idx]
        image_path = os.path.join(FLAGS.images_dir, image_file)
        print(idx, image_path)
        annotations_list = annotations_index[image_id]
        tf_example = create_tf_example(image_path, images[image_id - 1], annotations_list, category_index, include_masks)
        shard_idx = idx % num_shards
        output_tfrecords[shard_idx].write(tf_example.SerializeToString())   

def get_image_filename_list(images):
    image_filename_list = []
    for image in images:
        image_filename_list.append(image['file_name'])
    return image_filename_list;

def get_images_id_list(image_filename_list, image_files):
    images_id_list = []
    for image_file in image_files:
        idx = image_filename_list.index(image_file)
        images_id_list.append(idx + 1)
    return images_id_list

def get_annotations_indx(groundtruth_data, images_id_list):
    annotations_index = {}
    if 'annotations' not in groundtruth_data:
        return annotations_index
    for annotation in groundtruth_data['annotations']:
        image_id = annotation['image_id']
        if image_id not in images_id_list:
            continue
        if image_id not in annotations_index:
            annotations_index[image_id] = []
        annotations_index[image_id].append(annotation)
    return annotations_index

def main(_):
    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)
    train_images = dataset_util.read_examples_list(FLAGS.train_txt_path)
    val_images = dataset_util.read_examples_list(FLAGS.val_txt_path)
    annotations_file = FLAGS.annotations_file
    train_output_path = os.path.join(FLAGS.output_dir, 'train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'val.record')
    with contextlib2.ExitStack() as tf_record_close_stack, tf.gfile.GFile(annotations_file, 'r') as fid:
        groundtruth_data = json.load(fid)
        category_index = label_map_util.create_category_index(groundtruth_data['categories'])
        images = groundtruth_data['images']
        image_filename_list = get_image_filename_list(images)
        train_images_id_list = get_images_id_list(image_filename_list, train_images)
        val_images_id_list = get_images_id_list(image_filename_list, val_images)
        train_annotations_index = get_annotations_indx(groundtruth_data, train_images_id_list)
        val_annotations_index = get_annotations_indx(groundtruth_data, val_images_id_list)
        _create_tf_record_from_coco_annotations(
            tf_record_close_stack,
            category_index,
            images,
            train_images_id_list,
            train_images,
            train_annotations_index,
            train_output_path,
            FLAGS.include_masks,
            num_shards=1)
        _create_tf_record_from_coco_annotations(
            tf_record_close_stack,
            category_index,
            images,
            val_images_id_list,
            val_images,
            val_annotations_index,
            val_output_path,
            FLAGS.include_masks,
            num_shards=1)
        print('# Finished.')

if __name__ == '__main__':
    tf.app.run()
