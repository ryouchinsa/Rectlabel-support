import hashlib
import io
import json
import os
import contextlib2
import numpy as np
np.set_printoptions(threshold=np.nan)
import PIL.Image

from pycocotools import mask
import tensorflow as tf

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('train_image_dir', '', 'Full path to training images directory.')
flags.DEFINE_string('val_image_dir', '', 'Full path to validation images directory.')
flags.DEFINE_string('train_annotations_file', '', 'Full path to training annotations JSON file.')
flags.DEFINE_string('val_annotations_file', '', 'Full path to validation annotations JSON file.')
flags.DEFINE_string('output_dir', '/tmp/', 'Full path to output data directory.')
flags.DEFINE_boolean('include_masks', False, 'If you train Mask-RCNN, add --include_masks otherwise you can remove it. "segmentation" is expected to be RLE format.')
flags.DEFINE_boolean('dump_masks', False, 'Whether to dump mask images. default: False.')
FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def create_tf_example(image,
                      annotations_list,
                      image_dir,
                      category_index,
                      include_masks=False):
    image_height = image['height']
    image_width = image['width']
    filename = image['file_name']
    image_id = image['id']

    full_path = os.path.join(image_dir, filename)
    with tf.gfile.GFile(full_path, 'rb') as fid:
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
            binary_mask = mask.decode(object_annotations['segmentation'])
            mask_shape = binary_mask.shape
            binary_mask = np.ravel(binary_mask, order='F')
            binary_mask = binary_mask.reshape(mask_shape, order='C')

            if FLAGS.dump_masks:
                binary_mask[binary_mask > 0] = 255
                pil_image = PIL.Image.fromarray(binary_mask)
                pil_image.save(FLAGS.train_image_dir + "/mask_" + str(image_id) + "_"  + str(idx)+ ".png")
            else:
                pil_image = PIL.Image.fromarray(binary_mask)
            
            output_io = io.BytesIO()
            pil_image.save(output_io, format='PNG')
            encoded_mask_png.append(output_io.getvalue())
    feature_dict = {
        'image/height':
            dataset_util.int64_feature(image_height),
        'image/width':
            dataset_util.int64_feature(image_width),
        'image/filename':
            dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id':
            dataset_util.bytes_feature(str(image_id).encode('utf8')),
        'image/key/sha256':
            dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded':
            dataset_util.bytes_feature(encoded_jpg),
        'image/format':
            dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin':
            dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax':
            dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin':
            dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax':
            dataset_util.float_list_feature(ymax),
        'image/object/class/label':
            dataset_util.int64_list_feature(category_ids),
        'image/object/is_crowd':
            dataset_util.int64_list_feature(is_crowd),
        'image/object/area':
            dataset_util.float_list_feature(area),
    }
    if include_masks:
        feature_dict['image/object/mask'] = (dataset_util.bytes_list_feature(encoded_mask_png))
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def _create_tf_record_from_coco_annotations(annotations_file, image_dir, output_path, include_masks, num_shards):
    with contextlib2.ExitStack() as tf_record_close_stack, tf.gfile.GFile(annotations_file, 'r') as fid:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(tf_record_close_stack, output_path, num_shards)
        groundtruth_data = json.load(fid)
        images = groundtruth_data['images']
        category_index = label_map_util.create_category_index(groundtruth_data['categories'])

        annotations_index = {}
        if 'annotations' in groundtruth_data:
            for annotation in groundtruth_data['annotations']:
                image_id = annotation['image_id']
                if image_id not in annotations_index:
                    annotations_index[image_id] = []
                annotations_index[image_id].append(annotation)

        for idx, image in enumerate(images):
            annotations_list = annotations_index[image['id']]
            tf_example = create_tf_example(image, annotations_list, image_dir, category_index, include_masks)
            shard_idx = idx % num_shards
            output_tfrecords[shard_idx].write(tf_example.SerializeToString())


def main(_):
    assert FLAGS.train_image_dir, '`train_image_dir` missing.'
    assert FLAGS.val_image_dir, '`val_image_dir` missing.'
    assert FLAGS.train_annotations_file, '`train_annotations_file` missing.'
    assert FLAGS.val_annotations_file, '`val_annotations_file` missing.'

    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)
    train_output_path = os.path.join(FLAGS.output_dir, 'coco_train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'coco_val.record')

    _create_tf_record_from_coco_annotations(
        FLAGS.train_annotations_file,
        FLAGS.train_image_dir,
        train_output_path,
        FLAGS.include_masks,
        num_shards=1)

    _create_tf_record_from_coco_annotations(
        FLAGS.val_annotations_file,
        FLAGS.val_image_dir,
        val_output_path,
        FLAGS.include_masks,
        num_shards=1)


if __name__ == '__main__':
    tf.app.run()
