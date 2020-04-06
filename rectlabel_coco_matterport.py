"""

# Train a new model starting from pre-trained COCO weights
python matterport_coco_rectlabel.py train --dataset=/Users/ryo/rcam/Mask_RCNN-master/datasets/fashion --annotations=annotations_rle.json --weights=coco

"""

import os
import sys
import time
import numpy as np
import PIL.Image

from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Dump mask images for debug
DUMP_MASK_IMAGES = False

############################################################
#  Configurations
############################################################

class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + 1 class

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.5

############################################################
#  Dataset
############################################################

class CocoDataset(utils.Dataset):
    def load_coco(self, dataset_dir, annotations_file, subset):
        """Load the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        annotations_file: The COCO JSON file.
        subset: What to load (train, val)
        """
        image_dir = "{}/{}".format(dataset_dir, subset)
        coco = COCO("{}/{}".format(image_dir, annotations_file))
        
        # Load all classes
        class_ids = sorted(coco.getCatIds())
        image_ids = []
        for id in class_ids:
            image_ids.extend(list(coco.getImgIds(catIds=[id])))
        # Remove duplicates
        image_ids = list(set(image_ids))

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for idx, annotation in enumerate(annotations):
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info, idx)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return info["path"]
        else:
            super(CocoDataset, self).image_reference(image_id)

    def annToMask(self, ann, image_info, idx):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            height = image_info["height"]
            width = image_info["width"]
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
            m = maskUtils.decode(rle)
        else:
            # rle
            m = maskUtils.decode(segm)     
            mask_shape = m.shape
            m = np.ravel(m, order='F')
            m = m.reshape(mask_shape, order='C')           
        if DUMP_MASK_IMAGES:
            m[m > 0] = 255
            pil_image = PIL.Image.fromarray(m)
            save_path = image_info["path"].split('.')[0] + "_"  + str(idx)+ ".png"
            pil_image.save(save_path)
        return m

def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CocoDataset()
    dataset_train.load_coco(args.dataset, args.annotations, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CocoDataset()
    dataset_val.load_coco(args.dataset, args.annotations, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')

def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash
    
def detect_and_color_splash(model, image_path):
    assert image_path

    # Run model detection and generate the color splash effect
    print("Running on {}".format(args.image))
    # Read image
    image = skimage.io.imread(args.image)
    # Detect objects
    r = model.detect([image], verbose=1)[0]
    # Color splash
    splash = color_splash(image, r['masks'])
    # Save output
    file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
    skimage.io.imsave(file_name, splash)
    print("Saved to ", file_name)

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'inference'")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/dataset/",
                        help='Directory of the dataset')
    parser.add_argument('--annotations', required=True,
                        metavar="annotations.json",
                        help='The COCO JSON file')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the inference')
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Annotations: ", args.annotations)
    print("Logs: ", args.logs)
    print("Image: ", args.image)

    # Configurations
    if args.command == "train":
        config = CocoConfig()
    else:
        class InferenceConfig(CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or inference
    if args.command == "train":
        train(model)
    elif args.command == "inference":
        detect_and_color_splash(model, image_path=args.image)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'inference'".format(args.command))
