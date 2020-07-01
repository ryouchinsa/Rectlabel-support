"""

# Train a new model starting from pre-trained COCO weights
python rectlabel_coco_detectron2.py train --images_dir=${IMAGES_DIR} --annotations=${ANNOTATIONS} --weights=coco

# Resume training a model that you had trained earlier
python rectlabel_coco_detectron2.py train --images_dir=${IMAGES_DIR} --annotations=${ANNOTATIONS} --weights=last

# Apply inference to an image
python rectlabel_coco_detectron2.py inference --weights=last --image=${IMAGE}

"""

import os
import sys
import datetime

import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2
import random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

def setWeights(args, cfg):
    if args.weights.lower() == "last":
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

def train(args):
    register_coco_instances("my_dataset_train", {}, args.annotations, args.images_dir)
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ()
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    cfg.DATALOADER.NUM_WORKERS = 2
    setWeights(args, cfg)
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 500    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()

def inference(args):
    im = cv2.imread(args.image)
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    setWeights(args, cfg)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    MetadataCatalog.get("mydataset").thing_classes = ["shoes"]
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    print(outputs["instances"].pred_boxes)
    v = Visualizer(im[:, :, ::-1], 
        MetadataCatalog.get("mydataset"), 
        scale=0.8, 
        instance_mode=ColorMode.IMAGE_BW)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    file_name = "inference_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
    out.save(file_name)
    print("Saved to ", file_name)

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train on Detectron2.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'inference'")
    parser.add_argument('--images_dir', required=False,
                        metavar="images_dir",
                        help='images_dir')
    parser.add_argument('--annotations', required=False,
                        metavar="annotations.json",
                        help='annotations.json')
    parser.add_argument('--weights', required=True,
                        metavar="'coco' or 'last'",
                        help="'coco' or 'last'")
    parser.add_argument('--image', required=False,
                        metavar="image",
                        help='image')
    args = parser.parse_args()
    print("command: ", args.command)
    print("images_dir: ", args.images_dir)
    print("annotations: ", args.annotations)
    print("image: ", args.image)
    if args.command == "train":
        train(args)
    elif args.command == "inference":
        inference(args)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'inference'".format(args.command))
        
