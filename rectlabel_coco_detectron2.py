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
        if args.type.lower() == "maskrcnn":
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        else:
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")

def setNumClasses(cfg):
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # only has one class (ballon)

def setClasses(dataset_name):
    MetadataCatalog.get(dataset_name).thing_classes = ["person"]

def setKeypoints(dataset_name):
    MetadataCatalog.get(dataset_name).keypoint_names = ["head", "neck", "rshoulder", "relbow", "rwrist", "lshoulder", "lelbow", "lwrist", "rhip", "rknee", "rankle", "lhip", "lknee", "lankle"]
    MetadataCatalog.get(dataset_name).keypoint_flip_map = [("rshoulder", "lshoulder"), ("relbow", "lelbow"), ("rwrist", "lwrist"), ("rhip", "lhip"), ("rknee", "lknee"), ("rankle", "lankle")]
    MetadataCatalog.get(dataset_name).keypoint_connection_rules = [("head", "neck", (0, 255, 255)), ("neck", "lshoulder", (0, 255, 255)), ("lshoulder", "lelbow", (0, 255, 255)), ("lelbow", "lwrist", (0, 255, 255)), ("neck", "rshoulder", (0, 255, 255)), ("rshoulder", "relbow", (0, 255, 255)), ("relbow", "rwrist", (0, 255, 255)), ("neck", "lhip", (0, 255, 255)), ("lhip", "lknee", (0, 255, 255)), ("lknee", "lankle", (0, 255, 255)), ("neck", "rhip", (0, 255, 255)), ("rhip", "rknee", (0, 255, 255)), ("rknee", "rankle", (0, 255, 255))]

def train(args):
    dataset_name = "dataset_train"
    register_coco_instances(dataset_name, {}, args.annotations_path, args.images_dir)
    cfg = get_cfg()
    if args.type.lower() == "maskrcnn":
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    else:
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
        setKeypoints(dataset_name)
        cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 14
    cfg.DATASETS.TRAIN = (dataset_name,)
    cfg.DATASETS.TEST = ()                
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    cfg.DATALOADER.NUM_WORKERS = 2
    setWeights(args, cfg)
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 500    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    setNumClasses(cfg)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

def inference(args):
    dataset_name = "dataset_val"
    im = cv2.imread(args.image)
    cfg = get_cfg()
    if args.type.lower() == "maskrcnn":
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    else:
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
        setKeypoints(dataset_name)
        cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 14
    setWeights(args, cfg)
    setNumClasses(cfg)
    setClasses(dataset_name)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    print(outputs["instances"].pred_boxes)
    v = Visualizer(im[:, :, ::-1], 
        MetadataCatalog.get(dataset_name))
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
    parser.add_argument('--type', required=True,
                        metavar="'maskrcnn' or 'keypoint'",
                        help="'maskrcnn' or 'keypoint'")
    parser.add_argument('--images_dir', required=False,
                        metavar="images_dir",
                        help='images_dir')
    parser.add_argument('--annotations_path', required=False,
                        metavar="annotations_path",
                        help='annotations_path')
    parser.add_argument('--weights', required=True,
                        metavar="'coco' or 'last'",
                        help="'coco' or 'last'")
    parser.add_argument('--image', required=False,
                        metavar="image",
                        help='image')
    args = parser.parse_args()
    print("command: ", args.command)
    print("type: ", args.type)
    print("images_dir: ", args.images_dir)
    print("annotations_path: ", args.annotations_path)
    print("weights: ", args.weights)
    print("image: ", args.image)
    if args.command == "train":
        train(args)
    elif args.command == "inference":
        inference(args)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'inference'".format(args.command))
        
