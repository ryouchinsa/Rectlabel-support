from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import pprint
pp = pprint.PrettyPrinter(indent=2)
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

imgFolder = '/Users/ryo/rcam/test_annotations/test/donuts/images'
annFile ='/Users/ryo/rcam/test_annotations/test/donuts/coco_train.json'
imageIdToShow = 3

coco = COCO(annFile)
catIds = list(range(1, 256))
imgIds = coco.getImgIds();
img = coco.loadImgs(imgIds[imageIdToShow])[0]

I = io.imread(imgFolder + '/' + img['file_name'])
plt.axis('off')
plt.imshow(I);
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)
plt.show()

import cv2
image = cv2.imread(imgFolder + img['file_name'])
for i in anns:
	print(i['bbox'])
	[x,y,w,h] = i['bbox']
	cv2.rectangle(image, (x, y), (x+w, y+h), (255,0,0), 5)
	cv2.imshow('',image)
cv2.waitKey(0)