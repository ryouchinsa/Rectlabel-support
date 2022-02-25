import json

coco_json_read ='/Users/ryo/rcam/test_annotations/test_keypoints_1/annotations.json'
coco_json_save ='/Users/ryo/rcam/test_annotations/test_keypoints_1/annotations_tmp.json'
category_name = 'crab'
idx_list = [2, 0, 1]

def change_category_keypoints(keypoints, idx_list):
	keypoints_tmp = []
	for idx in idx_list:
		keypoints_tmp.append(keypoints[idx])
	return keypoints_tmp

def change_category_skeleton(skeleton, idx_list):
	skeleton_tmp = []
	for pair in skeleton:
		pair_tmp = []
		pair_tmp.append(idx_list.index(pair[0] - 1) + 1)
		pair_tmp.append(idx_list.index(pair[1] - 1) + 1)
		skeleton_tmp.append(pair_tmp)
	return skeleton_tmp

def change_annotation_keypoints(keypoints, idx_list):
	keypoints_tmp = []
	for idx in idx_list:
		for i in range(3):
			keypoints_tmp.append(keypoints[3 * idx + i])
	return keypoints_tmp

file = open(coco_json_read)
coco = json.load(file)
file.close()

category_id = -1
categories = coco['categories']
for category in categories:
	if category['name'] == category_name:
		category_id = category['id']
		keypoints = category['keypoints']
		category['keypoints'] = change_category_keypoints(keypoints, idx_list)
		skeleton = category['skeleton']
		category['skeleton'] = change_category_skeleton(skeleton, idx_list)
coco['categories'] = categories

annotations = coco['annotations']
for annotation in annotations:
	if annotation['category_id'] != category_id:
		continue
	if 'keypoints' not in annotation:
		continue
	keypoints = annotation['keypoints']
	annotation['keypoints'] = change_annotation_keypoints(keypoints, idx_list)
coco['annotations'] = annotations

file_write = open(coco_json_save,'w')
json.dump(coco,file_write,indent=2)






