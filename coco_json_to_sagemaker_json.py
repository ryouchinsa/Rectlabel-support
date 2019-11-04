import json
import os
import random
import shutil

images_path = '/Users/ryo/Desktop/test_annotations/raccoon_dataset-master/images';
coco_json_path = '/Users/ryo/Desktop/test_annotations/raccoon_dataset-master/annotations.json'

sagemaker_json_path = '/Users/ryo/Desktop/test_annotations/raccoon_dataset-master/sagemaker_tmp';
to_be_uploaded_to_s3_path = '/Users/ryo/Desktop/test_annotations/raccoon_dataset-master/sagemaker';
sagemaker_images_path_train = to_be_uploaded_to_s3_path + '/train';
sagemaker_images_path_val = to_be_uploaded_to_s3_path + '/validation';
sagemaker_json_path_train = to_be_uploaded_to_s3_path + '/train_annotation';
sagemaker_json_path_val = to_be_uploaded_to_s3_path + '/validation_annotation';

def makedir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

makedir(sagemaker_json_path)
makedir(to_be_uploaded_to_s3_path)
makedir(sagemaker_images_path_train)
makedir(sagemaker_images_path_val)
makedir(sagemaker_json_path_train)
makedir(sagemaker_json_path_val)

def fixCategoryId(category_id):
    return category_id - 1;

with open(coco_json_path) as f:
    js = json.load(f)
    images = js['images']
    categories = js['categories']
    annotations = js['annotations']
    for i in images:
        jsonFile = i['file_name']
        jsonFile = jsonFile.split('.')[0] + '.json'

        line = {}
        line['file'] = i['file_name']
        line['image_size'] = [{
            'width': int(i['width']),
            'height': int(i['height']),
            'depth': 3
        }]
        line['annotations'] = []
        line['categories'] = []
        for j in annotations:
            if j['image_id'] == i['id'] and len(j['bbox']) > 0:
                line['annotations'].append({
                    'class_id': fixCategoryId(int(j['category_id'])),
                    'top': int(j['bbox'][1]),
                    'left': int(j['bbox'][0]),
                    'width': int(j['bbox'][2]),
                    'height': int(j['bbox'][3])
                })
                class_name = ''
                for k in categories:
                    if int(j['category_id']) == k['id']:
                        class_name = str(k['name'])
                assert class_name is not ''
                line['categories'].append({
                    'class_id': fixCategoryId(int(j['category_id'])),
                    'name': class_name
                })
        if line['annotations']:
            with open(os.path.join(sagemaker_json_path, jsonFile), 'w') as p:
                json.dump(line, p)

jsons = os.listdir(sagemaker_json_path)
num_annotated_files = len(jsons)
train_split_pct = 0.90
num_train_jsons = int(num_annotated_files * train_split_pct)
random.seed(0)
random.shuffle(jsons)
train_jsons = jsons[:num_train_jsons]
val_jsons = jsons[num_train_jsons:]

count_train = 0
for i in train_jsons:
    file_name = i.split('.')[0]
    if len(file_name) == 0:
        continue
    image_file_path = images_path + '/' + file_name + '.jpg'
    shutil.copy(image_file_path, sagemaker_images_path_train)
    shutil.copy(sagemaker_json_path + '/'+i, sagemaker_json_path_train)
    count_train += 1

count_val = 0
for i in val_jsons:
    file_name = i.split('.')[0]
    if len(file_name) == 0:
        continue
    image_file_path = images_path + '/' + file_name + '.jpg'
    shutil.copy(image_file_path, sagemaker_images_path_val)
    shutil.copy(sagemaker_json_path + '/'+i, sagemaker_json_path_val)
    count_val += 1

print ('train_jsons {}'.format(count_train))
print ('val_jsons {}'.format(count_val))
