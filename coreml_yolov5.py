import yaml
import coremltools

yaml_file = '/Users/ryo/Downloads/yolov5/data/coco128.yaml'
coreml_file = '/Users/ryo/Downloads/yolov5/yolov5m.mlmodel'

with open(yaml_file) as file:
    obj = yaml.safe_load(file)
    names = obj['names']
    if type(names) == dict:
        labels = [v for (k, v) in names.items()]
    else:
        labels = names
    print(labels)

coreml_model = coremltools.models.MLModel(coreml_file)
coreml_model.user_defined_metadata['classes'] = ",".join(labels)
coreml_model.save(coreml_file)
