import coremltools

coreml_model = coremltools.converters.keras.convert('./model_data/yolo.h5', input_names='input1', image_input_names='input1', output_names=['output1', 'output2', 'output3'], image_scale=1/255.)

coreml_model.input_description['input1'] = 'Input image'
coreml_model.output_description['output1'] = 'The 13x13 grid (Scale1)'
coreml_model.output_description['output2'] = 'The 26x26 grid (Scale2)'
coreml_model.output_description['output3'] = 'The 52x52 grid (Scale3)'

coreml_model.author = 'Original paper: Joseph Redmon, Ali Farhadi'
coreml_model.license = 'Public Domain'
coreml_model.short_description = "The YOLOv3 network from the paper 'YOLOv3: An Incremental Improvement'"

labels = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]
coreml_model.user_defined_metadata['classes'] = ",".join(labels)
coreml_model.user_defined_metadata['confidence_threshold'] = '0.25'
coreml_model.user_defined_metadata['non_maximum_suppression_threshold'] = '0.45'

coreml_model.save('Yolov3.mlmodel')
