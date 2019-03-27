r"""
python train_turicreate.py "${EXPORTED_CSV_FILE}"
"""
import turicreate as tc
import sys
import os

csv_file = sys.argv[1]
csv = tc.SFrame(csv_file)

image_path = csv['path'][0]
image_folder = str(os.path.split(image_path)[0])

data = tc.image_analysis.load_images(image_folder, recursive=False)
data = data.join(csv)

model = tc.object_detector.create(data, max_iterations=3)
model.save('mymodel.model')

model.export_coreml('mymodel.mlmodel')