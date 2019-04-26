import turicreate as tc
import sys
import os

csv_file = sys.argv[1]
csv = tc.SFrame(csv_file)

image_folder = sys.argv[2]
data = tc.image_analysis.load_images(image_folder, recursive=True)
data = data.join(csv)

model = tc.object_detector.create(data, max_iterations=3)
model.save('mymodel.model')

model.export_coreml('mymodel.mlmodel')