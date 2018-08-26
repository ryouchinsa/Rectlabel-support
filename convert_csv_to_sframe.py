r"""
python convert_csv_to_sframe.py "${EXPORTED_CSV_FILE}"
"""
import turicreate as tc
from turicreate import SArray
import pandas as pd
import os
import sys

csv_file = sys.argv[1]

csv = pd.read_csv(csv_file)

for i, row in csv.iterrows():
    image_folder = str(os.path.split(row['path'])[0])
    break;

data = tc.image_analysis.load_images(image_folder, with_path=True)

annotations = []
for j, item in enumerate(data):
    has_annotation = False
    for i, row in csv.iterrows():
        if str(row['path']) == item['path']:
            annotations.append(eval(row['annotations']))
            has_annotation = True
            break
    if not has_annotation:
        annotations.append([])

data['annotations'] = SArray(data=annotations, dtype=list)

data.save('annotations.sframe')
print(data)



