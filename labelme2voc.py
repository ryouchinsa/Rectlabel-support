#!/usr/bin/env python

from __future__ import print_function
import argparse
import glob
import os
import os.path as osp
import sys

import labelme

try:
    import lxml.builder
    import lxml.etree
except ImportError:
    print("Please install lxml:\n\n    pip install lxml\n")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", help="input directory for labelme json files and images")
    parser.add_argument("output_dir", help="output directory for pascal voc xml files")
    args = parser.parse_args()
    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for filename in glob.glob(osp.join(args.input_dir, "*.json")):
        print("Converting:", filename)
        label_file = labelme.LabelFile(filename=filename)
        base = osp.splitext(osp.basename(filename))[0]
        out_xml_file = osp.join(args.output_dir, base + ".xml")
        img = labelme.utils.img_data_to_arr(label_file.imageData)
        maker = lxml.builder.ElementMaker()
        xml = maker.annotation(
            maker.folder(),
            maker.filename(),
            maker.size(
                maker.height(str(img.shape[0])),
                maker.width(str(img.shape[1])),
                maker.depth(str(img.ndim)),
            ),
        )
        for shape in label_file.shapes:
            if shape["shape_type"] != "polygon":
                continue
            polygon = maker.polygon()
            for i, xy in enumerate(shape["points"]):
                x, y = xy
                polygon.append(lxml.etree.XML("<x" + str(i + 1) + ">" + str(x) + "</x" + str(i + 1) + ">"))
                polygon.append(lxml.etree.XML("<y" + str(i + 1) + ">" + str(y) + "</y" + str(i + 1) + ">"))
            xml.append(
                maker.object(
                    maker.name(shape["label"]),
                    polygon,
                )
            )
        with open(out_xml_file, "wb") as f:
            f.write(lxml.etree.tostring(xml, pretty_print=True))

if __name__ == "__main__":
    main()
