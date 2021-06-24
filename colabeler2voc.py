#!/usr/bin/env python

from __future__ import print_function
import argparse
import glob
import os
import os.path as osp
import sys

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
    parser.add_argument("input_dir", help="input directory for colabeler xml files")
    parser.add_argument("output_dir", help="output directory for pascal voc xml files")
    args = parser.parse_args()
    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for filename in glob.glob(osp.join(args.input_dir, "*.xml")):
        print("Converting:", filename)
        base = osp.splitext(osp.basename(filename))[0]
        out_xml_file = osp.join(args.output_dir, base + ".xml")
        tree = lxml.etree.parse(filename)
        root = tree.getroot()
        size = root.find('size')
        width = size.find('width').text
        height = size.find('height').text
        depth = size.find('depth').text
        maker = lxml.builder.ElementMaker()
        xml = maker.annotation(
            maker.folder(),
            maker.filename(),
            maker.size(
                maker.width(width),
                maker.height(height),
                maker.depth(depth),
            ),
        )
        outputs = root.find('outputs')
        object = outputs.find('object')
        for item in object.iter('item'):
            name = item.find('name').text
            polygon_element = item.find('polygon')
            cubic_bezier_element = item.find('cubic_bezier')
            if polygon_element is not None:
                polygon = maker.polygon()
                for i, child in enumerate(polygon_element):
                    idx = int(i / 2) + 1
                    if i % 2 == 0:
                        polygon.append(lxml.etree.XML("<x" + str(idx) + ">" + child.text + "</x" + str(idx) + ">"))
                    else:
                        polygon.append(lxml.etree.XML("<y" + str(idx) + ">" + child.text + "</y" + str(idx) + ">"))
                xml.append(
                    maker.object(
                        maker.name(name),
                        polygon,
                    )
                )
            elif cubic_bezier_element is not None:
                cubic_bezier = maker.cubic_bezier()
                child_num = int(len(cubic_bezier_element.getchildren()) / 6)
                for i, child in enumerate(cubic_bezier_element):
                    idx = int(i / 6) + 1
                    if idx == child_num:
                        continue
                    if i % 6 == 0:
                        cubic_bezier.append(lxml.etree.XML("<x" + str(idx) + ">" + child.text + "</x" + str(idx) + ">"))
                    elif i % 6 == 1:
                        cubic_bezier.append(lxml.etree.XML("<y" + str(idx) + ">" + child.text + "</y" + str(idx) + ">"))
                xml.append(
                    maker.object(
                        maker.name(name),
                        cubic_bezier,
                    )
                )

        with open(out_xml_file, "wb") as f:
            f.write(lxml.etree.tostring(xml, pretty_print=True))

if __name__ == "__main__":
    main()
