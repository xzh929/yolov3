import xml.etree.ElementTree as ET
import numpy as np
import os

xml_path = r"F:\data2\outputs"
txt_path = os.path.join(r"F:\data2", "img.txt")

file = open(txt_path, "w")
file.write("filename class center_x center_y width height\n")
img_name = 1

for filename in os.listdir(xml_path):
    tree = ET.parse(os.path.join(xml_path, filename))
    root = tree.getroot()
    file.write("{}.jpg ".format(img_name))
    for elem in root.iter(tag="item"):
        name = elem.findtext("name")
        x1 = int(elem.findtext("bndbox/xmin"))
        y1 = int(elem.findtext("bndbox/ymin"))
        x2 = int(elem.findtext("bndbox/xmax"))
        y2 = int(elem.findtext("bndbox/ymax"))
        x = (x2 + x1) // 2
        y = (y2 + y1) // 2
        w = x2 - x1
        h = y2 - y1
        file.write("{0} {1} {2} {3} {4} ".format(name, x, y, w, h))
        print(filename, name, x, y, w, h)
    file.write("\n")
    img_name += 1
file.close()
