from PIL import Image
import cv2
import torch
import numpy as np
from torchvision import transforms as t

def black_pad(path):
    img = Image.open(path)
    w, h = img.size
    max_size = np.max((h, w))
    min_size = np.min((h, w))
    if max_size == h:
        img_pad = t.Pad([0, 0, (max_size - min_size), 0])(img)
    else:
        img_pad = t.Pad([0, 0, 0, (max_size - min_size)])(img)
    return img_pad


if __name__ == '__main__':
    img_path = r"F:\imgdata\52.jpg"
    img = Image.open(img_path)
    img_blackpad = black_pad(img_path)
    img_resize = t.Resize((416,416))(img_blackpad)
    factor = 416/np.max(img.size)
    img_blackpad.show()
    img_resize.show()
    print(factor)
    print(img.size)
    print(img_blackpad.size)
