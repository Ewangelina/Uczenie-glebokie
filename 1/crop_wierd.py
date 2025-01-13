import cv2
import numpy as np
from PIL import Image
from tempfile import TemporaryDirectory
import shutil
import os
from pathlib import Path

from skimage.data import lbp_frontal_face_cascade_filename
from skimage.feature import Cascade
def detect(frame, detector):
    detections = detector.detect_multi_scale(img=frame, scale_factor=1.2, step_ratio=1,
                                             min_size=(10, 10), max_size=(200, 200))
    boxes = []
    for detection in detections:
        x = detection['c']
        y = detection['r']
        w = detection['width']
        h = detection['height']
        boxes.append((x, y, w, h))
    return boxes

def crop(box, image):
    (x, y, w, h) = box
    return image[y:y+h, x:x+w]

directory = os.fsencode(".\\data\\wider\\not_sorted")
file = "./face.xml"
detector = Cascade(file)
num = 300
pathlist = Path(".\\data\\wider\\not_sorted").glob('**/*.jpg')
    
for file in pathlist:
    im = cv2.imread(file)
    boxes = detect(im, detector)
    for box in boxes:
        print(box)
        im1 = crop(box, im)
        filename = ".\\data\\wider\\cropped\\" + str(num) + ".jpg"
        num = num + 1
        cv2.imwrite(filename,im1)
print("DONE!!!!!")

