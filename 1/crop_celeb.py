import cv2
import numpy as np
from PIL import Image
from tempfile import TemporaryDirectory
import shutil

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
    (x, y, w, h) = box[0]
    return image[y:y+h, x:x+w]

file = "./face.xml"
detector = Cascade(file)

f = open(".\\data\\celebs\\Anno\\glasses2.csv")
last_student = -1
student_output = open(".\\data\\celebs\\Anno\\glasses.csv") #file not used
group = "train\\"
cecha = "glasses"

for task_line in f:
    task = task_line.split(";")
    file = task[0]
    if file == "162771.jpg":
        group = "val\\"
    if file == "182638.jpg":
        group = "test\\"
        
    glasses = int(task[1])
    if glasses == 1:
        cecha = "glasses\\"
    else:
        cecha = "no_glasses\\"

    newdir = ".\\data\\celebs\\images_sorted\\" + group + cecha + file
    olddir = ".\\data\\celebs\\Img\\img_align_celeba\\img_align_celeba\\" + file
    
    im = cv2.imread(olddir)
    #cv2.imshow('image',im)
    boxes = detect(im, detector)
    if not boxes == []:
        print(boxes)
        im1 = crop(boxes, im)
        #cv2.imshow('image',im1)
        cv2.imwrite(newdir,im1)    
f.close()
print("DONE!!!!!")

