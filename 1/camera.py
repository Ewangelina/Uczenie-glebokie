import cv2

from skimage.data import lbp_frontal_face_cascade_filename
from skimage.feature import Cascade

def detect(frame, detector):
    detections = detector.detect_multi_scale(img=frame, scale_factor=1.2, step_ratio=1,
                                             min_size=(100, 100), max_size=(200, 200))
    boxes = []
    for detection in detections:
        x = detection['c']
        y = detection['r']
        w = detection['width']
        h = detection['height']
        boxes.append((x, y, w, h))
    return boxes


def draw(frame, boxes):
    for x, y, w, h in boxes:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2)


if __name__ == '__main__':
    # file = lbp_frontal_face_cascade_filename()
    file = "./face.xml"
    detector = Cascade(file)

    cap = cv2.VideoCapture(0)
    skip = 5
    i = 0
    boxes = []
    while (True):
        ret, frame = cap.read()
        if i % skip == 0:
            boxes = detect(frame, detector)
        draw(frame, boxes)
        cv2.imshow('Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i += 1
    # cap.release()
    # cv2.destroyAllWindows()
