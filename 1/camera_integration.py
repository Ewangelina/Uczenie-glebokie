import cv2
import torch
from custom_cnn import CustomCNN
from utils.transforms import get_transforms

# Load the trained model
model = CustomCNN()
model.load_state_dict(torch.load("best_custom_cnn.pth"))
model.eval()

# Define image preprocessing
transform = get_transforms()

# Define face detection logic (existing from camera.py)
from camera import detect, draw

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    detector = Cascade("./face.xml")  # Load the cascade classifier

    while True:
        ret, frame = cap.read()
        boxes = detect(frame, detector)
        draw(frame, boxes, model, transform)
        cv2.imshow('Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
