import cv2
import torch
import pickle
from torchvision import datasets, models, transforms
from skimage.data import lbp_frontal_face_cascade_filename
from skimage.feature import Cascade
from camera import detect, draw
import PIL
import torchvision
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = pickle.load(open(".\\models_saved\\last.sav", 'rb'))
model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
classes = ["glasses", "no_glasses"]

# Define image preprocessing
transform = transforms.Compose([
        transforms.Resize(245),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
thickness              = 1
lineType               = 2

def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.cpu()
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def crop(box, image):
    (x, y, w, h) = box
    return image[y:y+h, x:x+w]

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    detector = Cascade("./face.xml")  # Load the cascade classifier

    while True:
        ret, frame = cap.read()
        boxes = detect(frame, detector)
        #frame = cv2.imread(".\\data\\celebs\\images_sorted\\test\\glasses\\182671.jpg")

        if not boxes == []:
            for box in boxes:
                my_frame = crop(box, frame)
                new_frame = PIL.Image.fromarray(my_frame)
                image_tensor = transform(new_frame).unsqueeze(0).to(device)
            
                #out = torchvision.utils.make_grid(image_tensor)
                #imshow(out)
            
                output = model(image_tensor)
                _, preds = torch.max(output, 1)
                
                (x, y, w, h) = box
                bottomLeftCornerOfText = (x ,y+20)
                cv2.putText(frame, classes[preds[0]], 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
        
        draw(frame, boxes)
        
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
