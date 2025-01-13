import cv2
import torch
from torchvision import transforms
import numpy as np
from custom_cnn import CustomCNN  # Replace with your actual model file

# Load Haar cascade
CASCADE_PATH = 'D:/1111/studia/2sem/Uczenie-glebokie/1/face.xml'
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# Load model
MODEL_PATH = 'D:/1111/studia/2sem/Uczenie-glebokie/1/best_custom_cnn.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CustomCNN()
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

# Transformation pipeline
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def predict_attributes(face, model):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_pil = transforms.ToPILImage()(face)
    face_tensor = transform(face_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(face_tensor)
        prediction = (output > 0.5).cpu().numpy().squeeze()

    return prediction

# Try opening the webcam
cap = cv2.VideoCapture(0)  # Default camera index
if not cap.isOpened():
    print("Error: Could not open webcam. Check camera connection or index.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        prediction = predict_attributes(face, model)
        label = f"Male: {prediction:.2f}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Face Attribute Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
