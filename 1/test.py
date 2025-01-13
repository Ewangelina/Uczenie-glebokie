import os
import pandas as pd
import torch
from PIL import Image

def test_model(model, test_loader, device='cuda'):
    # Ensure CUDA availability
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available. Switching to CPU.")
        device = 'cpu'

    model.to(device)
    model.eval()

    correct = 0
    total = 0

    print(f"Starting testing on {'GPU' if device == 'cuda' else 'CPU'}...")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

            outputs = model(images)
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % 10 == 0:
                print(f"Processed Batch [{batch_idx + 1}/{len(test_loader)}]")

    accuracy = correct / total * 100
    with open('celeba_acc.txt', 'w') as f:
        f.write(f"Test Accuracy: {accuracy:.2f}%\n")
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

def load_widerface_images(root_dir, annotations_file, num_images=100):
    images = []
    annotations = []

    # Read the bounding box annotation file
    with open(annotations_file, 'r') as file:
        lines = file.readlines()
    
    img_count = 0
    for line in lines:
        line = line.strip()
        if line.endswith('.jpg'):  # Image file
            img_path = os.path.join(root_dir, line)
            if os.path.exists(img_path):
                images.append(img_path)
                img_count += 1
                if img_count >= num_images:
                    break
        elif len(line.split()) > 1:  # Annotation line
            annotations.append(line)
    
    return images, annotations

def test_on_widerface(model, csv_file, root_dir, transform, device='cuda'):
    """
    Test the model on WIDERFace cropped and annotated dataset.

    Args:
        model (nn.Module): Trained model.
        csv_file (str): Path to the CSV file with annotations.
        root_dir (str): Directory containing cropped face images.
        transform (callable): Transformation to apply to images.
        device (str): Device to run the model on.
    """
    # Load annotations
    annotations = pd.read_csv(csv_file)
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    predictions = []

    print(f"Testing on {len(annotations)} cropped WIDERFace images...")
    with torch.no_grad():
        for _, row in annotations.iterrows():
            img_path = os.path.join(root_dir, row['file'])

            label = 1 if row['label'] == 'Male' else 0

            # Load and preprocess the image
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)

            # Predict using the model
            output = model(image_tensor).item()
            pred_label = 1 if output > 0.5 else 0

            predictions.append((img_path, pred_label))
            if pred_label == label:
                correct += 1
            total += 1

            print(f"Image: {img_path}, Prediction: {'Male' if pred_label == 1 else 'Female'}")

    accuracy = correct / total * 100
    print(f"Accuracy on WIDERFace: {accuracy:.2f}%")
    return predictions, accuracy
