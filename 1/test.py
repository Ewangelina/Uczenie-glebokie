import os
import torch
from PIL import Image
from torchvision.transforms import ToTensor

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

def test_on_widerface(model, widerface_root, annotations_file, transform, device='cuda'):
    model.to(device)
    model.eval()

    images, _ = load_widerface_images(widerface_root, annotations_file)
    results = []

    print(f"Testing on {len(images)} WIDERFace images...")
    with torch.no_grad():
        for img_path in images:
            # Load and preprocess the image
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)

            # Predict using the model
            output = model(image_tensor).item()
            label = "Male" if output > 0.5 else "Female"
            results.append((img_path, label))

            print(f"Image: {img_path}, Prediction: {label}")
    
    return results
