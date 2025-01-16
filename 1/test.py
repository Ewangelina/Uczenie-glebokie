import os
import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def save_confusion_matrix(y_true, y_pred, class_names, output_file):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, colorbar=False)
    plt.title("Confusion Matrix")
    plt.savefig(output_file)
    plt.close()


def save_misclassified_images(image_paths, predictions, labels, output_dir, num_examples=5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    misclassified_count = 0
    for img_path, pred, true in zip(image_paths, predictions, labels):
        if pred != true:
            misclassified_count += 1
            image = Image.open(img_path)
            output_path = os.path.join(output_dir, f"misclassified_{misclassified_count}_{os.path.basename(img_path)}")
            image.save(output_path)
            if misclassified_count >= num_examples:
                break

def test_model(model, test_loader, device='cuda'):
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available. Switching to CPU.")
        device = 'cpu'

    model.to(device)
    model.eval()

    correct = 0
    total = 0
    predictions = []
    labels = []
    image_paths = []

    print(f"Starting testing on {'GPU' if device == 'cuda' else 'CPU'}...")
    with torch.no_grad():
        for batch_idx, (images, batch_labels, paths) in enumerate(test_loader):
            images, batch_labels = images.to(device), batch_labels.to(device).float().unsqueeze(1)

            outputs = model(images)
            batch_predictions = (outputs > 0.5).float()
            predictions.extend(batch_predictions.cpu().numpy().flatten())
            labels.extend(batch_labels.cpu().numpy().flatten())
            image_paths.extend(paths)  # Collect image paths

            correct += (batch_predictions == batch_labels).sum().item()
            total += batch_labels.size(0)

            if (batch_idx + 1) % 10 == 0:
                print(f"Processed Batch [{batch_idx + 1}/{len(test_loader)}]")

    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Save confusion matrix
    save_confusion_matrix(
        y_true=labels, y_pred=predictions,
        class_names=["Female", "Male"],
        output_file="celeba_confusion_matrix.png"
    )

    # Save misclassified images
    save_misclassified_images(
        image_paths=image_paths, predictions=predictions, labels=labels,
        output_dir="celeba_misclassified", num_examples=5
    )

    return accuracy


def test_on_widerface(model, csv_file, root_dir, transform, device='cuda'):
    # Load annotations
    annotations = pd.read_csv(csv_file)
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    predictions = []
    labels = []
    image_paths = []

    print(f"Testing on {len(annotations)} cropped WIDERFace images...")
    with torch.no_grad():
        for _, row in annotations.iterrows():
            img_path = os.path.join(root_dir, row['file'])

            if not os.path.exists(img_path):
                print(f"File not found: {img_path}. Skipping.")
                continue

            label = 1 if row['label'] == 'Male' else 0
            image_paths.append(img_path)
            labels.append(label)

            # Load and preprocess the image
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)

            # Predict using the model
            output = model(image_tensor).item()
            pred_label = 1 if output > 0.5 else 0
            predictions.append(pred_label)

            if pred_label == label:
                correct += 1
            total += 1

    accuracy = correct / total * 100
    print(f"Accuracy on WIDERFace: {accuracy:.2f}%")

    # Save confusion matrix
    save_confusion_matrix(
        y_true=labels, y_pred=predictions, 
        class_names=["Female", "Male"], 
        output_file="widerface_confusion_matrix.png"
    )

    # Save misclassified images
    save_misclassified_images(
        image_paths=image_paths, predictions=predictions, labels=labels, 
        output_dir="widerface_misclassified", num_examples=5
    )

    return predictions, accuracy
