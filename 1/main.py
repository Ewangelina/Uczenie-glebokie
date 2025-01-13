import torch
from custom_cnn import CustomCNN
from utils.dataset import get_dataloaders
from train import train_model
from test import test_model, test_on_widerface
from utils.transforms import get_transforms
import pandas as pd

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {'GPU' if device == 'cuda' else 'CPU'}.")

    # Load model
    model = CustomCNN()

    # Get dataloaders
    print("Loading data...")
    train_loader, val_loader = get_dataloaders()
    print("Data loaded.")

    # Train the model
    print("Starting training...")
    train_model(model, train_loader, val_loader, epochs=10, patience=3, device=device)

    # Test the model on CelebA
    print("Starting testing on CelebA...")
    test_model(model, val_loader, device=device)
    print("Testing on CelebA complete.")

    # Test the model on WIDERFace cropped faces
    print("Starting testing on WIDERFace...")
    csv_file = './data/annotations.csv'
    root_dir = './data/WIDER/selected_faces'
    results, accuracy = test_on_widerface(model, csv_file, root_dir, get_transforms(), device=device)
    print(f"Testing on WIDERFace complete. Accuracy: {accuracy:.2f}%")

    # Save predictions to a CSV file
    results_df = pd.DataFrame(results, columns=['file', 'predicted_label'])
    results_df.to_csv('./data/widerface_predictions.csv', index=False)
    print("Predictions saved to widerface_predictions.csv")

