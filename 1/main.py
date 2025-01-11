import torch
from custom_cnn import CustomCNN
from utils.dataset import get_dataloaders
from train import train_model
from test import test_model, test_on_widerface
from utils.transforms import get_transforms

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

    # Test the model on WIDERFace
    print("Starting testing on WIDERFace...")
    widerface_root = './data/WIDER/WIDER_train/images'
    annotations_file = './data/WIDER/wider_face_split/wider_face_train_bbx_gt.txt'
    results = test_on_widerface(model, widerface_root, annotations_file, get_transforms(), device=device)
    print("Testing on WIDERFace complete.")