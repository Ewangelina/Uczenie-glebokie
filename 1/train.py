import torch
from torch.optim import Adam
from torch.nn import BCELoss
from pytorch_lightning.callbacks import EarlyStopping

def train_model(model, train_loader, val_loader, epochs=10, patience=3, device='cuda'):
    # Check if CUDA is available
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available. Switching to CPU.")
        device = 'cpu'

    model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = BCELoss()

    best_val_loss = float('inf')
    patience_counter = 0

    print(f"Starting training on {'GPU' if device == 'cuda' else 'CPU'}...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, (images, labels, img_paths) in enumerate(train_loader):
            labels = labels.to(device).float().unsqueeze(1)  # No need for a dictionary lookup
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()


            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for images, labels, img_paths in val_loader:
                images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()

        print(f"Epoch {epoch + 1} Summary: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_custom_cnn.pth")  # Save best model
            print("New best model saved.")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered!")
            break
    print("Training complete.")
