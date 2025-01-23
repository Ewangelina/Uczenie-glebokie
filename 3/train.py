import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam


def plot_embeddings_2d(model, embeddings, labels, title="2D Embeddings Visualization", task="classification"):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap="coolwarm", alpha=0.7, edgecolor="k")
    plt.colorbar(scatter, label="Labels")
    plt.title(title)
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")
    plt.grid(True)

    x_min, x_max = embeddings[:, 0].min() - 1, embeddings[:, 0].max() + 1
    y_min, y_max = embeddings[:, 1].min() - 1, embeddings[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(next(model.parameters()).device)
    with torch.no_grad():
        if task == "classification":
            Z = model.classifier(grid).cpu().numpy()
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.2, colors=["blue", "red"])
        elif task == "regression":
            Z = model.predictor(grid).cpu().numpy()
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z, levels=50, cmap="coolwarm", alpha=0.2)

    plt.show()

def plot_embeddings_1d(model, embeddings, labels, title="1D Embeddings Visualization", task="classification"):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embeddings, np.zeros_like(embeddings), c=labels, cmap="coolwarm", alpha=0.7, edgecolor="k")
    plt.colorbar(scatter, label="Labels")
    plt.title(title)
    plt.xlabel("Embedding Value")
    plt.yticks([])
    plt.grid(True)

    x_min, x_max = embeddings.min() - 1, embeddings.max() + 1
    x_vals = np.linspace(x_min, x_max, 100)
    x_vals_tensor = torch.tensor(x_vals, dtype=torch.float32).to(next(model.parameters()).device).view(-1, 1)
    with torch.no_grad():
        if task == "classification":
            y_vals = model.classifier(x_vals_tensor).cpu().numpy()
            plt.axvline(x=0.5, color='red', linestyle='--', label='Decision Boundary')
        elif task == "regression":
            y_vals = model.predictor(x_vals_tensor).cpu().numpy()
            plt.plot(x_vals, y_vals, color='red', linestyle='--', label='Approximation Function')
    plt.legend()

    plt.show()

def train_bace(model, train_loader, val_loader, device, num_epochs=100, lr=0.001, patience=10):
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data.x, data.edge_index, data.batch).squeeze()
            loss = criterion(output, data.y.float().squeeze())
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.num_graphs

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        embeddings = []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                output = model(data.x, data.edge_index, data.batch).squeeze()
                loss = criterion(output, data.y.float().squeeze())
                val_loss += loss.item() * data.num_graphs

                val_preds.extend(output.cpu().numpy())
                val_labels.extend(data.y.cpu().numpy())

                if model.embedding_size in [1, 2]:
                    embedding = model.get_embeddings(data.x, data.edge_index, data.batch)
                    embeddings.extend(embedding.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_auc = roc_auc_score(val_labels, val_preds)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    if embeddings:
        embeddings = np.array(embeddings)
        labels = np.array(val_labels)
        if model.embedding_size == 2:
            plot_embeddings_2d(model, embeddings, labels, title="2D Embeddings for Validation Set", task="classification")
        elif model.embedding_size == 1:
            plot_embeddings_1d(model, embeddings, labels, title="1D Embeddings for Validation Set", task="")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def train_qm9(model, train_loader, val_loader, device, num_epochs=10, lr=0.001, patience=3):
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data.x, data.edge_index, data.batch).squeeze()
            target = data.y[:, 0].float()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.num_graphs

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        embeddings = []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                output = model(data.x, data.edge_index, data.batch).squeeze()
                target = data.y[:, 0].float()
                loss = criterion(output, target)
                val_loss += loss.item() * data.num_graphs

                val_preds.extend(output.cpu().numpy())
                val_labels.extend(target.cpu().numpy())

                if model.embedding_size in [1, 2]:
                    embedding = model.get_embeddings(data.x, data.edge_index, data.batch)
                    embeddings.extend(embedding.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_mae = mean_absolute_error(val_labels, val_preds)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    if embeddings:
        embeddings = np.array(embeddings)
        labels = np.array(val_labels)
        if model.embedding_size == 2:
            plot_embeddings_2d(model, embeddings, labels, title="2D Embeddings for Validation Set", task="regression")
        elif model.embedding_size == 1:
            plot_embeddings_1d(model, embeddings, labels, title="1D Embeddings for Validation Set", task="regression")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()