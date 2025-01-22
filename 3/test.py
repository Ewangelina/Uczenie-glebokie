import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_absolute_error
import matplotlib.pyplot as plt

def compute_binary_metrics(preds, labels, threshold=0.5):
    binary_preds = (np.array(preds) >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(labels, binary_preds),
        "precision": precision_score(labels, binary_preds),
        "recall": recall_score(labels, binary_preds),
        "auc": roc_auc_score(labels, preds),
    }
    return metrics

def plot_embeddings_2d(model, embeddings, labels, title="2D Embeddings Visualization", task="classification"):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap="coolwarm", alpha=0.7, edgecolor="k")
    plt.colorbar(scatter, label="Labels")
    plt.title(title)
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")
    plt.grid(True)

    # Plot approximation function
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
    plt.yticks([])  # Hide y-axis ticks
    plt.grid(True)

    # Plot approximation function
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

def test_bace(model, test_loader, device):
    model = model.to(device)
    model.eval()
    criterion = torch.nn.BCELoss()  # Binary Cross-Entropy Loss

    test_loss = 0.0
    test_preds = []
    test_labels = []
    embeddings = []  # Collect embeddings if size == 2
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data.x, data.edge_index, data.batch).squeeze()
            loss = criterion(output, data.y.view(-1).float())  # Ensure target matches input size
            test_loss += loss.item() * data.num_graphs

            test_preds.extend(output.cpu().numpy())
            test_labels.extend(data.y.cpu().numpy())

            # Collect embeddings if embedding size == 2
            if model.embedding_size in [1, 2]:
                embedding = model.get_embeddings(data.x, data.edge_index, data.batch)
                embeddings.extend(embedding.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    metrics = compute_binary_metrics(test_preds, test_labels)

    print(f"Test Metrics - Loss: {test_loss:.4f}, AUC: {metrics['auc']:.4f}, "
          f"Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")

    # Visualize embeddings if size == 2
    if embeddings:
        embeddings = np.array(embeddings)
        labels = np.array(test_labels)
        if model.embedding_size == 2:
            plot_embeddings_2d(model, embeddings, labels, title="2D Embeddings for Validation Set", task="classification")
        elif model.embedding_size == 1:
            plot_embeddings_1d(model, embeddings, labels, title="1D Embeddings for Validation Set", task="classification")

    return test_loss, metrics


def test_qm9(model, test_loader, device):
    model = model.to(device)
    model.eval()
    criterion = torch.nn.MSELoss()  # Mean Squared Error Loss

    test_loss = 0.0
    test_preds = []
    test_labels = []
    embeddings = []  # Collect embeddings if size == 1 or 2
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data.x, data.edge_index, data.batch).squeeze()
            target = data.y[:, 0].float()
            loss = criterion(output, target)
            test_loss += loss.item() * data.num_graphs

            test_preds.extend(output.cpu().numpy())
            test_labels.extend(target.cpu().numpy())

            # Collect embeddings if embedding size == 1 or 2
            if model.embedding_size in [1, 2]:
                embedding = model.get_embeddings(data.x, data.edge_index, data.batch)
                embeddings.extend(embedding.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    test_mae = mean_absolute_error(test_labels, test_preds)

    print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

    # Visualize embeddings if size == 1 or 2
    if embeddings:
        embeddings = np.array(embeddings)
        labels = np.array(test_labels)
        if model.embedding_size == 2:
            plot_embeddings_2d(model, embeddings, labels, title="2D Embeddings for Validation Set", task="regression")
        elif model.embedding_size == 1:
            plot_embeddings_1d(model, embeddings, labels, title="1D Embeddings for Validation Set", task="regression")

    return test_loss, test_mae