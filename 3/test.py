import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
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

def plot_embeddings_2d(model, embeddings, labels, title="2D Embeddings Visualization"):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap="coolwarm", alpha=0.7, edgecolor="k")
    plt.colorbar(scatter, label="Labels")
    plt.title(title)
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")
    plt.grid(True)

    # Plot decision boundary
    x_min, x_max = embeddings[:, 0].min() - 1, embeddings[:, 0].max() + 1
    y_min, y_max = embeddings[:, 1].min() - 1, embeddings[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(next(model.parameters()).device)
    with torch.no_grad():
        Z = model.classifier(grid).cpu().numpy()
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.2, colors=["blue", "red"])

    plt.show()

def plot_embeddings_1d(model, embeddings, labels, title="1D Embeddings Visualization"):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embeddings, np.zeros_like(embeddings), c=labels, cmap="coolwarm", alpha=0.7, edgecolor="k")
    plt.colorbar(scatter, label="Labels")
    plt.title(title)
    plt.xlabel("Embedding Value")
    plt.yticks([])  # Hide y-axis ticks
    plt.grid(True)

    # Plot decision boundary
    threshold = 0.5
    plt.axvline(x=threshold, color='red', linestyle='--', label='Decision Boundary')
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
            plot_embeddings_2d(model, embeddings, labels, title="2D Embeddings for Validation Set")
        elif model.embedding_size == 1:
            plot_embeddings_1d(model, embeddings, labels, title="1D Embeddings for Validation Set")

    return test_loss, metrics
