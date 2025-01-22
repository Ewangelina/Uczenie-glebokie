import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from torch_geometric.transforms import BaseTransform
from train import train_bace, train_qm9
from test import test_bace, test_qm9
from model import GCNLinearClassifierModel, GCNNonlinearClassifierModel, TransformerLinearClassifierModel, TransformerNonlinearClassifierModel, GCNLinearRegressionModel, GCNNonlinearRegressionModel, TransformerLinearRegressionModel, TransformerNonlinearRegressionModel
import numpy as np
import random

# Function to split dataset into training, validation, and test sets
def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1):
    num_data = len(dataset)
    indices = list(range(num_data))
    train_idx, temp_idx = train_test_split(indices, train_size=train_ratio, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)

    return train_dataset, val_dataset, test_dataset

class ConvertToFloat(BaseTransform):
    def __call__(self, data):
        data.x = data.x.float()
        return data

def load_bace_dataset(batch_size=32):
    print("Loading BACE dataset...")
    from torch_geometric.datasets import MoleculeNet
    bace_dataset = MoleculeNet(root='data/BACE', name='BACE', transform=ConvertToFloat())

    # Split the dataset
    train_dataset, val_dataset, test_dataset = split_dataset(bace_dataset)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"BACE Dataset: {len(train_dataset)} training samples, {len(val_dataset)} validation samples, {len(test_dataset)} test samples.")
    return train_loader, val_loader, test_loader

# Load the QM9 dataset (Regression task)
def load_qm9_dataset(batch_size=32):
    print("Loading QM9 dataset...")
    qm9_dataset = QM9(root='data/QM9')

    # Split the dataset
    train_dataset, val_dataset, test_dataset = split_dataset(qm9_dataset)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"QM9 Dataset: {len(train_dataset)} training samples, {len(val_dataset)} validation samples, {len(test_dataset)} test samples.")
    return train_loader, val_loader, test_loader


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set the seed for reproducibility
set_seed(42)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model parameters
    hidden_dim = 64
    embedding_dim = 128
    output_dim = 1
    classifier_hidden_dim = 32  # For nonlinear classifier

    # Load datasets
    bace_train_loader, bace_val_loader, bace_test_loader = load_bace_dataset()
    bace_input_dim = 9

    # Train models for BACE (Binary Classification)
    gcn_linear_classifier_model = GCNLinearClassifierModel(bace_input_dim, hidden_dim, embedding_dim)
    train_bace(gcn_linear_classifier_model, bace_train_loader, bace_val_loader, device)
    test_bace(gcn_linear_classifier_model, bace_test_loader, device)

    gcn_nonlinear_classifier_model = GCNNonlinearClassifierModel(bace_input_dim, hidden_dim, embedding_dim, classifier_hidden_dim)
    train_bace(gcn_nonlinear_classifier_model, bace_train_loader, bace_val_loader, device)
    test_bace(gcn_nonlinear_classifier_model, bace_test_loader, device)

    transformer_linear_classifier_model = TransformerLinearClassifierModel(bace_input_dim, hidden_dim, embedding_dim)
    train_bace(transformer_linear_classifier_model, bace_train_loader, bace_val_loader, device)
    test_bace(transformer_linear_classifier_model, bace_test_loader, device)

    transformer_nonlinear_classifier_model = TransformerNonlinearClassifierModel(bace_input_dim, hidden_dim, embedding_dim, classifier_hidden_dim)
    train_bace(transformer_nonlinear_classifier_model, bace_train_loader, bace_val_loader, device)
    test_bace(transformer_nonlinear_classifier_model, bace_test_loader, device)

    # qm9_train_loader, qm9_val_loader, qm9_test_loader = load_qm9_dataset()

    # qm9_dataset = QM9(root='data/QM9')
    # input_dim = qm9_dataset.num_features

    # # Train models for QM9 (Regression)
    # print("Training GCNLinearRegressionModel...")
    # gcn_linear_model = GCNLinearRegressionModel(input_dim, hidden_dim, embedding_dim, output_dim)
    # train_qm9(gcn_linear_model, qm9_train_loader, qm9_val_loader, device)
    # test_qm9(gcn_linear_model, qm9_test_loader, device)

    # print("Training GCNNonlinearRegressionModel...")
    # gcn_nonlinear_model = GCNNonlinearRegressionModel(input_dim, hidden_dim, embedding_dim, classifier_hidden_dim, output_dim)
    # train_qm9(gcn_nonlinear_model, qm9_train_loader, qm9_val_loader, device)
    # test_qm9(gcn_nonlinear_model, qm9_test_loader, device)

    # print("Training TransformerLinearRegressionModel...")
    # transformer_linear_model = TransformerLinearRegressionModel(input_dim, hidden_dim, embedding_dim, output_dim)
    # train_qm9(transformer_linear_model, qm9_train_loader, qm9_val_loader, device)
    # test_qm9(transformer_linear_model, qm9_test_loader, device)

    # print("Training TransformerNonlinearRegressionModel...")
    # transformer_nonlinear_model = TransformerNonlinearRegressionModel(input_dim, hidden_dim, embedding_dim, classifier_hidden_dim, output_dim)
    # train_qm9(transformer_nonlinear_model, qm9_train_loader, qm9_val_loader, device)
    # test_qm9(transformer_nonlinear_model, qm9_test_loader, device)

