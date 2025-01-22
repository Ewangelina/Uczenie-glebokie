import torch
from torch.nn import Module, Linear, ReLU, Dropout, Sequential, BatchNorm1d
from torch_geometric.nn import global_max_pool, GCNConv, TransformerConv

# GCN-based GNN
class GCNModel(Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embedding_dim)
        self.bn1 = BatchNorm1d(hidden_dim)
        self.bn2 = BatchNorm1d(embedding_dim)
        self.relu = ReLU()

    def forward(self, x, edge_index, batch):
        x = x.float()  # Ensure input is of type float
        x = self.bn1(self.relu(self.conv1(x, edge_index)))
        x = self.bn2(self.relu(self.conv2(x, edge_index)))
        x = global_max_pool(x, batch)
        return x

# Transformer-based GNN
class TransformerModel(Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super(TransformerModel, self).__init__()
        self.conv1 = TransformerConv(input_dim, hidden_dim)
        self.conv2 = TransformerConv(hidden_dim, embedding_dim)
        self.bn1 = BatchNorm1d(hidden_dim)
        self.bn2 = BatchNorm1d(embedding_dim)
        self.relu = ReLU()

    def forward(self, x, edge_index, batch):
        x = self.bn1(self.relu(self.conv1(x, edge_index)))
        x = self.bn2(self.relu(self.conv2(x, edge_index)))
        x = global_max_pool(x, batch)
        return x

# Linear Predictor for Regression
class LinearPredictor(Module):
    def __init__(self, embedding_dim, output_dim):
        super(LinearPredictor, self).__init__()
        self.linear = Linear(embedding_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# Nonlinear Predictor for Regression
class NonlinearPredictor(Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(NonlinearPredictor, self).__init__()
        self.network = Sequential(
            Linear(embedding_dim, hidden_dim),
            ReLU(),
            Dropout(0.5),
            Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)
    
# Linear Classifier for Classification
class LinearClassifier(Module):
    def __init__(self, embedding_dim):
        super(LinearClassifier, self).__init__()
        self.linear = Linear(embedding_dim, 1)  # Output 1 for binary classification

    def forward(self, x):
        return torch.sigmoid(self.linear(x))  # Apply sigmoid for binary classification

# Nonlinear Classifier for Classification
class NonlinearClassifier(Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(NonlinearClassifier, self).__init__()
        self.network = Sequential(
            Linear(embedding_dim, hidden_dim),
            ReLU(),
            Dropout(0.5),
            Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.network(x))

# Combined Regression Models
class GCNLinearRegressionModel(Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, output_dim=1):
        super(GCNLinearRegressionModel, self).__init__()
        self.gcn = GCNModel(input_dim, hidden_dim, embedding_dim)
        self.predictor = LinearPredictor(embedding_dim, output_dim)  # output_dim should be 1
        self.embedding_size = embedding_dim 

    def forward(self, x, edge_index, batch):
        x = self.gcn(x, edge_index, batch)
        return self.predictor(x)

    def get_embeddings(self, x, edge_index, batch):
        return self.gcn(x, edge_index, batch)


class GCNNonlinearRegressionModel(Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, predictor_hidden_dim, output_dim=1):
        super(GCNNonlinearRegressionModel, self).__init__()
        self.gcn = GCNModel(input_dim, hidden_dim, embedding_dim)
        self.predictor = NonlinearPredictor(embedding_dim, predictor_hidden_dim, output_dim)  # output_dim=1
        self.embedding_size = embedding_dim 

    def forward(self, x, edge_index, batch):
        x = self.gcn(x, edge_index, batch)
        return self.predictor(x)

    def get_embeddings(self, x, edge_index, batch):
        return self.gcn(x, edge_index, batch)


class TransformerLinearRegressionModel(Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, output_dim=1):
        super(TransformerLinearRegressionModel, self).__init__()
        self.transformer = TransformerModel(input_dim, hidden_dim, embedding_dim)
        self.predictor = LinearPredictor(embedding_dim, output_dim)
        self.embedding_size = embedding_dim 

    def forward(self, x, edge_index, batch):
        x = self.transformer(x, edge_index, batch)
        return self.predictor(x)
    
    def get_embeddings(self, x, edge_index, batch):
        return self.transformer(x, edge_index, batch)

class TransformerNonlinearRegressionModel(Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, predictor_hidden_dim, output_dim=1):
        super(TransformerNonlinearRegressionModel, self).__init__()
        self.transformer = TransformerModel(input_dim, hidden_dim, embedding_dim)
        self.predictor = NonlinearPredictor(embedding_dim, predictor_hidden_dim, output_dim)
        self.embedding_size = embedding_dim 

    def forward(self, x, edge_index, batch):
        x = self.transformer(x, edge_index, batch)
        return self.predictor(x)
    
    def get_embeddings(self, x, edge_index, batch):
        return self.transformer(x, edge_index, batch)

    

# Combined Models for Classification
class GCNLinearClassifierModel(Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super(GCNLinearClassifierModel, self).__init__()
        self.gcn = GCNModel(input_dim, hidden_dim, embedding_dim)
        self.classifier = LinearClassifier(embedding_dim)
        self.embedding_size = embedding_dim  # Add this line

    def forward(self, x, edge_index, batch):
        x = self.gcn(x, edge_index, batch)
        return self.classifier(x)
    
    def get_embeddings(self, x, edge_index, batch):
        return self.gcn(x, edge_index, batch)

class GCNNonlinearClassifierModel(Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, classifier_hidden_dim):
        super(GCNNonlinearClassifierModel, self).__init__()
        self.gcn = GCNModel(input_dim, hidden_dim, embedding_dim)
        self.classifier = NonlinearClassifier(embedding_dim, classifier_hidden_dim)
        self.embedding_size = embedding_dim  # Add this line

    def forward(self, x, edge_index, batch):
        x = self.gcn(x, edge_index, batch)
        return self.classifier(x)
    
    def get_embeddings(self, x, edge_index, batch):
        return self.gcn(x, edge_index, batch)

class TransformerLinearClassifierModel(Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super(TransformerLinearClassifierModel, self).__init__()
        self.transformer = TransformerModel(input_dim, hidden_dim, embedding_dim)
        self.classifier = LinearClassifier(embedding_dim)
        self.embedding_size = embedding_dim  # Add this line

    def forward(self, x, edge_index, batch):
        x = self.transformer(x, edge_index, batch)
        return self.classifier(x)
    
    def get_embeddings(self, x, edge_index, batch):
        return self.transformer(x, edge_index, batch)

class TransformerNonlinearClassifierModel(Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, classifier_hidden_dim):
        super(TransformerNonlinearClassifierModel, self).__init__()
        self.transformer = TransformerModel(input_dim, hidden_dim, embedding_dim)
        self.classifier = NonlinearClassifier(embedding_dim, classifier_hidden_dim)
        self.embedding_size = embedding_dim  # Add this line

    def forward(self, x, edge_index, batch):
        x = self.transformer(x, edge_index, batch)
        return self.classifier(x)
    
    def get_embeddings(self, x, edge_index, batch):
        return self.transformer(x, edge_index, batch)