import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
# Check if MPS is available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
data = pd.read_excel('fall_processed_norm.xlsx')
application_id = data['Application Reference ID'].to_numpy()
inputs = data.drop(['admitted', 'Application Reference ID'], axis=1).to_numpy()
labels = data['admitted'].to_numpy()

train_inputs, test_inputs, train_labels, test_labels = train_test_split(
    inputs, labels, test_size=0.2, random_state=42, stratify=labels
)

# Convert to PyTorch tensors and move to device (MPS or CPU)
train_inputs = torch.from_numpy(train_inputs).float().to(device)
test_inputs = torch.from_numpy(test_inputs).float().to(device)
train_labels = torch.from_numpy(train_labels).int().to(device)
test_labels = torch.from_numpy(test_labels).int().to(device)

input_embedding_dimension = int(inputs[:, 0].max()) + 1
input_hs_embedding_dimension = int(inputs[:, 4].max()) + 1

# Dataset and DataLoader
train_dataset = TensorDataset(train_inputs, train_labels)
test_dataset = TensorDataset(test_inputs, test_labels)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class NeuralNetwork(nn.Module):
    def __init__(self, i, h_size, h_next_size, h_next_next_size=32, n_classes=2,
                 how_many_layers=4, embedding_dim=12, hs_embedding_dim=50):
        super(NeuralNetwork, self).__init__()

        features = i.shape[1]  # Total number of input features

        self.major_embedding = nn.Embedding(input_embedding_dimension, embedding_dim)

        # embedding layer for the high school codes
        self.hs_embedding = nn.Embedding(input_hs_embedding_dimension, hs_embedding_dim)

        print(f"Setting major embedding dim: {input_embedding_dimension}")
        print(f"Setting HS embedding dim: {input_hs_embedding_dimension}")

        # Input to fc1 will be (features - 2) continuous + embedding_dim
        self.fc1 = nn.Linear(features - 2 + embedding_dim + hs_embedding_dim, h_size)
        self.layers = how_many_layers

        if self.layers == 2:
            self.fc2 = nn.Linear(h_size, n_classes)
        elif self.layers == 3:
            self.fc3 = nn.Linear(h_size, h_next_size)
            self.fc4 = nn.Linear(h_next_size, n_classes)
        elif self.layers == 4:
            self.fc3 = nn.Linear(h_size, h_next_size)
            self.fc4 = nn.Linear(h_next_size, h_next_next_size)
            self.fc5 = nn.Linear(h_next_next_size, n_classes)

    def forward(self, X):
        # Ensure input is float and extract categorical/continuous features
        categorical_input = X[:, 0].long()  # First column: categorical (encoded)
        hs_input = X[:, 4].long()

        continuous_indices = [i for i in range(X.shape[1]) if i not in [0, 4]]
        continuous_input = X[:, continuous_indices].float()

        # Apply embedding
        embedded = self.major_embedding(categorical_input)  # Shape: [batch_size, embedding_dim]
        hs_embedded = self.hs_embedding(hs_input)

        # Concatenate with continuous features
        X = torch.cat((embedded, hs_embedded, continuous_input), dim=1)  # Shape: [batch_size, embedding_dim + (features-1)]

        if self.layers == 2:
            X = F.relu(self.fc1(X))
            X = self.fc2(X)
        elif self.layers == 3:
            X = F.relu(self.fc1(X))
            X = F.relu(self.fc3(X))
            X = self.fc4(X)
        elif self.layers == 4:
            X = F.relu(self.fc1(X))
            X = torch.tanh(self.fc3(X))
            X = F.sigmoid(self.fc4(X))
            X = self.fc5(X)

        return X


# Move the model to MPS or CPU
net = NeuralNetwork(inputs, h_size=16, h_next_size=22, how_many_layers=4).to(device)

n_epochs = 600
learning_rate = 0.0001
decay_rate = learning_rate / n_epochs
optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=decay_rate)
lambda_reg = 0.001
lambda_entropy = 0

def loss_fn(model, outputs, targets):
    # Convert labels to numpy (make sure to move to CPU before using numpy)
    y_train = train_labels.cpu().numpy()

    # Compute class weights
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    
    class_weights = np.array([1, 2]) #change
    # Convert to PyTorch tensor and move to device
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    cross_entropy = nn.functional.cross_entropy(outputs, targets, weight=class_weights_tensor)
    l2_regularization = 0

    for param in model.parameters():
        l2_regularization += torch.norm(param, p=2) ** 2

    loss = cross_entropy + lambda_reg * l2_regularization
    return loss

def test_instance(model):
    y_t = []
    y_s = []
    loss = 0
    acc = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss += loss_fn(model, outputs, labels.long())
            y_t.extend(labels.cpu().numpy())
            y_s.extend(torch.sigmoid(outputs).max(axis=1).indices.cpu().numpy())

    acc = accuracy_score(y_t, y_s)
    return loss, acc

iteration = 0
counter = 0

for epoch in range(n_epochs):
    running_loss = 0.0
    total = 0  # No. of total predictions
    correct = 0  # No. of correct predictions

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_fn(net, outputs, labels.long())
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / len(train_loader.dataset)  # Loss in every epoch
    epoch_acc = correct / total  # Accuracy for every epoch

    if epoch % 10 == 0 or epoch == n_epochs - 1:
        print(f'Epoch: {epoch + 1}/{n_epochs} | pLoss: {running_loss / len(inputs)} | Accuracy: {epoch_acc} | Loss: {epoch_loss}')

    if epoch % 50 == 0:
        test_loss, test_acc = test_instance(net)
        print(f'Epoch: {epoch + 1} | The test data Accuracy = {test_acc} | Test Loss = {test_loss}')

        if counter < test_acc:
            save_net = net
            counter = test_acc

torch.save(save_net.state_dict(), '/Users/bhaskaravanacharla/Downloads/Documents/Machine_learning/admissionnn/models/trained_model_5_8_25.pth')
torch.save(save_net,'/Users/bhaskaravanacharla/Downloads/Documents/Machine_learning/admissionnn/models/trained_full_model_5_8_25.pth')


# Test evaluation after training
y_true = []
y_scores = []
test_loss = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = save_net(inputs)
        test_loss += loss_fn(net, outputs, labels.long())
        predicted_labels = (torch.sigmoid(outputs) > 0.6).int() #change
        y_true.extend(labels.cpu().numpy())
        y_scores.extend(torch.sigmoid(outputs).max(axis=1).indices.cpu().numpy())

accuracy = accuracy_score(y_true, y_scores)
precision = precision_score(y_true, y_scores)
recall = recall_score(y_true, y_scores)
f1_val = f1_score(y_true, y_scores)
auc_roc = roc_auc_score(y_true, y_scores)

print('Accuracy: {:.4f}'.format(accuracy))
print('Precision: {:.4f}'.format(precision))
print('Recall: {:.4f}'.format(recall))
print('F1 Score: {:.4f}'.format(f1_val))
print('AUROC Score: {:.4f}'.format(auc_roc))

#originaltestingaccurcy
# Epoch: 600/600 | pLoss: 102.0592019289732 | Accuracy: 0.9850788932083238 | Loss: 0.11669243303106928
# Accuracy: 0.9655
# Precision: 0.8302
# Recall: 0.9270
# F1 Score: 0.8759
# AUROC Score: 0.9491

#testing on 5.8.25
# Epoch: 600/600 | pLoss: 97.52433138340712 | Accuracy: 0.9882231877429682 | Loss: 0.1115073535140717
# Accuracy: 0.9700
# Precision: 0.8801
# Recall: 0.8939
# F1 Score: 0.8870
# AUROC Score: 0.9377