import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# Define the NeuralNetwork class (same as in the original training script)
class NeuralNetwork(nn.Module):
    def __init__(self, i, h_size, h_next_size, h_next_next_size=16, n_classes=2,
                 how_many_layers=4, embedding_dim=300,hs_embedding_dim=300):
        super(NeuralNetwork, self).__init__()

        features = i.shape[1]  # Total number of input features

        self.major_embedding = nn.Embedding(input_embedding_dimension, embedding_dim)

        #embedding layer for the high school codes
        self.hs_embedding=nn.Embedding(input_hs_embedding_dimension,hs_embedding_dim)

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
        hs_input = X[:,4].long()

        #continuous_input = X[:, 1:].float() # Rest: continuous features
        continuous_indices = [i for i in range(X.shape[1]) if i not in [0, 4]]
        continuous_input = X[:, continuous_indices].float()

        # Apply embedding
        embedded = self.major_embedding(categorical_input)  # Shape: [batch_size, embedding_dim]
        hs_embedded = self.hs_embedding(hs_input)

        # Concatenate with continuous features
        X = torch.cat((embedded,hs_embedded, continuous_input), dim=1)  # Shape: [batch_size, embedding_dim + (features-1)]

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


data_2025 = pd.read_excel('/Users/bhaskaravanacharla/Downloads/Documents/Machine_learning/admissionnn/fall25/Fall 2025 5.13.25_processed_norm.xlsx')

# Ensure that new_data has the same columns as the training data except for the target column
print(data_2025.columns)


save_net=torch.load('/Users/bhaskaravanacharla/Downloads/Documents/Machine_learning/admissionnn/models/trained_full_model_4_17_25.pth',weights_only=False)
save_net.to(device) 
new_data = data_2025[data_2025.drop(['admitted','Application Reference ID'], axis=1).columns]

print(new_data.shape[1])
data_2025_labels = data_2025['admitted'].to_numpy()
# Make sure new_data is a PyTorch tensor
new_data_tensor = torch.from_numpy(new_data.to_numpy()).float().to(device)

# Perform inference on the new data
with torch.no_grad():
    # Assuming your model is saved in 'save_net' after training
    outputs = save_net(new_data_tensor)

# Convert logits to probabilities
probabilities = torch.softmax(outputs, dim=1)
print(probabilities)
# Apply custom threshold (e.g., 0.4)
threshold = 0.5
predictions = (probabilities[:, 1] >= threshold).int()  # Class 1 if probability >= 0.4, else class 0


# If you want the predicted probabilities, you can use:
# probabilities = torch.softmax(outputs, dim=1)

# Convert predictions to numpy for further processing
predictions = predictions.cpu().numpy()

print("Predictions on new data:")
print(predictions)
total_predictions=len(predictions)
count_ones = (predictions == 1).sum()
print(f"Number of 1s in predictions: {count_ones}")
print(f"Number of 0s in predictions: {total_predictions - count_ones}")
print(f"Total number of predictions: {total_predictions}")


accuracy = accuracy_score(data_2025_labels,predictions)
precision = precision_score(data_2025_labels,predictions)
recall = recall_score(data_2025_labels, predictions)
f1_val = f1_score(data_2025_labels, predictions)
auc_roc = roc_auc_score(data_2025_labels, predictions)

print('Accuracy: {:.4f}'.format(accuracy))
print('Precision: {:.4f}'.format(precision))
print('Recall: {:.4f}'.format(recall))
print('F1 Score: {:.4f}'.format(f1_val))
print('AUROC Score: {:.4f}'.format(auc_roc))



# Compute Confusion Matrix
cm = confusion_matrix(data_2025_labels,predictions)

# Print the matrix
print("Confusion Matrix:\n", cm)

# Plot the Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

cm = confusion_matrix(data_2025_labels,predictions)

# Extract TP, TN, FP, FN
TN = cm[0, 0]  # True Negative
FP = cm[0, 1]  # False Positive
FN = cm[1, 0]  # False Negative
TP = cm[1, 1]  # True Positive

print(f"True Positives (TP): {TP}")
print(f"False Positives (FP): {FP}")
print(f"True Negatives (TN): {TN}")
print(f"False Negatives (FN): {FN}")



#save file to excel
# Add predictions as a new column to the DataFrame
data_2025['predicted_admitted'] = predictions

# Filter the rows where the prediction is 1 (admitted)
admitted_applications = data_2025[data_2025['predicted_admitted'] == 1]

# Retrieve the Application Reference IDs of the admitted applications
application_ids = admitted_applications['Application Reference ID']

# Print the application IDs
print(application_ids)
#application_ids.to_excel('/Users/bhaskaravanacharla/Downloads/Documents/Machine_learning/admissionnn/filtered_application_ids/filtered_application_ids_050725.xlsx', index=False)