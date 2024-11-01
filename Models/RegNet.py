import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torchvision import datasets

# User inputs
batch_size = int(input("Enter batch size: "))
num_epochs = int(input("Enter number of epochs: "))
early_stop_patience = int(input("Enter early stop patience: "))
learning_rate = float(input("Enter learning rate: "))
dropout_rate = float(input("Enter dropout rate: "))

# Data loading
data_dir = r"please enter your dataset address here"  # Replace with your dataset path
transform = transforms.Compose([
    transforms.Resize(256),  # Resize to 256x256
    transforms.CenterCrop(224),  # Center crop to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load dataset
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Split dataset into train, validation, and test sets
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# Data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model setup
model = torch.hub.load('pytorch/vision:v0.10.0', 'regnety_800mf', pretrained=True)  # Using RegNet model
model.fc = nn.Sequential(
    nn.Dropout(dropout_rate),
    nn.Linear(model.fc.in_features, len(dataset.classes))  # Adjust for number of classes
)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Early stopping setup
best_val_loss = float('inf')
patience_counter = 0

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss, train_correct = 0.0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels).sum().item()

    train_loss /= len(train_loader)
    train_accuracy = train_correct / len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss, val_correct = 0.0, 0
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(outputs.softmax(dim=1).cpu().numpy())

    val_loss /= len(val_loader)
    val_accuracy = val_correct / len(val_loader.dataset)

    print(f'Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print("Early stopping!")
            break

# Testing
model.eval()
test_loss, test_correct = 0.0, 0
all_test_labels, all_test_preds, all_test_probs = [], [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        test_correct += (predicted == labels).sum().item()
        all_test_labels.extend(labels.cpu().numpy())
        all_test_preds.extend(predicted.cpu().numpy())
        all_test_probs.extend(outputs.softmax(dim=1).cpu().numpy())

test_loss /= len(test_loader)
test_accuracy = test_correct / len(test_loader.dataset)

# Evaluation metrics
f1 = f1_score(all_test_labels, all_test_preds, average='weighted', zero_division=0)
precision = precision_score(all_test_labels, all_test_preds, average='weighted', zero_division=0)
recall = recall_score(all_test_labels, all_test_preds, average='weighted', zero_division=0)

# AUC
try:
    # Ensure all_test_probs is properly formatted
    all_test_probs = np.array(all_test_probs)  # Convert to numpy array if not already
    auc = roc_auc_score(np.eye(len(dataset.classes))[all_test_labels], all_test_probs, multi_class='ovr')
except ValueError as e:
    print(f"AUC calculation error: {e}")
    auc = 0  # Handle cases where AUC cannot be computed

# Print results
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, F1 Score: {f1:.4f}, "
      f"Precision: {precision:.4f}, Recall: {recall:.4f}, AUC: {auc:.4f}")