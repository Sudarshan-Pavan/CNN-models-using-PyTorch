import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import os
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
from torch.nn import functional as F
import math
from torch.autograd import Variable

# Get user inputs for various parameters
batch_size = int(input("Enter the batch size: "))
num_epochs = int(input("Enter the number of epochs: "))
patience = int(input("Enter patience for early stopping: "))
learning_rate = float(input("Enter the learning rate: "))
dropout_rate = float(input("Enter the dropout rate: "))


import torch
import torch.nn as nn
import torch.nn.functional as F

class Shuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,W] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, C // g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)

class Bottleneck(nn.Module):
    def __init__(self, input_channel, output_channel, stride, groups, dropout_rate=dropout_rate):
        super().__init__()
        self.stride = stride
        self.dropout_rate = dropout_rate

        in_between_channel = int(output_channel / 4)
        g = 1 if input_channel == 24 else groups

        # Group Convolution
        self.conv1x1_1 = nn.Sequential(
            nn.Conv2d(input_channel, in_between_channel, kernel_size=1, groups=g, bias=False),
            nn.BatchNorm2d(in_between_channel), nn.ReLU(inplace=True))
        self.shuffle = Shuffle(groups=g)

        # Depthwise Convolution
        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(in_between_channel, in_between_channel, kernel_size=3, stride=stride, padding=1, groups=in_between_channel, bias=False),
            nn.BatchNorm2d(in_between_channel), nn.ReLU(inplace=True))
        self.conv1x1_3 = nn.Sequential(
            nn.Conv2d(in_between_channel, output_channel, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(output_channel))
        
        self.shortcut = nn.Sequential()
        if stride == 2:
            self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))

        # Dropout layer
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, x):
        out = self.conv1x1_1(x)
        out = self.shuffle(out)
        out = self.conv1x1_2(out)
        out = self.conv1x1_3(out)
        out = self.dropout(out)  # Apply dropout after bottleneck
        res = self.shortcut(x)
        out = F.relu(torch.cat([out, res], 1)) if self.stride == 2 else F.relu(out + res)
        return out

class ShuffleNet(nn.Module):
    def __init__(self, cfg, input_channel, n_classes, dropout_rate=dropout_rate):
        super().__init__()
        output_channels = cfg['out']
        n_blocks = cfg['n_blocks']
        groups = cfg['groups']
        self.in_channels = 24
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, 24, kernel_size=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )

        # Layers
        self.layer1 = self.make_layer(output_channels[0], n_blocks[0], groups, dropout_rate)
        self.layer2 = self.make_layer(output_channels[1], n_blocks[1], groups, dropout_rate)
        self.layer3 = self.make_layer(output_channels[2], n_blocks[2], groups, dropout_rate)
        
        # Pooling and classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(output_channels[2], n_classes)

    def make_layer(self, out_channel, n_blocks, groups, dropout_rate):
        layers = []
        for i in range(n_blocks):
            stride = 2 if i == 0 else 1
            cat_channels = self.in_channels if i == 0 else 0
            layers.append(Bottleneck(self.in_channels, out_channel - cat_channels, stride=stride, groups=groups, dropout_rate=dropout_rate))
            self.in_channels = out_channel
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class EarlyStopping:
    def __init__(self, patience=patience, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

# Define image transformations for training and validation
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),   # Resizes images to 224x224
    transforms.RandomHorizontalFlip(),   # Data augmentation with horizontal flip
    transforms.ToTensor(),               # Convert images to PyTorch tensors
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

# Load dataset from a single directory
data_dir = r'please enter your dataset address here'  # Root directory of the dataset
dataset = datasets.ImageFolder(data_dir, transform=transform)

# Split dataset
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

cfg = {
    'out': [240, 480, 960],  # Example output channels for each layer
    'n_blocks': [4, 8, 4],    # Number of bottleneck blocks for each layer
    'groups': 3               # Number of groups for group convolutions
}


# Initialize the MobileNetV2 model
num_classes = len(dataset.classes)  # Number of classes in the dataset
model = ShuffleNet(cfg, input_channel=3, n_classes=num_classes)

# Move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function and optimizer
learning_rate = learning_rate  # You can adjust this
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Learning rate scheduler (optional)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training function with early stopping
def train_model(model, criterion, optimizer, scheduler, num_epochs=num_epochs, patience=patience):
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                data_loader = train_loader
            else:
                model.eval()   # Set model to evaluation mode
                data_loader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass (track history only in train phase)
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Step scheduler (if using)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = running_corrects.double() / len(data_loader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}') #Acc% = Acc*10

            # Save to history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

                # Check early stopping
                early_stopping(val_loss=epoch_loss)
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    return model, history

    return model, history


# Train the model
history = train_model(model, criterion, optimizer, scheduler, num_epochs=num_epochs)

# Save the trained model
torch.save(model.state_dict(), 'shuffleNet.pth')

def evaluate_model(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(predicted.cpu().numpy())
    
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Accuracy: {acc:.4f}") #accuracy% = accuracy * 100
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("Confusion Matrix:")
    return cm

# Evaluate the model
cm = evaluate_model(model, test_loader)

import seaborn as sns

def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.show()

# Call this function after evaluating the model
plot_confusion_matrix(cm, dataset.classes)

# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.show()


# Plot the training history
plot_history(history)
