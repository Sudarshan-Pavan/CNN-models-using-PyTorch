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


import math
import torch
import torch.nn as nn
from torch.autograd import Variable

class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)

def roundChannels(c, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_c = max(min_value, int(c + divisor / 2) // divisor * divisor)
    if new_c < 0.9 * c:
        new_c += divisor
    
    return new_c

def roundRepeats(r):
    return int(math.ceil(r))

def dropPath(x, drop_probability, training):
    if drop_probability > 0 and training:
        keep_probability = 1 - drop_probability
        if x.is_cuda:
            mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_probability))
        else:
            mask = Variable(torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_probability))

        x.div_(keep_probability)
        x.mul_(mask)

    return x

def batchNorm(channels, eps=1e-3, momentum=0.01):
    return nn.BatchNorm2d(channels, eps=eps, momentum=momentum)

# CONV3x3
def conv3x3(in_channel, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channels, 3, stride, 1, bias=False),
        batchNorm(out_channels),
        Swish()
    )

# CONV1x1
def conv1x1(in_channel, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channels, 1, 1, 0, bias=False),
        batchNorm(out_channels),
        Swish()
    )

class SqueezeAndExcitation(nn.Module):
    def __init__(self, channel, squeeze_channel, se_ratio):
        super().__init__()
        squeeze_channel = squeeze_channel * se_ratio
        if not squeeze_channel.is_integer():
            raise ValueError('channels must be divisible by 1/se_ratio')

        squeeze_channel = int(squeeze_channel)
        self.se_reduce = nn.Conv2d(channel, squeeze_channel, 1, 1, 0, bias=True)
        self.non_linear1 = Swish()
        self.se_excite = nn.Conv2d(squeeze_channel, channel, 1, 1, 0, bias=True)
        self.non_linear2 = nn.Sigmoid()

    def forward(self, x):
        y = torch.mean(x, (2, 3), keepdim=True)
        y = self.non_linear1(self.se_reduce(y))
        y = self.non_linear2(self.se_excite(y))
        y = x * y
        return y

class MBConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, expand_ratio, se_ratio, drop_path_rate):
        super().__init__()
        expand = (expand_ratio != 1)
        expand_channel = in_channel * expand_ratio
        se = (se_ratio != 0)
        self.residual_connection = (stride == 1 and in_channel == out_channel)
        self.drop_path_rate = drop_path_rate

        conv = []

        if expand:
            pw_expansion = nn.Sequential(
                nn.Conv2d(in_channel, expand_channel, 1, 1, 0, bias=False),
                batchNorm(expand_channel),
                Swish()
            )
            conv.append(pw_expansion)

        # Depthwise convolution
        dw = nn.Sequential(
            nn.Conv2d(expand_channel, expand_channel, kernel_size, stride, kernel_size // 2, groups=expand_channel, bias=False),
            batchNorm(expand_channel),
            Swish()
        )
        conv.append(dw)

        if se:
            squeeze_excite = SqueezeAndExcitation(expand_channel, in_channel, se_ratio)
            conv.append(squeeze_excite)
        
        pw_projection = nn.Sequential(
            nn.Conv2d(expand_channel, out_channel, 1, 1, 0, bias=False),
            batchNorm(out_channel)
        )
        conv.append(pw_projection)
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        if self.residual_connection:
            return x + dropPath(self.conv(x), self.drop_path_rate, self.training)
        else:
            return self.conv(x)

class EfficientNet(nn.Module):
    cfg = [
        #(in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, repeats)
        [32,  16,  3, 1, 1, 0.25, 1],
        [16,  24,  3, 2, 6, 0.25, 2],
        [24,  40,  5, 2, 6, 0.25, 2],
        [40,  80,  3, 2, 6, 0.25, 3],
        [80,  112, 5, 1, 6, 0.25, 3],
        [112, 192, 5, 2, 6, 0.25, 4],
        [192, 320, 3, 1, 6, 0.25, 1]
    ]
    
    def __init__(self, input_channels, param, n_classes, stem_channels=32, feature_size=1280, drop_connect_rate=0.2, dropout_rate=dropout_rate):
        super().__init__()

        # Scaling width 
        width_coefficient = param[0]
        if width_coefficient != 1.0:
            stem_channels = roundChannels(stem_channels * width_coefficient)
            for conf in self.cfg:
                conf[0] = roundChannels(conf[0] * width_coefficient)
                conf[1] = roundChannels(conf[1] * width_coefficient)

        # Scaling depth
        depth_coefficient = param[1]
        if depth_coefficient != 1.0:
            for conf in self.cfg:
                conf[6] = roundRepeats(conf[6] * depth_coefficient)

        # Scaling resolution
        input_size = param[2]

        self.stem_conv = conv3x3(input_channels, stem_channels, 2)

        # Total blocks
        total_blocks = 0
        for conf in self.cfg:
            total_blocks += conf[6]

        blocks = []
        for in_channel, out_channel, kernel_size, stride, expand_ratio, se_ratio, repeats in self.cfg:
            drop_rate = drop_connect_rate * (len(blocks) / total_blocks)
            blocks.append(MBConvBlock(in_channel, out_channel, kernel_size, stride, expand_ratio, se_ratio, drop_rate))
            for _ in range(repeats - 1):
                drop_rate = drop_connect_rate * (len(blocks) / total_blocks)
                blocks.append(MBConvBlock(out_channel, out_channel, kernel_size, 1, expand_ratio, se_ratio, drop_rate))
        
        self.blocks = nn.Sequential(*blocks)

        self.head_conv = conv1x1(self.cfg[-1][1], feature_size)
        self.dropout = nn.Dropout(dropout_rate)  # Set dropout rate here
        self.classifier = nn.Linear(feature_size, n_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.blocks(x)
        x = self.head_conv(x)
        x = torch.mean(x, (2, 3))
        x = self.dropout(x)
        x = self.classifier(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def initialize_model(input_channels, n_classes, dropout_rate=dropout_rate):
    # Parameters for EfficientNet
    param = [1.0, 1.0, 224, dropout_rate]  # Use provided dropout_rate
    model = EfficientNet(input_channels=input_channels, param=param, n_classes=n_classes, dropout_rate=dropout_rate)
    return model

# Early stopping implementation
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

# Initialize the MobileNetV2 model
num_classes = len(dataset.classes)  # Number of classes in the dataset
input_channels = 3  # RGB images
model = initialize_model(input_channels=input_channels, n_classes=num_classes, dropout_rate=dropout_rate)

# Move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Learning rate scheduler (optional)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training and validation function
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
torch.save(model.state_dict(), 'effecientNet.pth')

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
