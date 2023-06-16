# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from matplotlib import pyplot as plt
import cv2
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
dataset_path = ''
#print(dataset_path)
from torch.utils.data import random_split

full_dataset = ImageFolder(dataset_path, transform=transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.2),
    transforms.ToTensor()
]))

#print(len(full_dataset))

# Calculate the split sizes based on the desired train-validation split ratio
train_size = int(0.9 * len(full_dataset))  # 80% of the data for training
val_size = len(full_dataset) - train_size  # Remaining data for validation

# Split the dataset into train and validation datasets
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

batch_size = 64
train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_dataset, batch_size=batch_size*2, num_workers=4, pin_memory=True)

def display_img(img, label):
    print(f"Label: {full_dataset.classes[label]}")
    plt.imshow(img.permute(1, 2, 0).numpy())
    plt.show()

#display the first image in the dataset
#display_img(*full_dataset[7])
def show_batch(dl):
    """Plot images grid of single batch"""
    for images, labels in dl:
        fig,ax = plt.subplots(figsize = (16,12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images,nrow=16).permute(1,2,0))
        break
        
if __name__ == '__main__':
    show_batch(train_dl)

class ImageClassification(nn.Module):

    def step(self, batch):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = self(images)
        _, predicted_labels = torch.max(outputs, 1)

        # Calculate loss
        loss = F.cross_entropy(outputs, labels)

        # Return the loss and predicted labels separately as tensors
        return loss, predicted_labels

    def validation_step(self, images, labels):
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        _, predicted_labels = torch.max(out, dim=1)  # Get predicted labels
        return {'val_loss': loss.detach(), 'val_acc': acc, 'val_predicted_labels': predicted_labels}


    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}


    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class HairModelClassification(ImageClassification):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # This creates a few nueral layers
            # Layer 1
            nn.Conv2d(3, 32, kernel_size=3, padding=(2, 2), stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=(2, 2), stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.5),
            
            # Layer 2
            nn.Conv2d(64, 128, kernel_size=3, padding=(2,2), stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=(2,2), stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.5),
            
            # layer 3
            nn.Conv2d(128, 128, kernel_size=3, padding=(2,2), stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.5),
            
            # Last layer
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)            
        )
        self.optimizer = torch.optim.Adam(self.parameters())
    def forward(self, xb):
        return self.network(xb)
    def train_step(self, batch):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = self(images)
        _, predicted_labels = torch.max(outputs, 1)

        # Calculate loss
        loss = F.cross_entropy(outputs, labels)

        # Return the loss and predicted labels separately as tensors
        return loss, predicted_labels, labels



# Create an instance of the model
model = HairModelClassification()
# Define the path for the saved model
model_path = ''

if Path(model_path).is_file():
    print("A saved model exists. Loading the model...")
    # Load the saved model
    model.load_state_dict(torch.load(model_path))
else:
    print("No saved model found. Starting training from scratch.")

# Move the model to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
# Create the validation data loader
num_workers = 4
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# ...

@torch.no_grad()
def evaluate(model, val_loader):
    model = model.to(device)
    model.eval()
    outputs = []
    predicted_labels = []  # List to store predicted labels
    for batch in val_loader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        output = model.validation_step(images, labels)  # Pass both images and labels
        outputs.append(output)
        predicted_labels.extend(output['val_predicted_labels'].tolist())  # Append predicted labels
        print("Output Dictionary:", output)
        # Check if 'val_predicted_labels' key exists in the output dictionary
        if 'val_predicted_labels' in output:
            print("Predicted Labels:", output['val_predicted_labels'])
            print("Actual Labels:", labels)

    val_losses = [x['val_loss'] for x in outputs]
    val_accs = [x['val_acc'] for x in outputs]
    return {
        'val_loss': torch.stack(val_losses).mean().item(),
        'val_acc': torch.stack(val_accs).mean().item(),
        'predicted_labels': predicted_labels  # Include predicted labels in the output dictionary
    }

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for batch in train_loader:
            loss, predicted_labels, labels = model.train_step(batch)  # Get loss and predictions from train_step function
            train_losses.append(loss.item())  # Append loss item to train_losses

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Convert actual and predicted labels tensors into hair types
            actual_labels = [full_dataset.classes[label.item()] for label in labels]
            predicted_labels = [full_dataset.classes[label.item()] for label in predicted_labels]

            # Print predicted labels and actual labels
            print("Predicted Labels:", predicted_labels)
            print("Actual Labels:", actual_labels)

        result = evaluate(model, val_loader)
        result['train_loss'] = torch.tensor(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)

    return history



def plot_accuracies(history):
    """ Plot the history of accuracies"""
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');
    


def plot_losses(history):
    """ Plot the losses in each epoch"""
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    
def train_model(): 
    num_epochs = 50
    opt_func = torch.optim.Adam
    lr = 0.001

    print("Running data training...")
    Run_Num = 1

    for Run in range(Run_Num):
        history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
        plot_accuracies(history)
        plot_losses(history)
        torch.save(model.state_dict(),model_path)
    print("Training set Finished...")

def validation_test():
    print("Unknown values now starting...")
    print("Getting location")
    Unknown_dataset_path = ''
    print("location found: " + Unknown_dataset_path)
    print("Transforming all the images that are unknown to 128x128 image sizes and converting to tensor values")
    Unknown_full_dataset = ImageFolder(Unknown_dataset_path, transform=transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.ToTensor()
    ]))
    
    # Create the DataLoader for the unknown dataset
    
    unknown_loader = torch.utils.data.DataLoader(Unknown_full_dataset, batch_size=batch_size, shuffle=False)

    # Evaluate unknown images
    predictions = evaluate(model, unknown_loader)
    
    # Get the predicted labels and convert them to hair types
    predicted_labels = predictions['predicted_labels']
    hair_types = ['Curly', 'Stright', 'Wavy']
    predicted_hair_types = [hair_types[label] for label in predicted_labels]

    print(predicted_hair_types)

if __name__ == '__main__':
    # Move the model and tensors to the GPU, this is to continue trainign the current model and to start from cold
    model = model.to(device)
    #train_model()
    print(validation_test())