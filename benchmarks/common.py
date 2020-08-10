import os
import random
import logging
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
from torchvision import datasets, transforms

from utils import progress_bar

parser = argparse.ArgumentParser(description='Benchmark')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--epochs', default=10, type=int)
args = parser.parse_args()


def fixed_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

fixed_seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

dataset = datasets.CIFAR10(
    root='./data',
    train=True, 
    download=True, 
    transform=transforms.ToTensor())

testset = datasets.CIFAR10(
    root='./data', 
    train=False, 
    download=True, 
    transform=transforms.ToTensor())

trainset, validset = torch.utils.data.random_split(dataset, [40000, 10000])

train_loader = torch.utils.data.DataLoader(
    trainset, 
    batch_size=128, 
    shuffle=True, 
    num_workers=4)

valid_loader = torch.utils.data.DataLoader(
    validset, 
    batch_size=128, 
    shuffle=True, 
    num_workers=2)

test_loader = torch.utils.data.DataLoader(
    trainset, 
    batch_size=100, 
    shuffle=False, 
    num_workers=2)

classes = (
    'plane', 'car', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
)

from lenet import LeNet
from resnet import ResNet, resnet_18



net = LeNet().to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
# net = resnet_18().to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


history = {
    'train': [],
    'valid': []
}


def train(model):
    model.train()
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        total = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total += labels.size(0)
            train_loss += loss.item() * labels.size(0)
            train_acc += torch.sum(preds == labels.data).item()

            
            progress_bar(epoch, 
                i, len(train_loader), 
                {'Loss': f'{train_loss/total:.4f}', 'Acc': f'{100.*train_acc/total:.2f}%', 'Progress': f'({total}/{len(train_loader.dataset)})'})
                # f'{train_loss/total:.3f} {100.*train_acc/total:.2f}% ({total}/{len(train_loader.dataset)})')


            # if (i+1) % 100 == 0:
            #     string = 'Train (epoch {:2}/{})[{:.2f}%]\t{:.4f}\t{:.4f}'
            #     string = string.format(
            #         epoch+1, args.epochs, (total/len(train_loader.dataset))*100, train_loss/total, train_acc/total)
            #     print(string)



        epoch_loss = train_loss / total
        epoch_acc = train_acc / total

        

        history['train'].append(epoch_loss)
        
        # print(f'Epoch {epoch+1} Train Loss: {epoch_loss:.3f} | Acc: {epoch_acc:.3f}')
        

        model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            valid_acc = 0
            total = 0
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                loss = criterion(outputs, labels)

                total += labels.size(0)
                valid_loss += loss.item() * labels.size(0)
                valid_acc += torch.sum(preds == labels.data).item()

        epoch_loss = valid_loss / total
        epoch_acc = valid_acc / total
        history['valid'].append(epoch_loss)
       
        # print(f'Epoch {epoch+1} Valid Loss: {epoch_loss:.3f} | Acc: {epoch_acc:.3f}')
        # print()

train(net)

# import matplotlib.pyplot as plt

# fig, ax = plt.subplots(dpi=200)

# ax.plot(range(len(history['train'])), history['train'], label='Train loss')
# ax.plot(range(len(history['train'])), history['valid'], label='Validation loss')
# plt.legend()
# plt.show()