import os
import sys
import re

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchsummary import summary

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2, ToTensor

sys.path.append('../')
from benchmark_base import parser_txt, PascalVocDataset
from detection import yolov1_base_config
from detection import darknet9, Yolov1Model, Yolov1Loss
from detection.utils import AverageMeter










# def parser_txt(path):
#     """Test labels parser

#     """
#     with open(path) as f:
#         lines = f.readlines()

#     # data = {
#     #     'image_ids': [],
#     #     'boxes': [],
#     #     'labels': [],
#     # }
#     data = {}
#     for line in lines:
#         line = line.strip().split()
#         # data['image_ids'].append(line[0])
#         image_id = line[0]
#         boxes, labels = [], []
#         for i in range((len(line)-1) // 5):
#             x_min = float(line[i*5 + 1])
#             y_min = float(line[i*5 + 2])
#             x_max = float(line[i*5 + 3])
#             y_max = float(line[i*5 + 4])
#             label = int(line[i*5 + 5])

#             boxes.append([x_min, y_min, x_max, y_max])
#             labels.append(label)
        
#         data[image_id] = {
#             'boxes': [],
#             'labels': [],
#         }
#         data[image_id]['boxes'] = boxes
#         data[image_id]['labels'] = labels

#     return data



# data = parser_txt(yolov1_base_config.dataset.train_labels)
# # print(data)

# print(len(data.keys()))
# ssibal = []
# for path in data.keys():
#     # print('./data/pascal-voc/VOC2007/JPEGImages/'+path)
#     try:
#         image = cv2.imread('./data/pascal-voc/VOC2007/train/JPEGImages/'+path, cv2.IMREAD_COLOR)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         # print(image)
#     except:
#         ssibal.append(path)
# print(len(ssibal))
# print(ssibal)
# sys.exit(0)


def get_transform(train=True):
    return A.Compose([
        A.Resize(448, 448),
        ToTensor(num_classes=20),
        # ToTensorV2(p=1.0)
        ], 
        bbox_params={'format': 'pascal_voc', 'label_fields': ['category_ids']})

trainset = PascalVocDataset(yolov1_base_config, get_transform())

def collate_fn(batch):
    return tuple(zip(*batch))

train_loader = DataLoader(
    trainset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Yolov1Model(yolov1_base_config).to(device)
optimizer = torch.optim.SGD(model.parameters(), 0.001)
criterion = Yolov1Loss()



num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    for i, (images, targets) in enumerate(train_loader):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # print(images)
        # print('='*100)
        # print(targets)
        # print('='*100)
        # sys.exit(0)
        optimizer.zero_grad()
        # print(targets[0]['boxes'])

        outputs = model(images)
        # print(outputs[0][0])
        # print('='*100)
        losses = criterion(outputs, targets)
        print('='*100)
        
        print(f"Epoch #{epoch} id: {i} loss: {losses}")
        losses = sum(loss for loss in losses.values())
        print(f"loss: {losses}")
        # sys.exit(0)
        
        losses.backward()
        optimizer.step()

        # if (i+1) % 40 == 0:
        #     break

model.eval()
for (images, targets) in train_loader:
    images = [image.to(device) for image in images]
    outputs = model(images)
    break

image = images[0]

print('#'*100)
    
    # print(outputs)

import matplotlib.pyplot as plt


fig, ax = plt.subplots(1, 1, figsize=(16, 8))

image = image.permute(1, 2, 0).detach().cpu().numpy()
for box in outputs[0]['boxes']:
    cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
    
ax.set_axis_off()
ax.imshow(image)
plt.show()

print(outputs[0]['boxes'])
print(torch.max(outputs[0]['labels'], 1))
# sys.exit(0)



        # print(len(outputs))
        # print(outputs[0]['boxes'].size())
        # print(outputs[0]['boxes'].view(2, -1, 4).size())

        # losses = criterion(outputs, targets)
        



        
        
        
        # loss_value = losses.item()

        # loss_hist.update(loss_value)

        
        # losses.backward()
        # optimizer.step()

    
    

# sys.exit(0)

# images, targets = next(iter(valid_loader))
# images = list(img.to(device) for img in images)
# targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

# boxes = targets[1]['boxes'].cpu().numpy().astype(np.int32)
# sample = images[1].permute(1,2,0).cpu().numpy()

# model.eval()
# cpu_device = torch.device("cpu")

# outputs = model(images)
# print(outputs)
# outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]



# fig, ax = plt.subplots(1, 1, figsize=(16, 8))

# for box in boxes:
#     cv2.rectangle(sample,
#                   (box[0], box[1]),
#                   (box[2], box[3]),
#                   (220, 0, 0), 3)
    
# ax.set_axis_off()
# ax.imshow(sample)
# plt.show()

# torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn.pth')