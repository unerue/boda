import os
import sys

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2



class PascalVocDataset(Dataset):
    """Pascal VOC dataset"""
    def __init__(self, df, image_dir, transform=None):
        super().__init__()
 
        self.df = df
        self.image_ids = df['image_id'].unique()
        self.image_dir = image_dir
        self.transform = transform
             
    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]

        image = cv2.imread(os.path.join(self.image_dir, f'{image_id}.jpg'), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        boxes = records[['x', 'y', 'w', 'h']].values.astype(np.int32)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((records.shape[0],20), dtype=torch.int64)
        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)
        
        target = {
            'image_id': torch.tensor([index]),
            'boxes': boxes,
            'labels': labels,
            'area': area,
            'iscrowd': iscrowd
        }
        if self.transform:
            sample = {
                'image': image,
                'bboxes': boxes,
                'labels': labels,
            }
            sample = self.transform(**sample)
            image = sample['image']
            target['boxes'] = torch.tensor(sample['bboxes'])
        
        return image, target

    def __len__(self) -> int:
        return len(self.image_ids)


def get_transform(train=True):
    return A.Compose([
        A.Resize(448, 448),
        ToTensorV2(p=1.0)], 
        bbox_params={
            'format': 'pascal_voc', 
            'label_fields': ['labels']})


train_labels = pd.read_csv(os.path.join(DIR_INPUT, 'train_labels.csv'))
valid_labels = pd.read_csv(os.path.join(DIR_INPUT, 'valid_labels.csv'))

trainset = WheatDataset(train_labels, DIR_TRAIN_IMAGES, get_transform())
validset = WheatDataset(valid_labels, DIR_VALID_IMAGES, get_transform())

def collate_fn(batch):
    return tuple(zip(*batch))

train_loader = DataLoader(
    trainset,
    batch_size=3,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn)

valid_loader = DataLoader(
    validset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn)


