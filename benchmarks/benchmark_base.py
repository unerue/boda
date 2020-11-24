#-*- coding:utf-8 -*-
import os
import sys
import glob
from typing import Tuple, List, Dict

import cv2
import numpy as np
from PIL import Image

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


def parser_txt(path):
    """Test labels parser

    """
    with open(path) as f:
        lines = f.readlines()

    # data = {
    #     'image_ids': [],
    #     'boxes': [],
    #     'labels': [],
    # }
    data = {}
    for line in lines:
        line = line.strip().split()
        # data['image_ids'].append(line[0])
        image_id = line[0]
        boxes, labels = [], []
        for i in range((len(line)-1) // 5):
            x_min = float(line[i*5 + 1])
            y_min = float(line[i*5 + 2])
            x_max = float(line[i*5 + 3])
            y_max = float(line[i*5 + 4])
            label = int(line[i*5 + 5])

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(label)
        
        data[image_id] = {
            'boxes': [],
            'labels': [],
        }
        data[image_id]['boxes'] = boxes
        data[image_id]['labels'] = labels

    return data

def parser_xml(path):
    return 


class PascalVocDataset(Dataset):
    """Pascal VOC dataset"""
    def __init__(self, config: Dict, transforms, sample: bool = True, is_train: bool = True):
        super().__init__()
        self.config = config
        self.transforms = transforms
        self.sample = sample
        self.is_train = is_train
        self.num_classes = self.config.dataset.num_classes

        if self.sample and self.is_train:
            self.image_dir = self.config.dataset.train_images
            self.labels = parser_txt(self.config.dataset.train_labels)
        elif self.sample and not self.is_train:
            self.image_dir = self.config.dataset.test_images
            self.labels = parser_txt(self.config.dataset.test_labels)
        
        self.image_ids = list(self.labels.keys())
        
    def __getitem__(self, index: int):
        """
        Returns:
            Tuple[List[Tensor], List[Dict[str, Tensor]]]
        """
        image_id = self.image_ids[index]
        data = self.labels[image_id]

        image = Image.open(os.path.join(self.image_dir, image_id)).convert('RGB')
        boxes = np.asarray(data['boxes'], dtype=np.float32)
        labels = torch.as_tensor(data['labels'], dtype=torch.int64)

        if self.transforms is not None:
            image, boxes, labels = self.transforms(image, boxes, labels)
       
        target = {
            'boxes': boxes,
            'labels': labels,
        }
        
        return image, target

    def __len__(self) -> int:
        return len(self.image_ids)
        
    
class CocoDataset(Dataset):
    """Pascal VOC dataset"""
    def __init__(self, config: Dict, transforms, sample: bool = True, is_train: bool = True):
        super().__init__()
        raise NotImplementedError

    def __getitem__(self):
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.image_ids)