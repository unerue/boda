#-*- coding:utf-8 -*-
import os
import sys
import glob
from typing import Tuple, List, Dict

import cv2
import numpy as np
from PIL import Image

import torch
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
        image_id = self.image_ids[index]
        data = self.labels[image_id]

        #TODO:
        # data
        
        # image = cv2.imread(os.path.join(self.image_dir, image_id), cv2.IMREAD_COLOR)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.open(os.path.join(self.image_dir, image_id)).convert('RGB')
        image = image.resize((448, 448))
        # print(image.shape)
        # mean = np.array([122.67891434, 116.66876762, 104.00698793])
        image = np.array(image).astype(np.float32)
        print(image.shape)
        image = np.array(image).astype(np.float32).transpose(2, 0, 1)
        print(image.shape)
        # print(image.shape)
        sys.exit(0)
        image /= 255.
        # image = (image - image.min()) / (image.max() - image.min())
        
        # image /= 255.

        boxes = np.asarray(data['boxes'], np.float64)
        labels = torch.as_tensor(data['labels'], dtype=torch.int64)#.view(-1, 1)
        
        target = {
            'boxes': boxes,
            'labels': labels,
        }
        
        if self.transforms is not None:
            sample = {
                'image': image,
                'bboxes': boxes,
                'category_ids': data['labels'],
            }
            sample = self.transforms(**sample)
            image = sample['image']
            target['boxes'] = torch.as_tensor(sample['bboxes'])

        
        
        image = torch.as_tensor(image)
        # target['boxes'] = torch.as_tensor(target['boxes'])

        # print(target)
        return image, target

    def __len__(self) -> int:
        return len(self.image_ids)
        
        
        # records = self.df[self.df['image_id'] == image_id]

        # image = cv2.imread(os.path.join(self.image_dir, f'{image_id}.jpg'), cv2.IMREAD_COLOR)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # image /= 255.0

        # boxes = records[['x', 'y', 'w', 'h']].values.astype(np.int32)
        # boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        # boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        
        # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # area = torch.as_tensor(area, dtype=torch.float32)
        # # there is only one class
        # labels = torch.ones((records.shape[0],20), dtype=torch.int64)
        # # suppose all instances are not crowd
        # iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)
        
        # target = {
        #     'image_id': torch.tensor([index]),
        #     'boxes': boxes,
        #     'labels': labels,
        #     'area': area,
        #     'iscrowd': iscrowd
        # }
        # if self.transform:
        #     sample = {
        #         'image': image,
        #         'bboxes': boxes,
        #         'labels': labels,
        #     }
        #     sample = self.transform(**sample)
        #     image = sample['image']
        #     target['boxes'] = torch.tensor(sample['bboxes'])
        
        # return image, target




def get_transform(train=True):
    return A.Compose([
        A.Resize(448, 448),
        ToTensorV2(p=1.0)], 
        bbox_params={
            'format': 'pascal_voc', 
            'label_fields': ['labels']})


# train_labels = pd.read_csv(os.path.join(DIR_INPUT, 'train_labels.csv'))
# valid_labels = pd.read_csv(os.path.join(DIR_INPUT, 'valid_labels.csv'))

# trainset = WheatDataset(train_labels, DIR_TRAIN_IMAGES, get_transform())
# validset = WheatDataset(valid_labels, DIR_VALID_IMAGES, get_transform())

# def collate_fn(batch):
#     return tuple(zip(*batch))

# train_loader = DataLoader(
#     trainset,
#     batch_size=3,
#     shuffle=False,
#     num_workers=4,
#     collate_fn=collate_fn)

# valid_loader = DataLoader(
#     validset,
#     batch_size=8,
#     shuffle=False,
#     num_workers=4,
#     collate_fn=collate_fn)


class CocoDataset(Dataset):
    """Pascal VOC dataset"""
    def __init__(self, config: Dict, transforms, sample: bool = True, is_train: bool = True):
        super().__init__()
        pass