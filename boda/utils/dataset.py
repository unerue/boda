import os
import json
from collections import defaultdict
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools import mask
import cv2


class CocoParser:
    def __init__(self, info_file):
        self.coco = self._from_json(info_file)
        self.image_info = {
            c['id']: {
                'file_name': c['file_name'], 
                'height': c['height'], 
                'width': c['width']} for c in self.coco['images']}
        self.annot_info = self._get_annot_info()
        print(f'Attached dataset... {len(self.image_info):,}')

    def _from_json(self, info_file):
        with open(info_file, 'r', encoding='utf-8') as f:
            info = json.load(f)

        return info

    def _get_annot_info(self):
        annot_info = defaultdict(list)
        for annot in self.coco['annotations']:
            annot_info[annot['image_id']].append({
                'id': annot['id'],
                'category_id': annot['category_id'],
                'bbox': annot['bbox'],
                'segmentation': annot['segmentation'],
                'area': annot['area'],
                'iscrowd': annot['iscrowd']
            })

        return annot_info

    def get_annots(self, image_id):
        return self.annot_info.get(image_id)

    # TODO: get_image_path or get_file_name    
    def get_file_name(self, image_id):
        return self.image_info.get(image_id)['file_name']

    def get_masks(self):
        raise NotImplementedError


class CocoDataset(Dataset):
    def __init__(self, image_dir, info_file, transforms=None):
        self.image_dir = image_dir
        # self.coco = COCO(info_file)
        # self.image_ids = list(self.coco.imgToAnns.keys())
        self.transforms = transforms
        self.coco = CocoParser(info_file)
        self.image_ids = list(self.coco.image_info.keys())
        # print(self.image_ids)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        targets = self.coco.get_annots(image_id)

        image = self.coco.get_file_name(image_id)
        # image = Image.open(os.path.join(self.image_dir, image)).convert('RGB')
        image = cv2.imread(os.path.join(self.image_dir, image))
        h, w, _ = image.shape

        boxes = []
        labels = []
        masks = []
        crowds = []
        areas = []
        for target in targets:
            boxes.append(target['bbox'])
            labels.append(target['category_id'])
            crowds.append(target['iscrowd'])
            areas.append(target['area'])
            
            if target['segmentation'] is not None:
                segment = target['segmentation']
                if isinstance(segment, list):
                    rles = mask.frPyObjects(segment, h, w)
                    rle = mask.merge(rles)
                elif isinstance(segment['count'], list):
                    rle = mask.frPyObjects(segment, h, w)
                else:
                    rle = segment
                
                masks.append(mask.decode(rle))

        boxes = np.asarray(boxes, dtype=np.float64)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        labels = np.asarray(labels, dtype=np.int64)
        crowds = np.asarray(crowds, dtype=np.int64)

        masks = np.vstack(masks).reshape(-1, h, w)
        # image = image.transpose((2, 0, 1))

        targets = {
            'boxes': boxes,
            'labels': labels,
            'crowds': crowds,
            'masks': masks
        }

        if self.transforms is not None:
            image, targets = self.transforms(image, targets)

        # image, targets = Resize((448, 448))(image, targets)
        # image = np.array(image).transpose(2, 0, 1)
        # image = image / 255.0

        # image = torch.as_tensor(image, dtype=torch.float32)

        return image, targets


if __name__ == '__main__':
    coco_dataset = CocoDataset('./benchmarks/samples/', './benchmarks/samples/annotations.json')
    coco_dataset[0]
