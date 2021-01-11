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
    def __init__(
        self,
        image_dir,
        info_file,
        mode: str = 'train',
        transforms=None):
        self.image_dir = image_dir
        # self.coco = COCO(info_file)
        # self.image_ids = list(self.coco.imgToAnns.keys())
        self.mode = mode
        self.transforms = transforms
        self.coco = CocoParser(info_file)
        self.image_ids = list(self.coco.image_info.keys())

        # self.label_map = { 1:  1,  2:  2,  3:  3,  4:  4,  5:  5,  6:  6,  7:  7,  8:  8,
        #            9:  9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
        #           18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24,
        #           27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32,
        #           37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40,
        #           46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48,
        #           54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56,
        #           62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64,
        #           74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
        #           82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}
        self.label_map = {
            1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6 
        }
        # print(self.image_ids)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        """
        Returns:
            image (ndarray[C, H, W])
            boxes [xyxy]
        """
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
            # labels.append(target['category_id'])
            labels.append(self.label_map.get(target['category_id'])-1)
            crowds.append(target['iscrowd'])
            areas.append(target['area'])
            
            if target['segmentation'] is not None:
                segment = target['segmentation']
                if isinstance(segment, list):
                    rles = mask.frPyObjects(segment, h, w)
                    rle = mask.merge(rles)
                elif isinstance(segment['counts'], list):
                    rle = mask.frPyObjects(segment, h, w)
                else:
                    rle = segment

                masks.append(mask.decode(rle))

        boxes = np.asarray(boxes, dtype=np.float32)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        labels = np.asarray(labels, dtype=np.int64)
        crowds = np.asarray(crowds, dtype=np.int64)

        masks = np.vstack(masks).reshape(-1, h, w)
        # image = image.transpose((2, 0, 1))

        targets = {
            'boxes': boxes,
            'masks': masks,
            'labels': labels,
            'crowds': crowds,
        }

        if self.transforms is not None:
            image, targets = self.transforms(image, targets)

        # image = np.array(image).transpose(2, 0, 1)
        # image = image / 255.0

        # image = torch.as_tensor(image, dtype=torch.float32)
        if self.mode == 'train':
            return image, targets
        else:
            return image, targets, h, w


if __name__ == '__main__':
    coco_dataset = CocoDataset('./benchmarks/samples/', './benchmarks/samples/annotations.json')
    coco_dataset[0]
