import os
import json
from collections import defaultdict
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO


class Resize:
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image, targets):
        w, h = image.size
        image = image.resize(self.size)
        # masks = TF.resize(masks, self.size, self.interpolation)

        targets['boxes'][:, [0, 2]] *= self.size[0] / w
        targets['boxes'][:, [1, 3]] *= self.size[1] / h

        return image, targets


class CocoParser:
    def __init__(self, info_file):
        self.coco = self._from_json(info_file)
        self.image_info = {c['id']: c['file_name'] for c in self.coco['images']}
        self.annot_info = self._get_annot_info()

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
                'iscrowd': annot['iscrowd']})

        return annot_info

    def get_annots(self, image_id):
        return self.annot_info.get(image_id)

    def get_file_name(self, image_id):
        return self.image_info.get(image_id)


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
        image = Image.open(os.path.join(self.image_dir, image)).convert('RGB')

        boxes = []
        labels = []
        for target in targets:
            boxes.append(target['bbox'])
            labels.append(target['category_id'])

        boxes = np.array(boxes)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        labels = np.array(labels)
        crowds = np.array([0])

        targets = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float64),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'crowds': torch.as_tensor(crowds, dtype=torch.int64)
        }

        image, targets = Resize((448, 448))(image, targets)
        image = np.array(image).transpose(2, 0, 1)
        image = image / 255.0

        image = torch.as_tensor(image, dtype=torch.float32)

        return image, targets


if __name__ == '__main__':
    coco_dataset = CocoDataset('./benchmarks/samples/', './benchmarks/samples/annotations.json')
    coco_dataset[0]
