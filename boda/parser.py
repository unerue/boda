import numpy as np
import os
import torch
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


class CocoParser(Dataset):
    def __init__(self, image_dir, info_file, transforms=None):
        self.image_dir = image_dir
        self.coco = COCO(info_file)
        self.image_ids = list(self.coco.imgToAnns.keys())
        self.transforms = transforms

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        annot_ids = self.coco.getAnnIds(imgIds=image_id)
        targets = [x for x in self.coco.loadAnns(annot_ids)]

        image = self.coco.loadImgs(image_id)[0]['file_name']
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

        targets = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float64),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
        }

        image, targets = Resize((448, 448))(image, targets)
        image = np.array(image).transpose(2, 0, 1)
        image = image / 255.0

        image = torch.as_tensor(image, dtype=torch.float32)

        return image, targets


# if __name__ == '__main__':
#     coco_dataset = CocoParser('./benchmarks/samples/', './benchmarks/samples/annotations.json')
#     coco_dataset[0]
