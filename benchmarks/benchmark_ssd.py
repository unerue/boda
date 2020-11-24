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
from torchsummary import summary

sys.path.append('../')
from benchmark_base import parser_txt, PascalVocDataset
from moeuda.arch.ssd import vgg16
# from moeuda.arch.ssd import PASCAL_CLASSES
# from detection import vgg, SsdPredictionHead
# from detection import Yolov1Model, Yolov1Loss
# from detection.utils import AverageMeter


config = [
    [64, 64],
    [ 'M', 128, 128],
    [ 'M', 256, 256, 256],
    [('M', {'kernel_size': 2, 'stride': 2, 'ceil_mode': True}), 512, 512, 512],
    [ 'M', 512, 512, 512],
    # [('M',  {'kernel_size': 3, 'stride':  1, 'padding':  1}),
    #  (1024, {'kernel_size': 3, 'padding': 6, 'dilation': 6}),
    #  (1024, {'kernel_size': 1})]
]

model = vgg16(config).to('cuda')
print(summary(model, input_data=(3, 300, 300), verbose=0))


# model = SsdPredictionHead()
# print(summary(model, input_data=(3, 300, 300), depth=2, verbose=0))


# class Compose:
#     """Composes several augmentations together.
#     Args:
#         transforms (List[Transform]): list of transforms to compose.
#     Example:
#         >>> augmentations.Compose([
#         >>>     transforms.CenterCrop(10),
#         >>>     transforms.ToTensor(),
#         >>> ])
#     """

#     def __init__(self, transforms):
#         self.transforms = transforms

#     def __call__(self, image, boxes=None, labels=None):
#         for t in self.transforms:
#             image, boxes, labels = t(image, boxes, labels)
#         return image, boxes, labels


# class Resize:
#     def __init__(self, size):
#         self.size = size

#     def __call__(self, image, boxes, labels=None):
#         w, h = image.size

#         image = image.resize(self.size)
#         boxes[:, [0, 2]] *= self.size[0] / w
#         boxes[:, [1, 3]] *= self.size[1] / h

#         return image, boxes, labels


# class Normalize:
#     def __init__(self, mean, std):
#         self.mean = np.array(mean, dtype=np.float32)
#         self.std = np.array(std,  dtype=np.float32)

#     def __call__(self, image, boxes, labels=None):
#         image = np.asarray(image, dtype=np.float32)
        

#         # if self.transform.normalize:
#         image = (image - self.mean) / self.std
#         # elif self.transform.subtract_means:
#         # image = (image - self.mean)
#         # elif self.transform.to_float:
#         image = image / 255


#         return image, boxes, labels 


# MEANS = (103.94, 116.78, 123.68)
# STD   = (57.38, 57.12, 58.40)

# class ToTensor:
#     def __call__(self, image, boxes=None, labels=None):
#         image = torch.from_numpy(image.transpose(2, 0, 1))
#         boxes = torch.from_numpy(boxes)
#         return image, boxes, labels

# def get_transforms(is_train=True):
#     return Compose([
#         Resize((448,448)),
#         Normalize(MEANS, STD),
#         ToTensor(),
#     ])



# trainset = PascalVocDataset(yolov1_base_config, get_transforms())
# # testset = PascalVocDataset(yolov1_base_config, get_transform(False))
# testset = PascalVocDataset(yolov1_base_config, get_transforms(False))

# def collate_fn(batch):
#     return tuple(zip(*batch))

# train_loader = DataLoader(
#     trainset,
#     batch_size=32,
#     shuffle=True,
#     num_workers=2,
#     collate_fn=collate_fn)

# test_loader = DataLoader(
#     testset,
#     batch_size=4,
#     shuffle=True,
#     num_workers=1,
#     collate_fn=collate_fn)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'


# from torch.optim.lr_scheduler import StepLR

# model = Yolov1Model(yolov1_base_config).to(device)
# optimizer = torch.optim.SGD(model.parameters(), 0.0005, momentum=0.9, weight_decay=0.0005)
# criterion = Yolov1Loss()
# scheduler = StepLR(optimizer, step_size=1, gamma=0.1)


# iteration = 0
# flag = 0
# num_epochs = 150
# for epoch in range(num_epochs):
#     model.train()
#     for i, (images, targets) in enumerate(train_loader):
#         images = [image.to(device) for image in images]
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

#         optimizer.zero_grad()
#         outputs = model(images)

#         losses = criterion(outputs, targets)
        
#         print(f'Epoch #{epoch} id: {i}\n loss_boxes: {losses["loss_boxes"]:.6f} loss_object: {losses["loss_object"]:.6f} loss_class: {losses["loss_class"]:.6f}')
#         losses = sum(loss for loss in losses.values())
#         print(f' total loss: {losses:.6f}')
#         print(losses)

#         # sys.exit()
#         losses.backward()
#         optimizer.step()

#         # if (i+1) > 200:
#         #     scheduler.step()

#         # if (i+1) % 200 == 0:
#         #     break
        
    
#         if iteration in (200, 400, 600, 20000, 30000):
#             scheduler.step()
    
#     if epoch in [40, 50, 80, 100]:
#         print('#'*100)
#         torch.save(model.state_dict(), f'./data/yolov1-{epoch}-1.pth')

#         i += 1

# torch.save(model.state_dict(), './data/yolov1-1.pth')

# def convert(targets):
#     nb = 2
#     gs = 7  # grid size
#     nc = 20
#     cell_size = 1.0 / gs
#     w, h = (448, 448) # Tuple[int, int]
#     # 모형에서 나온 아웃풋과 동일한 모양으로 변환
#     # x1, y1, x2, y2를 center x, center y, w, h로 변환하고
#     # 모든 0~1사이로 변환, cx, cy는 each cell안에서의 비율
#     # w, h는 이미지 대비 비율
#     transformed_targets = torch.zeros(len(targets), gs, gs, 5*nb+nc, device='cuda')
#     for b, target in enumerate(targets):
#         boxes = target['boxes']
#         norm_boxes = boxes / torch.Tensor([[w, h, w, h]]).expand_as(boxes).to('cuda')
#         # center x, y, width and height
#         xys = (norm_boxes[:, 2:] + norm_boxes[:, :2]) / 2.0
#         whs = norm_boxes[:, 2:] - norm_boxes[:, :2]

#         # cnt += boxes.size(0)
#         # print(boxes.size(0))
#         for box_id in range(boxes.size(0)):
#             xy = xys[box_id]
#             wh = whs[box_id]

#             ij = (xy / cell_size).ceil() - 1.0
#             x0y0 = ij * cell_size
#             norm_xy = (xy - x0y0) / cell_size

#             i, j = int(ij[0]), int(ij[1])
#             for k in range(0, 5*nb, 5):
#                 if transformed_targets[b, j, i, k+4] == 1.0:
#                     transformed_targets[b, j, i, k+5:k+5+4] = torch.cat([norm_xy, wh])
#                 else:
#                     transformed_targets[b, j, i, k:k+4] = torch.cat([norm_xy, wh])
#                     transformed_targets[b, j, i, k+4] = 1.0
                
#             # print(transformed_targets[b, j, i, :10])
#             indices = torch.as_tensor(target['labels'][box_id], dtype=torch.int64).view(-1, 1)
#             labels = torch.zeros(indices.size(0), 20).scatter_(1, indices, 1)
#             transformed_targets[b, j, i, 5*nb:] = labels.squeeze()
#     # sys.exit()
#     return transformed_targets


# model = Yolov1Model(yolov1_base_config).to(device)
# model.load_state_dict(torch.load('./data/yolov1-40-1.pth'))
# model.eval()
# for (images, targets) in test_loader:
#     images = [image.to(device) for image in images]
#     targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#     outputs = model(images)
    
#     # transformed = convert(targets)
#     break

# # print(transformed.size())

# # # -> torch.Size([batch_size, S, S, 5*B+C])
# # coord_mask = transformed[..., 4] > 0
# # coord_mask = coord_mask.unsqueeze(-1).expand_as(transformed)#.to(self.device)

# # coord_targets = transformed[coord_mask].view(-1, 30)
# # boxes_targets = coord_targets[:, :5*2].contiguous().view(-1, 5)
# # class_targets = coord_targets[:, 5*2:]


# # import pprint
# # print('Ground truth')
# # for target in targets:
# #     pprint.pprint(target['boxes'])
    
# # print()
# # print('Transform')
# # print(boxes_targets[:, :4])

# # print()
# # print('Convert')
# # w, h = (448, 448)
# # cell_size = 1./7
# # preds = []
# # for bi in range(transformed.size(0)):
# #     boxes = []
# #     scores = []
# #     labels = []
# #     for i in range(7):
# #         for j in range(7):
# #             label = transformed[bi, j, i, 5*2:]
# #             class_proba, _ = torch.max(transformed[bi, j, i, 5*2:], dim=0)
# #             for k in range(2):
# #                 score = transformed[bi, j, i, k*5+4]
            
# #                 proba = score * class_proba
# #                 proba = score * 1.0
# #                 if proba < 0.2:
# #                     continue

# #                 # ij = (xy / cell_size).ceil() - 1.0
# #                 # x0y0 = ij * cell_size
# #                 # norm_xy = (xy - x0y0) / cell_size

# #                 box = transformed[bi, j, i, k*5:k*5+4]

# #                 # print(box)
# #                 x0y0 = torch.FloatTensor([i, j]).to('cuda') * cell_size
# #                 # print(x0y0)
# #                 norm_xy = box[:2] * cell_size + x0y0 
# #                 norm_wh = box[2:]
# #                 # print(norm_xy, norm_wh)
# #                 xyxy = torch.zeros(4, device='cuda')
# #                 xyxy[:2] = norm_xy - 0.5 * norm_wh# * 0.5# * norm_wh
# #                 xyxy[2:] = norm_xy + 0.5 * norm_wh# * 0.5
# #                 # print(class_proba.size(), class_label.size())

# #                 xyxy[0], xyxy[1] = xyxy[0]*w, xyxy[1]*h
# #                 xyxy[2], xyxy[3] = xyxy[2]*w, xyxy[3]*h
# #                 # print(xyxy)
# #                 # sys.exit()
# #                 # sys.exit()
# #                 boxes.append(xyxy)
# #                 scores.append(score)
# #             # print(boxes)
# #             labels.append(label)
# #     preds.append({'boxes': torch.stack(boxes).to('cuda')})

# # for i in preds:
# #     pprint.pprint(i['boxes'])















# # image = images[0]
# # # print(image.size())
# # # print(images)
# # image = image.permute(1, 2, 0)
# # image = image.detach()
# # image = image.cpu()
# # image = image.numpy()
# # print('#'*100)
    


# fig, axes = plt.subplots(1, 4, figsize=(16, 8), dpi=300)
# # boxes2 = targets[0]['boxes'].numpy()
# # image = image.permute(1, 2, 0).detach().cpu().numpy()
# for i, (image, out1, out2) in enumerate(zip(images, outputs, targets)):
#     # print(image.shape)
#     # sys.exit(0)
#     # image = image.permute(1, 2, 0).detach().cpu().numpy()
#     image = image.permute(1, 2, 0).detach().cpu().numpy()
#     # for box, box2 in zip(outputs[0]['boxes'], targets[0]['boxes']):
#     # print(out1)
#     # sys.exit(0)
#     print(torch.max(out1['labels'], 1)[1])

#     for box in out1['boxes']:
#         cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
#         # label = PASCAL_CLASSES[torch.max(out1['labels'], 1)[1]]
#         # cv2.putText(image, label, (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9, color=(255, 0, 0), thickness=1, lineType=8)
        

#     for box2 in out2['boxes']:
        
#         # label = PASCAL_CLASSES[torch.max(out1['labels'], 1)[1]]
#         # cv2.putText(image, label, (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9, color=(255, 0, 0), thickness=1, lineType=8)
#         cv2.rectangle(image, (int(box2[0]), int(box2[1])), (int(box2[2]), int(box2[3])), (0, 0, 255), 2)


#     axes.flat[i].set_axis_off()
#     axes.flat[i].imshow(image)
# plt.show()





#         # print(len(outputs))
#         # print(outputs[0]['boxes'].size())
#         # print(outputs[0]['boxes'].view(2, -1, 4).size())

#         # losses = criterion(outputs, targets)
        



        
        
        
#         # loss_value = losses.item()

#         # loss_hist.update(loss_value)

        
#         # losses.backward()
#         # optimizer.step()

    
    

# # sys.exit(0)

# # images, targets = next(iter(valid_loader))
# # images = list(img.to(device) for img in images)
# # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

# # boxes = targets[1]['boxes'].cpu().numpy().astype(np.int32)
# # sample = images[1].permute(1,2,0).cpu().numpy()

# # model.eval()
# # cpu_device = torch.device("cpu")

# # outputs = model(images)
# # print(outputs)
# # outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]



# # fig, ax = plt.subplots(1, 1, figsize=(16, 8))

# # for box in boxes:
# #     cv2.rectangle(sample,
# #                   (box[0], box[1]),
# #                   (box[2], box[3]),
# #                   (220, 0, 0), 3)
    
# # ax.set_axis_off()
# # ax.imshow(sample)
# # plt.show()

# # torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn.pth')