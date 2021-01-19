import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from boda.utils.dataset import CocoDataset
from boda.utils.transforms import Compose, Resize, ToTensor, Normalize
from boda.models.configuration_yolact import YolactConfig
from boda.models.architecture_yolact import YolactModel
from boda.models.loss_yolact import YolactLoss
from boda.utils.trainer import Trainer
from boda.lib.torchsummary import summary


transforms = Compose([
    Resize((550, 550)),
    ToTensor(),
    Normalize()
])

# dataset = CocoDataset(
#     image_dir='./benchmarks/dataset/coco/train2014/',
#     info_file='./benchmarks/dataset/coco/annotations/instances_train2014.json',
#     transforms=transforms)

dataset = CocoDataset(
    image_dir='./benchmarks/dataset/custom/train/',
    info_file='./benchmarks/dataset/custom/train/annotations.json',
    transforms=transforms)

# validset = CocoDataset(
#     image_dir='./benchmarks/dataset/custom/valid/',
#     info_file='./benchmarks/dataset/custom/valid/annotations.json',
#     mode='valid',
#     transforms=transforms)


def collate_fn(batch):
    return tuple(zip(*batch))


train_loader = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=collate_fn)
# valid_loader = DataLoader(validset, batch_size=4, num_workers=0, collate_fn=collate_fn)

config = YolactConfig(num_classes=7)
model = YolactModel(config).to('cuda')
# print(summary(model, input_data=(3, 550, 550), verbose=0))


model.load_state_dict(torch.load('cache/test.pth'))
model.eval()

from boda.utils.detector import Detector

for images, targets in train_loader:
    images = [image.to('cuda') for image in images]
    outputs = model(images)
    outputs = Detector(7)(outputs)
    print('Done!!!!!!!!!!')
    break

print('Done2')
print(outputs.keys())
print('boxes:', outputs['boxes'].device, outputs['boxes'].size())
print('masks:', outputs['masks'].device, outputs['masks'].size())
print('scores:', outputs['scores'].device, outputs['scores'].size())
print('labels:', outputs['labels'].device, outputs['labels'].size())
print('proto:', outputs['proto_masks'].device, outputs['proto_masks'].size())
print('Done3')

# config = YolactConfig()
# model = YolactModel(config).to('cuda')
# print(summary(model, input_data=(3, 550, 550), verbose=0))

# state_dict = torch.load('yolact_base_54_800000.pth')
# # print(state_dict.keys())

# for key, value in state_dict.items():
#     if key[:8] != 'backbone':
#         print(key, value.size())

# state_dict = torch.load('yolact.pth')
# # print(state_dict.keys())
# print()
# for key, value in state_dict.items():
#     if key[:8] != 'backbone':
#         print(key, value.size())

# def collate_fn(batch):
#     return tuple(zip(*batch))