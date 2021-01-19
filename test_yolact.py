import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from boda.utils.dataset import CocoDataset
from boda.utils.transforms import Compose, Resize, ToTensor, Normalize
from boda.models import YolactConfig, YolactModel, YolactLoss
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


train_loader = DataLoader(dataset, batch_size=4, num_workers=0, collate_fn=collate_fn)
# valid_loader = DataLoader(validset, batch_size=4, num_workers=0, collate_fn=collate_fn)

config = YolactConfig(num_classes=80)
model = YolactModel(config).to('cuda')
print(summary(model, input_data=(3, 550, 550), verbose=0))

optimizer = optim.SGD(model.parameters(), 1e-4)
criterion = YolactLoss()

# trainer = Trainer(
#     train_loader, model, optimizer, criterion, num_epochs=50)
# trainer.train()


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