import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from boda.models.configuration_yolov1 import Yolov1Config
from boda.models.backbone_darknet import darknet, darknet21
from boda.models.architecture_yolov1 import Yolov1Model
from boda.models.loss_yolov1 import Yolov1Loss
from boda.utils.parser import CocoDataset
from torchsummary import summary
from boda.utils.timer import Timer


# model = darknet21()
# print(summary(model, input_data=(3, 448, 448), verbose=0))

config = Yolov1Config()
model = Yolov1Model(config).to('cuda')
criterion = Yolov1Loss(config)
optimizer = optim.SGD(model.parameters(), 0.001)


# with Timer('start'):
#     config = Yolov1Config()
#     model = Yolov1Model(config).to('cuda')
#     criterion = Yolov1Loss(config)
#     optimizer = optim.SGD(model.parameters(), 0.001)


def collate_fn(batch):
    return tuple(zip(*batch))


coco_dataset = CocoDataset(
    './benchmarks/dataset/train', 
    './benchmarks/dataset/train/annotations.json')
# coco_dataset[10]
train_loader = DataLoader(
    coco_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=2,
    collate_fn=collate_fn)


def run():
    model.train()
    num_epochs = 10
    for _ in range(num_epochs):
        for images, targets in train_loader:
            images = [image.to('cuda') for image in images]
            targets = [{k: v.to('cuda') for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            outputs = model(images)
            losses = criterion(outputs, targets)
            loss = sum(value for value in losses.values())
            loss.backward()

            optimizer.step()
        print(losses)


if __name__ == '__main__':
    run()

