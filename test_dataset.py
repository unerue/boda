import torch
from torch.utils.data import DataLoader
from boda.utils.dataset import CocoDataset
from boda.utils.transforms import Compose, Resize, ToTensor, Normalize
from boda.models.configuration_yolact import YolactConfig
from boda.models.architecture_yolact import YolactModel
from boda.models.loss_yolact import YolactLoss
from boda.lib.torchsummary import summary


transforms = Compose([
    Resize((550, 550)),
    ToTensor(),
    Normalize()
])

dataset = CocoDataset(
    image_dir='./benchmarks/dataset/coco/train2014/',
    info_file='./benchmarks/dataset/coco/annotations/instances_train2014.json',
    transforms=transforms)

def collate_fn(batch):
    return tuple(zip(*batch))

train_loader = DataLoader(dataset, batch_size=2, num_workers=0, collate_fn=collate_fn)

config = YolactConfig()
model = YolactModel(config).to('cuda')
# print(summary(model, input_data=(3, 550, 550), verbose=0))

criterion = YolactLoss()

num_epochs = 1
for epoch in range(num_epochs):
    for images, targets in train_loader:
        images = [image.to('cuda') for image in images]
        print(images[0].shape)
        targets = [{k: v.to('cuda') for k, v in t.items()} for t in targets]

        outputs = model(images)
        losses = criterion(outputs, targets)
        print(losses)
        break

