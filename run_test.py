from boda.models import YolactConfig, YolactModel
from boda.models.feature_extractor import resnet50, resnet101
# from boda.lib.torchinfo import summary
from boda.lib.torchsummary import summary
import torch

config = YolactConfig(num_classes=80)
model = YolactModel(config, backbone=resnet101()).to('cuda')
model.train()
print(model)
# print(summary(model, input_size=(16, 3, 550, 550), verbose=0))
print(summary(model, input_data=(3, 550, 550), verbose=0))

# model.load_weights('cache/yolact-base.pth')


from boda.models import PostprocessYolact
from PIL import Image
from torchvision import transforms

image = Image.open('test4.jpg')
model = YolactModel.from_pretrained('yolact-base').cuda()
model.eval()

aug = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    # transforms.Normalize([0.406, 0.456, 0.485], [0.225, 0.224, 0.229])
])

outputs = model([aug(image).cuda()])

print(outputs.keys())
post = PostprocessYolact()
outputs = post(outputs, outputs['image_sizes'])
print(outputs[0]['boxes'])
import cv2
import numpy as np

np_image = np.array(image)
np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
for box in outputs[0]['boxes']:
    # box = list(map(int, boxes[j, :]))
    x1, y1, x2, y2 = box.detach().cpu().numpy()
    # score = scores[j]
    # label = labels[j]
    cv2.rectangle(np_image, (x1, y1), (x2, y2), (0, 0, 255), thickness=1)

cv2.imwrite('test.jpg', np_image)