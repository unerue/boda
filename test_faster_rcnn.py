import cv2
import torch
from boda.lib.torchsummary import summary
from boda.models import FasterRcnnConfig, FasterRcnnModel

from boda.utils.transforms import Compose, ToTensor, Normalize

config = FasterRcnnConfig()
model = FasterRcnnModel(config).to('cuda')
model.load_weights('cache/faster_rcnn/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth')
model.eval()

images = [torch.randn((3, 1920, 1080), dtype=torch.float32).to('cuda') for _ in range(3)]

file_names = ['test1.jpg', 'test2.jpg', 'test3.jpg', 'test4.jpg', 'test5.jpg', 'test6.jpg', 'test7.jpg', 'test8.jpg']

tensors = []
for file_name in file_names:
    image = cv2.imread(file_name, cv2.COLOR_BGR2RGB)
    print(image.shape)
    image = image / 255.0
    # image = cv2.resize(image, (550, 550))
    image = image.transpose((2, 0, 1))
    # image /= 255
    print('input image', image.shape)
    tensor = torch.tensor(image, dtype=torch.float32).to('cuda')
    tensor, _ = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor, {})
    tensors.append(tensor)

# print(summary(model, input_data=(3, 1333, 800), verbose=0))
outputs = model(tensors)
print(len(outputs))
print(len(file_names))
for i, (output, fn) in enumerate(zip(outputs, file_names)):
    image = cv2.imread(fn, cv2.COLOR_BGR2RGB)
    mask = [output['scores'] > 0.15]
    boxes = output['boxes'][mask].detach().cpu().numpy()

    for box in boxes:
        box = list(map(int, box))
        x1, y1, x2, y2 = box
        # score = scores[j]
        # label = labels[j]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), thickness=1)

    cv2.imwrite(f'rcnn-result{i}.jpg', image)
