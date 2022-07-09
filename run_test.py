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

image = Image.open('test6.jpg')
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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.measure import find_contours
import adjustText

np_image = np.array(image)
np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
# for box in outputs[0]['boxes']:
#     # box = list(map(int, boxes[j, :]))
#     x1, y1, x2, y2 = box.detach().cpu().numpy()
#     # score = scores[j]
#     # label = labels[j]
#     cv2.rectangle(np_image, (x1, y1), (x2, y2), (0, 0, 255), thickness=1)

plt.imshow(image)
ax = plt.gca()
threshold = 0
COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush')

COLORS = {
    1: 'deepskyblue',
    2: 'orangered',
    3: 'yellowgreen',
    4: 'darkorange',
    5: 'chocolate',
    6: 'slategrey',
    7: 'darkgoldenrod',
    8: 'purple',
    9: 'saddlebrown',
    10: 'olive',
}

for output in outputs:
    boxes = output['boxes']
    scores = output['scores']
    labels = output['labels']
    masks = output['masks']
    print(scores)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.detach().cpu().numpy()
        score = scores[i].detach().cpu().numpy()
        label = labels[i].detach().cpu().numpy()
        mask = masks[i].detach().cpu().numpy().astype(np.int64)

        color = COLORS[(label+1) % 11]
        contours = find_contours(mask, 0.5)

        if score >= threshold:
            cx = x2 - x1
            cy = y2 - y1
            ax.text(x1, y1, f"{COCO_CLASSES[label]}", c='black', size=8, va='bottom', ha='left', alpha=0.5)

            rect = patches.Rectangle(
                (x1, y1),
                cx, cy,
                linewidth=1,
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(rect)

            for contour in contours:
                shapes = []
                for point in contour:
                    shapes.append([int(point[1]), int(point[0])])

                polygon_edge = patches.Polygon(
                        (shapes),
                        edgecolor=color,
                        facecolor='none',
                        linewidth=1,
                        fill=False,
                    )

                polygon_fill = patches.Polygon(
                    (shapes),
                    alpha=0.5,
                    edgecolor='none',
                    facecolor=color,
                    fill=True
                )

                ax.add_patch(polygon_edge)
                ax.add_patch(polygon_fill)
                

plt.axis('off')
plt.savefig('test.jpg' ,dpi=100, bbox_inches='tight', pad_inches=0)
