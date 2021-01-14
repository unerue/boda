import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from boda.utils.dataset import CocoDataset
from boda.utils.transforms import Compose, Resize, ToTensor, Normalize
from boda.models.configuration_yolov1 import Yolov1Config
from boda.models.backbone_darknet import darknet
from boda.models.architecture_yolov1 import Yolov1Model
from boda.models.loss_yolov1 import Yolov1Loss
from boda.utils.trainer import Trainer
from boda.lib.torchsummary import summary


# model = darknet().to('cuda')
# print(summary(model, input_data=(3, 448, 448), verbose=0))

# config = Yolov1Config()
# model = Yolov1Model(config).to('cuda')
# print(summary(model, input_data=(3, 448, 448), verbose=0))

# sys.exit()

transforms = Compose([
    Resize((448, 448)),
    ToTensor(),
    Normalize()
])

# dataset = CocoDataset(
#     image_dir='./benchmarks/dataset/coco/train2014/',
#     info_file='./benchmarks/dataset/coco/annotations/instances_train2014.json',
#     transforms=transforms)

dataset = CocoDataset(
    image_dir='./benchmarks/dataset/voc/train/',
    info_file='./benchmarks/dataset/voc/train/annotations.json',
    use_mask=False,
    transforms=transforms)

# validset = CocoDataset(
#     image_dir='./benchmarks/dataset/custom/valid/',
#     info_file='./benchmarks/dataset/custom/valid/annotations.json',
#     mode='valid',
#     use_mask=False,
#     transforms=transforms)


def collate_fn(batch):
    return tuple(zip(*batch))


train_loader = DataLoader(dataset, batch_size=4, num_workers=0, collate_fn=collate_fn)
valid_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)

config = Yolov1Config(num_classes=20)
model = Yolov1Model(config).to('cuda')
# print(summary(model, input_data=(3, 550, 550), verbose=0))

optimizer = optim.SGD(model.parameters(), 1e-4)
criterion = Yolov1Loss()

# trainer = Trainer(
#     train_loader, model, optimizer, criterion, num_epochs=400, verbose=10)
# trainer.train()

state_dict = torch.load('test.pth')
model.load_state_dict(state_dict)

model.eval()
for images, targets in valid_loader:
    images = [image.to('cuda') for image in images]
    preds = model(images)
    break

print(preds)
print(preds.size())

preds = preds.cpu().data
preds = preds.squeeze(0)
print(preds.size())

cell_size = 1.0 / 7
boxes, labels, confidences, class_scores = [], [], [], []
for i in range(7): # for x-dimension.
    for j in range(7): # for y-dimension.
        class_score, class_label = torch.max(preds[j, i, 5*2:], 0)

        for b in range(2):
            conf = preds[j, i, 5*b + 4]
            prob = conf * class_score
            if float(prob) < 0.005:
                continue

            # Compute box corner (x1, y1, x2, y2) from tensor.
            box = preds[j, i, 5*b : 5*b + 4]

            x0y0_normalized = torch.FloatTensor([i, j]).to(preds.device) * cell_size # cell left-top corner. Normalized from 0.0 to 1.0 w.r.t. image width/height.
            xy_normalized = box[:2] * cell_size + x0y0_normalized   # box center. Normalized from 0.0 to 1.0 w.r.t. image width/height.
            wh_normalized = box[2:] # Box width and height. Normalized from 0.0 to 1.0 w.r.t. image width/height.
            box_xyxy = torch.FloatTensor(4) # [4,]
            box_xyxy[:2] = xy_normalized - 0.5 * wh_normalized # left-top corner (x1, y1).
            box_xyxy[2:] = xy_normalized + 0.5 * wh_normalized # right-bottom corner (x2, y2).

            # Append result to the lists.
            boxes.append(box_xyxy)
            labels.append(class_label)
            confidences.append(conf)
            class_scores.append(class_score)

print(boxes)
if len(boxes) > 0:
    boxes = torch.stack(boxes, 0) # [n_boxes, 4]
    labels = torch.stack(labels, 0)             # [n_boxes, ]
    confidences = torch.stack(confidences, 0)   # [n_boxes, ]
    class_scores = torch.stack(class_scores, 0) # [n_boxes, ]
else:
    # If no box found, return empty tensors.
    boxes = torch.FloatTensor(0, 4)
    labels = torch.LongTensor(0)
    confidences = torch.FloatTensor(0)
    class_scores = torch.FloatTensor(0)

boxes_normalized_all = boxes
class_labels_all = labels
confidences_all = confidences
class_scores_all = class_scores

print()
print(boxes)
boxes_normalized, class_labels, probs = [], [], []

VOC_CLASS_BGR = {
    'person': (192, 128, 128),
    'bird': (128, 128, 0),
    'cat': (64, 0, 0),
    'cow': (64, 128, 0),
    'dog': (64, 0, 128),
    'horse': (192, 0, 128),
    'sheep': (128, 64, 0),
    'aeroplane': (128, 0, 0),
    'bicycle': (0, 128, 0),
    'boat': (0, 0, 128),
    'bus': (0, 128, 128),
    'car': (128, 128, 128),
    'motorbike': (64, 128, 128),
    'train': (128, 192, 0),
    'bottle': (128, 0, 128),
    'chair': (192, 0, 0),
    'diningtable': (192, 128, 0),
    'pottedplant': (0, 64, 0),
    'sofa': (0, 192, 0),
    'tvmonitor': (0, 64, 128)
}


def nms(boxes, scores, nms_thresh=0.5):
    """ Apply non maximum supression.
    Args:
    Returns:
    """
    threshold = nms_thresh

    x1 = boxes[:, 0] # [n,]
    y1 = boxes[:, 1] # [n,]
    x2 = boxes[:, 2] # [n,]
    y2 = boxes[:, 3] # [n,]
    areas = (x2 - x1) * (y2 - y1) # [n,]

    _, ids_sorted = scores.sort(0, descending=True) # [n,]
    ids = []
    while ids_sorted.numel() > 0:
        # Assume `ids_sorted` size is [m,] in the beginning of this iter.

        i = ids_sorted.item() if (ids_sorted.numel() == 1) else ids_sorted[0]
        ids.append(i)

        if ids_sorted.numel() == 1:
            break # If only one box is left (i.e., no box to supress), break.
        
        print(x1[i])
        inter_x1 = x1[ids_sorted[1:]].clamp(min=x1[i]) # [m-1, ]
        inter_y1 = y1[ids_sorted[1:]].clamp(min=y1[i]) # [m-1, ]
        inter_x2 = x2[ids_sorted[1:]].clamp(max=x2[i]) # [m-1, ]
        inter_y2 = y2[ids_sorted[1:]].clamp(max=y2[i]) # [m-1, ]
        inter_w = (inter_x2 - inter_x1).clamp(min=0) # [m-1, ]
        inter_h = (inter_y2 - inter_y1).clamp(min=0) # [m-1, ]

        inters = inter_w * inter_h # intersections b/w/ box `i` and other boxes, sized [m-1, ].
        unions = areas[i] + areas[ids_sorted[1:]] - inters # unions b/w/ box `i` and other boxes, sized [m-1, ].
        ious = inters / unions # [m-1, ]

        # Remove boxes whose IoU is higher than the threshold.
        ids_keep = (ious <= threshold).nonzero().squeeze() # [m-1, ]. Because `nonzero()` adds extra dimension, squeeze it.
        if ids_keep.numel() == 0:
            break # If no box left, break.

        ids_sorted = ids_sorted[ids_keep+1] # `+1` is needed because `ids_sorted[0] = i`.

    return torch.LongTensor(ids)



w, h = 448, 448

for class_label in range(len(VOC_CLASS_BGR.keys())):
    mask = (class_labels_all == class_label)
    if torch.sum(mask) == 0:
        continue # if no box found, skip that class.

    boxes_normalized_masked = boxes_normalized_all[mask]
    class_labels_maked = class_labels_all[mask]
    confidences_masked = confidences_all[mask]
    class_scores_masked = class_scores_all[mask]

    ids = nms(boxes_normalized_masked, confidences_masked, )

    boxes_normalized.append(boxes_normalized_masked[ids])
    class_labels.append(class_labels_maked[ids])
    probs.append(confidences_masked[ids] * class_scores_masked[ids])

boxes_normalized = torch.cat(boxes_normalized, 0)
class_labels = torch.cat(class_labels, 0)
probs = torch.cat(probs, 0)

# Postprocess for box, labels, probs.
boxes_detected, class_names_detected, probs_detected = [], [], []
for b in range(boxes_normalized.size(0)):
    box_normalized = boxes_normalized[b]
    class_label = class_labels[b]
    prob = probs[b]

    x1, x2 = w * box_normalized[0], w * box_normalized[2] # unnormalize x with image width.
    y1, y2 = h * box_normalized[1], h * box_normalized[3] # unnormalize y with image height.
    boxes_detected.append(((x1.item(), y1.item()), (x2.item(), y2.item())))

    class_label = int(class_label) # convert from LongTensor to int.
    class_name = list(VOC_CLASS_BGR.keys())[class_label]
    class_names_detected.append(class_name)

    prob = float(prob) # convert from Tensor to float.
    probs_detected.append(prob)

# print(prob)
print(boxes_detected, class_names_detected, probs_detected)

import cv2

print(images[0].size())
image = images[0].permute(2, 0, 1).detach().cpu().numpy()
# print(image.shape)
image = image.transpose(2, 0, 1)
print(image.shape)
image *= 255

def visualize_boxes(image_bgr, boxes, class_names, probs, name_bgr_dict=None, line_thickness=1):
    if name_bgr_dict is None:
        name_bgr_dict = VOC_CLASS_BGR

    image_boxes = image_bgr.copy()
    for box, class_name, prob in zip(boxes, class_names, probs):
        # Draw box on the image.
        left_top, right_bottom = box
        left, top = int(left_top[0]), int(left_top[1])
        right, bottom = int(right_bottom[0]), int(right_bottom[1])
        bgr = name_bgr_dict[class_name]
        cv2.rectangle(image_boxes, (left, top), (right, bottom), bgr, thickness=line_thickness)

        # Draw text on the image.
        text = '%s %.2f' % (class_name, prob)
        size, baseline = cv2.getTextSize(text,  cv2.FONT_HERSHEY_DUPLEX, fontScale=0.6, thickness=1)
        text_w, text_h = size

        x, y = left, top
        x1y1 = (x, y)
        x2y2 = (x + text_w + line_thickness, y + text_h + line_thickness + baseline)
        cv2.rectangle(image_boxes, x1y1, x2y2, bgr, -1)
        cv2.putText(image_boxes, text, (x + line_thickness, y + 2*baseline + line_thickness),
            cv2.FONT_HERSHEY_DUPLEX, fontScale=0.6, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    return image_boxes


image_boxes = visualize_boxes(image, boxes_detected, class_names_detected, probs_detected)
cv2.imwrite('test.jpg', image_boxes)


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