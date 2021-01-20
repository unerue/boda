from numpy.core.fromnumeric import sort
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

# dataset = CocoDataset(
#     image_dir='./benchmarks/dataset/custom/train/',
#     info_file='./benchmarks/dataset/custom/train/annotations.json',
#     transforms=transforms)

# validset = CocoDataset(
#     image_dir='./benchmarks/dataset/custom/valid/',
#     info_file='./benchmarks/dataset/custom/valid/annotations.json',
#     mode='valid',
#     transforms=transforms)


def collate_fn(batch):
    return tuple(zip(*batch))


# train_loader = DataLoader(dataset, batch_size=4, num_workers=0, collate_fn=collate_fn)
# valid_loader = DataLoader(validset, batch_size=4, num_workers=0, collate_fn=collate_fn)

config = YolactConfig(num_classes=80)
# model = YolactModel(config).to('cuda')
model = YolactModel.from_pretrained('yolact-base')
print(summary(model, input_data=(3, 550, 550), depth=3, verbose=0))

# model.load_weights('')
model.eval()

import cv2
# image = cv2.imread('./benchmarks/dataset/voc/train/JPEGImages/2007_000274.jpg')
image = cv2.imread('./benchmarks/dataset/voc/train/JPEGImages/2012_004245.jpg')
image = cv2.imread('tenis.jpg')
print(image.shape)
# image = image / 255.0
image = cv2.resize(image, (550, 550))
image = image.transpose((2, 0, 1))
# image /= 255
print(image.shape)

tensor = torch.tensor(image, dtype=torch.float32).to('cuda')
tensor, _ = Normalize()(tensor, {})
# print(tensor)
outputs = model([tensor])
# print(outputs[0]['boxes'])


from boda.utils.bbox import sanitize_coordinates


def postprocess(preds, size):
    # pred_boxes = preds['boxes']
    w, h = size
    boxes = preds['boxes']
    pred_masks = preds['masks']
    pred_scores = preds['scores']
    # prior_boxes = preds[0]['prior_boxes']
    proto_masks = preds['proto_masks']

    keep = pred_scores > 0.2
    
    boxes = boxes[keep]
    pred_scores = pred_scores[keep]

    masks = proto_masks @ pred_masks.t()
    masks = torch.sigmoid(masks)
    from boda.utils.bbox import crop
    import torch.nn.functional as F
    masks = crop(masks, boxes)
    masks = masks.permute(2, 0, 1).contiguous()

    masks = F.interpolate(masks.unsqueeze(0), (h, w), mode='bilinear', align_corners=False).squeeze(0)

    # # Binarize the masks
    masks.gt_(0.5)

    boxes[:, 0], boxes[:, 2] = \
        sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, cast=False)
    boxes[:, 1], boxes[:, 3] = \
        sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, cast=False)
    boxes = boxes.long()

    preds['boxes'] = boxes
    preds['proto_masks'] = proto_masks
    preds['masks'] = masks

    return preds


outputs = postprocess(outputs[0], (396, 594))

scores = outputs['scores'].detach().cpu()
sorted_index = scores.argsort(0, descending=True)[:5]

boxes = outputs['boxes'][sorted_index]
labels = outputs['labels'][sorted_index]
scores = scores[sorted_index]
masks = outputs['masks'][sorted_index]
print(masks.size())
num_dets_to_consider = min(5, labels.shape[0])
for j in range(num_dets_to_consider):
    if scores[j] < 0.0:
        num_dets_to_consider = j
        break

# def get_color(j, on_gpu=None):
#     global color_cache
#     color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)
    
#     if on_gpu is not None and color_idx in color_cache[on_gpu]:
#         return color_cache[on_gpu][color_idx]
#     else:
#         color = COLORS[color_idx]
#         if not undo_transform:
#             # The image might come in as RGB or BRG, depending
#             color = (color[2], color[1], color[0])
#         if on_gpu is not None:
#             color = torch.Tensor(color).to(on_gpu).float() / 255.
#             color_cache[on_gpu][color_idx] = color
#         return color

        
mask_alpha = 0.45
image2 = image.transpose((1, 2, 0))
image2 = cv2.resize(image2, (396, 594))
img_gpu = torch.Tensor(image2).cuda()
masks = masks[:num_dets_to_consider, :, :, None]
print(masks.shape)
colors = torch.cat([(torch.Tensor((0, 0, 0)).to('cuda').float() / 255).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha
inv_alph_masks = masks * (-mask_alpha) + 1

masks_color_summand = masks_color[0]
if num_dets_to_consider > 1:
    inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
    masks_color_cumul = masks_color[1:] * inv_alph_cumul
    masks_color_summand += masks_color_cumul.sum(dim=0)

img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand
image = (img_gpu * 255).byte().cpu().numpy()
print(image.shape)
# print(image)
# image = image * masks

# image = image.transpose((2, 0, 1))
print(image.shape)
# image = image / 255.0
# image = cv2.resize(image, (396, 594))

# image = image * masks
# image = image * 255.0
print(type(image))
for j in reversed(range(num_dets_to_consider)):
    box = boxes[j, :].detach().cpu().numpy()
    score = scores[j]
    label = labels[j]
    print(box, score, label)
    # cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 50), thickness=1)
    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 50))

cv2.imwrite('test.jpg', image)
# print(outputs)

# sorted_index = outputs[0]['scores'].argsort(0, descending=True)[:5]
# outputs[0]['boxes'] = outputs[0]['boxes'][sorted_index]

# # Must to be proto masks
# masks = outputs[0]['masks'][sorted_index]

# num_dets_to_consider = min(self.top_k, outputs['labels'].shape[0])
# for j in range(num_dets_to_consider):
#     if outputs['scores'][j] < self.score_threshold:
#         num_dets_to_consider = j
#         break

# classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]



# image = image.transpose((1, 2, 0))
# # image = image * 255
# print(image.shape)
# for box in outputs[0]['boxes']:
#     cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 50), thickness=1)

# cv2.imwrite('test.jpg', image)


# optimizer = optim.SGD(model.parameters(), 1e-4)
# criterion = YolactLoss()

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