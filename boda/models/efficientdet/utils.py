import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops.boxes import batched_nms


class BBoxTransform(nn.Module):
    def forward(self, anchors, regression):
        y_centers_a = (anchors[..., 0] + anchors[..., 2]) / 2
        x_centers_a = (anchors[..., 1] + anchors[..., 3]) / 2
        ha = anchors[..., 2] - anchors[..., 0]
        wa = anchors[..., 3] - anchors[..., 1]

        w = regression[..., 3].exp() * wa
        h = regression[..., 2].exp() * ha

        y_centers = regression[..., 0] * ha + y_centers_a
        x_centers = regression[..., 1] * wa + x_centers_a

        ymin = y_centers - h / 2.
        xmin = x_centers - w / 2.
        ymax = y_centers + h / 2.
        xmax = x_centers + w / 2.

        return torch.stack([xmin, ymin, xmax, ymax], dim=2)


class ClipBoxes(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, boxes, img):
        _, _, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width - 1)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height - 1)

        return boxes


class Resizer:
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, image):
        _, height, width = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = F.interpolate(
            image[None],
            size=(resized_height, resized_width),
            mode='bilinear',
            align_corners=False,
        )[0]
        
        new_image = torch.zeros((3, self.img_size, self.img_size))
        new_image[:, 0:resized_height, 0:resized_width] = image

        return new_image
    
    
def invert_affine(old_whs, input_size, preds):
    for i in range(len(preds)):
        if len(preds[i]['boxes']) == 0:
            continue
        else:
            old_w, old_h = old_whs[i]
            preds[i]['boxes'][:, [0, 2]] = preds[i]['boxes'][:, [0, 2]] / (input_size / old_w)
            preds[i]['boxes'][:, [1, 3]] = preds[i]['boxes'][:, [1, 3]] / (input_size / old_h)
    return preds


def postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold):
    transformed_anchors = regressBoxes(anchors, regression)
    transformed_anchors = clipBoxes(transformed_anchors, x)
    scores = torch.max(classification, dim=2, keepdim=True)[0]
    scores_over_thresh = (scores > threshold)[:, :, 0]
    out = []
    for i in range(x.shape[0]):
        if scores_over_thresh[i].sum() == 0:
            out.append({
                'boxes': torch.tensor(()),
                'labels': torch.tensor(()),
                'scores': torch.tensor(()),
            })
            continue

        classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)
        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
        scores_per = scores[i, scores_over_thresh[i, :], ...]
        scores_, classes_ = classification_per.max(dim=0)
        anchors_nms_idx = batched_nms(transformed_anchors_per, scores_per[:, 0], classes_, iou_threshold=iou_threshold)

        if anchors_nms_idx.shape[0] != 0:
            classes_ = classes_[anchors_nms_idx]
            scores_ = scores_[anchors_nms_idx]
            boxes_ = transformed_anchors_per[anchors_nms_idx, :]

            out.append({
                'boxes': boxes_,
                'labels': classes_,
                'scores': scores_,
            })
        else:
            out.append({
                'boxes': torch.tensor(()),
                'labels': torch.tensor(()),
                'scores': torch.tensor(()),
            })

    return out
