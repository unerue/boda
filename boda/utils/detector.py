from typing import Tuple
import numpy as np
import torch
from torch import nn, Tensor
from .bbox import crop, sanitize_coordinates
import torch.nn.functional as F


class Detector:
    def __init__(
        self,
        num_classes: int,
    ) -> None:
        self.num_classes = num_classes
        self.background_label = 0
        self.tok_k = 5
        self.nms_threshold = 0.5
        self.score_threshold = 0.2

        self.use_cross_class_nms = False
        self.use_fast_nms = False

    def __call__(self, preds):
        pred_boxes = preds['boxes']
        pred_scores = preds['scores']
        pred_masks = preds['masks']
        pred_priors = preds['priors']

        proto_masks = preds['prototype_masks']

        batch_size = preds['boxes'].size(0)
        num_prior_boxes = preds['priors'].size(0)

        pred_scores = preds['scores'].view(
            batch_size, num_prior_boxes, self.num_classes+1).transpose(2, 1).contiguous()

        for i in range(batch_size):
            decoded_boxes = self.decode(pred_boxes[i], pred_priors)
            results = self.detect(i, decoded_boxes, pred_masks, pred_scores)

            results['proto_masks'] = proto_masks[i]

        return results

    def decode(self, boxes, priors):
        # print(boxes.size(), priors.size())
        variances = [0.1, 0.2]

        boxes = torch.cat((
            priors[:, :2] + boxes[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(boxes[:, 2:] * variances[1])), dim=1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]

        return boxes

    def detect(
        self,
        batch_index,
        decoded_boxes,
        pred_masks,
        pred_scores,
    ):
        pred_scores = pred_scores[batch_index, 1:, :]
        scores, _ = torch.max(pred_scores, dim=0)

        keep = scores > self.score_threshold
        scores = pred_scores[:, keep]
        boxes = decoded_boxes[keep, :]
        masks = pred_masks[batch_index, keep, :]

        if scores.size(1) == 0:
            return None

        boxes, masks, scores, labels = self.nms(boxes, scores, masks)

        return_dict = {
            'boxes': boxes,
            'masks': masks,
            'scores': scores,
            'labels': labels,
        }

        return return_dict

    def nms(
        self,
        pred_boxes: Tensor,
        pred_scores: Tensor,
        pred_masks: Tensor = None,
        iou_threshold: float = 0.5,
        scores_threshold: float = 0.05,
        max_num_detections: int = 200
    ) -> Tuple[Tensor]:
        import pyximport
        pyximport.install(setup_args={'include_dirs': np.get_include()}, reload_support=True)

        from .cython_nms import nms as cnms

        num_classes = pred_scores.size(0)

        indexes = []
        labels = []
        scores = []

        max_size = 550
        pred_boxes = pred_boxes * max_size

        for i in range(num_classes):
            score = pred_scores[i, :]
            score_mask = score > scores_threshold
            index = torch.arange(score.size(0), device=pred_boxes.device)

            score = score[score_mask]
            index = index[score_mask]

            if score.size(0) == 0:
                continue

            preds = torch.cat(
                [pred_boxes[score_mask], score[:, None]], dim=1).detach().cpu().numpy()
            keep = cnms(preds, iou_threshold)
            keep = torch.Tensor(keep, device='cpu').long()

            indexes.append(index[keep])
            labels.append(keep * 0 + i)
            scores.append(score[keep])

        indexes = torch.cat(indexes, dim=0)
        labels = torch.cat(labels, dim=0)
        scores = torch.cat(scores, dim=0)

        scores, sorted_index = scores.sort(0, descending=True)
        sorted_index = sorted_index[:max_num_detections]
        scores = scores[:max_num_detections]

        indexes = indexes[sorted_index]
        labels = labels[indexes]

        pred_boxes = pred_boxes[indexes] / max_size
        pred_masks = pred_masks[indexes]

        return pred_boxes, pred_masks, scores, labels


def postprocess(
    det_output,
    w,
    h,
    batch_index=0,
    interpolation_mode='bilinear',
    visualize_lincomb=False,
    crop_masks=True,
    score_threshold=0
) -> None:
    """
    Postprocesses the output of Yolact on testing mode into a format that makes sense,
    accounting for all the possible configuration settings.
    Args:
        - det_output: The lost of dicts that Detect outputs.
        - w: The real with of the image.
        - h: The real height of the image.
        - batch_idx: If you have multiple images for this batch, the image's index in the batch.
        - interpolation_mode: Can be 'nearest' | 'area' | 'bilinear' (see torch.nn.functional.interpolate)
    Returns 4 torch Tensors (in the following order):
        - classes [num_det]: The class idx for each detection.
        - scores  [num_det]: The confidence score for each detection.
        - boxes   [num_det, 4]: The bounding box for each detection in absolute point form.
        - masks   [num_det, h, w]: Full image masks for each detection.
    """
    dets = det_output[batch_index]
    # dets = dets['detection']

    if dets is None:
        return [torch.Tensor()] * 4 # Warning, this is 4 copies of the same thing

    if score_threshold > 0:
        keep = dets['scores'] > score_threshold

        for k in dets:
            if k != 'proto':
                dets[k] = dets[k][keep]

        if dets['scores'].size(0) == 0:
            return [torch.Tensor()] * 4

    # Actually extract everything from dets now
    classes = dets['labels']
    boxes = dets['boxes']
    scores = dets['scores']
    masks = dets['masks']

    # At this points masks is only the coefficients
    proto_data = dets['prototype_masks']

    # # Test flag, do not upvote
    # if cfg.mask_proto_debug:
    #     np.save('scripts/proto.npy', proto_data.cpu().numpy())

    if visualize_lincomb:
        display_lincomb(proto_data, masks)

    masks = proto_data @ masks.t()
    masks = torch.sigmoid(masks)

    # Crop masks before upsampling because you know why
    if crop_masks:
        masks = crop(masks, boxes)

    # Permute into the correct output shape [num_dets, proto_h, proto_w]
    masks = masks.permute(2, 0, 1).contiguous()

    # Scale masks up to the full image
    masks = F.interpolate(masks.unsqueeze(0), (h, w), mode=interpolation_mode, align_corners=False).squeeze(0)

    # Binarize the masks
    masks.gt_(0.5)

    boxes[:, 0], boxes[:, 2] = \
        sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, cast=False)
    boxes[:, 1], boxes[:, 3] = \
        sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, cast=False)
    boxes = boxes.long()
    # Upscale masks
    full_masks = torch.zeros(masks.size(0), h, w)

    for jdx in range(masks.size(0)):
        x1, y1, x2, y2 = boxes[jdx, :]

        mask_w = x2 - x1
        mask_h = y2 - y1

        # Just in case
        if mask_w * mask_h <= 0 or mask_w < 0:
            continue

        mask = masks[jdx, :].view(1, 1, 550, 550)
        mask = F.interpolate(
            mask, (mask_h, mask_w), mode=interpolation_mode, align_corners=False)
        mask = mask.gt(0.5).float()
        full_masks[jdx, y1:y2, x1:x2] = mask

    masks = full_masks

    return classes, scores, boxes, masks


# class Detections:

#     def __init__(self):
#         self.bbox_data = []
#         self.mask_data = []

#     def add_bbox(self, image_id:int, category_id:int, bbox:list, score:float):
#         """ Note that bbox should be a list or tuple of (x1, y1, x2, y2) """
#         bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]

#         # Round to the nearest 10th to avoid huge file sizes, as COCO suggests
#         bbox = [round(float(x)*10)/10 for x in bbox]

#         self.bbox_data.append({
#             'image_id': int(image_id),
#             'category_id': get_coco_cat(int(category_id)),
#             'bbox': bbox,
#             'score': float(score)
#         })

#     def add_mask(self, image_id:int, category_id:int, segmentation:np.ndarray, score:float):
#         """ The segmentation should be the full mask, the size of the image and with size [h, w]. """
#         rle = pycocotools.mask.encode(np.asfortranarray(segmentation.astype(np.uint8)))
#         rle['counts'] = rle['counts'].decode('ascii') # json.dump doesn't like bytes strings

#         self.mask_data.append({
#             'image_id': int(image_id),
#             'category_id': get_coco_cat(int(category_id)),
#             'segmentation': rle,
#             'score': float(score)
#         })
    
#     def dump(self):
#         dump_arguments = [
#             (self.bbox_data, args.bbox_det_file),
#             (self.mask_data, args.mask_det_file)
#         ]

#         for data, path in dump_arguments:
#             with open(path, 'w') as f:
#                 json.dump(data, f)
    
#     def dump_web(self):
#         """ Dumps it in the format for my web app. Warning: bad code ahead! """
#         config_outs = ['preserve_aspect_ratio', 'use_prediction_module',
#                         'use_yolo_regressors', 'use_prediction_matching',
#                         'train_masks']

#         output = {
#             'info' : {
#                 'Config': {key: getattr(cfg, key) for key in config_outs},
#             }
#         }

#         image_ids = list(set([x['image_id'] for x in self.bbox_data]))
#         image_ids.sort()
#         image_lookup = {_id: idx for idx, _id in enumerate(image_ids)}

#         output['images'] = [{'image_id': image_id, 'dets': []} for image_id in image_ids]

#         # These should already be sorted by score with the way prep_metrics works.
#         for bbox, mask in zip(self.bbox_data, self.mask_data):
#             image_obj = output['images'][image_lookup[bbox['image_id']]]
#             image_obj['dets'].append({
#                 'score': bbox['score'],
#                 'bbox': bbox['bbox'],
#                 'category': cfg.dataset.class_names[get_transformed_cat(bbox['category_id'])],
#                 'mask': mask['segmentation'],
#             })

#         with open(os.path.join(args.web_det_path, '%s.json' % cfg.name), 'w') as f:
#             json.dump(output, f)