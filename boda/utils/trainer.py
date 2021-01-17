import math
from collections import deque
from typing import Tuple, List, Optional

import numpy as np
from numpy.core.defchararray import decode
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader


class MovingAverage():
    """Keeps an average window of the specified number of items."""

    def __init__(self, max_window_size=1000):
        self.max_window_size = max_window_size
        self.reset()

    def add(self, elem):
        """Adds an element to the window, removing the earliest element if necessary.
        """
        if not math.isfinite(elem):
            print('Warning: Moving average ignored a value of %f' % elem)
            return

        self.window.append(elem)
        self.sum += elem

        if len(self.window) > self.max_window_size:
            self.sum -= self.window.popleft()

    def append(self, elem):
        """Same as add just more pythonic."""
        self.add(elem)

    def reset(self):
        """Resets the MovingAverage to its initial state."""
        self.window = deque()
        self.sum = 0

    def get_avg(self):
        """Returns the average of the elements in the window."""
        return self.sum / max(len(self.window), 1)

    def __str__(self):
        return str(self.get_avg())
    
    def __repr__(self):
        return repr(self.get_avg())
    
    def __len__(self):
        return len(self.window)


class Trainer:
    def __init__(
        self,
        # config,
        train_loader,
        model,
        optimizer,
        criterion,
        valid_loader: Optional[DataLoader] = None,
        scheduler: Optional[str] = None,
        num_epochs = None,
        num_iterations = None,
        device: str = None,
        verbose: int = 1,
    ) -> None:
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.num_iterations = num_iterations
        self.device = device
        self.verbose = verbose
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # if config is not None:
        #     for key, value in config.items():
        #         setattr(self, key, value)

    def _init_trainer(self):
        ...

    def train(self):
        loss_averages = {k: MovingAverage(100) for k in ['B', 'M', 'C', 'S']}
        # loss_averages = {k: MovingAverage(100) for k in ['B', 'C', 'S']}
        self.model.train()
        for epoch in range(self.num_epochs):
            for i, (images, targets) in enumerate(self.train_loader):
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                self.optimizer.zero_grad()

                outputs = self.model(images)
                losses = self.criterion(outputs, targets)
                loss = sum(value for value in losses.values())
                loss.backward()

                self.optimizer.step()

                for k, v in losses.items():
                    loss_averages[k].add(v.item())

                loss = sum([loss_averages[k].get_avg() for k in losses.keys()])

                if (i+1) % self.verbose == 0:
                    print(f'{epoch:>{len(str(self.num_epochs))}}/{self.num_epochs} | T: {loss::>7.4f}', end=' | ')
                    # for k, v in losses.items():
                    #     print(f'{k}: {v.item():>7.4f}', end=' | ')
                    for k, v in loss_averages.items():
                        print(f'{k}: {v.get_avg():>7.4f}', end=' | ')
                    print()

            torch.save(self.model.state_dict(), 'test.pth')

    def train_one_step(self):
        raise NotImplementedError

    def train_one_epoch(self):
        raise NotImplementedError

    def valid(self):
        with torch.no_grad():
            self.model.eval()
            for i, (images, targets, h, w) in self.valid_loader:
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                outputs = self.model(images)

#     def compute_map(self, outputs, targets, h, w):
#         true_boxes = targets['boxes']
#         true_boxes[:, [0, 2]] *= w
#         true_boxes[:, [1, 3]] *= h
#         true_labels = targets['labels']
        
#         iou_thresholds = [x / 100 for x in range(50, 100, 5)]
        
#         pred_labels
#         true_labels
#         iou_types = [
#             ('box',  lambda i,j: bbox_iou_cache[i, j].item(),
#                      lambda i,j: crowd_bbox_iou_cache[i,j].item(),
#                      lambda i: box_scores[i], box_indices),
#             ('mask', lambda i,j: mask_iou_cache[i, j].item(),
#                      lambda i,j: crowd_mask_iou_cache[i,j].item(),
#                      lambda i: mask_scores[i], mask_indices)
#         ]

#         for _class in set(pred_labels + true_labels):
#             ap_per_iou = []
#             num_true_for_class = sum([1 for x in true_labels if x == _class])
            
#             for iouIdx in range(len(iou_thresholds)):
#                 iou_threshold = iou_thresholds[iouIdx]

#                 for iou_type, iou_func, crowd_func, score_func, indices in iou_types:
#                     gt_used = [False] * len(true_labels)
                    
#                     ap_obj = ap_data[iou_type][iouIdx][_class]
#                     ap_obj.add_gt_positives(num_true_for_class)

#                     for i in indices:
#                         if classes[i] != _class:
#                             continue
                        
#                         max_iou_found = iou_threshold
#                         max_match_idx = -1
#                         for j in range(num_gt):
#                             if gt_used[j] or gt_classes[j] != _class:
#                                 continue
                                
#                             iou = iou_func(i, j)

#                             if iou > max_iou_found:
#                                 max_iou_found = iou
#                                 max_match_idx = j
                        
#                         if max_match_idx >= 0:
#                             gt_used[max_match_idx] = True
#                             ap_obj.push(score_func(i), True)
#                         else:
#                             # If the detection matches a crowd, we can just ignore it
#                             matched_crowd = False

#                             if num_crowd > 0:
#                                 for j in range(len(crowd_classes)):
#                                     if crowd_classes[j] != _class:
#                                         continue
                                    
#                                     iou = crowd_func(i, j)

#                                     if iou > iou_threshold:
#                                         matched_crowd = True
#                                         break

#                             # All this crowd code so that we can make sure that our eval code gives the
#                             # same result as COCOEval. There aren't even that many crowd annotations to
#                             # begin with, but accuracy is of the utmost importance.
#                             if not matched_crowd:
#                                 ap_obj.push(score_func(i), False)


# def postprocess(det_output, w, h, batch_idx=0, interpolation_mode='bilinear',
#                 visualize_lincomb=False, crop_masks=True, score_threshold=0):
#     """
#     Postprocesses the output of Yolact on testing mode into a format that makes sense,
#     accounting for all the possible configuration settings.

#     Args:
#         - det_output: The lost of dicts that Detect outputs.
#         - w: The real with of the image.
#         - h: The real height of the image.
#         - batch_idx: If you have multiple images for this batch, the image's index in the batch.
#         - interpolation_mode: Can be 'nearest' | 'area' | 'bilinear' (see torch.nn.functional.interpolate)

#     Returns 4 torch Tensors (in the following order):
#         - classes [num_det]: The class idx for each detection.
#         - scores  [num_det]: The confidence score for each detection.
#         - boxes   [num_det, 4]: The bounding box for each detection in absolute point form.
#         - masks   [num_det, h, w]: Full image masks for each detection.
#     """
    
#     dets = det_output[batch_idx]
#     net = dets['net']
#     dets = dets['detection']

#     if dets is None:
#         return [torch.Tensor()] * 4 # Warning, this is 4 copies of the same thing

#     if score_threshold > 0:
#         keep = dets['score'] > score_threshold

#         for k in dets:
#             if k != 'proto':
#                 dets[k] = dets[k][keep]
        
#         if dets['score'].size(0) == 0:
#             return [torch.Tensor()] * 4
    
#     # Actually extract everything from dets now
#     classes = dets['class']
#     boxes   = dets['box']
#     scores  = dets['score']
#     masks   = dets['mask']

#     if cfg.mask_type == mask_type.lincomb and cfg.eval_mask_branch:
#         # At this points masks is only the coefficients
#         proto_data = dets['proto']
        
#         # Test flag, do not upvote
#         if cfg.mask_proto_debug:
#             np.save('scripts/proto.npy', proto_data.cpu().numpy())
        
#         if visualize_lincomb:
#             display_lincomb(proto_data, masks)

#         masks = proto_data @ masks.t()
#         masks = cfg.mask_proto_mask_activation(masks)

#         # Crop masks before upsampling because you know why
#         if crop_masks:
#             masks = crop(masks, boxes)

#         # Permute into the correct output shape [num_dets, proto_h, proto_w]
#         masks = masks.permute(2, 0, 1).contiguous()

#         if cfg.use_maskiou:
#             with timer.env('maskiou_net'):                
#                 with torch.no_grad():
#                     maskiou_p = net.maskiou_net(masks.unsqueeze(1))
#                     maskiou_p = torch.gather(maskiou_p, dim=1, index=classes.unsqueeze(1)).squeeze(1)
#                     if cfg.rescore_mask:
#                         if cfg.rescore_bbox:
#                             scores = scores * maskiou_p
#                         else:
#                             scores = [scores, scores * maskiou_p]

#         # Scale masks up to the full image
#         masks = F.interpolate(masks.unsqueeze(0), (h, w), mode=interpolation_mode, align_corners=False).squeeze(0)

#         # Binarize the masks
#         masks.gt_(0.5)

    
#     boxes[:, 0], boxes[:, 2] = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, cast=False)
#     boxes[:, 1], boxes[:, 3] = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, cast=False)
#     boxes = boxes.long()

#     if cfg.mask_type == mask_type.direct and cfg.eval_mask_branch:
#         # Upscale masks
#         full_masks = torch.zeros(masks.size(0), h, w)

#         for jdx in range(masks.size(0)):
#             x1, y1, x2, y2 = boxes[jdx, :]

#             mask_w = x2 - x1
#             mask_h = y2 - y1

#             # Just in case
#             if mask_w * mask_h <= 0 or mask_w < 0:
#                 continue
            
#             mask = masks[jdx, :].view(1, 1, cfg.mask_size, cfg.mask_size)
#             mask = F.interpolate(mask, (mask_h, mask_w), mode=interpolation_mode, align_corners=False)
#             mask = mask.gt(0.5).float()
#             full_masks[jdx, y1:y2, x1:x2] = mask
        
#         masks = full_masks

#     return classes, scores, boxes, masks


# class APDataObject:
#     """
#     Stores all the information necessary to calculate the AP for one IoU and one class.
#     Note: I type annotated this because why not.
#     """
#     def __init__(self):
#         self.data_points = []
#         self.num_gt_positives = 0

#     def push(self, score: float, is_true: bool):
#         self.data_points.append((score, is_true))
    
#     def add_gt_positives(self, num_positives: int):
#         """ Call this once per image. """
#         self.num_gt_positives += num_positives

#     def is_empty(self) -> bool:
#         return len(self.data_points) == 0 and self.num_gt_positives == 0

#     def get_ap(self) -> float:
#         """ Warning: result not cached. """

#         if self.num_gt_positives == 0:
#             return 0

#         # Sort descending by score
#         self.data_points.sort(key=lambda x: -x[0])

#         precisions = []
#         recalls = []
#         num_true = 0
#         num_false = 0

#         # Compute the precision-recall curve. The x axis is recalls and the y axis precisions.
#         for datum in self.data_points:
#             # datum[1] is whether the detection a true or false positive
#             if datum[1]:
#                 num_true += 1
#             else:
#                 num_false += 1

#             precision = num_true / (num_true + num_false)
#             recall = num_true / self.num_gt_positives

#             precisions.append(precision)
#             recalls.append(recall)

#         # Smooth the curve by computing [max(precisions[i:]) for i in range(len(precisions))]
#         # Basically, remove any temporary dips from the curve.
#         # At least that's what I think, idk. COCOEval did it so I do too.
#         for i in range(len(precisions)-1, 0, -1):
#             if precisions[i] > precisions[i-1]:
#                 precisions[i-1] = precisions[i]

#         # Compute the integral of precision(recall) d_recall from recall=0->1 using fixed-length riemann summation with 101 bars.
#         y_range = [0] * 101 # idx 0 is recall == 0.0 and idx 100 is recall == 1.00
#         x_range = np.array([x / 100 for x in range(101)])
#         recalls = np.array(recalls)

#         # I realize this is weird, but all it does is find the nearest precision(x) for a given x in x_range.
#         # Basically, if the closest recall we have to 0.01 is 0.009 this sets precision(0.01) = precision(0.009).
#         # I approximate the integral this way, because that's how COCOEval does it.
#         indices = np.searchsorted(recalls, x_range, side='left')
#         for bar_idx, precision_idx in enumerate(indices):
#             if precision_idx < len(precisions):
#                 y_range[bar_idx] = precisions[precision_idx]

#         # Finally compute the riemann sum to get our integral.
#         # avg([precision(x) for x in 0:0.01:1])
#         return sum(y_range) / len(y_range)


# class Evaluation:
#     def __init__(self):
#         pass

#     def evaluate(self):
#         return

#     def postprocess(self):
#         pass




# class Detection:
#     def __init__(self) -> None:
#         self.boxes = []
#         self.masks = []

#     def add_boxes(
#         self,
#     ):
#         bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]

#         # Round to the nearest 10th to avoid huge file sizes, as COCO suggests
#         bbox = [round(float(x)*10)/10 for x in bbox]

#         self.bbox_data.append({
#             'image_id': int(image_id),
#             'category_id': get_coco_cat(int(category_id)),
#             'bbox': bbox,
#             'score': float(score)
#         })

#     def add_masks(self):
#         rle = pycocotools.mask.encode(np.asfortranarray(segmentation.astype(np.uint8)))
#         rle['counts'] = rle['counts'].decode('ascii') # json.dump doesn't like bytes strings

#         self.mask_data.append({
#             'image_id': int(image_id),
#             'category_id': get_coco_cat(int(category_id)),
#             'segmentation': rle,
#             'score': float(score)
#         })


