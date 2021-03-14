from typing import Tuple
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from ...utils.bbox import decode, sanitize_coordinates, crop


class YolactInference:
    def __init__(
        self,
        num_classes: int = 81,
    ) -> None:
        self.config = None
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
        prior_boxes = preds['prior_boxes']

        proto_masks = preds['proto_masks']

        batch_size = pred_boxes.size(0)
        num_prior_boxes = prior_boxes.size(0)

        pred_scores = preds['scores'].view(
            batch_size, num_prior_boxes, self.num_classes).transpose(2, 1).contiguous()

        outputs = []
        for i in range(batch_size):
            decoded_boxes = decode(pred_boxes[i], prior_boxes)
            results = self.detect(i, decoded_boxes, pred_masks, pred_scores)

            results['proto_masks'] = proto_masks[i]
            outputs.append(results)

        return outputs

    def detect(
        self,
        batch_index,
        decoded_boxes,
        pred_masks,
        pred_scores,
    ):
        scores = pred_scores[batch_index, 1:, :]
        max_scores, _ = torch.max(scores, dim=0)

        keep = (max_scores > 0.05)
        scores = scores[:, keep]
        boxes = decoded_boxes[keep, :]
        masks = pred_masks[batch_index, keep, :]

        # print(boxes.size(), scores.size())
        if scores.size(1) == 0:
            return None

        boxes, masks, scores, labels = self.nms(boxes, scores, masks)
        # print(boxes.size(), masks.size(), scores.size(), labels.size())

        # import torchvision.ops as ops

        # keeps = []
        # for i in range(scores.size(0)):
        #     # print(scores[i, :])
        #     keep = ops.nms(boxes, scores, 0.5)
        #     keeps.append(keep)
        
        # print('after_nms', keeps)
        # keep = ops.nms(boxes, scores, 0.5)


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

        from ...utils.cython_nms import nms as cnms

        num_classes = pred_scores.size(0)

        indexes = []
        labels = []
        scores = []

        max_size = 550
        pred_boxes = pred_boxes * max_size

        for i in range(num_classes):
            score = pred_scores[i, :]
            score_mask = score > 0.05
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
        labels = labels[sorted_index]

        pred_boxes = pred_boxes[indexes] / max_size
        pred_masks = pred_masks[indexes]

        return pred_boxes, pred_masks, scores, labels


def postprocess(preds):
    # pred_boxes = preds['boxes']
    w, h = (550, 550)
    boxes = preds['boxes']
    pred_masks = preds['masks']
    pred_scores = preds['scores']
    # prior_boxes = preds[0]['prior_boxes']
    proto_masks = preds['proto_masks']

    masks = proto_masks @ pred_masks.t()
    masks = torch.sigmoid(masks)

    masks = crop(masks, boxes)

    masks = F.interpolate(masks.unsqueeze(0), (h, w), mode='bilinear', align_corners=False).squeeze(0)

    # Binarize the masks
    masks.gt_(0.5)

    boxes[:, 0], boxes[:, 2] = \
        sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, cast=False)
    boxes[:, 1], boxes[:, 3] = \
        sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, cast=False)
    boxes = boxes.long()

    preds['boxes'] = boxes
    preds['proto_masks'] = proto_masks

    return preds

#     def mask_added_color(self):
#         masks = masks[:num_dets_to_consider, :, :, None]
    
#         # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
#         colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
#         masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

#         # This is 1 everywhere except for 1-mask_alpha where the mask is
#         inv_alph_masks = masks * (-mask_alpha) + 1
    
#         # I did the math for this on pen and paper. This whole block should be equivalent to:
#         #    for j in range(num_dets_to_consider):
#         #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
#         masks_color_summand = masks_color[0]
#         if num_dets_to_consider > 1:
#             inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
#             masks_color_cumul = masks_color[1:] * inv_alph_cumul
#             masks_color_summand += masks_color_cumul.sum(dim=0)

#         img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand


#     def bounding_boxes(self):
#         x1, y1, x2, y2 = boxes[j, :]
#         color = get_color(j)
#         score = scores[j]

#         if args.display_bboxes:
#             cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

#         if args.display_text:
#             _class = cfg.dataset.class_names[classes[j]]
#             text_str = '%s: %.2f' % (_class, score) if args.display_scores else _class

#             font_face = cv2.FONT_HERSHEY_DUPLEX
#             font_scale = 0.6
#             font_thickness = 1

#             text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

#             text_pt = (x1, y1 - 3)
#             text_color = [255, 255, 255]

#             cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
#             cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)



# def visualize_lincomb(proto_data, masks):
#     out_masks = torch.matmul(proto_data, masks.t())

#     for kdx in range(1):
#         jdx = kdx + 0
    
#         coeffs = masks[jdx, :].cpu().numpy()
#         idx = np.argsort(-np.abs(coeffs))
#         # plt.bar(list(range(idx.shape[0])), coeffs[idx])
#         # plt.show()

#         coeffs_sort = coeffs[idx]
#         arr_h, arr_w = (4, 8)
#         proto_h, proto_w, _ = proto_data.size()
#         arr_img = np.zeros([proto_h*arr_h, proto_w*arr_w])
#         arr_run = np.zeros([proto_h*arr_h, proto_w*arr_w])
#         # test = torch.sum(proto_data, -1).cpu().numpy()

#         for y in range(arr_h):
#             for x in range(arr_w):
#                 i = arr_w * y + x

#                 if i == 0:
#                     running_total = proto_data[:, :, idx[i]].cpu().numpy() * coeffs_sort[i]
#                 else:
#                     running_total += proto_data[:, :, idx[i]].cpu().numpy() * coeffs_sort[i]

#                 running_total_nonlin = running_total
#                 if cfg.mask_proto_mask_activation == torch.sigmoid:
#                     running_total_nonlin = (1/(1+np.exp(-running_total_nonlin)))

#                 arr_img[y*proto_h:(y+1)*proto_h, x*proto_w:(x+1)*proto_w] = \
#                     (proto_data[:, :, idx[i]] / torch.max(proto_data[:, :, idx[i]])).cpu().numpy() * coeffs_sort[i]
#                 arr_run[y*proto_h:(y+1)*proto_h, x*proto_w:(x+1)*proto_w] = \
#                     (running_total_nonlin > 0.5).astype(np.float)
#         plt.imshow(arr_img)
#         plt.show()

#         plt.imshow(out_masks[:, :, jdx].cpu().numpy())
#         plt.show()