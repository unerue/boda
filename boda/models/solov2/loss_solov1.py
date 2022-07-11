import torch
import torch.nn.functional as F
from torch import nn, Tensor

from ...base_architecture import LossFunction
from ...ops.box import jaccard, cxywh_to_xyxy
from ...ops.loss import log_sum_exp


class Matcher:
    def __init__(self, config, threshold, variances):
        self.config = config

    def __call__(self, pred_boxes, pred_priors, true_boxes, idx) -> Tensor:
        raise NotImplementedError

    def encode(self, matched, priors, variances):
        raise NotImplementedError

    def decode(self, pred_boxes, pred_priors, variances):
        raise NotImplementedError


class Solov1Loss(LossFunction):
    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.config = config

    def forward(self, inputs, targets):
        """
        ins_preds,
        cate_preds,
        gt_bbox_list,
        gt_label_list,
        gt_mask_list,
        img_metas,
        cfg,
        gt_bboxes_ignore=None):
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in inputs]

        ins_label_list, cate_label_list, ins_ind_label_list = multi_apply(
            self.solo_target_single,
            gt_bbox_list,
            gt_label_list,
            gt_mask_list,
            featmap_sizes=featmap_sizes,
        )

        # ins
        ins_labels = [
            torch.cat(
                [
                    ins_labels_level_img[ins_ind_labels_level_img, ...]
                    for ins_labels_level_img, ins_ind_labels_level_img in zip(
                        ins_labels_level, ins_ind_labels_level
                    )
                ],
                0,
            )
            for ins_labels_level, ins_ind_labels_level in zip(
                zip(*ins_label_list), zip(*ins_ind_label_list)
            )
        ]

        ins_preds = [
            torch.cat(
                [
                    ins_preds_level_img[ins_ind_labels_level_img, ...]
                    for ins_preds_level_img, ins_ind_labels_level_img in zip(
                        ins_preds_level, ins_ind_labels_level
                    )
                ],
                0,
            )
            for ins_preds_level, ins_ind_labels_level in zip(
                ins_preds, zip(*ins_ind_label_list)
            )
        ]

        ins_ind_labels = [
            torch.cat(
                [
                    ins_ind_labels_level_img.flatten()
                    for ins_ind_labels_level_img in ins_ind_labels_level
                ]
            )
            for ins_ind_labels_level in zip(*ins_ind_label_list)
        ]
        flatten_ins_ind_labels = torch.cat(ins_ind_labels)

        num_ins = flatten_ins_ind_labels.sum()

        # dice loss
        loss_ins = []
        for input, target in zip(ins_preds, ins_labels):
            if input.size()[0] == 0:
                continue
            input = torch.sigmoid(input)
            loss_ins.append(dice_loss(input, target))
        loss_ins = torch.cat(loss_ins).mean()
        loss_ins = loss_ins * self.ins_loss_weight

        # cate
        cate_labels = [
            torch.cat(
                [
                    cate_labels_level_img.flatten()
                    for cate_labels_level_img in cate_labels_level
                ]
            )
            for cate_labels_level in zip(*cate_label_list)
        ]
        flatten_cate_labels = torch.cat(cate_labels)

        cate_preds = [
            cate_pred.permute(0, 2, 3, 1).reshape(-1, self.cate_out_channels)
            for cate_pred in cate_preds
        ]
        flatten_cate_preds = torch.cat(cate_preds)

        loss_cate = self.loss_cate(
            flatten_cate_preds, flatten_cate_labels, avg_factor=num_ins + 1
        )
        return dict(loss_ins=loss_ins, loss_cate=loss_cate)

    def solo_target_single(
        self, gt_bboxes_raw, gt_labels_raw, gt_masks_raw, featmap_sizes=None
    ):

        device = gt_labels_raw[0].device

        # ins
        gt_areas = torch.sqrt(
            (gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0])
            * (gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1])
        )

        ins_label_list = []
        cate_label_list = []
        ins_ind_label_list = []
        for (lower_bound, upper_bound), stride, featmap_size, num_grid in zip(
            self.scale_ranges, self.strides, featmap_sizes, self.seg_num_grids
        ):

            ins_label = torch.zeros(
                [num_grid ** 2, featmap_size[0], featmap_size[1]],
                dtype=torch.uint8,
                device=device,
            )
            cate_label = torch.zeros(
                [num_grid, num_grid], dtype=torch.int64, device=device
            )
            ins_ind_label = torch.zeros(
                [num_grid ** 2], dtype=torch.bool, device=device
            )

            hit_indices = (
                ((gt_areas >= lower_bound) & (gt_areas <= upper_bound))
                .nonzero()
                .flatten()
            )
            if len(hit_indices) == 0:
                ins_label_list.append(ins_label)
                cate_label_list.append(cate_label)
                ins_ind_label_list.append(ins_ind_label)
                continue
            gt_bboxes = gt_bboxes_raw[hit_indices]
            gt_labels = gt_labels_raw[hit_indices]
            gt_masks = gt_masks_raw[hit_indices.cpu().numpy(), ...]

            half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma
            half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma

            # mass center
            gt_masks_pt = torch.from_numpy(gt_masks).to(device=device)
            center_ws, center_hs = center_of_mass(gt_masks_pt)
            valid_mask_flags = gt_masks_pt.sum(dim=-1).sum(dim=-1) > 0

            output_stride = stride / 2
            for (
                seg_mask,
                gt_label,
                half_h,
                half_w,
                center_h,
                center_w,
                valid_mask_flag,
            ) in zip(
                gt_masks,
                gt_labels,
                half_hs,
                half_ws,
                center_hs,
                center_ws,
                valid_mask_flags,
            ):
                if not valid_mask_flag:
                    continue
                upsampled_size = (featmap_sizes[0][0] * 4, featmap_sizes[0][1] * 4)
                coord_w = int((center_w / upsampled_size[1]) // (1.0 / num_grid))
                coord_h = int((center_h / upsampled_size[0]) // (1.0 / num_grid))

                # left, top, right, down
                top_box = max(
                    0,
                    int(((center_h - half_h) / upsampled_size[0]) // (1.0 / num_grid)),
                )
                down_box = min(
                    num_grid - 1,
                    int(((center_h + half_h) / upsampled_size[0]) // (1.0 / num_grid)),
                )
                left_box = max(
                    0,
                    int(((center_w - half_w) / upsampled_size[1]) // (1.0 / num_grid)),
                )
                right_box = min(
                    num_grid - 1,
                    int(((center_w + half_w) / upsampled_size[1]) // (1.0 / num_grid)),
                )

                top = max(top_box, coord_h - 1)
                down = min(down_box, coord_h + 1)
                left = max(coord_w - 1, left_box)
                right = min(right_box, coord_w + 1)

                cate_label[top : (down + 1), left : (right + 1)] = gt_label
                # ins
                seg_mask = mmcv.imrescale(seg_mask, scale=1.0 / output_stride)
                seg_mask = torch.from_numpy(seg_mask).to(device=device)
                for i in range(top, down + 1):
                    for j in range(left, right + 1):
                        label = int(i * num_grid + j)
                        ins_label[
                            label, : seg_mask.shape[0], : seg_mask.shape[1]
                        ] = seg_mask
                        ins_ind_label[label] = True
            ins_label_list.append(ins_label)
            cate_label_list.append(cate_label)
            ins_ind_label_list.append(ins_ind_label)
        return ins_label_list, cate_label_list, ins_ind_label_list

    def get_seg(self, seg_preds, cate_preds, img_metas, cfg, rescale=None):
        assert len(seg_preds) == len(cate_preds)
        num_levels = len(cate_preds)
        featmap_size = seg_preds[0].size()[-2:]

        result_list = []
        for img_id in range(len(img_metas)):
            cate_pred_list = [
                cate_preds[i][img_id].view(-1, self.cate_out_channels).detach()
                for i in range(num_levels)
            ]
            seg_pred_list = [seg_preds[i][img_id].detach() for i in range(num_levels)]
            img_shape = img_metas[img_id]["img_shape"]
            scale_factor = img_metas[img_id]["scale_factor"]
            ori_shape = img_metas[img_id]["ori_shape"]

            cate_pred_list = torch.cat(cate_pred_list, dim=0)
            seg_pred_list = torch.cat(seg_pred_list, dim=0)

            result = self.get_seg_single(
                cate_pred_list,
                seg_pred_list,
                featmap_size,
                img_shape,
                ori_shape,
                scale_factor,
                cfg,
                rescale,
            )
            result_list.append(result)
        return result_list

    def get_seg_single(
        self,
        cate_preds,
        seg_preds,
        featmap_size,
        img_shape,
        ori_shape,
        scale_factor,
        cfg,
        rescale=False,
        debug=False,
    ):
        assert len(cate_preds) == len(seg_preds)

        # overall info.
        h, w, _ = img_shape
        upsampled_size_out = (featmap_size[0] * 4, featmap_size[1] * 4)

        # process.
        inds = cate_preds > cfg.score_thr
        # category scores.
        cate_scores = cate_preds[inds]
        if len(cate_scores) == 0:
            return None
        # category labels.
        inds = inds.nonzero()
        cate_labels = inds[:, 1]

        # strides.
        size_trans = cate_labels.new_tensor(self.seg_num_grids).pow(2).cumsum(0)
        strides = cate_scores.new_ones(size_trans[-1])
        n_stage = len(self.seg_num_grids)
        strides[: size_trans[0]] *= self.strides[0]
        for ind_ in range(1, n_stage):
            strides[size_trans[ind_ - 1] : size_trans[ind_]] *= self.strides[ind_]
        strides = strides[inds[:, 0]]

        # masks.
        seg_preds = seg_preds[inds[:, 0]]
        seg_masks = seg_preds > cfg.mask_thr
        sum_masks = seg_masks.sum((1, 2)).float()

        # filter.
        keep = sum_masks > strides
        if keep.sum() == 0:
            return None

        seg_masks = seg_masks[keep, ...]
        seg_preds = seg_preds[keep, ...]
        sum_masks = sum_masks[keep]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # maskness.
        seg_scores = (seg_preds * seg_masks.float()).sum((1, 2)) / sum_masks
        cate_scores *= seg_scores

        # sort and keep top nms_pre
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > cfg.nms_pre:
            sort_inds = sort_inds[: cfg.nms_pre]
        seg_masks = seg_masks[sort_inds, :, :]
        seg_preds = seg_preds[sort_inds, :, :]
        sum_masks = sum_masks[sort_inds]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        # Matrix NMS
        cate_scores = matrix_nms(
            seg_masks,
            cate_labels,
            cate_scores,
            kernel=cfg.kernel,
            sigma=cfg.sigma,
            sum_masks=sum_masks,
        )

        # filter.
        keep = cate_scores >= cfg.update_thr
        if keep.sum() == 0:
            return None
        seg_preds = seg_preds[keep, :, :]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # sort and keep top_k
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > cfg.max_per_img:
            sort_inds = sort_inds[: cfg.max_per_img]
        seg_preds = seg_preds[sort_inds, :, :]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        seg_preds = F.interpolate(
            seg_preds.unsqueeze(0), size=upsampled_size_out, mode="bilinear"
        )[:, :, :h, :w]
        seg_masks = F.interpolate(
            seg_preds, size=ori_shape[:2], mode="bilinear"
        ).squeeze(0)
        seg_masks = seg_masks > cfg.mask_thr
        return seg_masks, cate_labels, cate_scores
