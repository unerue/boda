import torch
import torch.nn.functional as F


def matrix_nms(
    seg_masks, cate_labels, cate_scores, kernel="gaussian", sigma=2.0, sum_masks=None
):
    """Matrix NMS for multi-class masks.

    Args:
        seg_masks (Tensor): shape (n, h, w)
        cate_labels (Tensor): shape (n), mask labels in descending order
        cate_scores (Tensor): shape (n), mask scores in descending order
        kernel (str):  'linear' or 'gauss'
        sigma (float): std in gaussian method
        sum_masks (Tensor): The sum of seg_masks

    Returns:
        Tensor: cate_scores_update, tensors of shape (n)
    """
    n_samples = len(cate_labels)
    if n_samples == 0:
        return []
    if sum_masks is None:
        sum_masks = seg_masks.sum((1, 2)).float()
    seg_masks = seg_masks.reshape(n_samples, -1).float()
    # inter.
    inter_matrix = torch.mm(seg_masks, seg_masks.transpose(1, 0))
    # union.
    sum_masks_x = sum_masks.expand(n_samples, n_samples)
    # iou.
    iou_matrix = (
        inter_matrix / (sum_masks_x + sum_masks_x.transpose(1, 0) - inter_matrix)
    ).triu(diagonal=1)
    # label_specific matrix.
    cate_labels_x = cate_labels.expand(n_samples, n_samples)
    label_matrix = (
        (cate_labels_x == cate_labels_x.transpose(1, 0)).float().triu(diagonal=1)
    )

    # IoU compensation
    compensate_iou, _ = (iou_matrix * label_matrix).max(0)
    compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)

    # IoU decay
    decay_iou = iou_matrix * label_matrix

    # matrix nms
    if kernel == "gaussian":
        decay_matrix = torch.exp(-1 * sigma * (decay_iou ** 2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou ** 2))
        decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
    elif kernel == "linear":
        decay_matrix = (1 - decay_iou) / (1 - compensate_iou)
        decay_coefficient, _ = decay_matrix.min(0)
    else:
        raise NotImplementedError

    # update the score.
    cate_scores_update = cate_scores * decay_coefficient
    return cate_scores_update


def get_seg(seg_preds, cate_preds, img_metas=[1]):
    assert len(seg_preds) == len(cate_preds)

    num_levels = len(cate_preds)
    featmap_size = seg_preds[0].size()[-2:]

    result_list = []
    for img_id in range(len(img_metas)):
        cate_pred_list = [
            cate_preds[i][img_id].view(-1, 80).detach()
            for i in range(num_levels)
            # cate_preds[i][img_id].view(-1, self.cate_out_channels).detach() for i in range(num_levels)
        ]
        seg_pred_list = [seg_preds[i][img_id].detach() for i in range(num_levels)]

        # img_shape = img_metas[img_id]['img_shape']
        # scale_factor = img_metas[img_id]['scale_factor']
        # ori_shape = img_metas[img_id]['ori_shape']
        size = (1333, 800, 3)
        # size = (800, 1333, 3)
        img_shape = size
        ori_shape = size

        cate_pred_list = torch.cat(cate_pred_list, dim=0)
        seg_pred_list = torch.cat(seg_pred_list, dim=0)

        result = get_seg_single(
            cate_pred_list, seg_pred_list, featmap_size, img_shape, ori_shape
        )

        result_list.append(result)

    return result_list


def get_seg_single(cate_preds, seg_preds, featmap_size, img_shape, ori_shape):
    assert len(cate_preds) == len(seg_preds)

    # test_seg_masks = seg_preds > 0.5 # cfg.mask_thr
    # test_masks = test_seg_masks.detach().cpu().numpy()[0] * 255
    # print(test_masks.shape)
    # import cv2
    # cv2.imwrite('solo-test12.jpg', test_masks)

    # overall info.
    h, w, _ = img_shape
    upsampled_size_out = (featmap_size[0] * 4, featmap_size[1] * 4)

    # process.
    inds = cate_preds > 0.1  # cfg.score_thr
    # category scores.
    cate_scores = cate_preds[inds]
    if len(cate_scores) == 0:
        return None
    # category labels.
    # inds = inds.nonzero()
    inds = inds.nonzero()
    # print(inds.nonzero())
    cate_labels = inds[:, 1]

    # strides.
    # size_trans = cate_labels.new_tensor(self.seg_num_grids).pow(2).cumsum(0)
    size_trans = cate_labels.new_tensor([40, 36, 24, 16, 12]).pow(2).cumsum(0)
    strides = cate_scores.new_ones(size_trans[-1])
    n_stage = len([40, 36, 24, 16, 12])  # len(self.seg_num_grids)
    strides[: size_trans[0]] *= (4, 8, 16, 32, 64)[0]  # self.strides[0]
    for ind_ in range(1, n_stage):
        # strides[size_trans[ind_ - 1]:size_trans[ind_]] *= self.strides[ind_]
        strides[size_trans[ind_ - 1] : size_trans[ind_]] *= (4, 8, 16, 32, 64)[ind_]
    strides = strides[inds[:, 0]]

    # masks.
    seg_preds = seg_preds[inds[:, 0]]
    seg_masks = seg_preds > 0.5  # cfg.mask_thr
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

    # print('#'*50)
    # print(seg_masks.size())
    test_seg_masks = seg_masks > 0.5  # cfg.mask_thr
    test_masks = test_seg_masks.detach().cpu().numpy()[0] * 255
    print(test_masks.shape)
    # test_masks = test_masks.transpose(1, 2, 0)
    import cv2

    cv2.imwrite("solo-test11.jpg", test_masks)

    # maskness.
    seg_scores = (seg_preds * seg_masks.float()).sum((1, 2)) / sum_masks
    cate_scores *= seg_scores

    # sort and keep top nms_pre
    sort_inds = torch.argsort(cate_scores, descending=True)
    if len(sort_inds) > 500:  # cfg.nms_pre
        sort_inds = sort_inds[:500]  # [:cfg.nms_pre]
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
        kernel="gaussian",
        sigma=2.0,
        sum_masks=sum_masks,
    )

    # filter.
    keep = cate_scores >= 0.05  # cfg.update_thr
    if keep.sum() == 0:
        return None
    seg_preds = seg_preds[keep, :, :]
    cate_scores = cate_scores[keep]
    cate_labels = cate_labels[keep]

    # sort and keep top_k
    sort_inds = torch.argsort(cate_scores, descending=True)
    if len(sort_inds) > 100:  # cfg.max_per_img:
        sort_inds = sort_inds[:100]  # [:cfg.max_per_img]
    seg_preds = seg_preds[sort_inds, :, :]
    cate_scores = cate_scores[sort_inds]
    cate_labels = cate_labels[sort_inds]

    print(seg_preds.size())
    print(upsampled_size_out)
    seg_preds = F.interpolate(
        seg_preds.unsqueeze(0), size=upsampled_size_out, mode="bilinear"
    )  # [:, :, :h, :w]

    # seg_masks = F.interpolate(
    #     seg_preds, size=ori_shape[:2], mode='bilinear').squeeze(0)
    size = (1333, 800)
    # size = (800, 1333)
    seg_masks = F.interpolate(seg_preds, size=size, mode="bilinear").squeeze(0)

    print("#" * 50)
    print(seg_masks.size())
    seg_masks = seg_masks > 0.5  # cfg.mask_thr

    test_masks = seg_masks.detach().cpu().numpy()[0] * 255
    print(test_masks.shape)
    # test_masks = test_masks.transpose(1, 2, 0)
    print(test_masks.shape)
    import cv2

    # test_masks = cv2.flip(test_masks, 1)
    # test_masks = cv2.rotate(test_masks, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # print(test_masks.shape)
    # test_masks = cv2.resize(test_masks, (1333, 800), cv2.INTER_AREA)
    # print(test_masks.shape)
    cv2.imwrite("solo-test1.jpg", test_masks)

    return seg_masks, cate_labels, cate_scores
