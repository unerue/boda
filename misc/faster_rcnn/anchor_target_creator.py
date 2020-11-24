import numpy as np
from model.utils.bbox_tools import bbox_iou
from model.utils.bbox_tools import bbox2delta

class AnchorTargetCreator(object):
    """
    This class will be used only in training phase to build rpn's loss function.
    Args:
        n_sample (int): The number of anchers to sample.
        pos_iou_thresh (float): Anchors with IoU above this
            threshold will be assigned as positive.
        neg_iou_thresh (float): Anchors with IoU below this
            threshold will be assigned as negative.
        pos_ratio (float): Ratio of positive regions in the
            sampled regions.
    """
    def __init__(self, n_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3, pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio


    def make_anchor_target(self, anchor, gt_bbox, image_size):
        """
        Assign ground truth supervision to sampled subset of anchors.
        Args:
            anchor:  (N1, 4)
            gt_bbox: (N2, 4)
        Return:
            target_dalta: (N1, 4), for bbox regression
            anchor_label: (N1, ), for classification
        """
        #---------- debug
        assert isinstance(anchor, np.ndarray)
        assert isinstance(gt_bbox, np.ndarray)
        assert len(anchor.shape) == len(gt_bbox.shape) == 2
        assert anchor.shape[1] == gt_bbox.shape[1] == 4
        #----------
        img_H, img_W = image_size
        n_anchor = len(anchor)

        index_inside_image = np.where(
            (anchor[:, 0] >= 0) &
            (anchor[:, 1] >= 0) &
            (anchor[:, 2] <= img_H) &
            (anchor[:, 3] <= img_W))[0]

        anchor = anchor[index_inside_image] # rule out anchors that are not fully included inside the image

        bbox_index_for_anchor, anchor_label = self._assign_targer_and_label_for_anchor(anchor, gt_bbox)

        # create targer delta for bbox regression
        target_delta = bbox2delta(anchor, gt_bbox[bbox_index_for_anchor])

        # expand the target_dalta and label to match original length of anchor
        target_delta = self._to_orignal_length(target_delta, n_anchor, index_inside_image, fill=0)
        anchor_label = self._to_orignal_length(anchor_label, n_anchor, index_inside_image, fill=-1)
        
        return target_delta, anchor_label


    def _assign_targer_and_label_for_anchor(self, anchor, gt_bbox):
        """ 
        assign a label for each anchor, and the targer bbox index(with max iou) for each anchor.
        label: 1 is positive, 0 is negative, -1 is don't care
        """
        #---------- debug
        assert len(anchor.shape) == len(gt_bbox.shape) == 2
        assert anchor.shape[1] == gt_bbox.shape[1] == 4
        #---------- debug
        
        label = np.zeros(anchor.shape[0], dtype=np.int32) - 1   # init label with -1
        
        bbox_index_for_anchor, max_iou_for_anchor, anchor_index_for_bbox = self._anchor_bbox_ious(anchor, gt_bbox)

        # 1. assign anchor with 0 whose max_iou is small than neg_iou_thresh
        label[max_iou_for_anchor < self.neg_iou_thresh] = 0

        # 2. for each gt_bbox, assign anchor with 1 who has max iou with the gt_bbox
        label[anchor_index_for_bbox] = 1

        # 3. assign anchor with 0 whose max_iou is large than pos_iou_thresh
        label[max_iou_for_anchor>self.pos_iou_thresh] = 1

        # subsample positive labels if we have too many
        n_pos = int(self.n_sample * self.pos_ratio)  # default: 128
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1
        
        # subsample negative labels if we have too many
        n_neg = int(self.n_sample - np.sum(label==1))
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        #---------- debug
        assert len(bbox_index_for_anchor.shape) == len(label.shape) == 1
        assert bbox_index_for_anchor.shape[0] == label.shape[0] == anchor.shape[0]
        # print(np.sum(label == 1)) # change anchor generate parameter, if neg samples and pos samples are not roughly equal to n_sample/2
        # print(np.sum(label == 0))
        assert np.sum(label == 0) + np.sum(label == 1) <= self.n_sample
        #---------- debug
        
        return bbox_index_for_anchor, label

    def _anchor_bbox_ious(self, anchor, gt_bbox):
        iou = bbox_iou(anchor, gt_bbox)
        
        bbox_index_for_anchor = iou.argmax(axis=1)   # (anchor.shape[0],)
        max_iou_for_anchor = iou.max(axis=1)

        anchor_index_for_bbox = iou.argmax(axis=0)   # (bbox.shape[0],)
        max_iou_for_bbox = iou.max(axis=0)

        return bbox_index_for_anchor, max_iou_for_anchor, anchor_index_for_bbox

    def _to_orignal_length(self, data, length, index, fill):
        shape = list(data.shape)
        shape[0] = length
        ret = np.empty(shape, dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
        return ret


if __name__ == '__main__':
    anchor_target_creator = AnchorTargetCreator()
    anchor = (np.random.randn(22500, 4) + [0,0,3,3]) * 100
    gt_bbox = (np.random.randn(10, 4) + [0,0,3,3]) * 100
    target_delta, anchor_label = anchor_target_creator.make_anchor_target(anchor, gt_bbox, image_size=(800,800))
    assert target_delta.shape == anchor.shape
    assert anchor_label.shape[0] == anchor.shape[0]