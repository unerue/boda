import numpy as np
from model.utils.bbox_tools import bbox_iou
from model.utils.bbox_tools import bbox2delta


class ProposalTargetCreator(object):
    """
    This class will be used only in training phase to build head's loss function.
    """
    def __init__(self, n_sample=128,
                 pos_ratio=0.25, 
                 pos_iou_thresh=0.5,
                 neg_iou_thresh_high=0.5,
                 neg_iou_thresh_low=0.0):
            self.n_sample = n_sample
            self.pos_ratio = pos_ratio
            self.pos_iou_thresh = pos_iou_thresh
            self.neg_iou_thresh_high = neg_iou_thresh_high
            self.neg_iou_thresh_low = neg_iou_thresh_low
        
    def make_proposal_target(self, roi, gt_bbox, gt_bbox_label):
        """
        Args:
            roi: (N1, 4)
            gt_bbox: (N2, 4)
            gt_bbox_label: (N2,)
        Note that gt_bbox_label class range from 0 ~ n_class-1, backdround is not included
        
        Returns:
            sample_roi : (Nx, 4)
            target_delta_for_sample_roi : (Nx, 4)
            bbox_bg_label_for_sample_roi : (Nx,)
        Note that bbox_bg_label_for_sample_roi class range from 0 ~ n_class, background(class 0) is included
        """
        #---------- debug
        assert isinstance(roi, np.ndarray)
        assert isinstance(gt_bbox, np.ndarray)
        assert isinstance(gt_bbox_label, np.ndarray)
        assert len(roi.shape) == len(gt_bbox.shape) == 2
        assert len(gt_bbox_label.shape) == 1
        assert roi.shape[1] == gt_bbox.shape[1] == 4
        assert gt_bbox.shape[0] == gt_bbox_label.shape[0]
        #---------- debug

        # concate gt_bbox as part of roi to be chose
        roi = np.concatenate((roi, gt_bbox), axis=0)   

        n_pos = int(self.n_sample * self.pos_ratio)

        iou = bbox_iou(roi, gt_bbox)
        bbox_index_for_roi = iou.argmax(axis=1)
        max_iou_for_roi = iou.max(axis=1)

        # note that bbox_bg_label_for_roi include background, class 0 stand for backdround
        # object class change from 0 ~ n_class-1 to 1 ~ n_class
        bbox_bg_label_for_roi = gt_bbox_label[bbox_index_for_roi] + 1
        
        # Select foreground(positive) RoIs as those with >= pos_iou_thresh IoU.
        pos_index = np.where(max_iou_for_roi >= self.pos_iou_thresh)[0]
        n_pos_real = int(min(n_pos, len(pos_index)))
        if n_pos_real > 0:
            pos_index = np.random.choice(pos_index, size=n_pos_real, replace=False)
        
        # Select background(negative) RoIs as those within [neg_iou_thresh_low, neg_iou_thresh_high).
        neg_index = np.where((max_iou_for_roi >= self.neg_iou_thresh_low) & (max_iou_for_roi < self.neg_iou_thresh_high))[0]
        n_neg = self.n_sample - n_pos_real
        n_neg_real = int(min(n_neg, len(neg_index)))
        if n_neg_real > 0:
            neg_index = np.random.choice(neg_index, size=n_neg_real, replace=False)
        
        keep_index = np.append(pos_index, neg_index)
        sample_roi = roi[keep_index]
        bbox_bg_label_for_sample_roi = bbox_bg_label_for_roi[keep_index]
        bbox_bg_label_for_sample_roi[n_pos_real:] = 0   # set negative sample's label to background 0

        target_delta_for_sample_roi = bbox2delta(sample_roi, gt_bbox[bbox_index_for_roi[keep_index]])

        target_delta_for_sample_roi = (target_delta_for_sample_roi - np.array([0., 0., 0., 0.])) / np.array([0.1, 0.1, 0.2, 0.2])
        return sample_roi, target_delta_for_sample_roi, bbox_bg_label_for_sample_roi

if __name__ == '__main__':
    proposal_target_creator = ProposalTargetCreator()
    roi = np.random.randn(2000,4) + [0,0,3,3]
    gt_bbox = np.random.randn(10, 4) + [0,0,3,3]
    gt_bbox_label = np.array([1,2,3,4,3,9,0,4,5,6])
    sample_roi, target_delta_for_sample_roi, bbox_bg_label_for_sample_roi = proposal_target_creator.make_proposal_target(roi, gt_bbox, gt_bbox_label)
    assert sample_roi.shape == target_delta_for_sample_roi.shape
    assert bbox_bg_label_for_sample_roi.shape[0] == sample_roi.shape[0]
    assert isinstance(sample_roi, np.ndarray) 
    assert isinstance(bbox_bg_label_for_sample_roi, np.ndarray)
    assert isinstance(target_delta_for_sample_roi, np.ndarray)