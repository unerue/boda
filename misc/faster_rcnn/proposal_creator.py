import numpy as np
from model.utils.bbox_tools import delta2bbox
from model.utils.nms_cpu import py_cpu_nms as nms
from model.utils.anchor_target_creator import AnchorTargetCreator



class ProposalCreator:
    """
    make proposal rois, this will be used both in training phase and in test phase.
    """
    def __init__(self, nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=2000,
                 n_test_pre_nms=6000,
                 n_test_post_nms=300,
                 min_roi_size=16):
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = 12000
        self.n_train_post_nms = 2000
        self.n_test_pre_nms = 6000
        self.n_test_post_nms = 300
        self.min_roi_size = 16

    def make_proposal(self, anchor, delta, score, image_size, is_training):
        """
        image_size used for clip anchor inside image field.
        anchor: (N, 4)
        delta:  (N, 4)
        score:   (N,)
        """
        #---------- debug
        assert isinstance(anchor, np.ndarray) and isinstance(delta, np.ndarray) and isinstance(score, np.ndarray)
        assert len(anchor.shape) == 2 and len(delta.shape) == 2 and len(score.shape) == 1
        assert anchor.shape[0] == delta.shape[0] == score.shape[0]
        #----------
        if is_training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms
        
        # 1. clip the roi into the size of image
        roi = delta2bbox(anchor, delta)
        roi[:,slice(0,4,2)] = np.clip(roi[:,slice(0,4,2)], a_min=0, a_max=image_size[0])
        roi[:,slice(1,4,2)] = np.clip(roi[:,slice(1,4,2)], a_min=0, a_max=image_size[1])
        
        # 2. remove roi where H or W is less than min_roi_size
        hs = roi[:, 2] - roi[:, 0]
        ws = roi[:, 3] - roi[:, 1]
        keep = np.where((hs >= self.min_roi_size) & (ws >= self.min_roi_size))[0]
        roi = roi[keep, :]
        score = score[keep]

        # 3. keep top n_pre_nms rois according to score, and the left roi are sorted according to score
        order = score.argsort()[::-1]
        order = order[:n_pre_nms]
        roi = roi[order,:]
        
        # 4. apply nms, ans keep top n_post_nms roi
        # note that roi is already sorted according to its score value
        keep = nms(roi, self.nms_thresh)
        keep = keep[:n_post_nms]
        roi = roi[keep,:]

        return roi


if __name__ == '__main__':
    proposal_creater = ProposalCreator()
    from model.utils.generate_anchor import generate_anchor
    anchor = generate_anchor(50,50,(600,800))   # (22500, 4)
    delta = np.random.randn(22500,4)
    score = np.random.randn(22500)
    roi = proposal_creater.make_proposal(anchor, delta, score, (600,800), True)
    assert roi.shape == (2000, 4)