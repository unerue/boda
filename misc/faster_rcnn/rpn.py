import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np

from model.utils.generate_anchor import generate_anchor
from model.utils.proposal_creator import ProposalCreator
from model.utils.anchor_target_creator import AnchorTargetCreator
from model.utils.loss import delta_loss



class RegionProposalNetwork(nn.Module):
    """Region Proposal Network (RPN)
    """
    def __init__(self, in_channel, mid_channel, ratio=[0.5, 1, 2], anchor_size = [128, 256, 512]):
        super().__init__()

        self.ratio = ratio
        self.anchor_size = anchor_size
        self.K = len(ratio)*len(anchor_size)    # default: 9 : 9 ahcnors per spatial channel in feature maps

        self.mid_layer = nn.Conv2d(in_channel, mid_channel, kernel_size=3, stride=1, padding=1) 
        self.score_layer = nn.Conv2d(mid_channel, 2*self.K, kernel_size=1, stride=1, padding=0)
        self.delta_layer = nn.Conv2d(mid_channel, 4*self.K, kernel_size=1, stride=1, padding=0)
        
        self._normal_init(self.mid_layer, 0, 0.01)
        self._normal_init(self.score_layer, 0, 0.01)
        self._normal_init(self.delta_layer, 0, 0.01)

        self.proposal_creator = ProposalCreator()
        self.anchor_target_creator = AnchorTargetCreator()

    def forward(self, features, image_size):
        """
        Batch size are fixed to one.
        features: (N-1, C, H, W)
        """
        #---------- debug
        assert isinstance(features, Variable)
        assert features.shape[0] == 1
        #---------- debug

        _, _, feature_height, feature_width = features.shape
        image_height, image_width = image_size[0], image_size[1]

        mid_features = F.relu(self.mid_layer(features))
        
        delta = self.delta_layer(mid_features)
        delta = delta.permute(0,2,3,1).contiguous().view([feature_height*feature_width*self.K, 4])
        
        score = self.score_layer(mid_features)
        score = score.permute(0,2,3,1).contiguous().view([feature_height*feature_width*self.K, 2])

        # ndarray: (feature_height*feature_width*K, 4)
        anchor = generate_anchor(feature_height, feature_width, image_size, self.ratio, self.anchor_size)
        #---------- debug
        assert isinstance(delta, Variable) and isinstance(score, Variable) and isinstance(anchor, np.ndarray)
        assert delta.shape == (feature_height*feature_width*self.K, 4)
        assert score.shape == (feature_height*feature_width*self.K, 2)
        #---------- debug
        return delta, score, anchor

    def loss(self, delta, score, anchor, gt_bbox, image_size):
        #---------- debug
        assert isinstance(delta, Variable)
        assert isinstance(score, Variable)
        assert isinstance(anchor, np.ndarray)
        assert isinstance(gt_bbox, np.ndarray)
        #---------- debug
        target_delta, anchor_label = self.anchor_target_creator.make_anchor_target(anchor, gt_bbox, image_size)
        target_delta = torch.FloatTensor(target_delta)
        anchor_label = torch.LongTensor(anchor_label)
        if torch.cuda.is_available():
            target_delta, anchor_label = target_delta.cuda(), anchor_label.cuda()

        rpn_delta_loss = delta_loss(delta, target_delta, anchor_label, 3)
        
        rpn_class_loss = F.cross_entropy(score, anchor_label, ignore_index=-1)   # ignore loss for label value -1

        return rpn_delta_loss + rpn_class_loss

    def predict(self, delta, score, anchor, image_size):
        #---------- debug
        assert isinstance(delta, Variable)
        assert isinstance(score, Variable)
        assert isinstance(anchor, np.ndarray)
        #---------- debug
        delta = delta.data.cpu().numpy()
        score = score.data.cpu().numpy()
        score_fg = score[:,1]
        roi = self.proposal_creator.make_proposal(anchor, delta, score_fg, image_size, is_training=self.training)
        
        #---------- debug
        assert isinstance(roi, np.ndarray)
        #---------- debug
        return roi


    def _normal_init(self, m, mean, stddev, truncated=False):
        """
        weight initalizer: truncated normal and random normal.
        """
        if truncated:
            m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
            m.weight.data.normal_(mean, stddev)
            m.bias.data.zero_()


if __name__ == '__main__':
    rpn = RegionProposalNetwork(512, 512)
    if torch.cuda.is_available():
        rpn_net = rpn.cuda()

    image_size = (500,500)
    features = torch.randn(1,512,50,50))
    if torch.cuda.is_available():
        features = features.cuda()

    delta, score, anchor = rpn_net.forward(features, image_size)
    
    gt_bbox = (np.random.rand(10,4) + [0,0,1,1])*240
    loss = rpn_net.loss(delta, score, anchor, gt_bbox, image_size)
    loss.backward()
    print(loss)

    rpn_net.train()
    roi = rpn_net.predict(delta, score, anchor, image_size)
    print(roi.shape)

    rpn_net.eval()
    roi = rpn_net.predict(delta, score, anchor, image_size)
    print(roi.shape)