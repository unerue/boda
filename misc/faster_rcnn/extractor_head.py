import torch
import torch.nn as nn
from torchvision.models import vgg16
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from model.utils.roipooling import RoIPool
from model.utils.loss import delta_loss
from model.utils.bbox_tools import delta2bbox
from model.utils.nms_cpu import py_cpu_nms

def get_vgg16_extractor_and_head(n_class, roip_size=7, vgg_pretrained=False):
    vgg16_net = vgg16(pretrained=True)
    features = list(vgg16_net.features)[0:30]
    
    for layer in features[0:10]:    # freeze top 4 conv2d layers
        for p in layer.parameters():
            p.requires_grad = False
    extractor = nn.Sequential(*features)
    output_feature_channel = 512

    classifier = list(vgg16_net.classifier)
    del(classifier[6])  # delete last fc layer
    classifier = nn.Sequential(*classifier)     # classifier : (N,25088) -> (N,4096); 25088 = 512*7*7 = C*H*W
    if torch.cuda.is_available():
        classifier = classifier.cuda()
    head = _VGG16Head(n_class_bg=n_class+1, roip_size=roip_size, classifier=classifier)
    if torch.cuda.is_available():
        extractor, head = extractor.cuda(), head.cuda()
    return extractor, head, output_feature_channel


class _VGG16Head(nn.Module):
    def __init__(self, n_class_bg, roip_size, classifier):
        """n_class_bg: n_class plus background = n_class + 1"""
        super(_VGG16Head, self).__init__()
        self.n_class_bg = n_class_bg
        self.roip_size = roip_size

        self.roip = RoIPool(roip_size, roip_size)
        self.classifier = classifier
        self.delta = nn.Linear(in_features=4096, out_features=n_class_bg*4)    # Note: predice a delta for each class
        self.score = nn.Linear(in_features=4096, out_features=n_class_bg)

        self._normal_init(self.delta, 0, 0.001)
        self._normal_init(self.score, 0, 0.01)

    def forward(self, feature_map, rois, image_size):
        """
        Args:
            feature_map: (N1=1,C,H,W)
            rois : (N2,4)
        """
        #---------- debug
        assert isinstance(feature_map, Variable)
        assert isinstance(rois, np.ndarray)
        assert len(feature_map.shape) == 4 and feature_map.shape[0] == 1    # batch size should be 1
        assert len(rois.shape) == 2 and rois.shape[1] == 4
        #---------- debug

        # this is important because rois are in image scale, we need to pass this ratio 
        # to roipooing layer to map roi into feature_map scale
        feature_image_scale = feature_map.shape[2] / image_size[0]  
        
        # meet roi_pooling's input requirement
        temp = np.zeros((rois.shape[0], 1), dtype=rois.dtype)
        rois = np.concatenate([temp, rois], axis=1) 

        rois = Variable(torch.FloatTensor(rois))
        if torch.cuda.is_available():
            rois = rois.cuda()

        roipool_out = self.roip(feature_map, rois, spatial_scale=feature_image_scale)

        roipool_out = roipool_out.view(roipool_out.size(0), -1) # (N, 25088)
        if torch.cuda.is_available():
            roipool_out = roipool_out.cuda()

        mid_output = self.classifier(roipool_out)   # (N, 4096)
        delta_per_class = self.delta(mid_output)    # (N, n_class_bg*4)
        score = self.score(mid_output)      # (N, n_class_bg)
        #---------- debug
        assert isinstance(delta_per_class, Variable) and isinstance(score, Variable)
        assert delta_per_class.shape[0] == score.shape[0] == rois.shape[0]
        assert delta_per_class.shape[1] == score.shape[1] * 4 == self.n_class_bg * 4
        assert len(delta_per_class.shape) == len(score.shape) == 2
        #---------- debug
        return delta_per_class, score

    def loss(self, score, delta_per_class, target_delta_for_sample_roi, bbox_bg_label_for_sample_roi):
        """
        Args:
            score: (N, 2)
            delta_per_class: (N, 4*n_class_bg)
            target_delta_for_sample_roi: (N, 4)
            bbox_bg_label_for_sample_roi: (N,)
        """
        #---------- debug
        assert isinstance(score, Variable)
        assert isinstance(delta_per_class, Variable)
        assert isinstance(target_delta_for_sample_roi, np.ndarray)
        assert isinstance(bbox_bg_label_for_sample_roi, np.ndarray)
        #---------- debug
        target_delta_for_sample_roi = Variable(torch.FloatTensor(target_delta_for_sample_roi))
        bbox_bg_label_for_sample_roi = Variable(torch.LongTensor(bbox_bg_label_for_sample_roi))
        if torch.cuda.is_available():
            target_delta_for_sample_roi = target_delta_for_sample_roi.cuda()
            bbox_bg_label_for_sample_roi = bbox_bg_label_for_sample_roi.cuda()

        n_sample = score.shape[0]
        delta_per_class = delta_per_class.view(n_sample, -1, 4)

        # get delta for roi w.r.t its corresponding bbox label
        index = torch.arange(0, n_sample).long()
        if torch.cuda.is_available():
            index = index.cuda()
        delta = delta_per_class[index, bbox_bg_label_for_sample_roi.data]

        head_delta_loss = delta_loss(delta, target_delta_for_sample_roi, bbox_bg_label_for_sample_roi, 1)
        head_class_loss = F.cross_entropy(score, bbox_bg_label_for_sample_roi)

        return head_delta_loss + head_class_loss

    def predict(self, roi, delta_per_class, score, image_size, prob_threshold=0.5):
        """
        Args:
            roi: (N, 4)
            delta_per_class: (N, 4*n_class_bg)
            score: (N, n_class_bg)
        """
        #---------- debug
        assert isinstance(roi, np.ndarray)
        assert isinstance(delta_per_class, Variable)
        assert isinstance(score, Variable)
        #---------- debug
        roi = torch.FloatTensor(roi)
        if torch.cuda.is_available():
            roi = roi.cuda()
        delta_per_class = delta_per_class.data
        prob = F.softmax(score, dim=1).data

        delta_per_class = delta_per_class.view(-1, self.n_class_bg, 4)
        
        #!!!!!
        delta_per_class = delta_per_class * torch.cuda.FloatTensor([0.1, 0.1, 0.2, 0.2]) + torch.cuda.FloatTensor([0., 0., 0., 0.])
        
        roi = roi.view(-1,1,4).expand_as(delta_per_class)
        bbox_per_class = delta2bbox(roi.cpu().numpy().reshape(-1,4), delta_per_class.cpu().numpy().reshape(-1,4))
        bbox_per_class = torch.FloatTensor(bbox_per_class)

        bbox_per_class[:,0::2] = bbox_per_class[:,0::2].clamp(min=0, max=image_size[0])
        bbox_per_class[:,1::2] = bbox_per_class[:,1::2].clamp(min=0, max=image_size[1])

        bbox_per_class = bbox_per_class.numpy().reshape(-1,self.n_class_bg,4)
        prob = prob.cpu().numpy()
        #---------- debug
        assert bbox_per_class.shape[0] == prob.shape[0]
        assert bbox_per_class.shape[2] == 4
        assert bbox_per_class.shape[1] == prob.shape[1] == self.n_class_bg
        #---------- debug
        
        # suppress:
        bbox_out = []
        class_out = []
        prob_out = []
        # skip class_id = 0 because it is the background class
        for t in range(1, self.n_class_bg):
            bbox_for_class_t = bbox_per_class[:,t,:]    #(N, 4)
            prob_for_class_t = prob[:,t]                #(N,)
            mask = prob_for_class_t > prob_threshold    #(N,)
            # debug:
            # print("mask", mask.sum())
            left_bbox_for_class_t = bbox_for_class_t[mask]  #(N2,4)
            left_prob_for_class_t = prob_for_class_t[mask]  #(N2,)
            keep = py_cpu_nms(left_bbox_for_class_t, score=left_prob_for_class_t)
            bbox_out.append(left_bbox_for_class_t[keep])
            prob_out.append(left_prob_for_class_t[keep])
            class_out.append((t-1)*np.ones(len(keep)))

        bbox_out = np.concatenate(bbox_out, axis=0).astype(np.float32)
        prob_out = np.concatenate(prob_out, axis=0).astype(np.float32)
        class_out = np.concatenate(class_out, axis=0).astype(np.int32)
        #---------- debug
        assert isinstance(bbox_out, np.ndarray)
        assert isinstance(prob_out, np.ndarray)
        assert isinstance(class_out, np.ndarray)
        #---------- debug
        return bbox_out, class_out, prob_out
    

    def _normal_init(self, m, mean, stddev, truncated=False):
        """
        weight initalizer: truncated normal and random normal.
        """
        if truncated:
            m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
            m.weight.data.normal_(mean, stddev)
            m.bias.data.zero_()




def _get_resnet50_extractor_and_head():
    pass


if __name__ == '__main__':
    from model.utils.proposal_target_creator import ProposalTargetCreator
    extractor, head, output_feature_channel = get_vgg16_extractor_and_head(20, 7)    
    features = Variable(torch.randn(1,512,50,50))
    if torch.cuda.is_available():
        extractor, head, features = extractor.cuda(), head.cuda(), features.cuda()
        
    rois = (np.random.rand(2000,4)+[0,0,1,1])*240
    gt_bbox = (np.random.rand(10,4) + [0,0,1,1])*240
    gt_bbox_label = np.random.randint(0,20,size=10)
    
    proposal_target_creator = ProposalTargetCreator()
    sample_roi, target_delta_for_sample_roi, bbox_bg_label_for_sample_roi = proposal_target_creator.make_proposal_target(rois, gt_bbox, gt_bbox_label)
    
    delta_per_class, score = head.forward(features, sample_roi, image_size=(500,500))
    loss = head.loss(score, delta_per_class,target_delta_for_sample_roi,bbox_bg_label_for_sample_roi)
    print(loss)
    loss.backward()

    rois = (np.random.rand(300,4)+[0,0,1,1])*240
    delta_per_class, score = head.forward(features, rois, image_size=(500,500))
    bbox_out, class_out, prob_out = head.predict(rois, delta_per_class, score, image_size=(500,500),prob_threshold=0.2)
    print(bbox_out.shape)
    print(class_out.shape)
    print(prob_out.shape)