import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class RoIPool(nn.Module):
    def __init__(self, pooled_height, pooled_width):
        super(RoIPool, self).__init__()
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)

    def forward(self, features, rois, spatial_scale):
        """
        Args:
            features: (N=1, C, H, W)
            rois: (N, 5); 5=[roi_index, x1, y1, x2, y2]
            spatial_scale: feature size / image size, this is important because rois are in image scale!
        Note: both features and rois are required to be Variable type.
              You should transform rois to Variable and set requires_grad to False before pass is to this function.
        """
        #---------- debug
        assert isinstance(features, Variable)
        assert isinstance(rois, Variable)
        #---------- debug
        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size()[0]
        outputs = Variable(torch.zeros(num_rois, num_channels, self.pooled_height, self.pooled_width))
        if torch.cuda.is_available():
            outputs = outputs.cuda()

        for roi_ind, roi in enumerate(rois):
            batch_ind = int(roi[0].data.item())
            roi_start_w, roi_start_h, roi_end_w, roi_end_h = np.round(
                roi[1:].data.cpu().numpy() * spatial_scale).astype(int)
            roi_width = max(roi_end_w - roi_start_w + 1, 1)
            roi_height = max(roi_end_h - roi_start_h + 1, 1)
            bin_size_w = float(roi_width) / float(self.pooled_width)
            bin_size_h = float(roi_height) / float(self.pooled_height)

            for ph in range(self.pooled_height):
                hstart = int(np.floor(ph * bin_size_h))
                hend = int(np.ceil((ph + 1) * bin_size_h))
                hstart = min(data_height, max(0, hstart + roi_start_h))
                hend = min(data_height, max(0, hend + roi_start_h))
                for pw in range(self.pooled_width):
                    wstart = int(np.floor(pw * bin_size_w))
                    wend = int(np.ceil((pw + 1) * bin_size_w))
                    wstart = min(data_width, max(0, wstart + roi_start_w))
                    wend = min(data_width, max(0, wend + roi_start_w))

                    is_empty = (hend <= hstart) or(wend <= wstart)
                    if is_empty:
                        outputs[roi_ind, :, ph, pw] = 0
                    else:
                        data = features[batch_ind]
                        data_pool = torch.max(data[:, hstart:hend, wstart:wend], 1)[0]
                        outputs[roi_ind, :, ph, pw] = torch.max(data_pool, 1)[0].view(-1)
        #---------- debug
        assert outputs.shape[0] == rois.shape[0]
        assert outputs.shape[1] == features.shape[1]
        assert outputs.shape[2] == self.pooled_height
        assert outputs.shape[3] == self.pooled_width
        assert isinstance(outputs, Variable)
        #---------- debug
        return outputs


if __name__ == '__main__':
    features = Variable(torch.ones(1,1,80,80), requires_grad=False)
    features[0,0,1,1] = 2
    features[0,0,1,70] = 3
    features[0,0,70,1] = 4
    features[0,0,70,70] = 5
    rois = Variable(torch.LongTensor([[0,0,0,750,750], [0,0,0,200,200]]),requires_grad=False)
    print(features.shape)
    print(rois.shape)

    roip = RoIPool(2,2)
    out = roip(features, rois, spatial_scale=0.1)
    print(out)
    print(out.shape)