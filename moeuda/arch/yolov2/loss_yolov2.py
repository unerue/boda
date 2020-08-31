import torch
from torch import nn, Tensor
import torch.nn.functional as F


# from utils.priorBox import PriorBox
# from utils.bbox_utils import box_iou



class Yolov2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def _transform_targets(self, loc_preds):
        '''
        loc_preds: (tensor) predicted locations, sized [N, 5, 4, fmsize, fmsize].
        '''
        N,anchor_num,_,fmsize,_ = loc_preds.size()
        priorbox = PriorBox(cfg).forward().cuda()
        loc_xy = loc_preds[:,:,:2,:,:]   #[N,5,2,13,13]
        grid_xy = priorbox[:,:2].contiguous().view(anchor_num,fmsize,fmsize,2).permute(0,3,1,2)
        box_xy = torch.sigmoid(loc_xy)+grid_xy.expand_as(loc_xy)

        loc_wh = loc_preds[:,:,2:4,:,:]   #[N,5,2,13,13]
        anchor_wh = priorbox[:,2:4].contiguous().view(anchor_num,fmsize,fmsize,2).permute(0,3,1,2)  #[2,13,13]
        box_wh = anchor_wh*torch.exp(loc_wh)
        box_preds = torch.cat((box_xy-box_wh*0.5,box_xy+box_wh*0.5),2)
        return box_preds   #[N,5,4,13,13]

    def forward(self, preds, loc_targets, cls_targets, box_targets):
        '''
            preds: [N,125,fmsize,fmsize]
            loc_target: [N,5,4,fmsize,fmsize]
            cls_targets: [N,5,20,fmsize,fmsize]
            box_targets: list [#obj,4]
        '''
        N,fm_num,fmsize,_ = preds.size()
        preds = preds.view(N,5,25,fmsize,fmsize)
        xy = torch.sigmoid(preds[:,:,:2,:,:])      #[N,5,2,13,13]
        wh = torch.exp(preds[:,:,2:4,:,:])         #[N,5,2,13,13]
        loc_preds = torch.cat((xy,wh),2)           #[N,5,4,13,13]

        pos = cls_targets.max(2)[0].squeeze()>0       #[N,5,13,13]
        num_pos = pos.data.long().sum()
        mask = pos.unsqueeze(2).expand_as(loc_preds)   #[N,5,4,13,13]
        loc_loss = F.smooth_l1_loss(loc_preds[mask], loc_targets[mask], size_average=False)
        
        iou_preds = torch.sigmoid(preds[:,:,4,:,:])  # [N,5,13,13]
        iou_targets = Variable(torch.zeros(iou_preds.size()),volatile=True).cuda()  #[N,5,13,13]
        box_preds = self.decode_loc(preds[:,:,:4,:,:])   #[N,5,4,13,13]
        box_preds = box_preds.permute(0,1,3,4,2).contiguous().view(N,-1,4)        #[N,5*13*13,4]
        for i in range(N):
            box_pred = box_preds[i]   #[5*13*13,4]
            box_target = box_targets[i]  #[#obj,4]
            iou_target = box_iou(box_pred, box_target)  # [5*13*13, #obj]
            iou_targets[i] = iou_target.max(1)[0].view(5,fmsize,fmsize)  # [5,13,13]

        mask = Variable(torch.ones(iou_preds.size())).cuda() * 0.1  # [N,5,13,13]
        mask[pos] = 1
        iou_loss = F.smooth_l1_loss(iou_preds*mask, iou_targets*mask, size_average=False)

        cls_preds = preds[:,:,5:,:,:]  # [N,5,20,13,13]
        cls_preds = cls_preds.permute(0,1,3,4,2).contiguous().view(-1,20)  # [N,5,20,13,13] -> [N,5,13,13,20] -> [N*5*13*13,20]
        cls_preds = F.softmax(cls_preds)  # [N*5*13*13,20]
        cls_preds = cls_preds.view(N,5,fmsize,fmsize,20).permute(0,1,4,2,3)  # [N*5*13*13,20] -> [N,5,20,13,13]
        pos = cls_targets > 0
        cls_loss = F.smooth_l1_loss(cls_preds[pos], cls_targets[pos], size_average=False)

        print('%f %f %f' % (loc_loss.data[0]/num_pos, iou_loss.data[0]/num_pos, cls_loss.data[0]/num_pos), end=' ')
        return (loc_loss + iou_loss + cls_loss) / num_pos