

import torch
import torch.nn as nn
import torch.nn.functional as F

def hard_negtives(logits, labels, pos, neg_radio):
    '''
    logits (batch, n, 21)
    labels (batch, n, )
    pos (batch, n,)
    Pick out the largest negative sample in this batch, because the number of negative samples in ssd is very large.
    If we randomly select negative samples for training, the model will converge quickly and the discriminating ability will be worse, 
    so we Choose the most difficult to identify negative samples to train
    '''
    
    num_batch, num_anchors, num_classes = logits.shape
    logits = logits.view(-1, num_classes)
    labels = labels.view(-1)
    
    #Get the loss of the corresponding category of the anchor
    losses = F.cross_entropy(logits, labels, reduction='none')

    losses = losses.view(num_batch, num_anchors)

    #Filter out positive samples because we only mine negative samples
    losses[pos] = 0

    #loss (batch, n)
    #The following two argsorts allow us to get the position where the loss should exist after sorting.
    #Example shows that the original loss is (3, 2, 1, 4)
    #(1,2,3,0) can be obtained by the following operation
    #why? Because the loss is sorted after (4,3,2,1)
    #And after 3 sorting, in the 1st position, 2 sorting and then 2nd position, 1 sorting in the 3rd position, 
    #4 sorting and then 0th position
    loss_idx = losses.argsort(1, descending=True)
    rank = loss_idx.argsort(1) #(batch, n)

    #By getting the number of positive samples for each picture of the batch, select the negative sample quantity according to the ratio, 
    #and the maximum cannot exceed the number of anchors.
    num_pos = pos.long().sum(1, keepdim=True)
    num_neg = torch.clamp(neg_radio*num_pos, max=pos.shape[1]-1) #(batch, 1)
    neg = rank < num_neg.expand_as(rank)   
    return neg
    
class MultiBoxLoss(nn.Module):

    def __init__(self, num_classes=10, neg_radio=3):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.neg_radio = neg_radio
    
    def forward(self, pred_loc, pred_label, gt_loc, gt_label):
        '''
        pred_loc : (batch, anchor_num, 4)
        pred_label : (batch, anchor_num, num_classes)
        gt_loc : (batch, anchor_num, 4)
        gt_label : (batch, anchor_num)
        '''

        num_batch = pred_loc.shape[0]

        #Select positive samples for coordinate regression
        pos_idx = gt_label > 0
        pos_loc_idx = pos_idx.unsqueeze(2).expand_as(pred_loc)
        pred_loc_pos = pred_loc[pos_loc_idx].view(-1, 4)
        gt_loc_pos = gt_loc[pos_loc_idx].view(-1, 4)

        loc_loss = F.smooth_l1_loss(pred_loc_pos, gt_loc_pos, reduction='sum')

        
        #Difficult negative sample mining
        logits = pred_label.detach()
        labels = gt_label.detach()
        neg_idx = hard_negtives(logits, labels, pos_idx, self.neg_radio) #neg (batch, n)

        #here, we use the difficult negative and positive samples for training classification.
        pos_cls_mask = pos_idx.unsqueeze(2).expand_as(pred_label)
        neg_cls_mask = neg_idx.unsqueeze(2).expand_as(pred_label)

        conf_p = pred_label[(pos_cls_mask+neg_cls_mask).gt(0)].view(-1, self.num_classes)
        target = gt_label[(pos_idx+neg_idx).gt(0)]

        cls_loss = F.cross_entropy(conf_p, target, reduction='sum')
        N = pos_idx.long().sum()

        loc_loss /= N
        cls_loss /= N


        return loc_loss, cls_loss
