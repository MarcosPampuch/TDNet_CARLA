"""
Lovasz-Softmax and Jaccard hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# diffrent losses to be tested for the training

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

class FocalLoss(nn.CrossEntropyLoss):
    ''' Focal loss for classification tasks on imbalanced datasets '''

    def __init__(self, gamma=0.5, alpha=None, ignore_index=254, reduction='none'):
        super().__init__(weight=alpha, ignore_index=ignore_index, reduction='none')
        self.reduction = reduction
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.indicador = 0

    def forward(self, input_, target, smooth_targets):        
        
        cross_entropy = cross_entropy_with_probs(input_, smooth_targets, self.alpha, reduction = "none", ignored_index = self.ignore_index)
#        cross_entropy = super().forward(input_, target)
        target = target * (target != self.ignore_index).long()
        input_prob = torch.gather(F.softmax(input_, 1), 1, target.unsqueeze(1))
        loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy
        return torch.mean(loss) 
    
def lovasz_softmax_flat(prb, lbl, ignore_index, only_present):
    """
    Multi-class Lovasz-Softmax loss
      prb: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      lbl: [P] Tensor, ground truth labels (between 0 and C - 1)
      ignore_index: void class labels
      only_present: average only on classes present in ground truth
    """
    C = prb.shape[0]
    prb = prb.permute(1, 2, 0).contiguous().view(-1, C)  # H * W, C
    lbl = lbl.view(-1)  # H * W
    if ignore_index is not None:
        mask = lbl != ignore_index
        if mask.sum() == 0:
            return torch.mean(prb * 0)
        prb = prb[mask]
        lbl = lbl[mask]

    total_loss = 0
    cnt = 0
    for c in range(C):
        fg = (lbl == c).float()  # foreground for class c
        if only_present and fg.sum() == 0:
            continue
        errors = (fg - prb[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        total_loss += torch.dot(errors_sorted, lovasz_grad(fg_sorted))
        cnt += 1
    return total_loss / cnt


class LovaszSoftmax(nn.Module): # a loss that can be used to fine tune a model already trained
    """
    Multi-class Lovasz-Softmax loss
      logits: [B, C, H, W] class logits at each prediction (between -\infty and \infty)
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      ignore_index: void class labels
      only_present: average only on classes present in ground truth
    """
    def __init__(self, ignore_index=None, only_present=True):
        super().__init__()
        self.ignore_index = ignore_index
        self.only_present = only_present

    def forward(self, logits, labels):
        probas = F.softmax(logits, dim=1)
        total_loss = 0
        batch_size = logits.shape[0]
        for prb, lbl in zip(probas, labels):
            total_loss += lovasz_softmax_flat(prb, lbl, self.ignore_index, self.only_present)
        return total_loss / batch_size

class OhemCELoss(nn.Module): # deprecated
    def __init__(self, thresh, n_min, ignore_lb=255, class_weight=None,*args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = thresh
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, weight=class_weight)

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        n_pixs = N * H * W
        logits = logits.permute(0, 2, 3, 1).contiguous().view(-1, C)
        labels = labels.view(-1)
        with torch.no_grad():
            scores = F.softmax(logits, dim=1)
            labels_cpu = labels
            invalid_mask = labels_cpu==self.ignore_lb
            labels_cpu[invalid_mask] = 0
            picks = scores[torch.arange(n_pixs), labels_cpu]
            picks[invalid_mask] = 1
            sorteds, _ = torch.sort(picks)
            thresh = self.thresh if sorteds[self.n_min]<self.thresh else sorteds[self.n_min]
            labels[picks>thresh] = self.ignore_lb
        ## TODO: here see if torch or numpy is faster
        labels = labels.clone()
        loss = self.criteria(logits, labels)
        return loss

class GeneralizedSoftDiceLoss(nn.Module): # mimics the mIoU as a loss

    def __init__(self,
                 p=1,
                 smooth=1,
                 reduction='mean',
                 weight=None,
                 ignore_lb=255):
        super(GeneralizedSoftDiceLoss, self).__init__()
        self.p = p
        self.smooth = smooth
        self.reduction = reduction
        self.weight = None if weight is None else torch.tensor(weight)
        self.ignore_lb = ignore_lb

    def forward(self, logits, label):
        '''
        args: logits: tensor of shape (N, C, H, W)
        args: label: tensor of shape(N, H, W)
        '''
        # overcome ignored label
        logits = logits.float()
        ignore = label.data.cpu() == self.ignore_lb
        label = label.clone()
        label[ignore] = 0
        lb_one_hot = torch.zeros_like(logits).scatter_(1, label.unsqueeze(1), 1)
        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        lb_one_hot[[a, torch.arange(lb_one_hot.size(1)).long(), *b]] = 0
        lb_one_hot = lb_one_hot.detach()

        # compute loss
        probs = torch.sigmoid(logits)
        numer = torch.sum((probs*lb_one_hot), dim=(2, 3))
        denom = torch.sum(probs.pow(self.p)+lb_one_hot.pow(self.p), dim=(2, 3))
        if not self.weight is None:
            numer = numer * self.weight.view(1, -1)
            denom = denom * self.weight.view(1, -1)
        numer = torch.sum(numer, dim=1)
        denom = torch.sum(denom, dim=1)
        loss = 1 - (2*numer+self.smooth)/(denom+self.smooth)

        if self.reduction == 'mean':
            loss = loss.mean()
        return loss
    

def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0, lb_ignore = 254): # apply uniform label smoothing
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
 #   true_labels[true_labels == lb_ignore] = 13
    label_shape = torch.Size((true_labels.size(0), classes+1, true_labels.size(1),true_labels.size(2)))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return true_dist[:,:-1]



def unigram_one_hot(true_labels: torch.Tensor, classes: int, weight: torch.Tensor, smoothing=0.0, lb_ignore = 254): # apply label smoothing with class weights to modify them
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    
    weight = weight/ weight.sum()
    weight = weight * smoothing
    weight = torch.cat ((weight, torch.tensor([0]).float().cuda()))
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    true_labels[true_labels == lb_ignore] = 13
    label_shape = torch.Size((true_labels.size(0), classes+1, true_labels.size(1),true_labels.size(2)))
    with torch.no_grad():
        
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        for i in range (0,classes):
            true_dist[:,i,:,:].fill_(weight[i]) 
#        true_dist.fill_(smoothing / (classes - 1))
        true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return true_dist[:,:-1]

def calculate_weigths( y, num_classes):
    
    z = torch.zeros((num_classes,)).cuda()
    
    mask = (y >= 0) & (y < num_classes)
    labels = y[mask].long()
    count_l = torch.bincount(labels, minlength=num_classes)
    
    total_frequency = count_l.sum()
    for index, frequency in enumerate(count_l):
        z[index] = 1 / (torch.log(1.02 + (frequency.float() / total_frequency.float())))

    return z

def cross_entropy_with_probs(input, target, weight = None, reduction = "mean", ignored_index = 254): # cross entropy with label smoothing

#    weighta = calculate_weigths(target, 20)
#    weighta = 2 * (weighta / weighta.max())
#    print(weighta)
    print("##########")
    print(input.device)
    print(target.device)
    batch, num_classes, h, w = input.shape
    cum_losses = input.new_zeros(batch, h,w)
    print(cum_losses.device)
    for y in range(num_classes):
        target_temp = input.new_full((batch, h,w), y, dtype=torch.long)
        y_loss = F.cross_entropy(input, target_temp,  reduction="none", ignore_index = ignored_index)# weight = weight.cuda(),
        print(y_loss.device)
        print(weight[y].device)
        if weight is not None:
            y_loss = y_loss * weight[y]
        if y != ignored_index:
            
            cum_losses += target[:, y].float() * y_loss

    if reduction == "none":
        return cum_losses
    elif reduction == "mean":
        return cum_losses.mean()
    elif reduction == "sum":
        return cum_losses.sum()
    else:
        raise ValueError("Keyword 'reduction' must be one of ['none', 'mean', 'sum']")
        



class OhemCE(nn.Module): # OHEM with label smoothing
    """
    Online hard example mining with cross entropy loss, for semantic segmentation.
    This is widely used in PyTorch semantic segmentation frameworks.
    Reference: https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/1b3ae72f6025bde4ea404305d502abea3c2f5266/lib/core/criterion.py#L29
    Arguments:
        ignore_label: Integer, label to ignore.
        threshold: Float, threshold for softmax score (of gt class), only predictions with softmax score
            below this threshold will be kept.
        min_kept: Integer, minimum number of pixels to be kept, it is used to adjust the
            threshold value to avoid number of examples being too small.
        weight: Tensor, a manual rescaling weight given to each class.
    """
    def __init__(self, ignore_label=-1, threshold=0.7,
                 min_kept=100000, weight=None):
        super(OhemCE, self).__init__()
        self.threshold = threshold
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label,
                                             reduction='none')

    def forward(self, logits, labels, smooth_labels, weight2 = None, **kwargs):
        predictions = F.softmax(logits, dim=1)
        if 'semantic_weights' in kwargs:
            pixel_losses = self.criterion(logits, labels) * kwargs['semantic_weights']
            pixel_losses = pixel_losses.contiguous().view(-1)
        else:
            pixel_losses = self.criterion(logits, labels).contiguous().view(-1)
            pixel_losses_2 = cross_entropy_with_probs(logits, smooth_labels, None, reduction = "none", ignored_index = self.ignore_label).contiguous().view(-1)

        mask = labels.contiguous().view(-1) != self.ignore_label

        tmp_labels = labels.clone()
        tmp_labels[tmp_labels == self.ignore_label] = 0
        # Get the score for gt class at each pixel location.
        predictions = predictions.gather(1, tmp_labels.unsqueeze(1))
        predictions, indices = predictions.contiguous().view(-1, )[mask].contiguous().sort()
        min_value = predictions[min(self.min_kept, predictions.numel() - 1)]
        threshold = max(min_value, self.threshold)

        pixel_losses_2 = pixel_losses_2[mask][indices]
        pixel_losses_2 = pixel_losses_2[predictions < threshold]
        return pixel_losses.mean()


class DeepLabCE(nn.Module): # Deelab loss with label smoothing
    
    def __init__(self, ignore_label=-1, top_k_percent_pixels=1.0, weight=None):
        super(DeepLabCE, self).__init__()
        self.top_k_percent_pixels = top_k_percent_pixels
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label,
                                             reduction='none')

    def forward(self, logits, labels, smooth_labels, weight2 = None, **kwargs):
        if 'semantic_weights' in kwargs:
            pixel_losses = self.criterion(logits, labels) * kwargs['semantic_weights']
            pixel_losses = pixel_losses.contiguous().view(-1)
        else:
            pixel_losses = self.criterion(logits, labels).contiguous().view(-1)
            pixel_losses_2 = cross_entropy_with_probs(logits, smooth_labels, weight2, reduction = "none", ignored_index = self.ignore_label).contiguous().view(-1)

        if self.top_k_percent_pixels == 1.0:
            return pixel_losses_2.mean()

        top_k_pixels = int(self.top_k_percent_pixels * pixel_losses.numel())
        pixel_losses_2, _ = torch.topk(pixel_losses_2, top_k_pixels)
        return pixel_losses_2.mean()
