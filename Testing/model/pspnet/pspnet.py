import torch
from torch import Tensor
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .resnet import resnet18,resnet34,resnet50,resnet101
import os
#from utils.dice_score import dice_loss

from torch.autograd import Variable

up_kwargs = {'mode': 'bilinear', 'align_corners': True}


class BatchNorm2d(nn.BatchNorm2d):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, num_features, activation='none'):
        super(BatchNorm2d, self).__init__(num_features=num_features)
        if activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'none':
            self.activation = lambda x:x
        else:
            raise Exception("Accepted activation: ['leaky_relu']")

    def forward(self, x):
        return self.activation(super(BatchNorm2d, self).forward(x))


up_kwargs = {'mode': 'bilinear', 'align_corners': True}

class pspnet(nn.Module):

    def __init__(self,
            nclass=21,
            norm_layer=BatchNorm2d,
            backbone='resnet101',
            dilated=True,
            aux=True,
            multi_grid=True,
            model_path=None,
            #loss_fn=None,
        ):
        super(pspnet, self).__init__()
        self.psp_path = model_path
#        self.loss_fn = loss_fn
        self.norm_layer = norm_layer
        self._up_kwargs = up_kwargs
        self.nclass = nclass
        # copying modules from pretrained models
        self.backbone = backbone
        
        if backbone == 'resnet18':
            self.pretrained = resnet18(dilated=dilated, multi_grid=multi_grid,
                                               deep_base=False)
            self.expansion = 1
        elif backbone == 'resnet34':
            self.pretrained = resnet34(dilated=dilated, multi_grid=multi_grid,
                                               deep_base=False)
            self.expansion = 1
        elif backbone == 'resnet50':
            self.pretrained = resnet50(dilated=dilated,multi_grid=multi_grid,
                                              norm_layer=norm_layer)
            self.expansion = 4
        elif backbone == 'resnet101':
            self.pretrained = resnet101(dilated=dilated,multi_grid=multi_grid,
                                               norm_layer=norm_layer)
            self.expansion = 4
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        # bilinear upsample options
        
        self.head = PSPHead(512*self.expansion, nclass, norm_layer, self._up_kwargs)
        self.pretrained_mp_load()
#        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, lbl=None, pos_id=None):
#        x = x[-1]
        _, _, h, w = x.size()

        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)

        x = self.head(c4)
        outputs = F.interpolate(x, (h,w), **self._up_kwargs)
        
#        if self.training: 
#            #loss = self.loss_fn(outputs,lbl)
##            loss = self.criterion(outputs, lbl) \
##                + self.dice_loss(F.softmax(outputs, dim=1).float(), F.one_hot(lbl, 13).permute(0, 3, 1, 2).float(), multiclass=True) 
#            
#            
#            
#            return loss
#        
#        else:
        return outputs
        
#    def get_params(self):
#        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
#        for name, child in self.named_children():
#            if isinstance(child, (OhemCELoss2D, SegmentationLosses, pspnet, nn.KLDivLoss)):
#                continue
#            child_wd_params, child_nowd_params = child.get_params()
#            if isinstance(child, (Encoding, Attention, PyramidPooling, FCNHead, Layer_Norm)):
#                lr_mul_wd_params += child_wd_params
#                lr_mul_nowd_params += child_nowd_params
#            else:
#                wd_params += child_wd_params
#                nowd_params += child_nowd_params
#        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params
#
#        return outputs

    def pretrained_mp_load(self):
        if self.psp_path is not None:
            if os.path.isfile(self.psp_path):
                print("Loading pretrained model from '{}'".format(self.psp_path))
                model_state = torch.load(self.psp_path)
                self.load_state_dict(model_state, strict=True)

            else:
                print("No pretrained found at '{}'".format(self.psp_path))
                
#    def dice_coeff(self, input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
#        # Average of Dice coefficient for all batches, or for a single mask
#        assert input.size() == target.size()
#        if input.dim() == 2 and reduce_batch_first:
#            raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')
#    
#        if input.dim() == 2 or reduce_batch_first:
#            inter = torch.dot(input.reshape(-1), target.reshape(-1))
#            sets_sum = torch.sum(input) + torch.sum(target)
#            if sets_sum.item() == 0:
#                sets_sum = 2 * inter
#    
#            return (2 * inter + epsilon) / (sets_sum + epsilon)
#        else:
#            # compute and average metric for each batch element
#            dice = 0
#            for i in range(input.shape[0]):
#                dice += self.dice_coeff(input[i, ...], target[i, ...])
#            return dice / input.shape[0]
#
#
#    def multiclass_dice_coeff(self, input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
#        # Average of Dice coefficient for all classes
#        assert input.size() == target.size()
#        dice = 0
#        for channel in range(input.shape[1]):
#            dice += self.dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)
#    
#        return dice / input.shape[1]
#    
#    
#    def dice_loss(self, input: Tensor, target: Tensor, multiclass: bool = False):
#        # Dice loss (objective to minimize) between 0 and 1
#        print(input.size())
#        print(target.size())
#        assert input.size() == target.size()
#        fn = self.multiclass_dice_coeff if multiclass else self.dice_coeff
#        return 1 - fn(input, target, reduce_batch_first=True)


class PSPHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(PSPHead, self).__init__()
        inter_channels = in_channels // 4

        self.norm_layer = norm_layer
        self.conv5 = nn.Sequential(PyramidPooling(in_channels, norm_layer, up_kwargs),
                                   nn.Conv2d(in_channels * 2, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(True),
                                   nn.Dropout2d(0.1, False),
                                   nn.Conv2d(inter_channels, out_channels, 1))
    def forward(self, x):
        return self.conv5(x)


class PyramidPooling(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, in_channels, norm_layer, up_kwargs):
        super(PyramidPooling, self).__init__()
        self.norm_layer = norm_layer
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)

        out_channels = int(in_channels/4)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True))
        # bilinear interpolate options
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.interpolate(self.conv1(self.pool1(x)), (h, w), **self._up_kwargs)
        feat2 = F.interpolate(self.conv2(self.pool2(x)), (h, w), **self._up_kwargs)
        feat3 = F.interpolate(self.conv3(self.pool3(x)), (h, w), **self._up_kwargs)
        feat4 = F.interpolate(self.conv4(self.pool4(x)), (h, w), **self._up_kwargs)
        return torch.cat((x, feat1, feat2, feat3, feat4), 1)



class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs={}, with_global=False):
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self._up_kwargs = up_kwargs
        self.norm_layer = norm_layer
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   nn.Dropout2d(0.1, False),
                                   nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        return self.conv5(x)
