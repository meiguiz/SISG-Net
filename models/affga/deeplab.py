import torch
import torch.nn as nn
import torch.nn.functional as F
from models.affga.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.affga.hasp import build_hasp
from models.affga.decoder import build_decoder
from models.affga.backbone import build_backbone
from models.affga.fpn import build_backbone_fpn
from model.detection import MaskRCNN
from model.detection.faster_rcnn import FastRCNNPredictor
from model.detection.mask_rcnn import MaskRCNNPredictor
from models import resnet
from model.detection.rpn import AnchorGenerator
from model.detection.image_list import ImageList
from model.detection.transform import GeneralizedRCNNTransform
from models.backbone import get_backbone_with_fpn
from torch.hub import load_state_dict_from_url
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
import os
import numpy as np
from PIL import Image
import torch
import cv2

from structure.img import Hmage
from structure.grasp import GraspMat, drawGrasp1
import yaml
import mmcv
from collections import OrderedDict
from torch import nn
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from torchvision.ops import misc as misc_nn_ops
from model.detection import MaskRCNN
from model.detection.faster_rcnn import FastRCNNPredictor
from model.detection.mask_rcnn import MaskRCNNPredictor
from models import resnet
from model.detection.rpn import AnchorGenerator
from model.detection.image_list import ImageList
from model.detection.transform import GeneralizedRCNNTransform

import torch
import torchvision
from torch.nn import functional as F
import numpy as np

from models.backbone import get_backbone_with_fpn
from torch.hub import load_state_dict_from_url


def forward(self, images, targets=None):
    for i in range(len(images)):
        image = images[i]
        target = targets[i] if targets is not None else targets
        if image.dim() != 3:
            raise ValueError("images is expected to be a list of 3d tensors "
                             "of shape [C, H, W], got {}".format(image.shape))
        # image = self.normalize(image)
        # image, target = self.resize(image, target)
        images[i] = image
        if targets is not None:
            targets[i] = target
    image_sizes = [img.shape[-2:] for img in images]
    images = self.batch_images(images)
    image_list = ImageList(images, image_sizes)
    return image_list, targets


def maskrcnn_resnet_fpn(input_modality, fusion_method, backbone_name,
                        pretrained=False, progress=True, num_classes=2,
                        pretrained_backbone=True, trainable_backbone_layers=3, **kwargs):
    assert trainable_backbone_layers <= 5 and trainable_backbone_layers >= 0
    # dont freeze any layers if pretrained model or backbone is not used
    if not (pretrained or pretrained_backbone):
        trainable_backbone_layers = 5
    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    backbone = get_backbone_with_fpn(input_modality, fusion_method,
                                     backbone_name, pretrained_backbone,
                                     trainable_backbone_layers)
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512)),
                                       aspect_ratios=((0.25, 0.5, 1.0, 2.0)))
    model = MaskRCNN(backbone, 32, rpn_anchor_generator=anchor_generator, **kwargs)

    return model


def get_model_instance_segmentation(cfg):
    GeneralizedRCNNTransform.forward = forward

    # initialize with pretrained weights
    model = maskrcnn_resnet_fpn(input_modality=cfg["input_modality"],
                                fusion_method=cfg["fusion_method"],
                                backbone_name=cfg["backbone_name"],
                                pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    # do category-agnostic instance segmentation (object or background)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=32)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes=32)

    return model
class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=256, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo

class DeepLab(nn.Module):
    def __init__(self, cfg, angle_cls, backbone='resnet',  output_stride=16, num_classes=21, sync_bn=False,
                 freeze_bn=False, size=320):
        super(DeepLab, self).__init__()
        BatchNorm = nn.BatchNorm2d
        self.aff = AFF(channels=256,r=4)
        self.backbone = build_backbone(backbone)
        self.backbone_fpn = build_backbone_fpn(backbone_name='resnet50',trainable_layers=3, pretrained_backbone=False)
        self.hasp = build_hasp(BatchNorm)              # HASP
        self.decoder = build_decoder(num_classes, backbone, BatchNorm, size, angle_cls=angle_cls)
        self.model = get_model_instance_segmentation(cfg)
    def forward(self, input, input_image, sg_image):

        loss_dict, feature_1, x_1, feat_2= self.model(input_image, sg_image)
        feat_1 = self.backbone_fpn(input)

        feature_1 = torch.repeat_interleave(feature_1, 256, 1)

        x_all = self.hasp(x_1,feature_1)


        able_pred, angle_pred, width_pred = self.decoder(feat_1, x_all)
        able_pred = F.interpolate(able_pred, size=input.size()[2:], mode='bilinear', align_corners=True)  # 上采样（双线性插值）
        angle_pred = F.interpolate(angle_pred, size=input.size()[2:], mode='bilinear', align_corners=True)  # 上采样（双线性插值）
        width_pred = F.interpolate(width_pred, size=input.size()[2:], mode='bilinear', align_corners=True)  # 上采样（双线性插值）

        return able_pred, angle_pred, width_pred, loss_dict

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.hasp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p


# if __name__ == "__main__":
#     model = DeepLab(angle_cls=120, device='cuda', backbone='resnet', output_stride=16)
#     model.eval()
#     input = torch.rand(1, 6, 512, 512)
#     output = model(input)
#     print(output)


