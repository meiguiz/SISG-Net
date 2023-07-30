from models.affga.backbone import xception, drn, mobilenet,convnext
from models.resnet import resnet50,resnet34

def build_backbone(backbone):

    if backbone == 'resnet':
        return resnet50()
    # elif backbone == 'convnext':
    #     return convnext.convnext_base()
    # elif backbone == 'xception':
    #     return xception.AlignedXception(output_stride, BatchNorm)
    # elif backbone == 'drn':
    #     return drn.drn_d_54(BatchNorm)
    # elif backbone == 'mobilenet':
    #     return mobilenet.MobileNetV2(output_stride, BatchNorm)
    # else:
    #     raise NotImplementedError
