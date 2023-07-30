from models.affga.fpn import FPN


def build_backbone_fpn(backbone_name,
                          trainable_layers, pretrained_backbone=True, extra_blocks=None, returned_layers=None):
    return FPN.get_backbone_with_fpn(backbone_name='resnet50',
                                  trainable_layers=3, pretrained_backbone=False)