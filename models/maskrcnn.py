from collections import OrderedDict

import yaml
from torch import nn
#from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
#from torchvision.ops import misc as misc_nn_ops
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from models import resnet
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.transform import GeneralizedRCNNTransform

import torch
import torchvision
from torch.nn import functional as F
import numpy as np

from .backbone import get_backbone_with_fpn
from torch.hub import load_state_dict_from_url
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
import os
import numpy as np
from PIL import Image
import torch
import cv2
from torchvision.transforms import functional as F
from structure.img import Hmage
from structure.grasp import GraspMat, drawGrasp1
import yaml
import mmcv
class OCIDDataset(Dataset):
    def __init__(self, output_size, angle_k, dataset_path, mode="train", argument=False,include_rgb=True, cfg=None, num_sample=None):
        self.mode = mode  # train mode or validation mode
        #self.rgb_path = os.path.join(dataset_path, "color_ims")
        self.depth_path = os.path.join(dataset_path, "depth_ims_numpy")
        self.seg_path = os.path.join(dataset_path, "modal_segmasks")
        self.grasp_path = os.path.join(dataset_path, "grasp_anno")
        self.grgb_path = os.path.join(dataset_path, "grgb_ims")
        self.output_size = output_size
        self.include_rgb = include_rgb
        self.angle_k = angle_k
        self.argument = argument
        # Read indices.npy
        mode = "test" if mode == "val" else mode
        indice_file = os.path.join(dataset_path, "{}_indices.npy".format(mode))
        indices = np.load(indice_file)
        #self.rgb_list = ["{:03d}r.png".format(idx) for idx in indices]
        self.depth_list = ["{:03d}rd.png.npy".format(idx) for idx in indices]
        self.seg_list = ["{:03d}rs.png".format(idx) for idx in indices]
        self.grasp_list = ["{:03d}grasp.mat".format(idx) for idx in indices]
        self.grgb_list = ["{:03d}r.png".format(idx) for idx in indices]
        if num_sample is not None:
            #self.rgb_list = self.rgb_list[:num_sample]
            self.depth_list = self.depth_list[:num_sample]
            self.seg_list = self.seg_list[:num_sample]
            self.grasp_list = self.grasp_list[:num_sample]
            self.grgb_list = self.grgb_list[:num_sample]
        assert len(self.grgb_list) == len(self.depth_list)
        print(mode, ":", len(self.grgb_list), "images")

        self.input_modality = cfg["input_modality"]
        self.width = cfg["width"]
        self.height = cfg["height"]
        self.min_depth = 0.0
        self.max_depth = 1.0

    @staticmethod
    def numpy_to_torch(s):
        """
        numpy转tensor
        """
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def __len__(self):
        return len(self.grgb_list)

    def __getitem__(self, idx):
        inputs = dict.fromkeys(["rgb", "depth", "val_mask"])
        image = Hmage(os.path.join(self.grgb_path, self.grgb_list[idx]))
        grasplabel = GraspMat(os.path.join(self.grasp_path, self.grasp_list[idx]))

        # if self.argument:
        scale = 0.8
        image.rescale(scale)
        grasplabel.rescale(scale)
        #     # crop_bbox = image.crop(self.output_size)
        #     # grasplabel.crop(crop_bbox)
        #     rota = 30
        #     rota = np.random.uniform(-1 * rota, rota)
        #     image.rotate(rota)
        #     grasplabel.rotate(rota)
        #     flip = True if np.random.rand() < 0.5 else False
        #     if flip:
        #         image.flip()
        #         grasplabel.flip()
        # color
        # image.color()
        image = image.nomalise(image)
        image = image.img.transpose((2, 0, 1))  # (320, 320, 3) -> (3, 320, 320)
        grasplabel.decode(angle_cls=self.angle_k)
        target = grasplabel.grasp  # (2 + angle_k, 320, 320)

        image = self.numpy_to_torch(image)
        target = self.numpy_to_torch(target)

        seg_mask_path = os.path.join(self.seg_path, self.seg_list[idx])
        seg_mask = Image.open(seg_mask_path).convert("L")

        seg_mask = np.array(seg_mask)
        seg_mask = mmcv.imrescale(seg_mask, scale, interpolation='bilinear')
        # instances are encoded as different colors
        obj_ids = np.unique(seg_mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        # split the color-encoded mask into a set
        # of binary masks
        seg_masks = seg_mask == obj_ids[:, None, None]
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        temp_obj_ids = []
        temp_masks = []
        boxes = []
        for i in range(num_objs):
            pos = np.where(seg_masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            if int(xmax - xmin) < 1 or int(ymax - ymin) < 1:
                continue
            temp_masks.append(seg_masks[i])
            temp_obj_ids.append(obj_ids[i])
            boxes.append([xmin, ymin, xmax, ymax])
        obj_ids = temp_obj_ids
        seg_masks = np.asarray(temp_masks)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = []
        for obj_id in obj_ids:
            if 1 <= obj_id:
                labels.append(obj_id)
            else:
                print("miss value error")
                exit(0)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        seg_masks = torch.as_tensor(seg_masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        targett = {}
        targett["boxes"] = boxes
        targett["labels"] = labels
        targett["masks"] = seg_masks
        targett["image_id"] = image_id
        targett["area"] = area
        targett["iscrowd"] = iscrowd
        return image, target, targett
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
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    cpu_device = torch.device("cpu")
    angle_k = 120
    output_size = 320
    dataset_path = '/media/meiguiz/HKTian/dataset/synthetic'
    with open('/media/meiguiz/HKTian/0619/cfgs/rgb_only.yaml') as cfg_file:
        cfg = yaml.safe_load(cfg_file)
    train_dataset = OCIDDataset(angle_k=angle_k, output_size=output_size, dataset_path=dataset_path, cfg=cfg)
    train_data = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
    )
    image, target, targett = train_dataset.__getitem__(2)
    image = image.unsqueeze(0)
    image = image.to(device)
    targett = [{k: v.to(device) for k, v in targett.items()}]
    model = get_model_instance_segmentation(cfg=cfg)
    model.to(device)
    model.eval()
    #input = torch.rand(1,3,512,512)
    outputs = model(image)
    outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
    print(image[0].shape)
    print(image[0])
    print(outputs[0])