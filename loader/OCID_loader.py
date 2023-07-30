from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
import os
import numpy as np
from PIL import Image
import torch
import cv2
from torchvision.transforms import functional as F
from structure.img import Hmage,Depth
from structure.grasp import GraspMat, drawGrasp1
import yaml
import mmcv
class OCIDDataset(Dataset):
    def __init__(self, output_size, angle_k, dataset_path, mode="train", argument=False,include_rgb=True, cfg=None, num_sample=None):
        self.mode = mode  # train mode or validation mode
        #self.rgb_path = os.path.join(dataset_path, "color_ims")
        self.depth_path = os.path.join(dataset_path, "depth")
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
        self.depth_list = ["{:03d}rd.png".format(idx) for idx in indices]
        self.seg_list = ["{:03d}rs.png".format(idx) for idx in indices]
        self.grasp_list = ["{:03d}grasp.mat".format(idx) for idx in indices]
        self.grgb_list = ["{:03d}r.png".format(idx) for idx in indices]
        if num_sample is not None:
            #self.rgb_list = self.rgb_list[:num_sample]
            self.depth_list = self.depth_list[:num_sample]
            self.seg_list = self.seg_list[:num_sample]
            self.grasp_list = self.grasp_list[:num_sample]
            self.grgb_list = self.grgb_list[:num_sample]
        #assert len(self.grgb_list) == len(self.depth_list)
        print(mode, ":", len(self.grgb_list), "images")

        self.input_modality = cfg["input_modality"]
        self.width = cfg["width"]
        self.height = cfg["height"]
        self.min_depth = 0.0
        self.max_depth = 1.0
        # self.rgb_transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(
        #         mean=[0.485, 0.456, 0.406],
        #         std=[0.229, 0.224, 0.225]),
        # ])

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
        #image = Image.open(os.path.join(self.grgb_path, self.grgb_list[idx])).convert("RGB")
        grasplabel = GraspMat(os.path.join(self.grasp_path, self.grasp_list[idx]))
        #image = image.resize((self.width, self.height))
        #image = self.rgb_transform(image)
        depth = Depth(os.path.join(self.depth_path, self.depth_list[idx]))
        # depth = np.load(os.path.join(self.depth_path, self.depth_list[idx])).astype(np.float32)
        # depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

        #image = torch.cat((image, depth), 0)
        #if self.argument:
        scale = 0.8
        grasplabel.rescale(scale)
        image.rescale(scale)
        depth.rescale(scale)
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
        #image.color()
        image.nomalise()
        image = image.img.transpose((2, 0, 1))  # (320, 320, 3) -> (3, 320, 320)
        # 获取target
        depth_out = depth.img
        img_depth = self.numpy_to_torch(depth_out)
        img_depth = torch.repeat_interleave(img_depth, 3, 0)
        img_rgb = self.numpy_to_torch(image)
        grasplabel.decode(angle_cls=self.angle_k)
        target = grasplabel.grasp  # (2 + angle_k, 320, 320)
        image= torch.cat((img_rgb, img_depth), dim=0)
        #image = self.numpy_to_torch(image)
        target = self.numpy_to_torch(target)



        seg_mask_path = os.path.join(self.seg_path, self.seg_list[idx])
        seg_mask = Image.open(seg_mask_path).convert("L")
        #seg_mask = seg_mask.resize((512, 384), Image.NEAREST )
        #seg_mask = np.array(seg_mask)

        seg_mask = seg_mask.resize((self.width, self.height), Image.NEAREST)
        seg_mask = np.array(seg_mask)
        #seg_mask = mmcv.imrescale(seg_mask, scale, interpolation='bilinear')
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



if __name__ == '__main__':
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
    image, target,  targett = train_dataset.__getitem__(1)
    print(image.shape)
    print(target.shape)
    print(targett['masks'].shape)