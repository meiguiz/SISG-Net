import numpy as np
import cv2
from utils.data import get_dataset
import torch
import torch.utils.data
import math
import random
import os
import copy
from PIL import Image
from utils.data.structure.img import Hmage
from utils.data.structure.grasp import GraspMat, drawGrasp1
#from .transforms import AddGaussianNoise, RandomErasing, SaltPepperNoise
import torchvision.transforms as transforms


class GraspDatasetBase(torch.utils.data.Dataset):
    def __init__(self, angle_k,output_size,include_depth=False, include_rgb=True,
                 argument=False):
        """
        :param output_size: int 输入网络的图像尺寸
        :param angle_k: 抓取角的分类数
        :param include_depth: 网络输入是否包括深度图
        :param include_rgb: 网络输入是否包括RGB图
        :param random_rotate: 是否随机旋转
        :param random_zoom: 是否随机缩放      # 后期可加是否随机平移
        """

        self.output_size = output_size
        self.include_depth = include_depth
        self.include_rgb = include_rgb
        self.angle_k = angle_k
        self.argument = argument


        self.input_modality = 'rgb'
        self.width = 320
        self.height = 320

        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
            ])
    @staticmethod
    def numpy_to_torch(s):
        """
        numpy转tensor
        """
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def __getitem__(self, idx):
        # 读取img和标签
        label_name = self.grasp_files[idx]
        rgb_name = label_name.replace('grasp.mat', 'r.png')
        seg_name = label_name.replace('grasp.mat', 'rs.png')
        image = Hmage(rgb_name)
        label = GraspMat(label_name)
        inputs = dict.fromkeys(["rgb", "depth", "val_mask"])
        if 'rgb' in self.input_modality:
            rgb = Image.open(rgb_name)
            rgb = rgb.resize((self.width, self.height))
            inputs["rgb"] = self.rgb_transform(rgb)
        img_s = inputs["rgb"]

        seg_mask = Image.open(seg_name).convert("L")
        seg_mask = seg_mask.resize((self.width, self.height), Image.NEAREST)
        seg_mask = np.array(seg_mask)
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
        target_s = {}
        target_s["boxes"] = boxes
        target_s["labels"] = labels
        target_s["masks"] = seg_masks
        target_s["image_id"] = image_id
        target_s["area"] = area
        target_s["iscrowd"] = iscrowd
        # 数据增强
        if self.argument:
            # resize
            scale = np.random.uniform(0.9, 1.1)
            image.rescale(scale)
            label.rescale(scale)
            # rotate
            rota = 30
            rota = np.random.uniform(-1 * rota, rota)
            image.rotate(rota)
            label.rotate(rota)
            # crop
            dist = 30  # 50
            x_offset = np.random.randint(-1 * dist, dist)
            y_offset = np.random.randint(-1 * dist, dist)
            crop_bbox = image.crop(self.output_size, x_offset, y_offset)
            label.crop(crop_bbox)
            # flip
            flip = True if np.random.rand() < 0.5 else False
            if flip:
                image.flip()
                label.flip()
            # color
            image.color()
        else:
            # crop
            dist = 30  # 50
            x_offset = np.random.randint(-1 * dist, dist)
            y_offset = np.random.randint(-1 * dist, dist)
            crop_bbox = image.crop(self.output_size, x_offset, y_offset)
            label.crop(crop_bbox)

        # img归一化
        image.nomalise()
        img = image.img.transpose((2, 0, 1))  # (320, 320, 3) -> (3, 320, 320)
        # 获取target
        label.decode(angle_cls=self.angle_k)
        target = label.grasp    # (2 + angle_k, 320, 320)

        img = self.numpy_to_torch(img)

        target = self.numpy_to_torch(target)
        img = torch.cat((img, img_s), 0)
        return img, target, target_s


    def __len__(self):
        return len(self.grasp_files)


if __name__ == '__main__':
    angle_cls = 120
    dataset = 'cornell'
    dataset_path = '/media/meiguiz/HKTian/dataset/cornell_clutter/'
    # 加载训练集
    print('Loading Dataset...')
    Dataset = get_dataset()
    train_dataset = Dataset(dataset_path,



                            test_mode='all-wise',
                            data='train',
                            data_list='train-test-all',
                            argument=True,
                            output_size=320,
                            angle_k=angle_cls)
    train_data = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=6)

    print('>> dataset: {}'.format(len(train_data)))

    count = 0
    max_w = 0
    for x, y, z in train_data:
        print(x.shape)
        print(y.shape)
