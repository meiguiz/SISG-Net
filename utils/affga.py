# -*- coding: utf-8 -*-
"""
@ Time ： 2020/3/2 11:33
@ Auth ： wangdx
@ File ：affga.py
@ IDE ：PyCharm
@ Function : 
"""

import cv2
import os
import torch
import time
from skimage.feature import peak_local_max
import numpy as np
from models.common import post_process_output
from models.loss import get_pred
import torchvision.transforms as transforms
from utils.data.evaluation import evaluation

def input_rgb(img):
    """
    对图像进行裁剪，保留中间(320, 320)的图像
    :param file: rgb文件
    :return: 直接输入网络的tensor, 裁剪区域的左上角坐标
    """

    # out_size_x = 640
    # out_size_y = 480
    #assert img.shape[0] >= out_size and img.shape[1] >= out_size, '输入的深度图必须大于等于(320, 320)'

    # 裁剪中间320*320作为输入
    # crop_x1 = int((img.shape[1] - out_size_x) / 2)
    # crop_y1 = int((img.shape[0] - out_size_y) / 2)
    # crop_x2 = crop_x1 + out_size_x
    # crop_y2 = crop_y1 + out_size_y
    # crop_rgb = img[crop_y1:crop_y2, crop_x1:crop_x2, :]

    # 归一化
    rgb = img.astype(np.float32) / 255.0
    rgb -= rgb.mean()

    #调整顺序，和网络输入一致
    rgb = rgb.transpose((2, 0, 1))  # (320, 320, 3) -> (3, 320, 320)
    rgb = torch.from_numpy(np.expand_dims(rgb, 0).astype(np.float32))  # np转tensor

    return rgb

def input_depth(depth):
    """
    对图像进行裁剪，保留中间(320, 320)的图像
    :param file: rgb文件
    :return: 直接输入网络的tensor, 裁剪区域的左上角坐标
    """

    # out_size = 480
    # #assert img.shape[0] >= out_size and img.shape[1] >= out_size, '输入的深度图必须大于等于(320, 320)'
    #
    # # 裁剪中间320*320作为输入
    # crop_x1 = int((img.shape[1] - out_size) / 2)
    # crop_y1 = int((img.shape[0] - out_size) / 2)
    # crop_x2 = crop_x1 + out_size
    # crop_y2 = crop_y1 + out_size
    # crop_depth = img[crop_y1:crop_y2, crop_x1:crop_x2]

    # 归一化
    #depth= crop_depth.astype(np.float32)

    img_depth = torch.from_numpy(np.expand_dims(depth, 0).astype(np.float32))

    img_depth = torch.repeat_interleave(img_depth, 3, 0)
    img_depth = torch.from_numpy(np.expand_dims(img_depth, 0).astype(np.float32)) # np转tensor

    return img_depth
def arg_thresh(array, thresh):
    """
    获取array中大于thresh的二维索引
    :param array: 二维array
    :param thresh: float阈值
    :return: array shape=(n, 2)
    """
    res = np.where(array > thresh)
    rows = np.reshape(res[0], (-1, 1))
    cols = np.reshape(res[1], (-1, 1))
    locs = np.hstack((rows, cols))
    for i in range(locs.shape[0]):
        for j in range(locs.shape[0])[i+1:]:
            if array[locs[i, 0], locs[i, 1]] < array[locs[j, 0], locs[j, 1]]:
                locs[[i, j], :] = locs[[j, i], :]

    return locs


class AFFGA:
    def __init__(self, model, device):
        self.t = 0
        self.num = 0
        # 加载模型
        print('>> loading AFFGA')
        self.device = device
        self.net = torch.load(model, map_location=torch.device(device))
        self.net.eval()
        print('>> load done')

    def fps(self):
        return 1.0 / (self.t / self.num)

    def predict(self, img, depth, image_sg, target, device, mode, thresh=0.5, peak_dist=1, angle_k=120):
        """
        预测抓取模型
        :param img: 输入图像 np.array (h, w, 3)
        :param thresh: 置信度阈值
        :param peak_dist: 置信度筛选峰值
        :param angle_k: 抓取角分类数
        :return:
            pred_grasps: list([row, col, angle, width])
            crop_x1
            crop_y1
        """
        # 预测
        results = {
            'correct': 0,
            'failed': 0,

        }
        rgb = input_rgb(img)
        depth = input_depth(depth)
        img = torch.cat((rgb,depth), dim=1)
        cpu_device = torch.device("cpu")


        t1 = time.time()
        # 预测
        self.able_out, self.angle_out, self.width_out, self.outputs = get_pred(self.net, img.to(device), image_sg)
        t2 = time.time() - t1
        # self.able = self.able_out.squeeze()
        # self.angle = self.angle_out.squeeze()
        # self.width = self.width_out.squeeze()


        outputs = self.outputs
        able_pred, angle_pred, width_pred = post_process_output(self.able_out, self.angle_out, self.width_out)
        #ret = evaluation(able_pred, angle_pred, width_pred, target, 120, 'max', desc='1')
        # if ret:
        #     print('true')
        # else:
        #     print('fail')
        outputs_1 = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        pred = outputs_1[0]
        # scores = pred["scores"].detach().cpu().numpy()
        # boxes = pred["boxes"].detach().cpu().numpy()
        # masks = pred["masks"].detach().cpu().numpy()
        # labels_s = pred["labels"].detach().cpu().numpy()
        # masks[masks >= 0.5] = 1
        # masks[masks < 0.5] = 0
        # cnd = scores[:] > 0.7
        # maskss = masks[cnd, :, :].squeeze()
        # mask = maskss[0]
        # masks_num = maskss.shape[0]
        # for i in range(1, masks_num):
        #     mask = mask + maskss[i]



        # 后处理
        # able_pred = able_pred * mask
        # width_pred = width_pred * mask
        if mode == 'peak':
            # 置信度峰值 抓取点
            pred_pts = peak_local_max(able_pred, min_distance=peak_dist, threshold_abs=thresh)
        elif mode == 'all':
            # 超过阈值的所有抓取点
            pred_pts = arg_thresh(able_pred, thresh=thresh)
        elif mode == 'max':
            # 置信度最大的点

            loc = np.argmax(able_pred)
            row = loc // able_pred.shape[1]
            col = loc % able_pred.shape[1]
            pred_pts = np.array([[row, col]])
        else:
            raise ValueError

        # 绘制预测的抓取
        pred_grasps = []
        for idx in range(pred_pts.shape[0]):
            row, col = pred_pts[idx]
            angle = angle_pred[row, col] / angle_k * 2 * np.pi  # 预测的抓取角弧度
            width = width_pred[row, col]
            # row += self.crop_y1
            # col += self.crop_x1

            pred_grasps.append([row, col, angle, width])

        self.t += t2
        self.num += 1

        return pred_grasps, pred


    def maps(self, img, image_sg,device):
        """绘制最终的特征图"""
        # 抓取置信度
        rgb = input_rgb(img)

        self.net.eval()
        with torch.no_grad():
            self.able_out, self.angle_out, self.width_out, self.outputs= get_pred(self.net, rgb.to(device),image_sg)

        able_map = self.able_out.detach().cpu().numpy().squeeze()
        able_featureMap = np.zeros((able_map.shape[0], able_map.shape[1], 3), dtype=np.float)
        """
        R: 0    ->    255
        G: 0    ->    0
        B: 255  ->    0
        """
        able_featureMap[:, :, 2] = able_map * 255.0
        able_featureMap[:, :, 1] = 0
        able_featureMap[:, :, 0] = able_map * -255.0 + 255.0
        able_featureMap = able_featureMap.astype(np.uint8)

        # 抓取角
        angle_map = self.angle_out.detach().cpu().numpy().squeeze()[0]
        angle_map = angle_map.copy() * (255.0 / angle_map.max())
        angle_featureMap = np.zeros((angle_map.shape[0], angle_map.shape[1], 3), dtype=np.float)
        angle_featureMap[:, :, 2] = 100
        angle_featureMap[:, :, 0] = 100
        angle_featureMap[:, :, 1] = angle_map * -1 + 255
        angle_featureMap = angle_featureMap.astype(np.uint8)

        # 抓取宽度
        width_map = self.width_out.detach().cpu().numpy().squeeze()
        width_featureMap = np.zeros((width_map.shape[0], width_map.shape[1], 3), dtype=np.float)
        width_featureMap[:, :, 2] = width_map * 800.0
        width_featureMap[:, :, 1] = 255
        width_featureMap[:, :, 0] = 255
        width_featureMap = width_featureMap.astype(np.uint8)

        return able_featureMap, angle_featureMap, width_featureMap
