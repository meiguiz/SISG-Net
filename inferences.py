import argparse
import cv2
import os
import pathlib
import yaml
from tqdm import tqdm
import numpy as np
import torch
from models import maskrcnn
from loader import OCIDDataset, WISDOMDataset
from utils.visualizer import draw_prediction
from utils.saver import Saver
import cv2
import os
import torch
import math
from utils.affga import AFFGA
import torchvision.transforms as transforms
from utils.affga import input_rgb
from PIL import Image
from structure.grasp import GraspMat, drawGrasp1
if __name__ == '__main__':

    # load arguments and cfgurations
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0", help="gpu number to use. 0, 1")
    parser.add_argument("--cfg", type=str, default='rgb_only', help="file name of configuration file")
    parser.add_argument("--eval_data", default='synthetic', choices=['synthetic', 'wisdom'])
    parser.add_argument("--dataset_path", default='/media/meiguiz/HKTian/dataset/synthetic',type=str, help="path to dataset for evaluation")
    parser.add_argument("--num_sample", type=int, default=None, help="the number of data for drawing")
    parser.add_argument("--thresh", type=float, default=0.75)
    parser.add_argument("--vis_depth", action="store_true", help="draw RGB and depth together if true")
    parser.add_argument("--epoch_list", type=str, default=None, help="list of comma splited epochs to evaluate")
    parser.add_argument("--weight_path", type=str, default=None, help="if it is given, evaluate this")
    parser.add_argument('--outdir', type=str, default='output', help='Training Output Directory')
    parser.add_argument('--modeldir', type=str, default='models', help='model保存地址')
    parser.add_argument('--logdir', type=str, default='tensorboard', help='summary保存文件夹')
    parser.add_argument('--imgdir', type=str, default='img', help='中间预测图保存文件夹')
    args = parser.parse_args()


    def calcAngle2(angle):
        """
        根据给定的angle计算与之反向的angle
        :param angle: 弧度
        :return: 弧度
        """
        return angle + math.pi - int((angle + math.pi) // (2 * math.pi)) * 2 * math.pi


    def drawGrasps(img, grasps, mode):
        """
        绘制grasp
        file: img路径
        grasps: list()	元素是 [row, col, angle, width]
        mode: arrow / region
        """
        assert mode in ['arrow', 'region']

        num = len(grasps)
        for i, grasp in enumerate(grasps):
            row, col, angle, width = grasp

            if mode == 'arrow':
                anglePi = math.pi - angle
                # 计算角度
                cosA = math.cos(anglePi)
                sinA = math.sin(anglePi)
                x = col
                y = row
                height = 20
                width = width
                x1 = x - 0.5 * width
                y1 = y - 0.5 * height

                x0 = x + 0.5 * width
                y0 = y1

                x2 = x1
                y2 = y + 0.5 * height

                x3 = x0
                y3 = y2

                x0n = (x0 - x) * cosA - (y0 - y) * sinA + x
                y0n = (x0 - x) * sinA + (y0 - y) * cosA + y

                x1n = (x1 - x) * cosA - (y1 - y) * sinA + x
                y1n = (x1 - x) * sinA + (y1 - y) * cosA + y

                x2n = (x2 - x) * cosA - (y2 - y) * sinA + x
                y2n = (x2 - x) * sinA + (y2 - y) * cosA + y

                x3n = (x3 - x) * cosA - (y3 - y) * sinA + x
                y3n = (x3 - x) * sinA + (y3 - y) * cosA + y

                # 根据得到的点，画出矩形框
                # if angle < math.pi:
                #     cv2.arrowedLine(img, (col, row), (int(col + dx), int(row - dy)), (0, 0, 255), 1, 8, 0, 0.5)
                # else:
                #     cv2.arrowedLine(img, (col, row), (int(col - dx), int(row + dy)), (0, 0, 255), 1, 8, 0, 0.5)
                #
                # if angle2 < math.pi:
                #     cv2.line(img, (col, row), (int(col + dx), int(row - dy)), (0, 0, 255), 1)
                # else:
                #     cv2.line(img, (col, row), (int(col - dx), int(row + dy)), (0, 0, 255), 1)
                cv2.line(img, (int(x0n), int(y0n)), (int(x1n), int(y1n)), (0, 0, 255), 1, 4)
                cv2.line(img, (int(x1n), int(y1n)), (int(x2n), int(y2n)), (0, 255, 0), 2, 4)
                cv2.line(img, (int(x2n), int(y2n)), (int(x3n), int(y3n)), (0, 0, 255), 1, 4)
                cv2.line(img, (int(x0n), int(y0n)), (int(x3n), int(y3n)), (0, 255, 0), 2, 4)
                color_b = 255 / num * i
                color_r = 0
                color_g = -255 / num * i + 255
                cv2.circle(img, (col, row), 2, (color_b, color_g, color_r), -1)
                # cv2.putText(img, str(point)[:4], (int(x0n), int(y0n-5)),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        return img


    def drawRect(img, rect):
        """
        绘制矩形
        rect: [x1, y1, x2, y2]
        """
        cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 1)


    # load model

    # load dataset

    cpu_device = torch.device("cpu")

    model_g = '/media/meiguiz/HIKVISION/0619/output/models/epoch_0100'
    input_path = 'demo/input'
    depth_path = "demo/depth"
    target_path = "deomo/target"
    output_path = 'demo/output'
    #input_seg= '/media/robot/HKTian/SF-Mask-RCNN-main/logs/rgb_only/vis_result_synthetic'
    #device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    affga = AFFGA(model_g, device=device)
    # inference

    with torch.no_grad():

        for file in os.listdir(input_path):
            print('processing ', file)

            img_file = os.path.join(input_path, file)
            #seg_file = os.path.join(input_seg, file)
            depth_file = os.path.join(depth_path, file)
            rgb_name = img_file.replace('r.png', 'grasp.mat')
            grasplabel = GraspMat(os.path.join(rgb_name))
            grasplabel.rescale(0.8)
            grasplabel.decode(angle_cls=120)
            target = grasplabel.grasp
            target = torch.from_numpy(target.astype(np.float32))
            target = target.unsqueeze(0)
            img = cv2.imread(img_file)

            img = cv2.resize(img, (512, 384), interpolation=cv2.INTER_NEAREST)
            depth = cv2.imread(depth_file, -1)
            depth = cv2.resize(depth, (512, 384), interpolation=cv2.INTER_NEAREST)
            #seg_img = cv2.imread(seg_file)
        #     inv_normalize = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(
        #         mean=[0.485, 0.456, 0.406],
        #         std=[0.229, 0.224, 0.225]),
        # ])
        #     sgimg = img.astype(np.float32) / 255.0
        #     sgimg -= sgimg.mean()

            sg_img = input_rgb(img)
            #sgimg = img_1.unsqueeze(0)

            #img_sg = img.transpose((2, 0, 1))

            #sgimg = sgimg.transpose((2, 0, 1))
            #img_sg = sgimg.unsqueeze(-1).permute(3,0,1,2)

            image_sg = sg_img.to(device)
            image_sg = list(image.to(device) for image in image_sg)
            # image_isg = sg_img.to(device)
            # image_isg = list(image.to(device) for image in image_isg)

            grasps, pred = affga.predict(img, depth, image_sg, target, device, mode='peak', thresh=0.5, peak_dist=5)
            #outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            #able, angle, width = affga.maps(img, image_sg,device)
            #seg_image = draw_prediction(img, pred, args.thresh)# 预测
            im_rest = drawGrasps(img, grasps, mode='arrow')
             # 绘制预测结果
            #im_rest = draw_prediction(img_1, pred, args.thresh)

            save_file = os.path.join(output_path, file)
            cv2.imwrite(save_file, im_rest)
            # save_file = os.path.join(target_path, file)
            # cv2.imwrite(save_file, seg_image)
            # save_1 = os.path.join('/home/meiguiz/下载/AFFGA-Net-main (1)/demo/output_cornell/able', file)
            # cv2.imwrite(save_1, able)
            # save_2 = os.path.join('/home/meiguiz/下载/AFFGA-Net-main (1)/demo/output_cornell/angle', file)
            # cv2.imwrite(save_2, angle)
            # save_3 = os.path.join('/home/meiguiz/下载/AFFGA-Net-main (1)/demo/output_cornell/width', file)
            # cv2.imwrite(save_3, width)
    print('FPS: ', affga.fps())