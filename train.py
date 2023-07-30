import os
import yaml
import pathlib
import pprint
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import datetime
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
from tensorboardX import SummaryWriter


from models import maskrcnn
from loader import OCIDDataset, WISDOMDataset
from utils import visualizer
from utils.engine import train_one_epoch, evaluate, collate_fn, test_one_epoch
from utils.coco_utils import get_coco_api_from_dataset
from utils.data.evaluation import evaluation
from utils.data import get_dataset
from utils.saver import Saver
from models import get_network
from models.common import post_process_output

from models.loss import compute_loss

if __name__ == '__main__':

    # load arguments and cfgurations
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0", help="gpu number to use. 0, 1")
    parser.add_argument("--cfg", type=str, default='rgb_depth', help="file name of configuration file")
    parser.add_argument("--resume", action="store_true", help="resume training")
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument('--outdir', type=str, default='output', help='Training Output Directory')
    parser.add_argument('--modeldir', type=str, default='models', help='model保存地址')
    parser.add_argument('--logdir', type=str, default='tensorboard', help='summary保存文件夹')
    parser.add_argument('--imgdir', type=str, default='img', help='中间预测图保存文件夹')
    parser.add_argument('--lr', type=float, default=1e-3, help='初始学习率')
    parser.add_argument('--lr-scheduler', type=str, default='poly', help='学习率衰减模式')
    parser.add_argument('--device-name', type=str, default='cuda:0', choices=['cpu', 'cuda:0', 'cuda:1'],
                        help='是否使用GPU')
    args = parser.parse_args()
    with open('cfgs/' + args.cfg + '.yaml') as cfg_file:
        cfg = yaml.safe_load(cfg_file)
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(cfg)


    def validate(epoch, net, device, val_data, saver, args):
        """
        Run validation.
        :param net: 网络
        :param device:
        :param val_data: 验证数据集
        :param saver: 保存器
        :param args:
        :return: Successes, Failures and Losses
        """
        net.eval()

        results = {
            'correct': 0,
            'failed': 0,
            'loss': 0,
            'graspable': 0,
            'fail': [],
            'losses': {
            }
        }

        ld = len(val_data)

        with torch.no_grad():  # 不计算梯度，不反向传播
            batch_idx = 0
            for x, y, z in val_data:
                batch_idx += 1
                print("\r Validating... {:.2f}".format(batch_idx / ld), end="")

                lossd = compute_loss(net, x.to(device), y.to(device), device)

                # 统计损失
                loss = lossd['loss']  # 损失和
                results['loss'] += loss.item() / ld  # 损失累加
                for ln, l in lossd['losses'].items():  # 添加单项损失
                    if ln not in results['losses']:
                        results['losses'][ln] = 0
                    results['losses'][ln] += l.item() / ld

                # 输出值预处理
                able_out, angle_out, width_out = post_process_output(lossd['pred']['able'], lossd['pred']['angle'],
                                                                     lossd['pred']['width'])

                # 保存预测图

                # saver.save_img(epoch, batch_idx, able_out)
                # saver.save_angle(epoch, batch_idx, angle_out)
                # saver.save_with(epoch, batch_idx, width_out)
                # 评估
                results['graspable'] += np.max(able_out) / ld

                ret = evaluation(able_out, angle_out, width_out, y, 120, 'max', desc='1')
                if ret:
                    results['correct'] += 1
                else:
                    results['failed'] += 1
                    results['fail'].append(batch_idx)

        return results
    # fix seed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True    
    torch.manual_seed(7777)

    # load dataset
    if cfg["dataset"] == 'synthetic':
        train_dataset = OCIDDataset(dataset_path=cfg["dataset_path"], mode="train", angle_k=120, output_size=320, cfg=cfg)
        val_dataset = OCIDDataset(dataset_path=cfg["dataset_path"], mode="val", angle_k=120, output_size=320, cfg=cfg)
    elif cfg["dataset"] == 'wisdom':
        train_dataset = WISDOMDataset(dataset_path=cfg["dataset_path"], mode="train", cfg=cfg)
    else:
        raise ValueError("Unsupported dataset type {} in your config file {}".format(cfg["dataset"], args.cfg))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=cfg["batch_size"], 
                                               num_workers=8, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=2, num_workers=8,
                                             shuffle=False, collate_fn=collate_fn)
    # logging
    now = datetime.datetime.now()
    logging_folder = os.path.join(pathlib.Path(__file__).parent.absolute(), 'logs', args.cfg )
    os.makedirs(logging_folder, exist_ok=True)
    summary = SummaryWriter(logdir=logging_folder)
    #visualizer.draw_sample_images(train_dataset, logging_folder, cfg, "train")

    # load model
    #model = maskrcnn.get_model_instance_segmentation(cfg=cfg)
    affga = get_network()


    device = args.device_name if torch.cuda.is_available() else "cpu"
    net = affga(cfg=cfg, angle_cls=120)
    print(net)
    print("Using", device, args.gpu)
    #model.to(device)
    net = net.to(device)

    saver = Saver(args.outdir, args.logdir, args.modeldir, args.imgdir)
    start_epoch = 0

    # construct an optimizer
    #params = [p for p in model.parameters() if p.requires_grad]
    #optimizer = torch.optim.Adam(params, lr=cfg["lr"], weight_decay=cfg["wd"])
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=cfg["wd"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 300, 320], gamma=0.5)
    print("Start training")
    for epoch in range(start_epoch, cfg["max_epoch"]):
        # train_one_epoch(model, optimizer, train_loader, device, epoch, 100, summary)
        train_one_epoch(net, optimizer, train_loader, device, epoch, 3, summary)
        scheduler.step()
        #test_results = test_one_epoch(net,val_loader,device,epoch,3)
        # print('>>> test_graspable = {:.5f}'.format(test_results['graspable']))
        # print(
        #     '>>> test_acc: %d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
        #                                   test_results['correct'] / (
        #                                           test_results['correct'] + test_results['failed'])))
        # print('>>> pred fail idx：', test_results['fail'])
        #lr_scheduler.step()

        if epoch % args.save_interval == 0:
            #torch.save(model.state_dict(), '{}/{}.tar'.format(logging_folder, epoch))
            saver.save_model(net, 'epoch_%04d' % (epoch) )
