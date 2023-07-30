import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from models.affga.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d



class Bottleneck(nn.Module):
    """瓶颈残差块"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, output_stride, batchNorm, device, pretrained=True):
        """
        :param block:   Bottleneck
        :param layers:  [3, 4, 23, 3]
        :param output_stride:   16
        :param BatchNorm:   nn.BatchNorm2d
        :param pretrained:  True
        """
        self.device = device

        self.inplanes = 64
        super(ResNet, self).__init__()
        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 2]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], BatchNorm=nn.BatchNorm2d)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], BatchNorm=nn.BatchNorm2d)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], BatchNorm=nn.BatchNorm2d)
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], BatchNorm=nn.BatchNorm2d)
        # self.groups = 8
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.weight = nn.Parameter(torch.zeros(1, 8, 1, 1))
        # self.bias = nn.Parameter(torch.zeros(1, 8, 1, 1))
        # self.sig = nn.Sigmoid()
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], BatchNorm=nn.BatchNorm2d)
        self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0]*dilation,
                            downsample=downsample, BatchNorm=BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i]*dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        #print(input.size())
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        #feat_1 = x
        # b, c, h, w = x.shape
        # x = x.view(b * self.groups, -1, h, w)  # bs*g,dim//g,h,w
        # xn = x * self.avg_pool(x)  # bs*g,dim//g,h,w
        # xn = xn.sum(dim=1, keepdim=True)  # bs*g,1,h,w
        # t = xn.view(b * self.groups, -1)  # bs*g,h*w
        #
        # t = t - t.mean(dim=1, keepdim=True)  # bs*g,h*w
        # std = t.std(dim=1, keepdim=True) + 1e-5
        # t = t / std  # bs*g,h*w
        # t = t.view(b, self.groups, h, w)  # bs,g,h*w
        #
        # t = t * self.weight + self.bias  # bs,g,h*w
        # t = t.view(b * self.groups, 1, h, w)  # bs*g,1,h*w
        # x = x * self.sig(t)
        # x = x.view(b, c, h, w)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)



def ResNet50(output_stride, device, BatchNorm, pretrained=True):
    """Constructs a ResNet-101 model.
    Args:
        output_stride: 16
        BatchNorm: nn.BatchNorm2d
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], output_stride, BatchNorm, device, pretrained=pretrained)
    return model


if __name__ == "__main__":
    import torch
    model = ResNet50(BatchNorm=nn.BatchNorm2d, pretrained=True, output_stride=16,device="cuda")
    input = torch.rand(1, 3, 640, 480)
    output = model(input)
          # [1, 256, 80, 80]
    print(output.size())                # [1, 2048, 20, 20]
