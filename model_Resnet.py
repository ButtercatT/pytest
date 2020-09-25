import torch.nn as nn
import torch


# class 1: basic mode of residual neural( used in low dimensions input)
# contains 2 layers in each neural
class BasicBlock(nn.Module):
    expansion = 1  # this para is used to squeeze the output dimensions of the data ,not used in BasicBlock

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # we usually don't use bias in Resnet for it does not make sense
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False)
        # conv1 might be used to squeeze the size of input(eg. 112*112->56)
        self.bn1 = nn.BatchNorm2d(out_channel)  # do a batch normalization
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1, bias=False)
        # conv2 don't need to accept a changeable parameter of stride.
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)

        return out


# this neural class is used to dealing with the high dimensions input
# contains 3 layers in a neural
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # the first layer is designed to squeeze the dimensions
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)  #
        # function of stride here is the same as what performs in BasicBlock
        self.bn2 = nn.BatchNorm2d(out_channel)

        # the third layer is designed to unsqueeze the dimensions
        self.conv3 = nn.Conv2d(out_channel, out_channel * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


# combine above two residual neural,program the ResNet
# todo:this resnet is only for color images
class ResNet(nn.Module):
    def __init__(self, block, blocks_num, classes_num=100, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channels = 64  # record the channels of next input data

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, blocks_num[0], stride=1)
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, classes_num)
        # this part aim to reset
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, blocks_num, stride=1):
        downsample = None  # residual system
        # if the size of residual does not match the size of the output int this layer ,we need to reshape it also,
        # if stride didn't equal to 1 ,that means the data size(input) will be change ,
        # so the residual needs to be change too
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channel * block.expansion, kernel_size=1, bias=False, stride=stride),
                nn.BatchNorm2d(channel * block.expansion)
            )
        layer = []
        layer.append(nn.Sequential(block(self.in_channels, channel, stride, downsample=downsample)))
        # only the first time we should consider the channels of data
        self.in_channels = block.expansion * channel
        for i in range(1, blocks_num):
            layer.append(block(self.in_channels, channel))
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet34(num_classes=100, include_top=True):
    """
    residual network model using 34 layers and BasicBlock
    :param num_classes: aim classes_num
    :param include_top: if contains full-forward net or not
    :return: model
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, include_top)


def resnet101(num_classes=1000, include_top=True):
    """
        residual network model using 101 layers and Bottleneck
        :param num_classes: aim classes_num
        :param include_top: if contains full-forward net or not
        :return: model
        """
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, include_top=include_top)
