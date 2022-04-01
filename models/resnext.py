from turtle import forward
import torch
import torch.nn as nn

class GroupBlock(nn.Module):
    """
    Group convolution with bottleneck block.
    """
    expand = 2
    def __init__(self, planes, cardinality, width, stride, downsample=None):
        super(GroupBlock, self).__init__()

        out_planes = cardinality * width
        self.conv1 = nn.Conv2d(planes, out_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, 
                        groups=cardinality, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv3 = nn.Conv2d(out_planes, out_planes * self.expand, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes * self.expand)
        self.act = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.act(out)

        return out

class ResNeXt(nn.Module):
    def __init__(self, blocks, cardinality, width, num_classes=10):
        super().__init__()

        self.cardinality = cardinality
        self.width = width

        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)

        self.layer1 = self._make_layer(blocks[0], stride=1)
        self.layer2 = self._make_layer(blocks[1], stride=2)
        self.layer3 = self._make_layer(blocks[2], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_planes, num_classes)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _make_layer(self, repeat, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != GroupBlock.expand * self.cardinality * self.width:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, GroupBlock.expand * self.cardinality * self.width, 
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(GroupBlock.expand * self.cardinality * self.width)
            )
        
        layers = [GroupBlock(self.in_planes, self.cardinality, self.width, stride, downsample)]
        self.in_planes = GroupBlock.expand * self.cardinality * self.width

        for _ in range(1, repeat):
            layers += [GroupBlock(self.in_planes, self.cardinality, self.width, stride=1)]
        
        self.width *= 2

        return nn.Sequential(*layers)

def ResNeXt29_2x64d(num_classes):
    return ResNeXt(blocks=[3, 3, 3], cardinality=2, width=64, num_classes=num_classes)

def ResNeXt29_4x64d(num_classes):
    return ResNeXt(blocks=[3, 3, 3], cardinality=4, width=64, num_classes=num_classes)

def ResNeXt29_8x64d(num_classes):
    return ResNeXt(blocks=[3, 3, 3], cardinality=8, width=64, num_classes=num_classes)

def ResNeXt29_32x4d(num_classes):
    return ResNeXt(blocks=[3, 3, 3], cardinality=32, width=4, num_classes=num_classes)
