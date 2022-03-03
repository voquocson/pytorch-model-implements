import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """
    Pre-activation basic block in Pre-Activation Resnet.
    """
    expand = 1
    def __init__(self, in_planes, out_planes, stride, downsample=None):
        super(BasicBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        
        out = self.conv1(self.act(self.bn1(x)))
        out = self.conv2(self.act(self.bn2(out)))

        if self.downsample is not None:
            identity = self.downsample(identity)
    
        out += identity
        return out

class BottleneckBlock(nn.Module):
    """
    Pre-activation bottleneck block in Pre-Activation Resnet.
    """
    expand = 4
    def __init__(self, in_planes, out_planes, stride, downsample=None):
        super(BottleneckBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.conv3 = nn.Conv2d(out_planes, out_planes * self.expand, kernel_size=1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(self.act(self.bn1(x)))
        out = self.conv2(self.act(self.bn2(out)))
        out = self.conv3(self.act(self.bn3(out)))

        if self.downsample is not None:
            identity = self.downsample(identity)
        
        out += identity
        return out

class Resnet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(Resnet, self).__init__()

        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0]) # 16
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) # 8
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) # 4
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2) # 2

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # global average pooling
        self.fc = nn.Linear(512 * block.expand, num_classes)

        self._initialize_weights()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expand:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expand, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * block.expand)
            )

        layers = [block(self.in_planes, planes, stride=stride, downsample=downsample)]

        self.in_planes = planes * block.expand
        for _ in range(blocks - 1):
            layers += [block(self.in_planes, planes, stride=1)]
        
        return nn.Sequential(*layers)
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def Resnet18(num_classes=10):
    return Resnet(BasicBlock, [2, 2, 2, 2], num_classes)

def Resnet34(num_classes):
    return Resnet(BasicBlock, [3, 4, 6, 3], num_classes)

def Resnet50(num_classes):
    return Resnet(BottleneckBlock, [3, 4, 6, 3], num_classes)

def Resnet101(num_classes):
    return Resnet(BottleneckBlock, [3, 4, 23, 3], num_classes)

def Resnet152(num_classes):
    return Resnet(BottleneckBlock, [3, 8, 36, 3], num_classes)