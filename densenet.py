from math import floor
import torch
import torch.nn as nn

class TransitionLayer(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(TransitionLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.avg_pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.avg_pool(x)

        return x

class BasicLayer(nn.Module):
    def __init__(self, in_planes, planes):
        super(BasicLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, planes, kernel_size=3, 
                    stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv(out)

        out = torch.cat((x, out), 1)
        return out


class BottleneckLayer(nn.Module):
    expand = 4
    def __init__(self, in_planes, planes):
        super(BottleneckLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes * self.expand, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * self.expand)
        self.conv2 = nn.Conv2d(planes * self.expand, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = torch.cat((x, out), 1)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, blocks, growth_rate=32, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        in_planes = growth_rate * 2
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)

        features = []
        for repeat in blocks[:-1]:
            features += [self._make_layer(block, in_planes, repeat)]
            in_planes += self.growth_rate * repeat
            out_planes = int(floor(in_planes * reduction))
            features += [TransitionLayer(in_planes, out_planes)]
            in_planes = out_planes
        
        features += [self._make_layer(block, in_planes, blocks[-1])]
        self.features = nn.Sequential(*features)
        in_planes += growth_rate * blocks[-1]

        self.bn2 = nn.BatchNorm2d(in_planes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_planes, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.features(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

    
    def _make_layer(self, block, in_planes, repeat):
        layers = []
        for _ in range(repeat):
            layers += [block(in_planes, self.growth_rate)]
            in_planes += self.growth_rate
        
        return nn.Sequential(*layers)


def DenseNet121():
    return DenseNet(BottleneckLayer, [6,12,24,16], growth_rate=32)

def DenseNet169():
    return DenseNet(BottleneckLayer, [6,12,32,32], growth_rate=32)

def DenseNet201():
    return DenseNet(BottleneckLayer, [6,12,48,32], growth_rate=32)

def DenseNet161():
    return DenseNet(BottleneckLayer, [6,12,36,24], growth_rate=48)


