from models.mobilenetv1 import MobileNetV1
from models.densenet import DenseNet121, DenseNet161, DenseNet169, DenseNet201
from models.resnet import Resnet50, Resnet101, Resnet34, Resnet18, Resnet152
from models.vgg import VGG
# from models.preactresnet import Resnet18, Resnet34, Resnet50, Resnet101, Resnet152

import torch

x = torch.randn(1, 3, 32, 32)
net = MobileNetV1()
print(net(x).shape)