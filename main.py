from mobilenetv1 import MobileNetV1
import torch

net = MobileNetV1()
x = torch.randn((1, 3, 224, 224))
print(net(x).shape)
