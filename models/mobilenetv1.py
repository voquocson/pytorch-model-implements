import torch
import torch.nn as nn

class DepthwiseConv(nn.Module):
    def __init__(self, planes, kernel_size, stride, padding):
        super(DepthwiseConv, self).__init__()
        self.conv = nn.Conv2d(planes, planes, kernel_size=kernel_size,
                    stride=stride, padding=padding, groups=planes, bias=False)
        self.norm = nn.BatchNorm2d(planes)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.act(self.norm(self.conv(x)))
        return out



class MobileNetV1(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetV1, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU(inplace=True)

        self.features = nn.Sequential(
            self._depth_separable_conv(32, 64, 1),
            self._depth_separable_conv(64, 128, 2),
            self._depth_separable_conv(128, 128, 1),
            self._depth_separable_conv(128, 256, 2),
            self._depth_separable_conv(256, 256, 1),
            self._depth_separable_conv(256, 512, 2),
            self._depth_separable_conv(512, 512, 1),
            self._depth_separable_conv(512, 512, 1),
            self._depth_separable_conv(512, 512, 1),
            self._depth_separable_conv(512, 512, 1),
            self._depth_separable_conv(512, 512, 1),
            self._depth_separable_conv(512, 1024, 2),
            self._depth_separable_conv(1024, 1024, 1)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.features(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

    
    def _depth_separable_conv(self, in_planes, out_planes, stride):
        ds_conv = nn.Sequential(
            DepthwiseConv(in_planes, kernel_size=3, stride=stride, padding=1),
            nn.Conv2d(in_planes, out_planes, kernel_size=1,
                        stride=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )

        return ds_conv