from matplotlib.pyplot import isinteractive
import torch
import torch.nn as nn

cfgs = {
    '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, cfg, in_planes=3, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()

        self.features = self._make_layer(cfg, in_planes)
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

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
    
    def _make_layer(self, cfg, in_planes=3):
        in_planes = in_planes
        layers = []
        for layer in cfg:
            if layer == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                out_planes = int(layer)
                layers += [
                    nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_planes),
                    nn.ReLU(inplace=True)
                ]
                in_planes = out_planes
        
        return nn.Sequential(*layers)

x = torch.randn((1, 3, 224, 224))
vgg = VGG(cfgs['16'])
print(vgg(x).shape)