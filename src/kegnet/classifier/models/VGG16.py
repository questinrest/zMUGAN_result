from torch import nn
from torch.nn import functional as func


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes, num_channels = 3,  return_activations = False):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name], channels = num_channels)
        self.classifier = nn.Linear(512, num_classes)
        self.return_activations = return_activations

    def forward(self, x):      
        if not self.return_activations:
            out = self.features(x)
            out = out.view(out.size(0), -1)
            out = self.classifier(out)
            return out

        activation_list = []
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.Conv2d) and (x.numel() > 0):
                activation_list.append(x)        
        
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        activation_list.append(x)
        return x, activation_list

    def _make_layers(self, cfg, channels=3):
        layers = []
        in_channels = channels
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
    
    def compress_blocks(self, block):
        pass

    def compress(self, option):
        pass


def VGG16(num_classes = 10,num_channels = 3, return_activations = False):
    return VGG('VGG16',num_classes=num_classes, return_activations=return_activations, num_channels=num_channels)