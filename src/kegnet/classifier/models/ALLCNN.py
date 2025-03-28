import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self,x):
        return x.view(x.size(0), -1)

#class ConvStandard(nn.Conv2d):

    #def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, output_padding=0, w_sig =\
                 #np.sqrt(1.0)):
        #super(ConvStandard, self).__init__(in_channels, out_channels,kernel_size)
        #self.in_channels=in_channels
        #self.out_channels=out_channels
        #self.kernel_size=kernel_size
        #self.stride=stride
        #self.padding=padding
        #self.w_sig = w_sig
       # self.reset_parameters()

    #def reset_parameters(self):
        #torch.nn.init.normal_(self.weight, mean=0, std=self.w_sig/(self.in_channels*np.prod(self.kernel_size)))
        #if self.bias is not None:
           # torch.nn.init.normal_(self.bias, mean=0, std=0)

    #def forward(self, input):
        #return F.conv2d(input,self.weight,self.bias,self.stride,self.padding)

class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, output_padding=0,
                 activation_fn=nn.ReLU, batch_norm=True, transpose=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        model = []
        if not transpose:
#             model += [ConvStandard(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
#                                 )]
            model += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                bias=not batch_norm)]
        else:
            model += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                         output_padding=output_padding, bias=not batch_norm)]
        if batch_norm:
            model += [nn.BatchNorm2d(out_channels, affine=True)]
        model += [activation_fn()]
        super(Conv, self).__init__(*model)

class AllCNN(nn.Module):
    def __init__(self, filters_percentage=1., n_channels=3 , num_classes=10, dropout=False, batch_norm=True):
        super(AllCNN, self).__init__()
        n_filter1 = int(96 * filters_percentage)
        n_filter2 = int(192 * filters_percentage)

        self.conv1 = Conv(n_channels, n_filter1, kernel_size=3, batch_norm=batch_norm)
        self.conv2 = Conv(n_filter1, n_filter1, kernel_size=3, batch_norm=batch_norm)
        self.conv3 = Conv(n_filter1, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm)

        self.dropout1 = nn.Sequential(nn.Dropout(inplace=False) if dropout else Identity())

        self.conv4 = Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv5 = Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv6 = Conv(n_filter2, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm)

        self.dropout2  = nn.Sequential(nn.Dropout(inplace=False) if dropout else Identity()) #self.features

        self.conv7 = Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv8 = Conv(n_filter2, n_filter2, kernel_size=1, stride=1, batch_norm=batch_norm)
        if n_channels == 3:
            self.pool = nn.AvgPool2d(8)
        elif n_channels == 1:
            self.pool = nn.AvgPool2d(7)
        self.flatten = Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(n_filter2, num_classes),
        )

    def forward(self, x):
        out = self.conv1(x)
        actv1 = out

        out = self.conv2(out)
        actv2 = out

        out = self.conv3(out)
        actv3 = out

        out = self.dropout1(out)

        out = self.conv4(out)
        actv4 = out

        out = self.conv5(out)
        actv5 = out

        out = self.conv6(out)
        actv6 = out

        out = self.dropout2(out)

        out = self.conv7(out)
        actv7 = out

        out = self.conv8(out)
        actv8 = out

        out = self.pool(out)

        out = self.flatten(out)

        out = self.classifier(out)

        return out, actv1, actv2, actv3, actv4, actv5, actv6, actv7, actv8

