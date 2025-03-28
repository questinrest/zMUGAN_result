"""
Knowledge Extraction with No Observable Data (NeurIPS 2019)

Authors:
- Jaemin Yoo (jaeminyoo@snu.ac.kr), Seoul National University
- Minyong Cho (chominyong@gmail.com), Seoul National University
- Taebum Kim (k.taebum@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.
"""
import numpy as np

from kegnet.classifier.models import lenet, resnet, linear, VGG16, ALLCNN
from kegnet.utils import data


def init_classifier(dataset):
    """
    Initialize a classifier based on the dataset.
    """
    d = data.to_dataset(dataset)
    if dataset == 'mnist':
        return lenet.LeNet5()
    elif dataset in ('svhn', 'fashion', 'cifar10'):
        # return resnet.ResNet(d.nc, d.ny)
         return ALLCNN.AllCNN()
        #return VGG16.VGG16()
    else:
        return linear.MLP(d.nx, d.ny)


def count_parameters(model):
    """
    Count the parameters of a classifier.
    """
    size = 0
    for parameter in model.parameters():
        size += np.prod(parameter.shape)
    return size
