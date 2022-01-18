''' Creates the Network which detects face attributes 
    This model is based off the ImageNet classification model
'''

import torch.nn as nn
import torch
from Helper import create_conv_layer, create_double_conv

class AttributesNetwork(nn.Module):
    ''' Constructor for the class
        All layers of the network and their parameters get defined here
    '''
    def __init__(self, num_attributes: int):
        super(AttributesNetwork, self).__init__()
        self._num_attributes = num_attributes

        #Create the layers for the network
        self._relu = nn.ReLU()
        self._max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self._dropout = nn.Dropout(p=0.5)

        self._conv1 = create_conv_layer(in_chan=3, out_chan=32)
        self._conv2 = create_conv_layer(in_chan=32, out_chan=64)
        self._conv3 = create_conv_layer(in_chan=64, out_chan=96)
        self._conv4 = create_conv_layer(in_chan=96, out_chan=64)
        self._conv5 = create_conv_layer(in_chan=64, out_chan=32)

        self._res1 = create_conv_layer(in_chan=32, out_chan=32)
        self._res2 = create_conv_layer(in_chan=64, out_chan=64)
        self._res3 = create_conv_layer(in_chan=96, out_chan=96)

        self._batchnorm1 = nn.BatchNorm2d(32)
        self._batchnorm2 = nn.BatchNorm2d(64)
        self._batchnorm3 = nn.BatchNorm2d(96)

        self._lin1 = nn.Linear(in_features=32*16*16, out_features=256)
        self._lin2 = nn.Linear(in_features=256, out_features=self._num_attributes)
        

    ''' Performs a forward pass through all the layers of the network
        Input is a image of size 512x512 with 3 input channels
    '''
    def forward(self, x) -> torch.Tensor:
        x = self._conv1(x) #Outputs Batch x 512 x 512 x 16
        #x = self._batchnorm1(x)
        x = self._relu(x)
        x = self._max_pool(x) #Outputs Batch x 256 x 256 x 16

        x = self._perform_residual(self._res1, x)

        x = self._conv2(x) #Outputs Batch x 256 x 256 x 32
        #x = self._batchnorm2(x)
        x = self._relu(x)
        x = self._max_pool(x) #Outputs Batch x 128 x 128 x 32

        x = self._perform_residual(self._res2, x)

        x = self._conv3(x) #Outputs Batch x 128 x 128 x 64
        #x = self._batchnorm3(x)
        x = self._relu(x)
        x = self._max_pool(x) #Outputs Batch x 64 x 64 x 64

        x = self._perform_residual(self._res3, x)

        x = self._conv4(x)
        #x = self._batchnorm2(x)
        x = self._relu(x)
        x = self._max_pool(x)

        x = self._perform_residual(self._res2, x)

        x = self._conv5(x)
        #x = self._batchnorm1(x)
        x = self._relu(x)
        x = self._max_pool(x)

        x = self._perform_residual(self._res1, x)

        x = x.view(x.size(0), -1) #Flatten input so it can be passed to linear layers

        x = self._lin1(x) #Outputs Batch x 512
        x = self._dropout(x)
        x = self._relu(x)

        x = self._lin2(x) #Outputs Batch x 64
        return x

    ''' Performs a residual connection step using a given Conv layer

    '''
    def _perform_residual(self, conv_layer: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
        inital_x = x
        new_x = conv_layer(x)
        new_x = self._relu(new_x)
        new_x = conv_layer(x)
        new_x = new_x + inital_x
        return new_x