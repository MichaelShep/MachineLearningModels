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

        '''self._conv1 = create_double_conv(in_chan=3, out_chan=64)
        self._conv2 = create_double_conv(in_chan=64, out_chan=128)
        self._conv3 = create_double_conv(in_chan=128, out_chan=256)
        self._conv4 = create_double_conv(in_chan=256, out_chan=128)
        self._conv5 = create_double_conv(in_chan=128, out_chan=64)
        self._conv6 = create_double_conv(in_chan=64, out_chan=32)'''

        self._conv1 = create_conv_layer(in_chan=3, out_chan=64)
        self._conv2 = create_conv_layer(in_chan=64, out_chan=128)
        self._conv3 = create_conv_layer(in_chan=128, out_chan=64)
        self._conv4 = create_conv_layer(in_chan=64, out_chan=32)
        self._conv5 = create_conv_layer(in_chan=32, out_chan=16)

        self._lin1 = nn.Linear(in_features=16*32*32, out_features=512)
        self._lin2 = nn.Linear(in_features=512, out_features=64)
        self._lin3 = nn.Linear(in_features=64, out_features=self._num_attributes)
        

    ''' Performs a forward pass through all the layers of the network
        Input is a image of size 512x512 with 3 input channels
    '''
    def forward(self, x) -> torch.Tensor:
        x = self._conv1(x) #Outputs Batch x 512 x 512 x 16
        x = self._max_pool(x) #Outputs Batch x 256 x 256 x 16

        x = self._conv2(x) #Outputs Batch x 256 x 256 x 32
        x = self._max_pool(x) #Outputs Batch x 128 x 128 x 32

        x = self._conv3(x) #Outputs Batch x 128 x 128 x 64
        x = self._max_pool(x) #Outputs Batch x 64 x 64 x 64

        x = self._conv4(x) #Outputs Batch x 64 x 64 x 128
        x = self._max_pool(x) #Outputs Batch x 32 x 32 x 128

        x = self._conv5(x) #Outputs Batch x 32 x 32 x 256

        x = x.view(x.size(0), -1) #Flatten input so it can be passed to linear layers

        x = self._lin1(x) #Outputs Batch x 512
        x = self._relu(x)

        x = self._lin2(x) #Outputs Batch x 64
        x = self._relu(x)

        x = self._lin3(x) #Outputs Batch x 64

        return x