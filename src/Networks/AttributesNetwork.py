''' Creates the Network which detects face attributes 
    This model is based off the ImageNet classification model
'''

import torch.nn as nn
import torch
from Helper import create_conv_layer, perform_residual

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

        self._conv_layers = nn.ModuleList([
            create_conv_layer(in_chan=3, out_chan=32), create_conv_layer(in_chan=32, out_chan=64),
            create_conv_layer(in_chan=64, out_chan=96), create_conv_layer(in_chan=96, out_chan=64),
            create_conv_layer(in_chan=64, out_chan=32)
        ])

        self._res_layers = nn.ModuleList([
            create_conv_layer(in_chan=32, out_chan=32), create_conv_layer(in_chan=64, out_chan=64),
            create_conv_layer(in_chan=96, out_chan=96), create_conv_layer(in_chan=64, out_chan=64),
            create_conv_layer(in_chan=32, out_chan=32)
        ])

        self._lin_layers = nn.ModuleList([
            nn.Linear(in_features=32*16*16, out_features=256),
            nn.Linear(in_features=256, out_features=self._num_attributes)
        ])

    ''' Performs a conv layer for the network - followed by relu, pooling and residual
    '''
    def _perform_conv_layer(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        x = self._conv_layers[idx](x)
        x = self._relu(x)
        x = self._max_pool(x)
        x = perform_residual(self._res_layers[idx], x)
        return x

    ''' Performs an inner linear layer for the network - followed by dropout and relu
    '''
    def _perform_linear_layer(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        x = self._lin_layers[idx](x)
        x = self._dropout(x)
        x = self._relu(x)
        return x    

    ''' Performs a forward pass through all the layers of the network
        Input is a image of size 512x512 with 3 input channels
    '''
    def forward(self, x) -> torch.Tensor:
        for i in range(len(self._conv_layers)):
            x = self._perform_conv_layer(x, i)

        #Flatten input so it can be passed to linear layers
        x = x.view(x.size(0), -1)

        for i in range(len(self._lin_layers) - 1):
            x = self._perform_linear_layer(x, i)

        #We want the raw output (without activiation) from the final layer of the network
        x = self._lin_layers[len(self._lin_layers) - 1](x)
        return x