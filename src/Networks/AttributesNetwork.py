''' Creates the Network which detects face attributes 
    This model is based off the ImageNet classification model
'''

import torch.nn as nn
import torch
from Helper import create_conv_layer

class AttributesNetwork(nn.Module):
    ''' Constructor for the class
        All layers of the network and their parameters get defined here
    '''
    def __init__(self, num_attributes: int, device: str):
        super(AttributesNetwork, self).__init__()
        self._num_attributes = num_attributes
        self._device = device

        #Create the layers for the network
        self._relu = nn.ReLU()
        self._max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self._dropout = nn.Dropout(p=0.2)
        self._sigmoid = nn.Sigmoid()

        self._res_layers = nn.ModuleList([
            create_conv_layer(in_chan=3, out_chan=8),
            create_conv_layer(in_chan=8, out_chan=16),
            create_conv_layer(in_chan=16, out_chan=32),
            create_conv_layer(in_chan=32, out_chan=64),
            create_conv_layer(in_chan=64, out_chan=128),
            create_conv_layer(in_chan=128, out_chan=256),
        ])

        self._lin_layers = nn.ModuleList([
            nn.Linear(in_features=256*8*8, out_features=256),
            nn.Linear(in_features=256, out_features=64),
            nn.Linear(in_features=64, out_features=self._num_attributes)
        ])

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
        for i in range(0, len(self._res_layers)):
            x = self._res_layers[i](x) 
            x = self._max_pool(x)

        #Flatten input so it can be passed to linear layers
        x = x.view(x.size(0), -1)

        for i in range(len(self._lin_layers) - 1):
            x = self._perform_linear_layer(x, i)

        #Apply sigmoid to final layer to get values in range 0-1
        x = self._lin_layers[len(self._lin_layers) - 1](x)
        x = self._sigmoid(x)
        return x

    ''' Calculates how accurate our predictions were for a specific image
    '''
    def evaluate_prediction_accuracy(self, predictions: torch.Tensor, actual: torch.Tensor) -> float:
        correct_predictions = 0
        for i in range(len(predictions)):
            if predictions[i] == actual[i]:
                correct_predictions += 1

        return float(correct_predictions) / len(predictions)