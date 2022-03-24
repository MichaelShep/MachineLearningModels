import torch.nn as nn
import torch
from Helper import create_conv_layer

class AttributesNetwork(nn.Module):
    ''' Creates a network for the attribute detection task - will predict a list of attribute for a given face '''

    def __init__(self, num_attributes: int, device: str):
        ''' Initialises all the layers of the network
        
        Parameters
        ----------
        num_attributes: int
            The number of attributes that it is possible for each face to have
        device: str
            The device which our model will run on - will be CUDA or cpu
        '''
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

    def _perform_linear_layer(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        ''' Performs a single linear layer with dropout enabled followed by a ReLU activation function
        
        Parameters
        ----------
        x: torch.Tensor
            The input for this layer of the network
        idx: int
            The index of the linear layer we are wanting to perform
        Returns
        -------
        torch.Tensor
            The output from this linear layer and its activation
        '''
        x = self._lin_layers[idx](x)
        x = self._dropout(x)
        x = self._relu(x)
        return x    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' Performs a forward pass of the network - passes an input through all the layers in our network

        Parameters
        ----------
        x: torch.Tensor
            The input image that will be passed through the network
        Returns
        -------
        torch.Tensor
            The output from the network - will be a list of attributes that the model thinks this face has
        '''
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

    def evaluate_prediction_accuracy(self, predictions: torch.Tensor, actual: torch.Tensor) -> float:
        ''' Used to see how accurate the predictions of our model were for a single image
        Parameters
        ----------
        predictions: torch.Tensor
            A list containing all the predictions our model made (list of 0s and 1s)
        actual: torch.Tensor
            A list containing all the actual outputs that our model was aiming for
        Returns
        -------
        float
            Value defining how accurate our model was - simply number of correct predictions out of the total
        '''
        correct_predictions = 0
        for i in range(len(predictions)):
            if predictions[i] == actual[i]:
                correct_predictions += 1

        return float(correct_predictions) / len(predictions)