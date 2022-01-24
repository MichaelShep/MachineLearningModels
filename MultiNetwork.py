''' Creates the network for the multi-task learning - both face segmentation and attribute detection
'''

import torch.nn as nn
import torch

class MultiNetwork(nn.Module):
    ''' Constructor for the class
        All layers of the network and their parameters get defined here
    '''
    def __init__(self):
        super(MultiNetwork, self).__init__()

    ''' Performs a forward pass through all the layers of the network
        Input is a image of size 512x512 with 3 input channels
    '''
    def forward(x: torch.Tensor) -> torch.Tensor:
        return x