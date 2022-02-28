''' Creates the network for the multi-task learning - both face segmentation and attribute detection
'''

import torch.nn as nn
import torch
from Helper import create_conv_layer, create_double_conv

class MultiNetwork(nn.Module):
    ''' Constructor for the class
        All layers of the network and their parameters get defined here
    '''
    def __init__(self, num_output_masks: int, num_attributes: int):
        super(MultiNetwork, self).__init__()
        self._num_output_masks = num_output_masks
        self._num_attributes = num_attributes

        #Network layers which are used by both paths
        self._conv1 = create_double_conv(in_chan=3, out_chan=8)
        self._conv2 = create_double_conv(in_chan=8, out_chan=16)
        self._conv3 = create_double_conv(in_chan=16, out_chan=32)
        self._conv4 = create_double_conv(in_chan=32, out_chan=64)
        self._conv5 = create_double_conv(in_chan=64, out_chan=128)
        self._conv6 = create_double_conv(in_chan=128, out_chan=256)

        self._max_pool =  nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        #Layers that will only be used for the segmentation part of the network
        self._skip1 = torch.Tensor()
        self._skip2 = torch.Tensor()
        self._skip3 = torch.Tensor()
        self._skip4 = torch.Tensor()
        self._skip5 = torch.Tensor()

        self._upsample = nn.Upsample(scale_factor=2)

        self._conv7 = create_double_conv(in_chan=256, out_chan=128)
        self._conv8 = create_double_conv(in_chan=128, out_chan=64)
        self._conv9 = create_double_conv(in_chan=64, out_chan=32)
        self._conv10 = create_double_conv(in_chan=32, out_chan=16)
        self._conv11 = create_double_conv(in_chan=16, out_chan=8)
        self._conv12 = create_conv_layer(8, self._num_output_masks, kernal_size=1, padding=0)

        #Layers that will only be used for the attributes part of the network
        self._lin1 = nn.Linear(in_features=256*8*8, out_features=256)
        self._lin2 = nn.Linear(in_features=256, out_features=64)
        self._lin3 = nn.Linear(in_features=64, out_features=self._num_attributes)

        self._dropout = nn.Dropout(p=0.2)
        self._relu = nn.ReLU()
        self._sigmoid = nn.Sigmoid()

    ''' Performs a forward pass through all the layers of the network
        Input is a image of size 512x512 with 3 input channels
    '''
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._perform_joint_path(x)

        segmentation_output = self._perform_segmentation_path(x)
        attributes_output = self._perform_attributes_path(x)

        return (segmentation_output, attributes_output)

    ''' Creates the joint section of the network - the skip connections will only get used for the Segmentation part
    '''
    def _perform_joint_path(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv1(x)
        self._skip1 = x
        x2 = self._max_pool(x)

        x2 = self._conv2(x2)
        self._skip2 = x2
        x3 = self._max_pool(x2)

        x3 = self._conv3(x3)
        self._skip3 = x3
        x4 = self._max_pool(x3)

        x4 = self._conv4(x4)
        self._skip4 = x4
        x5 = self._max_pool(x4)

        x5 = self._conv5(x5)
        self._skip5 = x5
        x6 = self._max_pool(x5)

        x6 = self._conv6(x6)
        return x6

    ''' Performs the segmentation specific part of the network
    '''
    def _perform_segmentation_path(self, x: torch.Tensor) -> torch.Tensor:
        x = self._upsample(x)
        x = self._conv7(x)
        x = x + self._skip5

        x = self._upsample(x)
        x = self._conv8(x)
        x = x + self._skip4

        x = self._upsample(x)
        x = self._conv9(x)
        x = x + self._skip3

        x = self._upsample(x)
        x = self._conv10(x)
        x = x + self._skip2

        x = self._upsample(x)
        x = self._conv11(x)
        x = x + self._skip1

        x = self._conv12(x)
        return x
    
    ''' Performs the attribute specific part of the network
    '''
    def _perform_attributes_path(self, x: torch.Tensor) -> torch.Tensor:
        x = self._max_pool(x)

        x = x.view(x.size(0), -1) #Flatten input so it can be passed to linear layers

        x = self._lin1(x)
        x = self._dropout(x)
        x = self._relu(x)

        x = self._lin2(x)
        x = self._dropout(x)
        x = self._relu(x)

        x = self._lin3(x)
        x = self._sigmoid(x)
        return x