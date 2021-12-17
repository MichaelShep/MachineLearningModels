''' Creates the Face Segmentation CNN '''

import torch.nn as nn
import torch
from Helper import create_conv_layer, create_double_conv

class SegmentationNetwork(nn.Module):
  ''' Constructor for the class
      All layers of the network and their parameters get defined here
  '''
  def __init__(self, num_output_masks: int):
    super(SegmentationNetwork, self).__init__()
    self._num_output_masks = num_output_masks

    #Create variables which will be used for creating skip connections
    self._skip1 = torch.Tensor()
    self._skip2 = torch.Tensor()
    self._skip3 = torch.Tensor()
    self._skip4 = torch.Tensor()
    self._skip5 = torch.Tensor()

    #Create the layers for the network - Creating an architecture very similar to that used in U-Net
    self._max_pool =  nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    self._upsample = nn.Upsample(scale_factor=2)

    #Parts of the contracting path
    self._conv1 = create_double_conv(in_chan=3, out_chan=32)
    self._conv2 = create_double_conv(in_chan=32, out_chan=64)
    self._conv3 = create_double_conv(in_chan=64, out_chan=128)
    self._conv4 = create_double_conv(in_chan=128, out_chan=256)
    self._conv5 = create_double_conv(in_chan=256, out_chan=512)
    self._conv6 = create_double_conv(in_chan=512, out_chan=1024)

    #Parts of the expanding path
    self._conv7 = create_double_conv(in_chan=1024, out_chan=512)
    self._conv8 = create_double_conv(in_chan=512, out_chan=256)
    self._conv9 = create_double_conv(in_chan=256, out_chan=128)
    self._conv10 = create_double_conv(in_chan=128, out_chan=64)
    self._conv11 = create_double_conv(in_chan=64, out_chan=32)
    self._conv12 = create_conv_layer(32, self._num_output_masks, kernal_size=1, padding=0)

  ''' Creates the contracting section of the network - also sets up the skip connections
  '''
  def _perform_contracting_path(self, x: torch.Tensor) -> torch.Tensor:
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

  ''' Creates the expanding section of the network - makes use of created skip connections
  '''
  def _perform_expanding_path(self, x: torch.Tensor) -> torch.Tensor:
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

  ''' Performs a forward pass through the network for some data
      Method should not be called directly as only gets called as part of the lifecycle of the network
  '''
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self._perform_contracting_path(x)
    x = self._perform_expanding_path(x)

    return x