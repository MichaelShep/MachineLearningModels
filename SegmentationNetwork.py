''' Creates the Face Segmentation CNN '''

import torch.nn as nn
import torch

''' Class defining the Segmentation Network
    All the layers of the network are setup here and the flow through the network is also defined
'''
class SegmentationNetwork(nn.Module):
  ''' Constructor for the class
      All layers of the network and their parameters get defined here
  '''
  def __init__(self, num_output_masks: int):
    super(SegmentationNetwork, self).__init__()
    self._num_output_masks = num_output_masks

    #Create variables which will be used for creating skip connections
    self.skip1 = torch.Tensor()
    self.skip2 = torch.Tensor()
    self.skip3 = torch.Tensor()
    self.skip4 = torch.Tensor()
    self.skip5 = torch.Tensor()

    #Create the layers for the network - Creating an architecture very similar to that used in U-Net
    self.max_pool =  nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    self.upsample = nn.Upsample(scale_factor=2)

    #Parts of the contracting path
    self.conv1 = self._create_double_conv(in_chan=3, out_chan=32)
    self.conv2 = self._create_double_conv(in_chan=32, out_chan=64)
    self.conv3 = self._create_double_conv(in_chan=64, out_chan=128)
    self.conv4 = self._create_double_conv(in_chan=128, out_chan=256)
    self.conv5 = self._create_double_conv(in_chan=256, out_chan=512)
    self.conv6 = self._create_double_conv(in_chan=512, out_chan=1024)

    #Parts of the expanding path
    self.conv7 = self._create_double_conv(in_chan=1024, out_chan=512)
    self.conv8 = self._create_double_conv(in_chan=512, out_chan=256)
    self.conv9 = self._create_double_conv(in_chan=256, out_chan=128)
    self.conv10 = self._create_double_conv(in_chan=128, out_chan=64)
    self.conv11 = self._create_double_conv(in_chan=64, out_chan=32)
    self.conv12 = self._create_conv_layer(32, self._num_output_masks, kernal_size=1, padding=0)

  ''' Creates a new convolution layer with some default parameters that are used for this network
  '''
  def _create_conv_layer(self, in_chan: int, out_chan: int, kernal_size=3, stride=1, padding=1) -> nn.Conv2d:
    return nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=kernal_size, stride=stride, padding=padding)

  ''' Creates the double conv setup that is used for our network
  '''
  def _create_double_conv(self, in_chan: int, out_chan: int) -> nn.Sequential:
    return nn.Sequential(
      self._create_conv_layer(in_chan=in_chan, out_chan=out_chan),
      nn.ReLU(),
      self._create_conv_layer(in_chan=out_chan, out_chan=out_chan),
      nn.ReLU()
    )

  ''' Creates the contracting section of the network - also sets up the skip connections
  '''
  def _perform_contracting_path(self, x: torch.Tensor) -> torch.Tensor:
    x = self.conv1(x)
    self.skip1 = x
    x2 = self.max_pool(x)
    x2 = self.conv2(x2)
    self.skip2 = x2
    x3 = self.max_pool(x2)
    x3 = self.conv3(x3)
    self.skip3 = x3
    x4 = self.max_pool(x3)
    x4 = self.conv4(x4)
    self.skip4 = x4
    x5 = self.max_pool(x4)
    x5 = self.conv5(x5)
    self.skip5 = x5
    x6 = self.max_pool(x5)
    x6 = self.conv6(x6)
    return x6

  ''' Creates the expanding section of the network - makes use of created skip connections
  '''
  def _perform_expanding_path(self, x: torch.Tensor) -> torch.Tensor:
    x = self.upsample(x)
    x = self.conv7(x)
    x = x + self.skip5
    x = self.upsample(x)
    x = self.conv8(x)
    x = x + self.skip4
    x = self.upsample(x)
    x = self.conv9(x)
    x = x + self.skip3
    x = self.upsample(x)
    x = self.conv10(x)
    x = x + self.skip2
    x = self.upsample(x)
    x = self.conv11(x)
    x = x + self.skip1
    x = self.conv12(x)
    return x

  ''' Performs a forward pass through the network for some data
      Method should not be called directly as only gets called as part of the lifecycle of the network
  '''
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self._perform_contracting_path(x)
    x = self._perform_expanding_path(x)

    return x