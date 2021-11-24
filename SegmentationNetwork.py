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

    #Create the layers for the network - Creating an architecture very similar to that used in U-Net
    self._contracting_path = nn.Sequential(
      self._create_double_conv(in_chan=3, out_chan=64),
      nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
      self._create_double_conv(in_chan=64, out_chan=128),
      nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
      self._create_double_conv(in_chan=128, out_chan=256),
      nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
      self._create_double_conv(in_chan=256, out_chan=512),
      nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
      self._create_double_conv(in_chan=512, out_chan=1024),
      nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    )

    self._expanding_path = nn.Sequential(
      nn.Upsample(scale_factor=2),
      self._create_double_conv(in_chan=1024, out_chan=512),
      nn.Upsample(scale_factor=2),
      self._create_double_conv(in_chan=512, out_chan=256),
      nn.Upsample(scale_factor=2),
      self._create_double_conv(in_chan=256, out_chan=128),
      nn.Upsample(scale_factor=2),
      self._create_double_conv(in_chan=128, out_chan=64),
      nn.Upsample(scale_factor=2),
      self._create_double_conv(in_chan=64, out_chan=32),
      self._create_conv_layer(32, self._num_output_masks, kernal_size=1, padding=0)
    )

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

  ''' Performs a forward pass through the network for some data
      Method should not be called directly as only gets called as part of the lifecycle of the network
  '''
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self._contracting_path(x)
    x = self._expanding_path(x)

    return x