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
  def __init__(self):
    super(SegmentationNetwork, self).__init__()

    #Create the layers for the network - Creating an architecture very similar to that used in U-Net

    #Define reLU activation which will be used as activation for all Conv layers in network
    self.relu = nn.ReLU()

    #Define max pool layer with 2x2 filter and 2 stride
    self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    #Create all the conv layers for the contracting path with differing values of in and out channels
    self.conv364 = self._create_conv_layer(3, 64)
    self.conv6464 = self._create_conv_layer(64, 64)
    self.conv64128 = self._create_conv_layer(64, 128)
    self.conv128128 = self._create_conv_layer(128, 128)
    self.conv128256 = self._create_conv_layer(128, 256)
    self.conv256256 = self._create_conv_layer(256, 256)
    self.conv256512 = self._create_conv_layer(256, 512)
    self.conv512512 = self._create_conv_layer(512, 512)
    self.conv5121024 = self._create_conv_layer(512, 1024)
    self.conv10241024 = self._create_conv_layer(1024, 1024)

    #Create the upsample layer which increases the size of the input by a factor of 2
    self.upsample = torch.nn.Upsample(scale_factor=2)

    #Create all the upsample and conv layers for the expanding path of the network
    self.conv1024512 = self._create_conv_layer(1024, 512)
    self.conv512256 = self._create_conv_layer(512, 256)
    self.conv256128 = self._create_conv_layer(256, 128)
    self.conv12864 = self._create_conv_layer(128, 64)
    self.conv6432 = self._create_conv_layer(64, 32)
    self.conv3232 = self._create_conv_layer(32, 32)
    self.conv321 = self._create_conv_layer(32, 1, kernal_size=1, padding=0)

  ''' Creates a new convolution layer with some default parameters that are used for this network
  '''
  def _create_conv_layer(self, in_chan: int, out_chan: int, kernal_size=3, stride=1, padding=1) -> nn.Conv2d:
    return nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=kernal_size, stride=stride, padding=padding)

  ''' Passes an input tensor through all the contracting layers of the network
  '''
  def _perform_contracting_path(self, x: torch.Tensor) -> torch.Tensor:
    x = self.conv364(x) #3 input Channels, 64 output channels, 512x512 tensor size
    x = self.relu(x) 
    x = self.conv6464(x) #64 input Channels, 64 output channels, 512x512 tensor size
    x = self.relu(x)

    x = self.max_pool(x) #512x512 reduced to 256x256

    x = self.conv64128(x) #64 input channels, 128 output channels, 256x256 tensor size
    x = self.relu(x)
    x = self.conv128128(x) #128 input channels, 128 output channels, 256x256 tensor size
    x = self.relu(x)

    x = self.max_pool(x) #256x256 reduced to 128x128

    x = self.conv128256(x) #128 input channels, 256 output channels, 128x128 tensor size
    x = self.relu(x)
    x = self.conv256256(x) #256 input channels, 256 output channels, 128x128 tensor size
    x = self.relu(x)

    x = self.max_pool(x) #128x128 reduced to 64x64

    x = self.conv256512(x) #256 input channels, 512 output channels, 64x64 tensor size
    x = self.relu(x)
    x = self.conv512512(x) #512 input channels, 512 output channels, 64x64 tensor size
    x = self.relu(x)
    
    x = self.max_pool(x) #64x64 reduced to 32x32

    x = self.conv5121024(x) #512 input channels, 1024 output channels, 32x32 tensor size
    x = self.relu(x)
    x = self.conv10241024(x) #1024 input channels, 1024 output channels, 32x32 tensor size
    x = self.relu(x)

    return x

  ''' Takes an input that has already been passed through the contracting part of the network and expands it
  '''
  def _perform_expanding_path(self, x: torch.Tensor) -> torch.Tensor:
    x = self.upsample(x) #Upsample from 32x32 to 64x64

    x = self.conv1024512(x) #1024 input channels, 512 output channels, 64x64 tensor size
    x = self.relu(x)
    x = self.conv512512(x) #512 input channels, 512 output channels, 64x64 tensor size
    x = self.relu(x)

    x = self.upsample(x) #Upsample from 64x64 to 128x128

    x = self.conv512256(x) #512 input channels, 256 output channels, 128x128 tensor size
    x = self.relu(x)
    x = self.conv256256(x) #256 input channels, 256 output channels, 128x128 tensor size
    x = self.relu(x)

    x = self.upsample(x) #Upsample from 128x128 to 256x256

    x = self.conv256128(x) #256 input channels, 128 output channels, 256x256 tensor size
    x = self.relu(x)
    x = self.conv12864(x) #128 input channels, 64 output channels, 256x256 tensor size
    x = self.relu(x)

    x = self.upsample(x) #Upsample from 256x256 to 512x512

    x = self.conv6432(x) #64 input channels, 32 output channels, 512x512 tensor size
    x = self.relu(x)
    x = self.conv3232(x) #32 input channels, 32 output channels, 512x512 tensor size
    x = self.relu(x)

    x = self.conv321(x) #32 input channels, 1 output channel, 512x512 tensor size

    return x

  ''' Performs a forward pass through the network for some data
      Method should not be called directly as only gets called as part of the lifecycle of the network
  '''
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self._perform_contracting_path(x)
    x = self._perform_expanding_path(x)

    return x