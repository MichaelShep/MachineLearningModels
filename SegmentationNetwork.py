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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self._num_output_masks = num_output_masks

    #Create the layers for the network - Creating an architecture very similar to that used in U-Net
    self._skip_connections = [torch.Tensor()] * 5
    self._max_pool =  nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    self._upsample = nn.Upsample(scale_factor=2)

    self._contracting_path = nn.ModuleList([
      create_double_conv(in_chan=3, out_chan=32, device=device), create_double_conv(in_chan=32, out_chan=64, device=device),
      create_double_conv(in_chan=64, out_chan=128, device=device), create_double_conv(in_chan=128, out_chan=256, device=device),
      create_double_conv(in_chan=256, out_chan=512, device=device), create_double_conv(in_chan=512, out_chan=1024, device=device),
    ])

    self._expanding_path = nn.ModuleList([
      create_double_conv(in_chan=1024, out_chan=512, device=device), create_double_conv(in_chan=512, out_chan=256, device=device),
      create_double_conv(in_chan=256, out_chan=128, device=device), create_double_conv(in_chan=128, out_chan=64, device=device),
      create_double_conv(in_chan=64, out_chan=32, device=device), create_conv_layer(32, self._num_output_masks, device, kernal_size=1, padding=0)
    ])

  ''' Runs the contracting part of the network
  '''
  def _perform_contracting_path(self, x: torch.Tensor) -> torch.Tensor:
    for i in range(len(self._contracting_path) - 1):
      x = self._perform_contracting_layer(x, i)

    #For final part of path, don't want to perform skip connection or max pooling operations
    x = self._contracting_path[len(self._contracting_path) - 1](x)
    return x

  ''' Runs the expanding part of the network
  '''
  def _perform_expanding_path(self, x: torch.Tensor) -> torch.Tensor:
    for i in range(len(self._expanding_path) - 1):
      x = self._perform_expanding_layer(x, i)

    #For final part of expanding path, do not need to upsample or use skip connection
    x = self._expanding_path[len(self._expanding_path) - 1](x)
    return x

  ''' Creates the setup for one of the contracting layers - plus assigns the relevant skip connection
  '''
  def _perform_contracting_layer(self, x: torch.Tensor, idx: int) -> torch.Tensor:
    x = self._contracting_path[idx](x)
    self._skip_connections[idx] = x
    return self._max_pool(x)

  ''' Creates the setup for one of the expanding layers - makes use of created skip connections
  '''
  def _perform_expanding_layer(self, x: torch.Tensor, idx: int) -> torch.Tensor:
    x = self._upsample(x)
    x = self._expanding_path[idx](x)
    x = x + self._skip_connections[len(self._skip_connections) - 1 - idx]
    return x

  ''' Performs a forward pass through the network for some data
      Method should not be called directly as only gets called as part of the lifecycle of the network
  '''
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self._perform_contracting_path(x)
    x = self._perform_expanding_path(x)

    return x