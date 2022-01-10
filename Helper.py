''' File contains a series of helper functions that perform useful operations '''

from typing import List, Tuple
import torch
import matplotlib.pyplot as plt # type: ignore
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

''' Takes a 2D list and returns a 1D list with a specific index extracted from each sublist
'''
def extract_element_from_sublist(main_list: List[Tuple[torch.Tensor, torch.Tensor]], index: int) -> List[torch.Tensor]:
  return [item[index] for item in main_list]
  
''' Displays a comparison of the predicted masks and the actual masks
'''  
def plot_predicted_and_actual(input_image: torch.Tensor, predicted: torch.Tensor, actual: torch.Tensor) -> None:
  fig = plt.figure(figsize=(20, 4))
  display_data_element(input_image, actual, fig, 1)
  display_data_element(input_image, predicted, fig, 2)
  plt.show()

''' Displays a single image with matplotlib
'''
def display_image(image: torch.Tensor):
  fig = plt.figure(figsize=(5, 5))
  plt.imshow(image.detach().permute(1, 2, 0))
  plt.axis('off')
  plt.show()

''' Displays a single input image and all the output masks related with that
    Requires already created figure object and does not call plt.show()
'''
def display_data_element(input_image: torch.Tensor, output_masks: torch.Tensor, fig: plt.figure, row: int) -> None:
  fig.add_subplot(2, len(output_masks) + 1, row + ((row - 1) * len(output_masks)))
  plt.imshow(input_image.detach().permute(1, 2, 0))
  plt.axis('off')

  for i in range(len(output_masks)):
    fig.add_subplot(2, len(output_masks) + 1, i + 1 + row + ((row - 1) * len(output_masks)))
    plt.imshow(output_masks[i].detach().unsqueeze(0).permute(1, 2, 0))
    plt.axis('off')

''' Creates a new convolution layer with some default parameters that are used within our networks
'''
def create_conv_layer(in_chan: int, out_chan: int, kernal_size=3, stride=1, padding=1) -> nn.Conv2d:
  return nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=kernal_size, stride=stride, padding=padding)

''' Creates the double conv setup that is used within our networks
'''
def create_double_conv(in_chan: int, out_chan: int) -> nn.Sequential:
  return nn.Sequential(
    create_conv_layer(in_chan=in_chan, out_chan=out_chan),
    nn.ReLU(),
    create_conv_layer(in_chan=out_chan, out_chan=out_chan),
    nn.ReLU()
  )

''' Saves and optimizes a model so that it can be used with PyTorch mobile
'''
def save_model_for_mobile(model: torch.nn.Module, model_name: str, example_input: torch.Tensor):
  model = model.to('cpu')
  example_input = example_input.to('cpu')
  model.eval()
  script_model = torch.jit.trace(model, example_input)
  optimized_script_model = optimize_for_mobile(script_model)
  optimized_script_model._save_for_lite_interpreter(model_name + '.ptl')
  print('Mobile Model Saved')

''' Uses matplotlib to plot a curve for loss values - using the data indexes as the x axis
'''
def plot_loss_list(training_losses: List[float], validation_losses: List[float]):
  indexes = list(range(0, len(training_losses)))
  plt.plot(indexes, training_losses, color='green')
  plt.plot(indexes, validation_losses, color='blue')
  plt.show()
