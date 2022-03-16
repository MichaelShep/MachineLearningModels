''' File contains a series of helper functions that perform useful operations '''

from re import L
from tkinter import NE
from typing import List, Tuple
import torch
import matplotlib.pyplot as plt # type: ignore
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile
import pickle

from Networks.NetworkType import NetworkType

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
def create_conv_layer(in_chan: int, out_chan: int, device: str = 'cpu', kernal_size=3, stride=1, padding=1) -> nn.Conv2d:
  return nn.Conv2d(in_channels=in_chan, out_channels=out_chan, 
                  kernel_size=kernal_size, stride=stride, padding=padding).to(device)

''' Creates the double conv setup that is used within our networks
'''
def create_double_conv(in_chan: int, out_chan: int, device: str = 'cpu') -> nn.Sequential:
  return nn.Sequential(
    create_conv_layer(in_chan=in_chan, out_chan=out_chan, device=device),
    nn.ReLU(),
    create_conv_layer(in_chan=out_chan, out_chan=out_chan, device=device),
    nn.ReLU()
  ).to(device)

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
def plot_loss_list(training_losses: List[float], validation_losses: List[float], network_type: NetworkType) -> None:
  indexes = list(range(0, len(training_losses)))
  fig = plt.figure(figsize=(10, 7))
  ax = fig.add_subplot(111) 
  plt_title = ''
  if network_type == NetworkType.ATTRIBUTE:
    plt_title = 'Attributes Model'
  elif network_type == NetworkType.SEGMENTATION:
    plt_title = 'Segmentation Model'
  else:
    plt_title = 'Multi-Learning Model'
  ax.set_title(plt_title)
  ax.set_xlabel('Epoch')
  ax.set_ylabel('Loss')
  ax.plot(indexes, training_losses, label='Training Loss', color='green')
  ax.plot(indexes, validation_losses, label='Validation Loss', color='blue')
  ax.legend(loc='best', fancybox=True, framealpha=0.5)
  plt.show()

''' Converts the floating point outputs of our model into 0 or 1 based on a threshold value
'''
def threshold_outputs(network_type: NetworkType, model_output: torch.Tensor, 
                      threshold_level: int, threshold_level_2: int = 0) -> torch.Tensor:
  if network_type == NetworkType.SEGMENTATION:
    threshold_output = (model_output>threshold_level).float()
    return threshold_output
  elif network_type == NetworkType.ATTRIBUTE:
    threshold_output = (model_output>threshold_level).float()
    return threshold_output
  else:
    model_output_0 = (model_output[0]>threshold_level).float()
    model_output_1 = (model_output[1]>threshold_level_2).float()
    return (model_output_0, model_output_1) 

''' Evaluates the prediction accuracy of one of the models
'''
def evaluate_model_accuracy(model: nn.Module, network_type: NetworkType,
                            input_data: torch.Tensor, output_data: torch.Tensor, threshold_level: int, for_multi_segmentation_output: bool = False) -> None:
  model_predictions = model(input_data)
  #For Multi model, get which part of the output we are currently dealing with and treat model as that form of network
  if network_type == NetworkType.MULTI:
      if for_multi_segmentation_output:
          network_type = NetworkType.SEGMENTATION
          model_predictions = model_predictions[0]
      else:
          network_type = NetworkType.ATTRIBUTE
          model_predictions = model_predictions[1]
  model_predictions = threshold_outputs(network_type, model_predictions, threshold_level)

  correct_predictions = 0
  total_predictions = 0
  for i in range(len(model_predictions)):
    if network_type == NetworkType.ATTRIBUTE:
      for j in range(len(model_predictions[i])):
        if model_predictions[i][j] == output_data[i][j]:
          correct_predictions += 1
        total_predictions += 1
    if network_type == NetworkType.SEGMENTATION:
      for j in range(len(model_predictions[i])):
        for k in range(len(model_predictions[i][j])):
          for x in range(len(model_predictions[i][j][k])):
            if model_predictions[i][j][k][x] == output_data[i][j][k][x]:
              correct_predictions += 1
            total_predictions += 1
  
  return correct_predictions / total_predictions

def compare_model_accuracies(segmentation_file_name: str, attributes_file_name: str, multi_file_name: str):
    with open(f'{segmentation_file_name}.pt', 'rb') as segmentation_file:
        segmentation_accuracies = pickle.load(segmentation_file)['accuracies']
    with open(f'{attributes_file_name}.pt', 'rb') as attributes_file:
        attributes_accuracies = pickle.load(attributes_file)['accuracies']
    with open(f'{multi_file_name}.pt', 'rb') as multi_file:
        multi_accuracies = pickle.load(multi_file)['accuracies']

    #Create box plot for comparing standard segmentation model output to multi model
    fig = plt.figure(figsize=(10, 7))
    segmentation_display = fig.add_subplot(111)
    segmentation_display.set_title('Segmentation Outputs Comparison')
    segmentation_display.set_xticklabels(['Segmentation', 'Multi'])
    segmentation_display.set_xlabel('Model Type')
    segmentation_display.set_ylabel('Accuracy')

    segmentation_display.boxplot([segmentation_accuracies, [item[0] for item in multi_accuracies]])
    plt.show()

    fig = plt.figure(figsize=(10, 7))
    attributes_display = fig.add_subplot(111)
    attributes_display.set_title('Attributes Outputs Comparison')
    attributes_display.set_xticklabels(['Attribute', 'Multi'])
    attributes_display.set_xlabel('Model Type')
    attributes_display.set_ylabel('Accuracy')

    attributes_display.boxplot([attributes_accuracies, [item[1] for item in multi_accuracies]])
    plt.show()
