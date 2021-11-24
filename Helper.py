''' File contains a series of helper functions that perform useful operations '''

from typing import List, Tuple
import torch
import matplotlib.pyplot as plt # type: ignore

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