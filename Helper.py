''' File contains a series of helper functions that perform useful operations '''

from typing import List
import torch
import matplotlib.pyplot as plt

''' Takes a 2D list and returns a 1D list with a specific index extracted from each sublist
'''
def extract_element_from_sublist(main_list: List[torch.Tensor], index: int) -> List[torch.Tensor]:
  return [item[index] for item in main_list]
  
''' Creates and display a matplotlib display displaying some of the output maps created for an image
'''
def display_outputs(input_tensor: torch.Tensor, output: str) -> None:
  num_outputs_to_display = len(output) if len(output) < 10 else 10
  fig = plt.figure('Output Channels', figsize=(20, 1))

  fig.add_subplot(1, num_outputs_to_display + 1, 1)
  plt.imshow(input_tensor.permute(1, 2, 0))
  plt.axis('off')

  for i in range(1, num_outputs_to_display + 1):
    fig.add_subplot(1, num_outputs_to_display + 1, i + 1)
    plt.imshow(output[i - 1])
    plt.axis('off')

  plt.tight_layout()
  plt.show()