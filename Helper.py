''' File contains a series of helper functions that perform useful operations '''

from typing import List, Tuple
import torch
import matplotlib.pyplot as plt # type: ignore

''' Takes a 2D list and returns a 1D list with a specific index extracted from each sublist
'''
def extract_element_from_sublist(main_list: List[Tuple[torch.Tensor, torch.Tensor]], index: int) -> List[torch.Tensor]:
  return [item[index] for item in main_list]
  
''' Creates and display a matplotlib display displaying some of the output maps created for an image
'''
def display_outputs(output: torch.Tensor) -> None:
  num_outputs_to_display = len(output) if len(output) < 10 else 10
  fig = plt.figure('Output Channels', figsize=(20, 1))

  for i in range(0, num_outputs_to_display):
    fig.add_subplot(1, num_outputs_to_display, i + 1)
    plt.imshow(output[i])
    plt.axis('off')

  plt.tight_layout()
  plt.show()