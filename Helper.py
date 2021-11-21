''' File contains a series of helper functions that perform useful operations '''

from typing import List, Tuple
import torch
import matplotlib.pyplot as plt # type: ignore

''' Takes a 2D list and returns a 1D list with a specific index extracted from each sublist
'''
def extract_element_from_sublist(main_list: List[Tuple[torch.Tensor, torch.Tensor]], index: int) -> List[torch.Tensor]:
  return [item[index] for item in main_list]
  
def plot_predicted_and_actual(predicted: torch.Tensor, actual: torch.Tensor) -> None:
  predicted = predicted.detach().permute(1, 2, 0)
  actual = actual.detach().permute(1, 2, 0)
  fig = plt.figure(figsize=(5, 5))

  fig.add_subplot(1, 2, 1)
  plt.imshow(predicted)
  plt.axis('off')

  fig.add_subplot(1, 2, 2)
  plt.imshow(actual)
  plt.axis('off')

  plt.show()