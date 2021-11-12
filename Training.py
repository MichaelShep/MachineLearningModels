''' File where all the training of our model occurs '''

import torch
from Helper import extract_element_from_sublist
from typing import List, Tuple
from SegmentationNetwork import SegmentationNetwork

class Training():
    def __init__(self, model: SegmentationNetwork, training_data: List[Tuple[torch.Tensor, torch.Tensor]]):
        self._model = model
        self._input_data = torch.stack(extract_element_from_sublist(training_data, 0))
        self._output_data = torch.stack(extract_element_from_sublist(training_data, 1))

    def train(self) -> torch.Tensor:
        return self._model(self._input_data)