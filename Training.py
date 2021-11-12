''' File where all the training of our model occurs '''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Helper import extract_element_from_sublist
from typing import List, Tuple
from SegmentationNetwork import SegmentationNetwork
from tqdm import tqdm

class Training():
    _MINI_BATCH_SIZE = 1
    _NUM_EPOCHS = 2

    def __init__(self, model: SegmentationNetwork, training_data: List[Tuple[torch.Tensor, torch.Tensor]]):
        self._model = model
        self._input_data = torch.stack(extract_element_from_sublist(training_data, 0))
        self._output_data = torch.stack(extract_element_from_sublist(training_data, 1))

        #Put the data into a DataLoader so that we can use shuffled random batches for better training
        self._data_loader = DataLoader(training_data, batch_size=self._MINI_BATCH_SIZE, shuffle=True, num_workers=1)
        #Use Stochastic Gradient Descent as our optimization method with a Learning Rate of 0.1
        self._optimizer = torch.optim.SGD(self._model.parameters(), lr=0.1)

        #Using Pixel-Wise Cross Entropy Loss as our loss function
        self._loss_func = nn.CrossEntropyLoss()

    def train(self) -> torch.Tensor:
        model_output = self._model(self._input_data)

        for epoch in range(self._NUM_EPOCHS):
            for _, (input_image, actual_output) in enumerate(self._data_loader):
                output = self._model(input_image)
                print(output, actual_output)
                loss = self._loss_func(output, actual_output)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                print('Epoch:', epoch, 'Loss:', loss)


        return model_output