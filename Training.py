''' File where all the training of our model occurs '''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from SegmentationNetwork import SegmentationNetwork
from CelebADataset import CelebADataset
from typing import Tuple
import time

class Training():
  _MINI_BATCH_SIZE = 1
  _NUM_EPOCHS = 1
  _NUM_TRAINING_EXAMPLES = 20000

  ''' Splits the data into training and testing data and sets up data loader so data can be dealt with in
      batches
  '''
  def __init__(self, model: SegmentationNetwork, dataset: CelebADataset):
    self._model = model
    self._dataset = dataset
    self._training_examples, self._testing_examples = dataset.get_train_test_split(self._NUM_TRAINING_EXAMPLES)

    self._training_loader = DataLoader(self._training_examples, batch_size=self._MINI_BATCH_SIZE, shuffle=True, num_workers=1)
    self._testing_loader = DataLoader(self._testing_examples, batch_size=self._MINI_BATCH_SIZE, shuffle=False, num_workers=1)

    #Using Per Pixel Cross Entropy Loss for our loss and Stochastic Gradient Descent for our optimiser
    self._loss_func = nn.CrossEntropyLoss(reduction='none')
    self._optim = torch.optim.SGD(self._model.parameters(), lr=0.1)

  ''' Gets all the training tuples of data (input and output) for a given tensor containing indexes
  '''
  def _get_data_for_indexes(self, indexes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    input_values = []
    output_values = []

    for index_tensor in indexes:
      element = self._dataset[index_tensor.item()]
      input_values.append(element[0])
      output_values.append(element[1])

    return (torch.stack(input_values), torch.stack(output_values))


  ''' Performs the actual training using our training data and model
  '''
  def train(self) -> None:
    for _ in range(self._NUM_EPOCHS):
      for i, data_indexes in enumerate(self._training_loader):
        if i == 0:
          input_data, output_data = self._get_data_for_indexes(data_indexes)
          print('Starting to run model')
          start_time = time.time()
          model_output = self._model(input_data)
          print('Time taken to run on single mini batch of size', self._MINI_BATCH_SIZE, ':', (time.time() - start_time), 'seconds')
  
          '''Loss is not currently working correctly - need to find a better way to handle this
          torch.set_printoptions(profile='full')
          loss = self._loss_func(model_output[0], output_data[0])
          print('Loss value of:', loss)'''
