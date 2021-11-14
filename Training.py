''' File where all the training of our model occurs '''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from SegmentationNetwork import SegmentationNetwork
from DataSet import ImageDataset

class Training():
  _MINI_BATCH_SIZE = 50
  _NUM_EPOCHS = 1
  _NUM_TRAINING_EXAMPLES = 20000

  ''' Splits the data into training and testing data and sets up data loader so data can be dealt with in
      batches
  '''
  def __init__(self, model: SegmentationNetwork, dataset: ImageDataset):
    self._model = model
    self._training_examples, self._testing_examples = dataset.get_train_test_split(self._NUM_TRAINING_EXAMPLES)

    self._training_loader = DataLoader(self._training_examples, batch_size=self._MINI_BATCH_SIZE, shuffle=True, num_workers=1)
    self._testing_loader = DataLoader(self._testing_examples, batch_size=self._MINI_BATCH_SIZE, shuffle=False, num_workers=1)

  ''' Performs the actual training using our training data and model
  '''
  def train(self) -> None:
    for _ in range(self._NUM_EPOCHS):
      for i, (input_image_indexes, output_map_locations) in enumerate(self._training_loader):
        print('Batch', i, 'Indexes:', input_image_indexes)

  '''  def train(self) -> torch.Tensor:
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


        return model_output'''
