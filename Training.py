''' File where all the training of our model occurs '''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from SegmentationNetwork import SegmentationNetwork
from CelebADataset import CelebADataset
from typing import Tuple
from Helper import plot_predicted_and_actual

class Training():
  _MINI_BATCH_SIZE = 7
  _NUM_EPOCHS = 20
  _NUM_TRAINING_EXAMPLES = 20000
  _OUTPUT_THRESHOLD = 0.8

  ''' Splits the data into training and testing data and sets up data loader so data can be dealt with in
      batches
  '''
  def __init__(self, model: SegmentationNetwork, dataset: CelebADataset):
    self._model = model
    self._dataset = dataset
    self._training_examples, self._testing_examples = dataset.get_train_test_split(self._NUM_TRAINING_EXAMPLES)

    self._training_loader = DataLoader(self._training_examples, batch_size=self._MINI_BATCH_SIZE, shuffle=True, num_workers=2)
    self._testing_loader = DataLoader(self._testing_examples, batch_size=self._MINI_BATCH_SIZE, shuffle=False, num_workers=2)

    #Using Per Pixel Mean Squared Error for our loss and Stochastic Gradient Descent for our optimiser
    self._loss_func = nn.BCEWithLogitsLoss()
    self._optim = torch.optim.Adam(self._model.parameters(), lr=0.0001)

  ''' Gets all the training tuples of data (input and output) for a given tensor containing indexes
  '''
  def _get_data_for_indexes(self, indexes: torch.Tensor, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    input_values = []
    output_values = []

    for index_tensor in indexes:
      element = self._dataset[index_tensor.item()]
      input_values.append(element[0])
      output_values.append(element[1])
 
    return (torch.stack(input_values).to(device), torch.stack(output_values).to(device))


  ''' Performs the actual training using our training data and model
  '''
  def train(self) -> None:
    for epoch in range(self._NUM_EPOCHS):
      self._model.train()
      for i, data_indexes in enumerate(self._training_loader):
        input_data, output_data = self._get_data_for_indexes(data_indexes, 'cuda' if torch.cuda.is_available() else 'cpu')
        model_output = self._model(input_data)

        #Perform actual learning - calculate loss value, perform back-prop and grad descent step
        loss = self._loss_func(model_output, output_data)
        self._optim.zero_grad()
        loss.backward()
        self._optim.step()

        #Once all operations on model have been done, convert each pixel output to binary values
        model_output = (model_output>self._OUTPUT_THRESHOLD).float()        

        if i % 50 == 0:
          print('Epoch:', epoch, 'Batch:', i, 'Loss:', loss.item())
        if i % 1000 == 0 and i != 0:
          print('Saving Model...')
          torch.save(self._model.state_dict(), 'MODEL_SKIP.pt')
          print('Model Saved.')
        if i % 500 == 0:
          plot_predicted_and_actual(input_data[0].cpu(), model_output[0].cpu(), output_data[0].cpu())

        #Clear all unneeded memory - without this will get a memory error
        del input_data, output_data, model_output, loss
        torch.cuda.empty_cache()

      print('Saving Model...')  
      torch.save(self._model.state_dict(), 'MODEL_SKIP.pt')
      print('Model Saved.')


      print('Running testing data')
      #Perform model evaluation on testing data
      self._model.eval()
      with torch.no_grad():
        test_loss = 0
        for i, data_indexes in enumerate(self._testing_loader):
          input_data, output_data = self._get_data_for_indexes(data_indexes, 'cuda' if torch.cuda.is_available() else 'cpu')
          model_output = self._model(input_data)

          test_loss += self._loss_func(model_output, output_data)
        test_loss /= len(self._training_examples)
        print(f'Test loss at epoch {epoch}: {test_loss}')

