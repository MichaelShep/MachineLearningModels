''' File where all the training of our model occurs '''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from SegmentationNetwork import SegmentationNetwork
from CelebADataset import CelebADataset
from typing import Tuple
from Helper import plot_predicted_and_actual

class Training():
  _NUM_TRAINING_EXAMPLES = 20000
  _OUTPUT_THRESHOLD = 0.8

  ''' Splits the data into training and testing data and sets up data loader so data can be dealt with in
      batches
  '''
  def __init__(self, model: SegmentationNetwork, dataset: CelebADataset, for_segmentation: bool,
               batch_size: int, learning_rate: float, save_name: str, num_epochs: int, display_outputs: bool):
    self._model = model
    self._dataset = dataset
    self._for_segmentation = for_segmentation
    self._save_name = save_name
    self._display_outputs = display_outputs
    self._num_epochs = num_epochs
    self._training_examples, self._validation_examples = dataset.get_train_test_split(self._NUM_TRAINING_EXAMPLES)

    self._training_loader = DataLoader(self._training_examples, batch_size=batch_size, shuffle=True, num_workers=2)
    self._validation_loader = DataLoader(self._validation_examples, batch_size=batch_size, shuffle=False, num_workers=2)

    #Using Per Pixel Mean Squared Error for our loss and Stochastic Gradient Descent for our optimiser
    self._loss_func = nn.BCEWithLogitsLoss()
    self._optim = torch.optim.Adam(self._model.parameters(), lr=learning_rate)

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
    for epoch in range(self._num_epochs):
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
          torch.save(self._model.state_dict(), self._save_name)
          print('Model Saved.')

        if i % 500 == 0 and self._for_segmentation and self._display_outputs:
          plot_predicted_and_actual(input_data[0].cpu(), model_output[0].cpu(), output_data[0].cpu())
        elif i % 500 == 0 and self._display_outputs:
          print('Example Predicted Values: ', model_output[0].cpu())
          print('Example Actual Values:', output_data[0].cpu())

        #Clear all unneeded memory - without this will get a memory error
        del input_data, output_data, model_output, loss
        torch.cuda.empty_cache()

      print('Saving Model...')  
      torch.save(self._model.state_dict(), self._save_name)
      print('Model Saved.')
    
  ''' Runs our model on unseen validation data to check that it still performs well on unseen data
  '''
  def run_on_valdiation_data(self, display_outputs: bool = False, for_segmentaiton=False) -> None:
    print('Running Validation Step...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self._model = self._model.to(device)
    for i, data_indexes in enumerate(self._validation_loader):
      self._model.eval()
      with torch.no_grad():
        total_validation_loss = 0
        input_data, output_data = self._get_data_for_indexes(data_indexes, device)
        model_output = self._model(input_data)
        loss = self._loss_func(model_output, output_data)  
        model_output = (model_output>self._OUTPUT_THRESHOLD).float()
        total_validation_loss += loss
        if i % 50 == 0:
          print(f'Batch {i}, Current Batch Loss: {loss}')
        if i % 500 == 0 and display_outputs and for_segmentaiton:
          plot_predicted_and_actual(input_data[0].cpu(), model_output[0].cpu(), output_data[0].cpu())
        elif i % 500 == 0 and display_outputs:
          print('Example Predicted Values: ', model_output[0].cpu())
          print('Example Actual Values:', output_data[0].cpu())
    total_validation_loss /= len(self._validation_examples)
    print(f'Validation Loss: {total_validation_loss}')
    if display_outputs and for_segmentaiton:
      plot_predicted_and_actual(input_data[0].cpu(), model_output[0].cpu(), output_data[0].cpu())
    elif display_outputs:
      print('Example Predicted Values: ', model_output[0].cpu())
      print('Example Actual Values:', output_data[0].cpu())

