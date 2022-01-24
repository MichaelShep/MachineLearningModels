''' File where all the training of our model occurs '''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from CelebADataset import CelebADataset
from typing import Tuple
from Helper import plot_predicted_and_actual, display_image, plot_loss_list

class Training():
  _NUM_TRAINING_EXAMPLES = 20000
  _OUTPUT_THRESHOLD = 0.8

  ''' Splits the data into training and testing data and sets up data loader so data can be dealt with in
      batches
  '''
  def __init__(self, model: nn.Module, dataset: CelebADataset,batch_size: int, learning_rate: float, 
                save_name: str, num_epochs: int, display_outputs: bool, for_segmentation: bool = False,
                for_multi: bool = False,):
    self._model = model
    self._dataset = dataset
    self._for_segmentation = for_segmentation
    self._for_multi = for_multi
    self._save_name = save_name
    self._display_outputs = display_outputs
    self._num_epochs = num_epochs
    self._training_examples, self._validation_examples = dataset.get_train_test_split(self._NUM_TRAINING_EXAMPLES)

    self._training_loader = DataLoader(self._training_examples, batch_size=batch_size, shuffle=True, num_workers=2)
    self._validation_loader = DataLoader(self._validation_examples, batch_size=batch_size, shuffle=True, num_workers=2)

    #Using Per Pixel Mean Squared Error for our loss and Stochastic Gradient Descent for our optimiser
    self._loss_func = nn.BCEWithLogitsLoss()
    self._optim = torch.optim.Adam(self._model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    self._per_epoch_training_loss = []
    self._per_epoch_validation_loss = []

  ''' Gets all the training tuples of data (input and output) for a given tensor containing indexes
  '''
  def _get_data_for_indexes(self, indexes: torch.Tensor, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    input_values = []
    output_values = []

    #Used for the multi network
    segmentation_values = []
    attribute_values = []

    for index_tensor in indexes:
      element = self._dataset[index_tensor.item()]
      input_values.append(element[0])
      if not self._for_multi:
        output_values.append(element[1])
      else:
        segmentation_values.append(element[1][0])
        attribute_values.append(element[1][1])
 
    if not self._for_multi:
      return (torch.stack(input_values).to(device), torch.stack(output_values).to(device))
    else:
      return (torch.stack(input_values).to(device), torch.stack(segmentation_values).to(device), torch.stack(attribute_values).to(device))


  ''' Performs the actual training using our training data and model
  '''
  def train(self) -> None:
    #Run on validation data before doing any training so that we get an inital value for our loss
    self.run_on_validation_data(display_outputs=self._display_outputs, for_segmentation=self._for_segmentation)
    for epoch in range(self._num_epochs):
      self._model.train()
      total_epoch_loss = 0
      for i, data_indexes in enumerate(self._training_loader):
        input_data, output_data = self._get_data_for_indexes(data_indexes, 'cuda' if torch.cuda.is_available() else 'cpu')
        model_output = self._model(input_data)

        #Perform actual learning - calculate loss value, perform back-prop and grad descent step
        loss = self._loss_func(model_output, output_data)
        #Add an initial value of our tracked loss to be used as the starting point for the loss
        if len(self._per_epoch_training_loss) == 0:
          self._per_epoch_training_loss.append(loss.item())
        total_epoch_loss += (loss.item() * len(data_indexes))
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
          self._display_output_for_attributes_model(input_data[0].cpu(), output_data[0].cpu(), model_output[0].cpu())

        #Clear all unneeded memory - without this will get a memory error
        del input_data, output_data, model_output, loss
        torch.cuda.empty_cache()

      print('Saving Model...')  
      total_epoch_loss /= len(self._training_examples)
      self._per_epoch_training_loss.append(total_epoch_loss)
      print('Average Loss for epoch:', total_epoch_loss)
      torch.save(self._model.state_dict(), self._save_name)
      print('Model Saved.')
      self.run_on_validation_data(display_outputs=self._display_outputs, for_segmentation=self._for_segmentation)
    
    #Show training loss curve once the model has been trained
    plot_loss_list(self._per_epoch_training_loss, self._per_epoch_validation_loss)

  ''' Training loop for the multi learning model
  '''
  def train_multi(self) -> None:
    for epoch in range(self._num_epochs):
      self._model.train()
      total_epoch_loss = 0
      for i, data_indexes in enumerate(self._training_loader):
        input_data, segmentation_output, attribute_output = self._get_data_for_indexes(data_indexes, 'cuda' if torch.cuda.is_available() else 'cpu')
        model_output = self._model(input_data)

        segmentation_loss = self._loss_func(model_output[0], segmentation_output)
        attribute_loss = self._loss_func(model_output[1], attribute_output)
        joint_loss = segmentation_loss + attribute_loss

        total_epoch_loss += (segmentation_loss.item() + attribute_loss.item()) * len(data_indexes)

        self._optim.zero_grad()
        joint_loss.backward()
        self._optim.step()

        if i % 50 == 0:
          print('Epoch:', epoch, 'Batch:', i, 'Segmentation Loss:', segmentation_loss.item(), 'Attribute Loss:', attribute_loss.item())

        del input_data, segmentation_output, attribute_output, model_output, segmentation_loss, attribute_loss
        torch.cuda.empty_cache()

      print('Saving Model...')
      torch.save(self._model.state_dict(), self._save_name)
      print('Model Saved')
    
  ''' Runs our model on unseen validation data to check we are not overfitting to training data
  '''
  def run_on_validation_data(self, display_outputs: bool = False, for_segmentation: bool = False) -> None:
    print('Running Validation Step...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self._model = self._model.to(device)
    total_epoch_validation_loss = 0
    for i, data_indexes in enumerate(self._validation_loader):
      self._model.eval()
      with torch.no_grad():
        input_data, output_data = self._get_data_for_indexes(data_indexes, device)
        model_output = self._model(input_data)
        loss = self._loss_func(model_output, output_data)
        total_epoch_validation_loss += (loss.item() * len(data_indexes))

        model_output = (model_output>self._OUTPUT_THRESHOLD).float()
        if i % 50 == 0:
          print(f'Validation Batch {i}, Current Batch Loss: {loss}')
        if i % 500 == 0 and display_outputs and for_segmentation:
          plot_predicted_and_actual(input_data[0].cpu(), model_output[0].cpu(), output_data[0].cpu())
        elif i % 500 == 0 and display_outputs:
          self._display_output_for_attributes_model(input_data[0].cpu(), output_data[0].cpu(), model_output[0].cpu())
    total_epoch_validation_loss /= len(self._validation_examples)
    self._per_epoch_validation_loss.append(total_epoch_validation_loss)
    print(f'Validation Loss: {total_epoch_validation_loss}')
    if display_outputs and for_segmentation:
      plot_predicted_and_actual(input_data[0].cpu(), model_output[0].cpu(), output_data[0].cpu())
    elif display_outputs:
      self._display_output_for_attributes_model(input_data[0].cpu(), output_data[0].cpu(), model_output[0].cpu())
    print('Finished Validation Step')
    print()

  ''' Displays an example of the output from the attributes model - displays input image alongside this
  '''
  def _display_output_for_attributes_model(self, input_data, actual_output, predicted_output):
    print('Actual Values for this image:\n', self._dataset.attribute_list_to_string(actual_output))
    print('Predicted Values for this image:\n', self._dataset.attribute_list_to_string(predicted_output))
    display_image(input_data)

