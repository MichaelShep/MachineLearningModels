import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from CelebADataset import CelebADataset
from Helper import plot_predicted_and_actual, display_image, plot_loss_list, threshold_outputs
from Networks.NetworkType import NetworkType

class Training():
    ''' Class where all the training of models happens'''
    _NUM_TRAINING_EXAMPLES = 20000
    _SEGMENTATION_OUTPUT_THRESHOLD = 0.8
    _ATTRIBUTES_OUTPUT_THRESHOLD = 0.5

    def __init__(self, model: nn.Module, dataset: CelebADataset, batch_size: int, learning_rate: float,
                    save_name: str, num_epochs: int, display_outputs: bool, network_type: NetworkType, device: str):
        ''' Initialises all member variables - splits data into training and validation and sets up data loader
        
        Parameters
        ----------
        model: nn.Module
            The model which we are training
        dataset: CelebADataset
            The dataset which contains the training and validation images used to train this dataset
        batch_size: int
            The amount of images that we pass through the network in a single forward pass
        learning_rate: float
            The size of the steps taken when updating the network weights through backprop
        save_name: str
            The name that will we give to the file containing all the saved data for this model
        num_epochs: int
            The number of times we will pass through the full dataset through the model
        display_outputs: bool
            Whether we should display example outputs from our model during the training process
        network_type: NetworkType
            The type of network which we are training
        device: str
            The device we are performing training on: cuda if using the gpu, cpu otherwise '''

        self._model = model
        self._dataset = dataset
        self._network_type = network_type
        self._save_name = save_name
        self._display_outputs = display_outputs
        self._num_epochs = num_epochs
        self._device = device
        self._training_examples, self._validation_examples = dataset.get_train_test_split(self._NUM_TRAINING_EXAMPLES)

        self._training_loader = DataLoader(self._training_examples, batch_size=batch_size, shuffle=True, num_workers=2)
        self._validation_loader = DataLoader(self._validation_examples, batch_size=batch_size, shuffle=True, num_workers=2)

        #Using different loss functions for the different forms of network
        if network_type == NetworkType.ATTRIBUTE:
            self._loss_func = nn.MSELoss()
        elif network_type == NetworkType.SEGMENTATION:
            self._loss_func = nn.BCEWithLogitsLoss()
        else:
            self._mse_loss = nn.MSELoss()
            self._bce_loss = nn.BCEWithLogitsLoss()
        self._optim = torch.optim.Adam(self._model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        self._per_epoch_training_loss = []
        self._per_epoch_validation_loss = []

    def train(self) -> None:
        ''' Performs the actual training process of our model '''
        self.run_on_validation_data()
        for epoch in range(self._num_epochs):
            print('Epoch:', epoch)
            self._model.train()
            total_epoch_loss = 0
            for i, data_indexes in enumerate(self._training_loader):
                input_data, output_one, output_two = self._dataset.get_data_for_indexes(data_indexes)
                model_output = self._model(input_data)

                loss = self._compute_loss_and_display(model_output, output_one, output_two, i)

                #Add an initial value of our tracked loss to be used as the starting point for the loss
                if len(self._per_epoch_training_loss) == 0:
                    self._per_epoch_training_loss.append(loss.item())
                total_epoch_loss += (loss.item() * len(data_indexes))
                self._optim.zero_grad()
                loss.backward()
                self._optim.step()

                if i % 1000 == 0 and i != 0:
                    print('Saving Model...')
                    torch.save(self._model.state_dict(), self._save_name + '.pt')
                    print('Model Saved.')
                elif i == 0 and self._network_type == NetworkType.ATTRIBUTE:
                    print('Pre processed predicted output:', model_output[0])
                    model_output = threshold_outputs(self._network_type, model_output, self._ATTRIBUTES_OUTPUT_THRESHOLD)
                    accuracy = self._model.evaluate_prediction_accuracy(model_output[0], output_one[0])
                    print('Example prediction accurracy:', accuracy)
                    print('Model Output:', model_output[0])
                    print('Actual Output:', output_one[0])
                
                if self._display_outputs:
                    self._display_outputs(input_data, model_output, output_one, output_two, i)

                #Clear all unneeded memory - without this will get a memory error
                del input_data, output_one, output_two, model_output, loss
                torch.cuda.empty_cache()

        print('Saving Model...')  
        total_epoch_loss /= len(self._training_examples)
        self._per_epoch_training_loss.append(total_epoch_loss)
        print('Average Loss for epoch:', total_epoch_loss)
        torch.save(self._model.state_dict(), self._save_name + '.pt')
        print('Model Saved.')
        self.run_on_validation_data()
        
        #Show training loss curve once the model has been trained
        plot_loss_list(self._per_epoch_training_loss, self._per_epoch_validation_loss, self._network_type)
        
    def run_on_validation_data(self) -> None:
        ''' Runs our model on unseen validation data to check we are not overfitting to training data '''
        print('Running Validation Step...')
        self._model = self._model.to(self._device)
        total_epoch_validation_loss = 0
        for i, data_indexes in enumerate(self._validation_loader):
            self._model.eval()
            with torch.no_grad():
                input_data, output_one, output_two = self._dataset.get_data_for_indexes(data_indexes)
                model_output = self._model(input_data)
                loss = self._compute_loss_and_display(model_output, output_one, output_two, i)
                total_epoch_validation_loss += (loss.item() * len(data_indexes))

                if self._display_outputs:
                    self._display_outputs(input_data, model_output, output_one, output_two, i)
        total_epoch_validation_loss /= len(self._validation_examples)
        self._per_epoch_validation_loss.append(total_epoch_validation_loss)
        print(f'Validation Loss: {total_epoch_validation_loss}')
        print('Finished Validation Step')
        print()

    ''' Computes the loss for a specific set of outputs and displays them to the screen for certain batches
    '''
    def _compute_loss_and_display(self, model_output: torch.Tensor, output_one: torch.Tensor, output_two: torch.Tensor, batch_index: int) -> torch.Tensor:
        ''' Computes the loss for a specific set of outputs and displays them to the screen for certain batches
        
        Parameters
        ----------
        model_output: torch.Tensor
            The output from our models - its predection
        output_one: torch.Tensor
            What the actual output is for the corrosponding input
        output_two: torch.Tensor
            Only used for multi model - contains the attributes part of the output as the first output is segmentation
        batch_index: int
            The value of the current batch in the epoch that we are currently processing '''

        if self._network_type == NetworkType.MULTI:
            segmentation_loss = self._bce_loss(model_output[0], output_one)
            attribute_loss = self._mse_loss(model_output[1], output_two)
            if batch_index % 50 == 0:
                print('Batch:', batch_index, 'Segmentation Loss:', segmentation_loss.item(), 'Attribute Loss:', attribute_loss.item())
            return segmentation_loss + attribute_loss
        
        #Section for segmentation and attribute models
        loss = self._loss_func(model_output, output_one)
        if batch_index % 50 == 0:
            print('Batch:', batch_index, 'Loss:', loss.item())
        return loss

    def _display_outputs(self, input_data: torch.Tensor, model_output: torch.Tensor, 
                        output_one: torch.Tensor, output_two: torch.Tensor, batch_index: int) -> None:   
        ''' Displays an example output from our model
        
        Parameters
        ----------
        input_data: torch.Tensor
            The input data we are using for our example
        model_output: torch.Tensor
            The output that our model gave on the provided input
        output_one: torch.Tensor
            The actual output that should have been given for the provided input
        output_two: torch.Tensor
            Only used for Multi model - contains the attribute output as first output contains segmentation
        batch_index: int
            The value of the current batch that we are processing '''                 
        if batch_index % 500 == 0:
            if self._network_type == NetworkType.SEGMENTATION:
                model_output = threshold_outputs(self._network_type, model_output, self._SEGMENTATION_OUTPUT_THRESHOLD)
                plot_predicted_and_actual(input_data[0].cpu(), model_output[0].cpu(), output_one[0].cpu())
            elif self._network_type == NetworkType.ATTRIBUTE:
                model_output = threshold_outputs(self._network_type, model_output, self._ATTRIBUTES_OUTPUT_THRESHOLD)
                self._display_output_for_attributes_model(input_data[0].cpu(), output_one[0].cpu(), model_output[0].cpu())
            else:
                model_output = threshold_outputs(self._network_type, model_output, self._SEGMENTATION_OUTPUT_THRESHOLD, self._ATTRIBUTES_OUTPUT_THRESHOLD)
                plot_predicted_and_actual(input_data[0].cpu(), model_output[0][0].cpu(), output_one[0].cpu())
                self._display_output_for_attributes_model(input_data[0].cpu(), output_two[0].cpu(), model_output[0][1].cpu())

    def _display_output_for_attributes_model(self, input_data: torch.Tensor, actual_output: torch.Tensor, predicted_output: torch.Tensor):
        ''' Displays an example of the output from the attributes model - displays input image alongside this
        
        Parameters
        ----------
        input_data: torch.Tensor
            The input which we are displaying the model output for
        actual_output: torch.Tensor
            The correct output value from the dataset for this input
        predicted_output: torch.Tensor
            The output that was predicted for this input by our model '''
        print('Actual Values for this image:\n', self._dataset.attribute_list_to_string(actual_output))
        print('Predicted Values for this image:\n', self._dataset.attribute_list_to_string(predicted_output))
        display_image(input_data)

