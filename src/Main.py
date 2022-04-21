import sys
import os.path
import torch
import torch.nn as nn
from enum import Enum

from CelebADataset import CelebADataset
from Networks.SegmentationNetwork import SegmentationNetwork
from Networks.AttributesNetwork import AttributesNetwork
from Networks.MultiNetwork import MultiNetwork
from Training import Training
from Helper import save_model_for_mobile
from Networks.NetworkType import NetworkType
import Testing

class RunMode(Enum):
    ''' Enum Class used to define which version of the code we are currently running '''
    Training = 'Training'
    Testing = 'Testing'
    Comparing = 'Comparing'

def train_model(model: nn.Module, dataset: CelebADataset, batch_size: int, learning_rate: int, 
             num_epochs: int, save_name: str, network_type: NetworkType, device: str, display_outputs: bool = False):
    ''' Trains one of our models and saves all results from training 
    
    Parameters
    ----------
    model: nn.Module
        The model to be trained
    dataset: CelebADataset
        The dataset that we will use to train the model with
    batch_size: int
        The amount of images processed at a time by the model
    learning_rate: int
        The size of the backpropogation steps used to update the network after each batch
    num_epochs: int
        The number of times we will pass over the complete dataset
    save_name: str
        The name of the file which we will use to save the trained model when completed
    network_type: NetworkType
        The type of network that we are training
    device: str
        The device we are going to use for training - will be either cpu or cuda (for GPU enabled)
    display_outputs: bool
        Whether we want to display what the outputs we look like as we progress through training '''
    
    if os.path.exists(save_name + '.pt'):
        model.load_state_dict(torch.load(save_name + '.pt', map_location=device))
    
    training = Training(model, dataset, batch_size, learning_rate, save_name, num_epochs, 
                        display_outputs, network_type, device)
    training.train()
    save_model_for_mobile(model, save_name, dataset[0][0].unsqueeze(dim=0))

def test_model(model: nn.Module, dataset: CelebADataset, saved_name: str, 
                                network_type: NetworkType, num_samples: int, device: str):
    ''' Tests an already trained model to get how accurate it is for unseen data
    
    Parameters
    ----------
    model: nn.Module
        The model that has been trained which will be used for testing
    dataset: CelebADataset
        The dataset which contains the unseen data that we will be testing with
    saved_name: str
        The name of the file where all the saved data for the model is
    network_type: NetworkType
        The type of network we are working with
    num_samples: int
        The number of images we will be getting the accuracy for in each batch
    device: str
        The device which we are currently using (cuda or cpu) - needed to ensure we can load the data correctly '''

    if not os.path.exists(saved_name + '.pt'):
        print(f'Model file {saved_name}.pt cannot be found')
        return
    model.load_state_dict(torch.load(saved_name + '.pt', map_location=device))
    Testing.test_model(model, dataset, network_type, num_samples)

def start_program():
    ''' Starting point for the program which will choose what code should be run based on the run settings that a user provides '''
    base_directory = os.path.dirname(sys.path[0])
    dataset_directory = os.path.join(os.path.split(base_directory)[0], 'CelebAMask-HQ')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #Change this to change which sort of model is run and what we are doing with the model
    network_type = NetworkType.ATTRIBUTE
    run_mode = RunMode.Comparing

    dataset = CelebADataset(dataset_directory, network_type, device)
    #Run the code corrosponding to our model and run mode selection
    if run_mode != RunMode.Comparing:
        if network_type == NetworkType.SEGMENTATION:
            model = SegmentationNetwork(dataset.get_num_output_masks()).to(device)
            if run_mode == RunMode.Training:
                train_model(model, dataset, 5, 0.0001, 10, 'segmentation_model', network_type, device, False)
            elif run_mode == RunMode.Testing:
                test_model(model, dataset, 'segmentation_model', network_type, 20, device)
        elif network_type == NetworkType.ATTRIBUTE:
            model = AttributesNetwork(dataset.get_num_attributes(), device=device).to(device)
            if run_mode == RunMode.Training:
                train_model(model, dataset, 20, 0.001, 20, 'attributes_model', network_type, device, False)
            elif run_mode == RunMode.Testing:
                test_model(model, dataset, 'attributes_model', network_type, 20, device)
        else:
            model = MultiNetwork(dataset.get_num_output_masks(), dataset.get_num_attributes()).to(device)
            if run_mode == RunMode.Training:
                train_model(model, dataset, 7, 0.0001, 20, 'multi_model', network_type, device, False)
            elif run_mode == RunMode.Testing:
                test_model(model, dataset, 'multi_model', network_type, 20, device)
    else:
        Testing.compare_model_accuracies('accuracies_segmentation', 'accuracies_attribute', 'accuracies_multi')

#Entry point for the program
if __name__ == '__main__':
    start_program()
        
    