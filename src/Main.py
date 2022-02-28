''' Main script, brings everything together - creates, trains and tests model '''

import sys
import os.path

from CelebADataset import CelebADataset
from Networks.SegmentationNetwork import SegmentationNetwork
from Networks.AttributesNetwork import AttributesNetwork
from Networks.MultiNetwork import MultiNetwork
from Training import Training
import torch
import torch.nn as nn
from Helper import save_model_for_mobile, evaluate_model_accuracy
from Networks.NetworkType import NetworkType
import random

''' Trains a specific type of model
'''
def train_model(model: nn.Module, dataset: CelebADataset, batch_size: int, learning_rate: int, 
             num_epochs, save_name: str, display_outputs: bool = False):
    if os.path.exists(save_name + '.pt'):
        model.load_state_dict(torch.load(save_name + '.pt'))
    
    training = Training(model, dataset, batch_size, learning_rate, save_name, num_epochs, 
                        display_outputs, network_type, device)
    training.train()
    save_model_for_mobile(model, save_name, dataset[0][0].unsqueeze(dim=0))

''' Tests a model that we have already trained to get the accuracy of its predictions
'''
def test_model(model: nn.Module, dataset: CelebADataset, saved_name: str, 
                                network_type: NetworkType, num_samples: int):
    if not os.path.exists(saved_name + '.pt'):
        print(f'Model file {saved_name}.pt cannot be found')
        return
    model.load_state_dict(torch.load(saved_name + '.pt'))
    print('Loading required data...')
    #Generate 100 random indexes that will be used for testing - using 20000-30000 to use validation data
    index_values = random.sample(range(20000, 30000), num_samples)
    data_indexes = torch.Tensor(index_values)
    input_data, output_one, output_two = dataset.get_data_for_indexes(data_indexes)
    print('Data loaded')
    #Currently does not work for multi model, need to rework code to get this working
    if network_type == NetworkType.ATTRIBUTE:
        evaluate_model_accuracy(model, network_type, input_data, output_one, 0.5)
    elif network_type == NetworkType.SEGMENTATION:
        evaluate_model_accuracy(model, network_type, input_data, output_one, 0.8)
    evaluate_model_accuracy(model, network_type, input_data, output_two)

''' Entry point for the program
'''
if __name__ == '__main__':
    base_directory = os.path.dirname(sys.path[0])
    dataset_directory = os.path.join(os.path.split(base_directory)[0], 'CelebAMask-HQ')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #Change this to change which sort of model is run
    network_type = NetworkType.SEGMENTATION
    #Change this parameter to change between testing or training a model
    testing_model = True

    dataset = CelebADataset(dataset_directory, network_type, device)
    #Run the code for a training a model
    if network_type == NetworkType.SEGMENTATION:
        model = SegmentationNetwork(dataset.get_num_output_masks()).to(device)
        if not testing_model:
            train_model(model, dataset, 7, 0.0001, 10, 'segmentation_model', False)
        else:
            test_model(model, dataset, 'segmentation_model', network_type, 10)
    elif network_type == NetworkType.ATTRIBUTE:
        model = AttributesNetwork(dataset.get_num_attributes(), device=device).to(device)
        if not testing_model:
            train_model(model, dataset, 20, 0.001, 20, 'attributes_model', False)
        else:
            test_model(model, dataset, 'attributes_model', network_type, 10)
    else:
        model = MultiNetwork(dataset.get_num_output_masks(), dataset.get_num_attributes()).to(device)
        if not testing_model:
            train_model(model, dataset, 7, 0.0001, 20, 'multi_model', False)
        else:
            test_model(model, dataset, 'multi_model', network_type, 10)
        
    