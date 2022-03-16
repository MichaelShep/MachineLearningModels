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
from Helper import save_model_for_mobile, evaluate_model_accuracy, compare_model_accuracies
from Networks.NetworkType import NetworkType
import random
import pickle
from enum import Enum

''' Enum which is used to state which version of the code we are currently running '''
class RunMode(Enum):
    Training = 'Training'
    Testing = 'Testing'
    Comparing = 'Comparing'

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
                                network_type: NetworkType, num_samples: int, device: str):
    if not os.path.exists(saved_name + '.pt'):
        print(f'Model file {saved_name}.pt cannot be found')
        return
    model.load_state_dict(torch.load(saved_name + '.pt', map_location=device))
    print('Loading required data...')
    #Generate random indexes that will be used for testing - using 20000-30000 to use validation data
    #Will be using 50 batches so need 50 * the amount of images we want to use for each batch
    index_values = random.sample(range(20000, 30000), num_samples * 50)
    #Store all the accurracies from each batch so that can be used statistical test
    batch_accuracies = []

    for i in range(0, len(index_values), num_samples):
        print(f'Running for batch {(i/num_samples) + 1}')
        data_indexes = torch.Tensor(index_values[i:i+num_samples])
        input_data, output_one, output_two = dataset.get_data_for_indexes(data_indexes)
        #Currently does not work for multi model, need to rework code to get this working
        if network_type == NetworkType.ATTRIBUTE:
            accuracy = evaluate_model_accuracy(model, network_type, input_data, output_one, 0.5)
            print('Accuracy for this batch: ', accuracy)
        elif network_type == NetworkType.SEGMENTATION:
            accuracy = evaluate_model_accuracy(model, network_type, input_data, output_one, 0.8)
            print('Accuracy for this batch: ', accuracy)
        else:
            segmentation_accuracy = evaluate_model_accuracy(model, network_type, input_data, output_one, 0.8, True)
            attributes_accuracy = evaluate_model_accuracy(model, network_type, input_data, output_two, 0.5, False)
            accuracy = (segmentation_accuracy, attributes_accuracy)
            print(f'Segmentation Accuracy: {segmentation_accuracy}, Attributes Accuracy: {attributes_accuracy}, Overall Accuracy: {(segmentation_accuracy + attributes_accuracy) / 2}')
        batch_accuracies.append(accuracy)

    #Saved all the collected accuracies
    save_data = dict(accuracies=batch_accuracies)
    with open(f'accuracies_{network_type.value.lower()}.pt', 'wb') as save_file:
        pickle.dump(save_data, save_file)
    print('Accuracy Data saved')

''' Entry point for the program
'''
if __name__ == '__main__':
    base_directory = os.path.dirname(sys.path[0])
    dataset_directory = os.path.join(os.path.split(base_directory)[0], 'CelebAMask-HQ')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #Change this to change which sort of model is run and what we are doing with the model
    network_type = NetworkType.ATTRIBUTE
    run_mode = RunMode.Comparing

    dataset = CelebADataset(dataset_directory, network_type, device)
    #Run the code for a training a model
    if run_mode != RunMode.Comparing:
        if network_type == NetworkType.SEGMENTATION:
            model = SegmentationNetwork(dataset.get_num_output_masks()).to(device)
            if run_mode == RunMode.Training:
                train_model(model, dataset, 5, 0.0001, 10, 'segmentation_model', False)
            elif run_mode == RunMode.Testing:
                test_model(model, dataset, 'segmentation_model', network_type, 10, device)
        elif network_type == NetworkType.ATTRIBUTE:
            model = AttributesNetwork(dataset.get_num_attributes(), device=device).to(device)
            if run_mode == RunMode.Training:
                train_model(model, dataset, 20, 0.001, 20, 'attributes_model', False)
            elif run_mode == RunMode.Testing:
                test_model(model, dataset, 'attributes_model', network_type, 10, device)
        else:
            model = MultiNetwork(dataset.get_num_output_masks(), dataset.get_num_attributes()).to(device)
            if run_mode == RunMode.Training:
                train_model(model, dataset, 7, 0.0001, 20, 'multi_model', False)
            elif run_mode == RunMode.Testing:
                test_model(model, dataset, 'multi_model', network_type, 10, device)
    else:
        compare_model_accuracies('accuracies_segmentation', 'accuracies_attribute', 'accuracies_multi')
        
    