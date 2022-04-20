from audioop import mul
import torch.nn as nn
import torch
import pickle
import random
import matplotlib.pyplot as plt
from typing import List
from scipy.stats import mannwhitneyu

from CelebADataset import CelebADataset
from Networks.NetworkType import NetworkType
from Helper import threshold_outputs

def test_model(model: nn.Module, dataset: CelebADataset, network_type: NetworkType, num_samples: int):
    ''' Tests a model that has been trained by getting the accuracy of its predictions

    Parameters
    ----------
    model: nn.Module
        The model that we are going to test
    dataset: CelebADataset
        The dataset we are going to use for testing
    network_type: NetworkType
        The type of the network that has been trained
    num_samples: int
        The number of testing samples we want to collect
    '''
    #Generate random indexes that will be used for testing - using 20000-30000 to use validation data
    #Will be using 50 batches so need 50 * the amount of images we want to use for each batch
    index_values = random.sample(range(20000, 30000), num_samples * 50)
    #Store all the accurracies from each batch so that can be used statistical test
    batch_accuracies = []

    for i in range(0, len(index_values), num_samples):
        print(f'Running for batch {(i/num_samples) + 1}')
        data_indexes = torch.Tensor(index_values[i:i+num_samples])
        input_data, output_one, output_two = dataset.get_data_for_indexes(data_indexes)
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

def evaluate_model_accuracy(model: nn.Module, network_type: NetworkType,
                            input_data: torch.Tensor, output_data: torch.Tensor, threshold_level: int, for_multi_segmentation_output: bool = False) -> float:
    ''' Evaluates the prediction accuracy of a model

    Parameters
    ----------
    model: nn.Module
        The model we are testing
    network_type: NetworkType
        The type of the model that we are testing
    input_data: torch.Tensor
        The input data which we are going to use to evaulate the accuracy with
    output_data: torch.Tensor
        The actual output data that should be produced for the given input data
    threshold_level: int
        The threshold level we will use to convert our raw floating point model outputs into binary
    for_multi_segmentation_output: bool
        Only used for multi model and is a flag for whether we are currently getting segmentation or attribute output

    Returns
    -------
    float:
        The computed accuracy of the provided model
    '''
    model_predictions = model(input_data)
    #For Multi model, get which part of the output we are currently dealing with and treat model as that form of network
    if network_type == NetworkType.MULTI:
        if for_multi_segmentation_output:
            network_type = NetworkType.SEGMENTATION
            model_predictions = model_predictions[0]
        else:
            network_type = NetworkType.ATTRIBUTE
            model_predictions = model_predictions[1]
    model_predictions = threshold_outputs(network_type, model_predictions, threshold_level)

    #Keeps track of the prediction variables for the attribute output
    correct_predictions = 0
    total_predictions = 0
    
    #Keeps track of the prediction variables for the segmentation output
    total_intersection_over_union = 0
    total_output_masks = 0
    for i in range(len(model_predictions)):
        if network_type == NetworkType.ATTRIBUTE:
            for j in range(len(model_predictions[i])):
                if model_predictions[i][j] == output_data[i][j]:
                    correct_predictions += 1
                total_predictions += 1
        if network_type == NetworkType.SEGMENTATION:
            for j in range(len(model_predictions[i])):
                total_intersection_over_union += compute_intersection_over_union(model_predictions[i][j], output_data[i][j])
                total_output_masks += 1
  
    if network_type == NetworkType.ATTRIBUTE:
        return correct_predictions / total_predictions
    return total_intersection_over_union / total_output_masks

def compute_intersection_over_union(predicted_mask: torch.Tensor, actual_mask: torch.Tensor) -> float:
    ''' Computes the intersection over union metric for a predicted mask and actual mask - used for evaulating segmentation output
    Only looks at the areas of the image where the output is 1 (displays white when viewing)

    Parameters
    ----------
    predicted_mask: torch.Tensor
        Tensor containing the predicted mask
    actual_mask: torch.Tensor
        Tensor containing the actual mask that we are aiming for with our model

    Returns
    -------
    float:
        The intersection over union of the two masks
    '''
    common_pixels = 0
    total_pixels = 0

    for y in range(len(predicted_mask)):
        for x in range(len(predicted_mask[y])):
            if predicted_mask[y][x] == 1 and actual_mask[y][x] == 1:
                common_pixels += 1
                total_pixels += 1
            elif predicted_mask[y][x] == 1 or actual_mask[y][x] == 1:
                total_pixels += 1
    
    #If there are no pixels in the given masks, this means for this was an empty mask that was correctly predicted by our model, so should output 1 (fully accurate)
    if total_pixels == 0:
        return 1
    return common_pixels / total_pixels

def compare_model_accuracies(segmentation_file_name: str, attributes_file_name: str, multi_file_name: str):
    ''' Compares accuracies of all 3 types of model by loading the saved accuracy data from file
    
    Parameters
    ----------
    segmentation_file_name: str
        The file where all the segmentation accuracies are stored
    attributes_file_name: str
        The file where all the attributes accuracies are stored
    multi_file_name: str
        The file where all the multi-model accuracies are stored
    '''
    with open(f'{segmentation_file_name}.pt', 'rb') as segmentation_file:
        segmentation_accuracies = pickle.load(segmentation_file)['accuracies']
    with open(f'{attributes_file_name}.pt', 'rb') as attributes_file:
        attributes_accuracies = pickle.load(attributes_file)['accuracies']
    with open(f'{multi_file_name}.pt', 'rb') as multi_file:
        multi_accuracies = pickle.load(multi_file)['accuracies']

    #Create box plot for comparing standard segmentation model output to multi model
    fig = plt.figure(figsize=(10, 7))
    segmentation_display = fig.add_subplot(111)
    segmentation_display.set_title('Segmentation Outputs Comparison')
    segmentation_display.set_xticklabels(['Segmentation', 'Multi'])
    segmentation_display.set_xlabel('Model Type')
    segmentation_display.set_ylabel('Accuracy')

    segmentation_display.boxplot([segmentation_accuracies, [item[0] for item in multi_accuracies]])
    plt.show()

    fig = plt.figure(figsize=(10, 7))
    attributes_display = fig.add_subplot(111)
    attributes_display.set_title('Attributes Outputs Comparison')
    attributes_display.set_xticklabels(['Attribute', 'Multi'])
    attributes_display.set_xlabel('Model Type')
    attributes_display.set_ylabel('Accuracy')

    attributes_display.boxplot([attributes_accuracies, [item[1] for item in multi_accuracies]])
    plt.show()

    perform_statistical_tests(segmentation_accuracies, attributes_accuracies, multi_accuracies)

def perform_statistical_tests(segmentation_accuracies: List[int], attributes_accuracies: List[int], multi_accuracies: List[int]):
    ''' Performs statistical tests between the multi model and individual models to see if there is any significant difference

    Parameters
    ----------
    segmentation_accuracies: List[int]
        The list containing all the segmentation model accuracies
    attributes_accuracies: List[int]
        The list containing all the attributes model accuracies
    multi_accuracies: List[int]
        The list containing all the multi model accuracies
    '''
    segmentation_u, segmentation_p = mannwhitneyu([item[0] for item in multi_accuracies], segmentation_accuracies)
    print(f'Segmentation U-Value: {segmentation_u} , P-Value: {segmentation_p:.10f}')

    attributes_u, attributes_p = mannwhitneyu([item[1] for item in multi_accuracies], attributes_accuracies)
    print(f'Attributes U-Value: {attributes_u} P-Value: {attributes_p:.10f}')
