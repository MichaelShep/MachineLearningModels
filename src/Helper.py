from typing import List, Tuple
import torch
import matplotlib.pyplot as plt # type: ignore
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

from Networks.NetworkType import NetworkType
  
def plot_predicted_and_actual(input_image: torch.Tensor, predicted: torch.Tensor, actual: torch.Tensor):
    ''' Creates a plot showing the difference between the predicted and actual outputs for the segmentation model

    Parameters
    ----------
    input_image: torch.Tensor
        The image which we are making the predictions from
    predicted: torch.Tensor
        The predictions that our model made
    actual: torch.Tensor
        The actual output for the image
    '''
    fig = plt.figure(figsize=(20, 4))
    display_data_element(input_image, actual, fig, 1)
    display_data_element(input_image, predicted, fig, 2)
    plt.show()

def display_image(image: torch.Tensor):
    ''' Displays a single image from a tensor

    Parameters
    ----------
    image: torch.Tensor
        The image to be displayed
    '''
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(image.detach().permute(1, 2, 0))
    plt.axis('off')
    plt.show()

def display_data_element(input_image: torch.Tensor, output_masks: torch.Tensor, fig: plt.figure, row: int):
    ''' Displays an input image along with a series masks associated with that image

    Parameters
    ----------
    input_image: torch.Tensor
        The input image we are going to display
    output_masks: torch.Tensor
        The list of masks that we are displaying with the image
    fig: plt.figure
        The matplotlib figure we are displaying the image on
    row: int
        The row that we should display our images on in the figure
    '''
    fig.add_subplot(2, len(output_masks) + 1, row + ((row - 1) * len(output_masks)))
    plt.imshow(input_image.detach().permute(1, 2, 0))
    plt.axis('off')

    for i in range(len(output_masks)):
        fig.add_subplot(2, len(output_masks) + 1, i + 1 + row + ((row - 1) * len(output_masks)))
        plt.imshow(output_masks[i].detach().unsqueeze(0).permute(1, 2, 0))
        plt.axis('off')

def create_conv_layer(in_chan: int, out_chan: int, device: str = 'cpu', kernal_size=3, stride=1, padding=1) -> nn.Conv2d:
    ''' Creates a convolution layer for our network - makes it easier to define our layers as contains default parameters

    Parameters
    ----------
    in_chan: int
        The number of input channels for our layer
    out_chan: int
        The number of output channels for our layer
    device: str
        The device which our layer is being run on - CUDA or cpu
    kernal_size: int
        The size of the kernal that will contain all the weights that will be passed over our image - will be square so only need to define 1 dimension
    stride: int
        The amount we move the kernal after each step
    padding: int
        The amount of padding we are adding around the image - will be padded with 0s
    Returns
    -------
    nn.Conv2d
        The convolutional layer formed from all our inputs
    '''
    return nn.Conv2d(in_channels=in_chan, out_channels=out_chan, 
                  kernel_size=kernal_size, stride=stride, padding=padding).to(device)

def create_double_conv(in_chan: int, out_chan: int, device: str = 'cpu') -> nn.Sequential:
    ''' Creates the double conv setup (two conv layers straight after each other) used in our segmentation network

    Parameters
    ----------
    in_chan: int
        The number of input channels to our layer
    out_chan: int
        The number of output channels from our layer
    device: str
        The device that our layer will be run on - CUDA or cpu
    Returns
    -------
    nn.Sequential
        A sequence of layers that make up the double conv structure
    '''
    return nn.Sequential(
        create_conv_layer(in_chan=in_chan, out_chan=out_chan, device=device),
        nn.ReLU(),
        create_conv_layer(in_chan=out_chan, out_chan=out_chan, device=device),
        nn.ReLU()
    ).to(device)

def save_model_for_mobile(model: torch.nn.Module, model_name: str, example_input: torch.Tensor):
    ''' Saves and optimizes a trained model in the mobile format using PyTorch Mobile

    Parameters
    ----------
    model: torch.nn.Module
        The model that we are wanting to save
    model_name: str
        The name that we have given to the model which will be used as the file name
    example_input: torch.Tensor
        An example input which could be passed through the model (needed for PyTorch Model)
    '''
    model = model.to('cpu')
    example_input = example_input.to('cpu')
    model.eval()
    script_model = torch.jit.trace(model, example_input)
    optimized_script_model = optimize_for_mobile(script_model)
    optimized_script_model._save_for_lite_interpreter(model_name + '.ptl')
    print('Mobile Model Saved')

def plot_loss_list(training_losses: List[float], validation_losses: List[float], network_type: NetworkType):
    ''' Use matplotlib to plot a curve showing how the loss values changed whilst training a model

    Parameters
    ----------
    training_losses: List[float]
        List containing the training losses from each epoch of training
    validation_losses: List[float]
        List containing the validation losses from each epoch of training
    network_type: NetworkType
        The type of network that has been trained
    '''
    indexes = list(range(0, len(training_losses)))
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111) 
    plt_title = ''
    if network_type == NetworkType.ATTRIBUTE:
        plt_title = 'Attributes Model'
    elif network_type == NetworkType.SEGMENTATION:
        plt_title = 'Segmentation Model'
    else:
        plt_title = 'Multi-Learning Model'
    ax.set_title(plt_title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.plot(indexes, training_losses, label='Training Loss', color='green')
    ax.plot(indexes, validation_losses, label='Validation Loss', color='blue')
    ax.legend(loc='best', fancybox=True, framealpha=0.5)
    plt.show()

def threshold_outputs(network_type: NetworkType, model_output: torch.Tensor, 
                      threshold_level: int, threshold_level_2: int = 0) -> torch.Tensor:
    ''' Converts the raw floating point outputs of a model into binary values based on a threshold

    Parameters
    ----------
    network_type: NetworkType
        The type of network that the outputs are from
    model_output: torch.Tensor
        The raw output that we want to apply thresholding to
    threshold_level: int
        The value which we want to threshold from - values above this will be 1, everything else 0
    threshold_level_2: int
        This is only used for multi network and is so that we can have different thresholds for segmentation and attribute
    Returns
    -------
    torch.Tensor
        The result of thresholding the original output
    '''
    if network_type != NetworkType.MULTI:
        threshold_output = (model_output>threshold_level).float()
        return threshold_output
    else:
        model_output_0 = (model_output[0]>threshold_level).float()
        model_output_1 = (model_output[1]>threshold_level_2).float()
        return (model_output_0, model_output_1) 
