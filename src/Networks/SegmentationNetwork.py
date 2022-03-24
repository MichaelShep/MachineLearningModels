import torch.nn as nn
import torch
from Helper import create_conv_layer, create_double_conv

class SegmentationNetwork(nn.Module):
    ''' Creates a network for the face segmentation task - takes an input face and creates masks containing the different features of the face '''
    
    def __init__(self, num_output_masks: int):
        ''' Initalizes all the layers of the network
        
        Parameters
        ----------
        num_output_masks: int
            The number of masks which each image has
        '''
        super(SegmentationNetwork, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._num_output_masks = num_output_masks

        #Create the layers for the network - Creating an architecture very similar to that used in U-Net
        self._skip_connections = [torch.Tensor()] * 5
        self._max_pool =  nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self._upsample = nn.Upsample(scale_factor=2)

        self._contracting_path = nn.ModuleList([
        create_double_conv(in_chan=3, out_chan=8, device=device), create_double_conv(in_chan=8, out_chan=16, device=device),
        create_double_conv(in_chan=16, out_chan=32, device=device), create_double_conv(in_chan=32, out_chan=64, device=device),
        create_double_conv(in_chan=64, out_chan=128, device=device), create_double_conv(in_chan=128, out_chan=256, device=device),
        ])

        self._expanding_path = nn.ModuleList([
        create_double_conv(in_chan=256, out_chan=128, device=device), create_double_conv(in_chan=128, out_chan=64, device=device),
        create_double_conv(in_chan=64, out_chan=32, device=device), create_double_conv(in_chan=32, out_chan=16, device=device),
        create_double_conv(in_chan=16, out_chan=8, device=device), create_conv_layer(8, self._num_output_masks, device, kernal_size=1, padding=0)
        ])

    def _perform_contracting_path(self, x: torch.Tensor) -> torch.Tensor:
        ''' Runs all the layers which reduce the dimensions of the image - first part of the network
        
        Parameters 
        ----------
        x: torch.Tensor
            The input to this part of the network - since this is the first section will just be the input image
        Returns
        -------
        torch.Tensor
            The output from this part of the network
        '''
        for i in range(len(self._contracting_path) - 1):
            x = self._perform_contracting_layer(x, i)

        #For final part of path, don't want to perform skip connection or max pooling operations
        x = self._contracting_path[len(self._contracting_path) - 1](x)
        return x

    def _perform_expanding_path(self, x: torch.Tensor) -> torch.Tensor:
        ''' Runs all the layers which take the small dimension image and convert this into the output mask

        Parameters
        ----------
        x: torch.Tensor
            The input to this part of the network - will be the output of the contracting path
        Returns
        -------
        torch.Tensor
            The output of this part of the network - will be the final output masks
        '''
        for i in range(len(self._expanding_path) - 1):
            x = self._perform_expanding_layer(x, i)

        #For final part of expanding path, do not need to upsample or use skip connection
        x = self._expanding_path[len(self._expanding_path) - 1](x)
        return x

    def _perform_contracting_layer(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        ''' Performs a single layer from the contracting part of the network - also stores a skip connection for use later

        Parameters
        ----------
        x: torch.Tensor
            The input for this layer
        idx: int
            The index of the layer that we are wanting to perform
        Returns
        -------
        torch.Tensor
            The output from this layer
        '''
        x = self._contracting_path[idx](x)
        self._skip_connections[idx] = x
        return self._max_pool(x)

    def _perform_expanding_layer(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        ''' Performs a single layer from the expanding part of the network - makes use of previous stored skip connections

        Parameters
        ----------
        x: torch.Tensor
            The input for this layer
        idx: int
            The index of the layer that we are wanting to perform
        Returns
        -------
        torch.Tensor
            The output from this layer
        '''
        x = self._upsample(x)
        x = self._expanding_path[idx](x)
        x = x + self._skip_connections[len(self._skip_connections) - 1 - idx]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' Performs a full forward pass through the network

        Parameters
        ----------
        x: torch.Tensor
            The input image that we are wanting to segment
        Returns
        -------
        torch.Tensor
            The output of the network - all the segmentation masks for the image
        '''
        x = self._perform_contracting_path(x)
        x = self._perform_expanding_path(x)

        return x

    def evaluate_prediction_accuracy(predicted_image: torch.Tensor, actual_image: torch.Tensor) -> float:
        ''' Evaluates how accurate out model output for one of the masks is compared to the actual mask that it should be

        Parameters
        ----------
        predicted_image: torch.Tensor
            The prediction from our model - consists of 1s and 0s which make up an output mask
        actual_image: torch.Tensor
            The output that our model is aiming for
        Returns
        -------
        float
            The accuracy of our model - the amount of correct predictions out of total predictions
        '''
        correct_pixels = 0
        for row in range(len(predicted_image)):
            for col in range(len(predicted_image[row])):
                if predicted_image[row][col] == actual_image[row][col]:
                    correct_pixels += 1

        #Get the total number of pixels by multiplying the length of rows by length of cols
        total_num_pixels = len(predicted_image) * len(predicted_image[0])
        accuracy = float(correct_pixels) / float(total_num_pixels)

        return accuracy