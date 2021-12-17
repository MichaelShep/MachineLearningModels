''' Main script, brings everything together - creates, trains and tests model '''

import sys
import os.path
from CelebADataset import CelebADataset
from LfwDataset import LfwDataset
from SegmentationNetwork import SegmentationNetwork
from AttributesNetwork import AttributesNetwork
from Training import Training
import torch
from Helper import display_image

''' Runs the code to start the Segmentation Network
'''
def run_segmentation_network(dataset_directory: str) -> None:
    dataset = CelebADataset(dataset_directory)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.empty_cache()
    model = SegmentationNetwork(dataset.get_num_output_masks()).to(device=device)
    model.load_state_dict(torch.load('MODEL_SKIP.pt'))
    model_training = Training(model, dataset)
    model_training.train()

''' Runs the code to start the Attributes network
'''
def run_attributes_network(dataset_directory: str) -> None:
    dataset = CelebADataset(dataset_directory)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AttributesNetwork(dataset.get_num_attributes()).to(device=device)

''' Entry point for the program
'''
if __name__ == '__main__':
    current_directory = sys.path[0]
    dataset_directory = os.path.join(os.path.split(current_directory)[0], 'CelebAMask-HQ')
    #run_segmentation_network(current_directory)
    run_attributes_network(dataset_directory)
    