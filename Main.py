''' Main script, brings everything together - creates, trains and tests model '''

import sys
import os.path
from CelebADataset import CelebADataset
from LfwDataset import LfwDataset
from SegmentationNetwork import SegmentationNetwork
from AttributesNetwork import AttributesNetwork
from Training import Training
import torch
from Helper import save_model_for_mobile

''' Runs the code to start the Segmentation Network
'''
def run_segmentation_network(dataset_directory: str) -> None:
    dataset = CelebADataset(dataset_directory, for_segmentation=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.empty_cache()
    model = SegmentationNetwork(dataset.get_num_output_masks()).to(device=device)
    model_save_name = 'SegmentationModelNew'
    if os.path.exists(model_save_name + '.pt'):
        model.load(model_save_name + '.pt')
    model_training = Training(model, dataset, for_segmentation=True, batch_size=7, 
                                learning_rate=0.0001, save_name='SegmentationModel.pt')
    model_training.train()
    save_model_for_mobile(model, model_save_name, dataset[0][0].unsqueeze(dim=0))

''' Runs the code to start the Attributes network
'''
def run_attributes_network(dataset_directory: str) -> None:
    dataset = CelebADataset(dataset_directory, for_segmentation=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AttributesNetwork(dataset.get_num_attributes()).to(device=device)
    model_save_name = 'AttributesModel'
    if os.path.exists(model_save_name + '.pt'):
        model.load(model_save_name + '.pt')
    model_training = Training(model, dataset, for_segmentation=False, batch_size=50, 
                                learning_rate=0.01, save_name=model_save_name + '.pt')
    model_training.train()
    save_model_for_mobile(model, model_save_name)

''' Entry point for the program
'''
if __name__ == '__main__':
    current_directory = sys.path[0]
    dataset_directory = os.path.join(os.path.split(current_directory)[0], 'CelebAMask-HQ')
    run_segmentation_network(dataset_directory)
    