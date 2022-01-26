''' Main script, brings everything together - creates, trains and tests model '''

import sys
import os.path

from CelebADataset import CelebADataset
from SegmentationNetwork import SegmentationNetwork
from AttributesNetwork import AttributesNetwork
from MultiNetwork import MultiNetwork
from Training import Training
import torch
from Helper import save_model_for_mobile
from NetworkType import NetworkType

''' Runs and trains a model
'''
def run_model(network_type: NetworkType, batch_size: int, learning_rate: int, 
             num_epochs, save_name: str, display_outputs: bool = False):
    current_directory = sys.path[0]
    dataset_directory = os.path.join(os.path.split(current_directory)[0], 'CelebAMask-HQ')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = CelebADataset(dataset_directory, network_type)

    if network_type == NetworkType.SEGMENTATION:
        model = SegmentationNetwork(dataset.get_num_output_masks())
    elif network_type == NetworkType.ATTRIBUTE:
        model = AttributesNetwork(dataset.get_num_attributes())
    else:
        model = MultiNetwork(dataset.get_num_output_masks(), dataset.get_num_attributes())
    model = model.to(device)
    
    if os.path.exists(save_name + '.pt'):
        model.load_state_dict(torch.load(save_name + '.pt'))
    
    training = Training(model, dataset, batch_size, learning_rate, save_name, num_epochs, 
                        display_outputs, network_type, device)
    training.train()
    save_model_for_mobile(model, save_name, dataset[0][0].unsqueeze(dim=0))


''' Entry point for the program
'''
if __name__ == '__main__':
    #Uncomment this line to run the segmentation network
    #run_model(NetworkType.SEGMENTATION, 7, 0.0001, 10, 'segmentation_model', False)
    #Uncomment this line to run the atttributes network
    #run_model(NetworkType.ATTRIBUTE, 20, 0.0001, 20, 'attributes_model', False)
    #Uncomment this line to run the multi-learning network
    run_model(NetworkType.MULTI, 7, 0.0001, 20, 'multi_model', False)
    