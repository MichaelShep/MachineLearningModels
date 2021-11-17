''' Main script, brings everything together - creates, trains and tests model '''

import sys
import os.path
from CelebADataset import CelebADataset
from SegmentationNetwork import SegmentationNetwork
from Training import Training
import torch
from Helper import plot_predicted_and_actual


def predict_output(model: SegmentationNetwork, data):
    (input_element, output_element) = data
    model_output = model(input_element)
    print('About to display model')
    plot_predicted_and_actual(model_output, output_element)

if __name__ == '__main__':
    '''current_directory = sys.path[0]
    dataset_directory = os.path.join(os.path.split(current_directory)[0], 'CelebAMask-HQ')
    image_directory = os.path.join(dataset_directory, 'CelebA-HQ-img')
    features_directory = os.path.join(dataset_directory, 'CelebAMask-HQ-mask-anno')

    #Load all image and map data and split it randomly into testing and training data
    dataset = CelebADataset(image_directory, features_directory)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.empty_cache()
    model = SegmentationNetwork().to(device=device)
    model.load_state_dict(torch.load('MODEL.pt'))
    model_training = Training(model, dataset)
    model_training.train()'''

    model = SegmentationNetwork()
    model.load_state_dict(torch.load('MODEL.pt'))
    predict_output()