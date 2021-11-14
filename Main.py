''' Main script, brings everything together - creates, trains and tests model '''

import sys
import os.path
from DataSet import ImageDataset
from SegmentationNetwork import SegmentationNetwork
from Training import Training

if __name__ == '__main__':
    NUM_TRAINING_EXAMPLES = 50
    MINI_BATCH_SIZE = 1

    current_directory = sys.path[0]
    dataset_directory = os.path.join(os.path.split(current_directory)[0], 'CelebAMask-HQ')
    image_directory = os.path.join(dataset_directory, 'CelebA-HQ-img')
    features_directory = os.path.join(dataset_directory, 'CelebAMask-HQ-mask-anno')

    #Load all image and map data and split it randomly into testing and training data
    dataset = ImageDataset(image_directory, features_directory)
    model = SegmentationNetwork()
    model_training = Training(model, dataset)
    model_training.train()