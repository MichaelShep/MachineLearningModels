''' Main script, brings everything together - creates, trains and tests model '''

import sys
import os.path
from DataLoader import ImageDataset
from SegmentationNetwork import SegmentationNetwork
from Training import Training
import Helper

if __name__ == '__main__':
    current_directory = sys.path[0]
    dataset_directory = os.path.join(os.path.split(current_directory)[0], 'CelebAMask-HQ')
    image_directory = os.path.join(dataset_directory, 'CelebA-HQ-img')
    features_directory = os.path.join(dataset_directory, 'CelebAMask-HQ-mask-anno')

    #Note: Images in this dataset are 1024 x 1024
    dataset = ImageDataset(image_directory, features_directory)
    train_data = dataset.load_data(0, 0)

    model = SegmentationNetwork()
    model_training = Training(model, train_data)

    output = model_training.train()

    print('Output Shape:', output.shape)
    Helper.display_outputs(output.detach()[0])