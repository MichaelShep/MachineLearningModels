''' Main script, brings everything together - creates, trains and tests model '''

import sys
import os.path

from torchvision import transforms

from DataSet import ImageDataset
from SegmentationNetwork import SegmentationNetwork
from Training import Training
import Helper
from torch.utils.data import DataLoader, random_split

if __name__ == '__main__':
    NUM_TRAINING_EXAMPLES = 50
    MINI_BATCH_SIZE = 1

    current_directory = sys.path[0]
    dataset_directory = os.path.join(os.path.split(current_directory)[0], 'CelebAMask-HQ')
    image_directory = os.path.join(dataset_directory, 'CelebA-HQ-img')
    features_directory = os.path.join(dataset_directory, 'CelebAMask-HQ-mask-anno')

    #Load all image and map data and split it randomly into testing and training data
    dataset = ImageDataset(image_directory, features_directory)
    train_data, test_data = random_split(dataset.ids, [NUM_TRAINING_EXAMPLES, len(dataset.ids) - NUM_TRAINING_EXAMPLES])
    train_loader = DataLoader(train_data, batch_size=MINI_BATCH_SIZE, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_data, batch_size=MINI_BATCH_SIZE, shuffle=False, num_workers=1)

    input_image_idx, output_map_name = next(iter(train_loader))

    #model = SegmentationNetwork()
    #model_training = Training(model, train_data)

    #output = model_training.train()

    #Helper.display_outputs(output.detach()[0])s