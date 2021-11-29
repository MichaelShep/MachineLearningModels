''' Loads all out data from the Lfw Dataset and formats it ready for use with Pytorch '''

from typing import Tuple
import torch
import os
from torchvision import transforms
from PIL import Image

class LfwDataset:
    def __init__(self, base_location: str):
        self._base_location = base_location
        self._load_image_locations()
        self.tensor_transform = transforms.ToTensor()

    ''' Sets up array containing all the image names so that each can be loaded when needed
    '''
    def _load_image_locations(self):
        file_location = os.path.join(self._base_location, 'peopleDevTrain.txt')
        with open(file_location) as f:
            lines = f.readlines()
        
        self._image_data = []
        for i in range(len(lines)):
            #First line contains the amount of training data we are using
            if i == 0:
                self.NUM_TRAINING_DATA = int(lines[i])
                continue

            split_line = lines[i].rstrip('\n').split('\t')
            #First element will be the label, second will be the index of the image
            self._image_data.append((split_line[0], split_line[1]))

    ''' Allows for array style indexing - will get a specific element
        Element will contain who the image is of (label) and the actual image itself as a tensor 
    '''
    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        element = self._image_data[idx]
        image_name = element[0] + '_' + element[1].zfill(4) + '.jpg'
        image = self.tensor_transform(Image.open(os.path.join(self._base_location, element[0], image_name)))
        return (element[0], image)