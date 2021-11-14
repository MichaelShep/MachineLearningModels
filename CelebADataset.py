''' Loads all our data from the CelebAHQ Dataset and formats it ready for use with Pytorch '''

import os
from torch.utils.data import random_split, Subset
import torch
from typing import Tuple
from torchvision import transforms # type: ignore
from PIL import Image # type: ignore
import glob

#For now, just loading the skin image as the output and the original face as the input

class CelebADataset:
  _ITEMS_PER_MAP_FOLDER = 2000

  ''' Constructor for the ImageDataset class
      Takes the location of all the images as well as all the maps used in the dataset 
  '''
  def __init__(self, img_dir: str, feature_map_dir: str):
    self._img_dir = img_dir
    self._feature_map_dir = feature_map_dir
    self._tensor_transform = transforms.ToTensor()

    self._calc_dataset_size()

  ''' Gets the full path of the output file for a given index - currently only 1 output file which is the skin
  '''
  def _get_output_file_path(self, index: int) -> str:
    folder_value = int(index / self._ITEMS_PER_MAP_FOLDER)
    return os.path.join(self._feature_map_dir, str(folder_value), str(index).zfill(5) + '_skin.png')

  ''' Gets the number of items we have in our dataset by checking the image locations
  '''
  def _calc_dataset_size(self) -> None:
    #Get the total amaount of images used in the dataset by reading image folder
    image_files = glob.glob(os.path.join(self._img_dir, '*.jpg'), recursive=True)
    self._num_images = len(image_files)

  ''' Used to allow indexing of the dataset - returns the actual input image and for now returns the skin output map
  '''
  def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
    if idx >= self._num_images:
      print('Invalid Index')
      return 

    input_image = self._tensor_transform(Image.open(os.path.join(self._img_dir, str(idx) + '.jpg')))
    #For now, output only ever has 1 element as will always be the skin map
    output_image = self._tensor_transform(Image.open(self._get_output_file_path(idx)))

    return (input_image, output_image)

  ''' Splits our dataset indicies randomly into training and testing data
  '''
  def get_train_test_split(self, num_training_examples) -> Tuple[Subset, Subset]:
    index_array = list(range(self._num_images))
    return random_split(index_array, [num_training_examples, self._num_images - num_training_examples])
