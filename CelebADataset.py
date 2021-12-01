''' Loads all our data from the CelebAHQ Dataset and formats it ready for use with Pytorch '''

import os
from torch.utils.data import random_split, Subset
import torch
from typing import Tuple
from torchvision import transforms # type: ignore
from PIL import Image # type: ignore
import glob

class CelebADataset:
  _ITEMS_PER_MAP_FOLDER = 2000
  _REDUCED_IMAGE_SIZE = 512
  #List containing all the possible output masks for a given input face
  _mask_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear',
                'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

  ''' Constructor for the ImageDataset class
      Takes the location of all the images as well as all the maps used in the dataset 
  '''
  def __init__(self, img_dir: str, feature_mask_dir: str):
    self._img_dir = img_dir
    self._feature_mask_dir = feature_mask_dir
    self._tensor_transform = transforms.ToTensor()
    self._greyscale_transform = transforms.Grayscale()

    self._calc_dataset_size()

  ''' Gets the base path of the output file for a given index - just need to append the actual mask name to get the full path
  '''
  def _get_base_output_file_path(self, index: int) -> str:
    folder_value = int(index / self._ITEMS_PER_MAP_FOLDER)
    return os.path.join(self._feature_mask_dir, str(folder_value), str(index).zfill(5))

  ''' Gets all the output masks for a single input image - will return a tensor of shape [18, 1, 512, 512]
      since there are 18 possible output masks - will return a 0 image where a certain mask does not exist
  '''
  def _get_output_images(self, index: int) -> torch.Tensor:
    output_images = []
    base_path = self._get_base_output_file_path(index)

    for name in self._mask_list:
      full_path = base_path + '_' + name + '.png'
      if os.path.exists(full_path):
        output_images.append(self._tensor_transform(self._greyscale_transform(Image.open(full_path))).squeeze(0))
      else:
        #Return an image of all 0s when the mask does not exist for the current input image
        zero_image = torch.zeros(size=[self._REDUCED_IMAGE_SIZE, self._REDUCED_IMAGE_SIZE])
        output_images.append(zero_image)

    return torch.stack(output_images) 

  ''' Gets the number of items we have in our dataset by checking the image locations
  '''
  def _calc_dataset_size(self) -> None:
    #Get the total amaount of images used in the dataset by reading image folder
    image_files = glob.glob(os.path.join(self._img_dir, '*.jpg'), recursive=True)
    self._num_images = len(image_files)

  ''' Used to allow indexing of the dataset - returns the input image and all its output masks
  '''
  def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if idx >= self._num_images:
      print('Invalid Index')
      return 

    input_image = self._tensor_transform(Image.open(os.path.join(self._img_dir, str(idx) + '.jpg')).resize((self._REDUCED_IMAGE_SIZE, self._REDUCED_IMAGE_SIZE)))
    #For now, output only ever has 1 element as will always be the skin map
    output_images = self._get_output_images(idx)

    return (input_image, output_images)

  ''' Splits our dataset indicies randomly into training and testing data
  '''
  def get_train_test_split(self, num_training_examples) -> Tuple[Subset, Subset]:
    index_array = list(range(self._num_images))
    return random_split(index_array, [num_training_examples, self._num_images - num_training_examples])

  ''' Gets the amount of output masks our system is working with
  '''
  def get_num_output_masks(self) -> int:
    return len(self._mask_list)
