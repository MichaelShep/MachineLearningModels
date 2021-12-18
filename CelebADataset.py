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
      Takes the location of the dataset and uses this to get the location of images, masks and annotations
  '''
  def __init__(self, base_dir: str, for_segmentation: bool):
    self._for_segmentation = for_segmentation
    self._img_dir = os.path.join(base_dir, 'CelebA-HQ-img')
    self._feature_mask_dir = os.path.join(base_dir, 'CelebAMask-HQ-mask-anno')
    self._attribute_anno_file = os.path.join(base_dir, 'CelebAMask-HQ-attribute-anno.txt')

    self._tensor_transform = transforms.ToTensor()
    self._greyscale_transform = transforms.Grayscale()

    self._calc_dataset_size()
    self._load_attribute_data()

  ''' Loads all the attribute data about each image from the attibute file
  '''
  def _load_attribute_data(self):
    with open(self._attribute_anno_file) as f:
      lines = f.read().splitlines()
      num_elements = int(lines[0])
      self._attribute_names = lines[1].split(' ')
      self._attributes = []
      for i in range(2, num_elements + 2):
        image_attributes = lines[i].split(' ')
        #Remove element for image name and also second whitespace
        image_attributes.pop(0)
        image_attributes.pop(0)

        #Convert all values to ints and replace -1's with 0 as easier to work with in network
        for i in range(len(image_attributes)):
          image_attributes[i] = int(image_attributes[i])
          if image_attributes[i] == -1:
            image_attributes[i] = 0

        self._attributes.append(image_attributes)
      

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

    #For segmentation network, need to return output images, for attributes need to return attributes list
    if self._for_segmentation:
      #For now, output only ever has 1 element as will always be the skin map
      output = self._get_output_images(idx)
    else:
      output = torch.Tensor(self._attributes[idx])

    return (input_image, output)

  ''' Splits our dataset indicies randomly into training and testing data
  '''
  def get_train_test_split(self, num_training_examples) -> Tuple[Subset, Subset]:
    index_array = list(range(self._num_images))
    return random_split(index_array, [num_training_examples, self._num_images - num_training_examples])

  ''' Gets the amount of output masks our system is working with
  '''
  def get_num_output_masks(self) -> int:
    return len(self._mask_list)

  ''' Gets the amount of attributes that each image has 
  '''
  def get_num_attributes(self) -> int:
    return len(self._attribute_names)
