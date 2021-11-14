''' Loads all our data from the CelebAHQ Dataset and formats it ready for use with Pytorch '''

import os
from torch.utils.data import random_split, Subset
import torch
from typing import Tuple
from torchvision import transforms # type: ignore
from PIL import Image # type: ignore
import glob

#For now, just loading the skin image as the output and the original face as the input

class ImageDataset:
  _ITEMS_PER_MAP_FOLDER = 2000

  ''' Constructor for the ImageDataset class
      Takes the location of all the images as well as all the maps used in the dataset 
  '''
  def __init__(self, img_dir: str, feature_map_dir: str):
    self._img_dir = img_dir
    self._feature_map_dir = feature_map_dir
    self._tensor_transform = transforms.ToTensor()

    self._load_data()

  ''' Converts a full file path into just the name of the file
  '''
  def _get_file_name_string(self, path: str) -> str:
    path_components = os.path.normpath(path).split(os.sep)
    file_name_with_type = path_components[len(path_components) - 1]
    return file_name_with_type.split('.')[0]

  ''' Loads the dataset and stores a list of all the input image locations and their relevant output maps
  '''
  def _load_data(self) -> None:
    #Get the names of all the files from the image and mask directorys (other than hidden files)
    image_files = glob.glob(os.path.join(self._img_dir, '*.jpg'), recursive=True)
    map_files = glob.glob(os.path.join(self._feature_map_dir, '*', '*.png'), recursive=True)

    files_dictionary = {}
    for i in range(len(image_files)):
      files_dictionary[int(self._get_file_name_string(image_files[i]))] = []

    #For now, only load the skin map as the output map - others will be added later
    for i in range(len(map_files)):
      if('skin' in self._get_file_name_string(map_files[i])):
        files_dictionary[int(self._get_file_name_string(map_files[i]).split('_')[0])].append(map_files[i])

    self._ids = list(sorted(files_dictionary.items()))

  ''' Used to allow indexing of the dataset - returns the actual input image and for now returns the skin output map
  '''
  def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
    if not self._ids:
      print('Need to load data before can be accessed')
      return

    image_number = self._ids[idx][0]
    input_image = self._tensor_transform(Image.open(os.path.join(self._img_dir, str(image_number) + '.jpg')))

    #For now, output only ever has 1 element as will always be the skin map
    output_element = self._ids[idx][1][0]
    output_image = self._tensor_transform(Image.open(output_element))

    return (input_image, output_image)

  ''' Splits our dataset randomly into training and testing data
  '''
  def get_train_test_split(self, num_training_examples) -> Tuple[Subset, Subset]:
    return random_split(self._ids, [num_training_examples, len(self._ids) - num_training_examples])
