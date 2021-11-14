''' Loads all our data from the CelebAHQ Dataset and formats it ready for use with Pytorch '''

import os
import torch
from torchvision import transforms # type: ignore
from PIL import Image # type: ignore
from typing import List, Tuple
import glob

#For now, just loading the hair image as the output and the original face as the input - change in future

class ImageDataset:
  _ITEMS_PER_MAP_FOLDER = 2000

  ''' Constructor for the ImageDataset class
  Takes the location of all the images as well as all the maps used in the dataset - sets up all private properties for the object
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

    #For now, only load the hair map as the output map - others will be added later
    for i in range(len(map_files)):
      if('hair' in self._get_file_name_string(map_files[i])):
        files_dictionary[int(self._get_file_name_string(map_files[i]).split('_')[0])].append(map_files[i])

    self.ids = list(sorted(files_dictionary.items()))

  ''' Used to allow indexing of the dataset - returns the actual input image and for now returns the hair output map
  '''
  def __getitem__(self, idx):
    if not self.ids:
      print('Need to load data before can be accessed')
      return

    image_number = self.ids[idx][0]
    input_image = self._tensor_transform(Image.open(os.path.join(self._img_dir, str(image_number) + '.jpg')))

    #For now, output only ever has 1 element as will always be the hair map
    output_element = self.ids[idx][1][0]
    output_image = self._tensor_transform(Image.open(output_element))

    return (input_image, output_image)
