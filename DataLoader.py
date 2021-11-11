import os
import torch
from torchvision import transforms
from PIL import Image

#For now, just loading the hair image as the output and the original face as the input - change in future

class ImageDataset:
  ''' Constructor for the ImageDataset class
  Takes the location of all the images as well as all the maps used in the dataset - sets up all private properties for the object
  '''
  def __init__(self, img_dir, feature_map_dir):
    self._ITEMS_PER_MAP_FOLDER = 2000
    self._img_dir = img_dir
    self._feature_map_dir = feature_map_dir
    self._tensor_transform = transforms.ToTensor()

  def load_data(self, start_index, end_index):
    # Initialize tensor with all 1's
    output_list = []
    for i in range(0, end_index + 1):
      image_tensor = self._tensor_transform(Image.open(os.path.join(self._img_dir, str(i) + '.jpg')))
      #Need to get the correct map folder
      map_folder = str(int(i / self._ITEMS_PER_MAP_FOLDER))
      map_path = f'{os.path.join(self._feature_map_dir, map_folder, str(i).zfill(5) + "_hair")}.png'
      #If the map file does not exist, return -1 as the map value for this image
      if (os.path.isfile(map_path)):
        map_tensor = self._tensor_transform(Image.open(map_path))
        output_list.append((image_tensor, map_tensor))
      
    return output_list
