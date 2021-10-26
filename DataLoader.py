import matplotlib.pyplot as plt
from torchvision.io import read_image
import os

IMAGE_FOLDER_PATH = '/Users/michaelshepherd/Documents/University/Project Module/CelebAMask-HQ/CelebA-HQ-img'
FEATURES_FOLDER_PATH = '/Users/michaelshepherd/Documents/University/Project Module/CelebAMask-HQ/CelebAMask-HQ-mask-anno'

class ImageDataset:
  ''' Constructor for the ImageDataset class
  Takes the location of all the images as well as all the maps used in the dataset - sets up all private properties for the object
  '''
  def __init__(self, img_dir, feature_map_dir):
    self._MAP_NAMES = ['hair', 'l_brow', 'l_eye', 'l_lip', 'mouth', 'neck', 'nose', 'r_brow', 'r_eye', 'skin', 'u_lip']
    self._ITEMS_PER_MAP_FOLDER = 2000
    self._img_dir = img_dir
    self._feature_map_dir = feature_map_dir

  ''' Function to allow array style indexing on the ImageDataset object
  Will return the image and all the maps associated with the provided index
  '''
  def __getitem__(self, idx):
    img_path = os.path.join(self._img_dir, str(idx) + '.jpg')
    image_list = []
    image_list.append(read_image(img_path))
    for map in self._MAP_NAMES:
      map_folder = str(int(idx / self._ITEMS_PER_MAP_FOLDER)) #Maps are split into 14 folders each with 2000 image indexes in
      map_path = f'{os.path.join(self._feature_map_dir, map_folder, str(idx).zfill(5) + "_" + map)}.png'
      if(os.path.isfile(map_path)): #Check map file exists since not all image indexes contain all map options
        image_list.append(read_image(map_path))
    return image_list


if __name__ == '__main__':
  dataset = ImageDataset(IMAGE_FOLDER_PATH, FEATURES_FOLDER_PATH)
  _, plot_array = plt.subplots(1, 12)
  for img_idx in range(len(dataset[20000])):
    plot_array[img_idx].imshow(dataset[20000][img_idx].permute(1, 2, 0))
  plt.show()