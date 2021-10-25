import matplotlib.pyplot as plt
from torchvision.io import read_image
import os

IMAGE_FOLDER_PATH = '/Users/michaelshepherd/Documents/University/Project Module/CelebAMask-HQ/CelebA-HQ-img'
FEATURES_FOLDER_PATH = '/Users/michaelshepherd/Documents/University/Project Module/CelebAMask-HQ/CelebAMask-HQ-mask-anno/0'

MAP_NAMES = ['hair', 'l_brow', 'l_eye', 'l_lip', 'mouth', 'neck', 'nose', 'r_brow', 'r_eye', 'skin', 'u_lip']

class ImageDataset:
  def __init__(self, img_dir, feature_map_dir, transform=None, target_transform=None):
    self.img_dir = img_dir
    self.feature_map_dir = feature_map_dir
    self.transform = transform
    self.target_transform = target_transform

  def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, str(idx))
    image_list = []
    image_list.append(read_image(f'{img_path}.jpg'))
    for map in MAP_NAMES:
      map_path = os.path.join(self.feature_map_dir, f'{str(idx).zfill(5)}_{map}')
      image_list.append(read_image(f'{map_path}.png'))
    return image_list


if __name__ == '__main__':
  dataset = ImageDataset(IMAGE_FOLDER_PATH, FEATURES_FOLDER_PATH)
  _, plot_array = plt.subplots(1, len(MAP_NAMES) + 1)
  for img_idx in range(len(dataset[0])):
    plot_array[img_idx].imshow(dataset[0][img_idx].permute(1, 2, 0))
  plt.show()