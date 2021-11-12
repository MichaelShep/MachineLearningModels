''' Main script, brings everything together - creates, trains and tests model '''

import sys
import os.path
from DataLoader import ImageDataset
from SegmentationNetwork import SegmentationNetwork
import torch
import Helper

if __name__ == '__main__':
  current_directory = sys.path[0]
  dataset_directory = os.path.join(os.path.split(current_directory)[0], 'CelebAMask-HQ')
  image_directory = os.path.join(dataset_directory, 'CelebA-HQ-img')
  features_directory = os.path.join(dataset_directory, 'CelebAMask-HQ-mask-anno')

  #Note: Images in this dataset are 1024 x 1024
  dataset = ImageDataset(image_directory, features_directory)
  train_data = dataset.load_data(0, 0)

  train_data_input = torch.stack(Helper.extract_element_from_sublist(train_data, 0))
  train_data_output = torch.stack(Helper.extract_element_from_sublist(train_data, 1))

  network = SegmentationNetwork()
  output = network(train_data_input)

  print('Output Shape:', output.shape)
  Helper.display_outputs(train_data_input[0], output.detach()[0])