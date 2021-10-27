from DataLoader import ImageDataset
import torch.nn as nn
import torchvision
import torch

IMAGE_FOLDER_PATH = '/Users/michaelshepherd/Documents/University/Project Module/CelebAMask-HQ/CelebA-HQ-img'
FEATURES_FOLDER_PATH = '/Users/michaelshepherd/Documents/University/Project Module/CelebAMask-HQ/CelebAMask-HQ-mask-anno'
IMAGE_SIZE = 1048576 #Size of all the images in the CelebA Dataset

''' Class defining the Segmentation Network
    All the layers of the network are setup here and the flow through the network is also defined
'''
class SegmentationNetwork(nn.Module):
  ''' Constructor for the class
      All layers of the network and their parameters get defined here
  '''
  def __init__(self):
    super(SegmentationNetwork, self).__init__()

    self.conv1 = nn.Linear(IMAGE_SIZE, 128)

  ''' Performs a forward pass through the network for some data
      Method should not be called directly as only gets called as part of the lifecycle of the network
  '''
  def forward(self, x):
    x = torch.flatten(x)
    x = self.conv1(x)
    return x

if __name__ == '__main__':
  dataset = ImageDataset(IMAGE_FOLDER_PATH, FEATURES_FOLDER_PATH)
  network = SegmentationNetwork()
  output = network(dataset[0][0][0].to(None, dtype=torch.float32))
  print(output)