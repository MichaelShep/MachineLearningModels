import os
from torch.utils.data import random_split, Subset
import torch
from typing import Tuple, List
from torchvision import transforms # type: ignore
from PIL import Image # type: ignore
import glob

from Networks.NetworkType import NetworkType

class CelebADataset:
    ''' Class to manage the Datase used for our images - dataset is called CelebA-HQ '''
    _ITEMS_PER_MAP_FOLDER = 2000
    _REDUCED_IMAGE_SIZE = 512
    #List containing all the possible output masks for a given input face
    _mask_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear',
                  'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

    def __init__(self, base_dir: str, network_type: NetworkType, device: str):
        ''' Takes the location of the dataset and uses this to get the location of images, masks and annotations
        
        Parameters
        ----------
        base_dir: str
            The location in the users file system of the base location of the dataset foler
        network_type: NetworkType
            The type of network that we are currently working with
        device: str
            The device we are running our model on - cpu or cuda (gpu) '''

        self._network_type = network_type
        self._device = device
        self._img_dir = os.path.join(base_dir, 'CelebA-HQ-img')
        self._feature_mask_dir = os.path.join(base_dir, 'CelebAMask-HQ-mask-anno')
        self._attribute_anno_file = os.path.join(base_dir, 'CelebAMask-HQ-attribute-anno.txt')

        self._tensor_transform = transforms.ToTensor()
        self._greyscale_transform = transforms.Grayscale()
        self._normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self._calc_dataset_size()
        self._load_attribute_data()

    def _load_attribute_data(self):
        ''' Loads all the data about the attributes displayed in each image '''
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
      
    def _get_base_output_file_path(self, index: int) -> str:
        ''' Used for the segmentation masks - used to get the location of the masks - are in different numbered folders
        
        Parameters
        ----------
        index: int
            The index of the image that we want the file path for '''

        folder_value = int(index / self._ITEMS_PER_MAP_FOLDER)
        return os.path.join(self._feature_mask_dir, str(folder_value), str(index).zfill(5))

    def _get_output_images(self, index: int) -> torch.Tensor:
        ''' Gets all the output masks for a single input image - will return a tensor of shape [18, 1, 512, 512]
        since there are 18 possible output masks - will return a 0 image where a certain mask does not exist 
        
        Parameters
        ----------
        index: int
            The index of the image which we want to get the segmentation masks for'''
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

    def _calc_dataset_size(self):
        ''' Calculates the number of images we have in our dataset'''
        image_files = glob.glob(os.path.join(self._img_dir, '*.jpg'), recursive=True)
        self._num_images = len(image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ''' Used to allowing indexing of this class - will return the image at that index as well as its outputs
        
        Parameters
        ----------
        idx: int
            The index of the data that we want to collect
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Returns tuple containing the input image as first element, output segmentation masks as second'''

        if idx >= self._num_images:
            print('Invalid Index')
            return 

        input_image = self._normalize_transform(self._tensor_transform(Image.open(os.path.join(self._img_dir, str(idx) + '.jpg')).resize((self._REDUCED_IMAGE_SIZE, self._REDUCED_IMAGE_SIZE))))

        #For segmentation network, need to return output images, for attributes need to return attributes list
        if self._network_type == NetworkType.SEGMENTATION:
            output = self._get_output_images(idx)
        #For multi network, need both the segmentation masks for the image and the list of attributes
        elif self._network_type == NetworkType.MULTI:
            image_output = self._get_output_images(idx)
            attribute_output = torch.Tensor(self._attributes[idx])
            #Output is a tuple - first element is the image outputs and second is attributes
            output = (image_output, attribute_output)
        #For attributes network, output is the list of attributes associated with that image
        else:
            output = torch.Tensor(self._attributes[idx])

        return (input_image, output)

    def get_train_test_split(self, num_training_examples: int) -> Tuple[Subset, Subset]:
        ''' Splits our dataset indicies randomly into training and testing data
        
        Parameters
        ----------
        num_training_examples: int
            The size of the training section of the dataset - the rest will be used for validation
        Returns
        -------
        Tuple[Subset, Subset]
            Returns tuple where the first element contains all training data, second element contains validation'''

        index_array = list(range(self._num_images))
        return random_split(index_array, [num_training_examples, self._num_images - num_training_examples])

    ''' Gets the amount of output masks our system is working with
    '''
    def get_num_output_masks(self) -> int:
        ''' Returns the amount of output masks for a single image'''
        return len(self._mask_list)

    def get_num_attributes(self) -> int:
        ''' Returns the number of attributes for a single image '''
        return len(self._attribute_names)

    ''' Takes an attribute list consisting of 0s and 1s and converts it into a string listing the attributes that a certain image has.
    '''
    def attribute_list_to_string(self, attribute_list: List[int]) -> str:
        ''' Takes list containing 0's and 1's and converts this into text containing all the attributes a image has
        
        Parameters
        ----------
        attribute_list: List[int]
            The list of 0s and 1s containing all the attributes that are present in an image
        Returns
        -------
        str:
            Text listing all the attributes that are present in this face image'''
        #Check that the input list is in the correct format
        if len(attribute_list) != self.get_num_attributes():
            print('Invalid Attribute List')
            return
        
        #Create output by getting all the names of the features that this face has
        output_string = 'This face image has the following attributes: '
        for i in range(len(attribute_list)):
            if attribute_list[i] == 1:
                output_string += self._attribute_names[i] + ', '
        return output_string

    def get_data_for_indexes(self, indexes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ''' Gets all the data for a tensor containing a list of indexes
        
        Parameters
        ----------
        indexes: torch.Tensor
            Tensor containing a list of indexes which we want to get the data for
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple where first element is the list of input data for indexes, second element is the output data for these indexes'''
        input_values = []
        output_values = []

        #Used for the multi network
        segmentation_values = []
        attribute_values = []

        for index_tensor in indexes:
            element = self[int(index_tensor.item())]
            input_values.append(element[0])
            if self._network_type != NetworkType.MULTI:
                output_values.append(element[1])
            else:
                segmentation_values.append(element[1][0])
                attribute_values.append(element[1][1])
        
        if self._network_type != NetworkType.MULTI:
            return (torch.stack(input_values).to(self._device), torch.stack(output_values).to(self._device), 
                torch.tensor(0).to(self._device))
        else:
            return (torch.stack(input_values).to(self._device), torch.stack(segmentation_values).to(self._device), 
                torch.stack(attribute_values).to(self._device))
