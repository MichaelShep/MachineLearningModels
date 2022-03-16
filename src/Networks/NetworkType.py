from enum import Enum

class NetworkType(Enum):
    ''' Enum class which contains all the possibly different forms of neural network '''
    SEGMENTATION = 'Segmentation'
    ATTRIBUTE = 'Attribute'
    MULTI = 'Multi'