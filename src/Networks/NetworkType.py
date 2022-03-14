''' Creates the enum used to keep track of what network we are working with
'''

from enum import Enum

class NetworkType(Enum):
    SEGMENTATION = 'Segmentation'
    ATTRIBUTE = 'Attribute'
    MULTI = 'Multi'