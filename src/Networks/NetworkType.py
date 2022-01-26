''' Creates the enum used to keep track of what network we are working with
'''

from enum import Enum

class NetworkType(Enum):
    SEGMENTATION = 0
    ATTRIBUTE = 1
    MULTI = 2