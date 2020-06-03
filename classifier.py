# PROGRAMMER: Justin Bellucci 
# DATE CREATED: 06_01_2020                                  
# REVISED DATE: 

# Import python modules
import torch
from torch import nn
import torch.nn.functional as F

### ----------------------------------------------
# Neural network model Classifier class
class Classifier(nn.Module):
    """ Classifier for Densenet121 neural network model.
        
        Attributes:
            - Input dimensions: 1024
            - Hidden dimensions: [512, 256]
            - Output dimensions: 133
            
        Args:
            None
            
        Returns:
            Self
    """
    def __init__(self):
        super().__init__()
        # fully connected linear layers
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 133)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # flatten convolutional output
        x = x.view(x.shape[0], -1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1) # used for topk funcionality
        return x