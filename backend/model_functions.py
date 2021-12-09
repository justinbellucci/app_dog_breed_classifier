# PROGRAMMER: Justin Bellucci 
# DATE CREATED: 06_01_2020                                  
# REVISED DATE: 

import ast
import torch
from torchvision import models
from classifier import Classifier

### ----------------------------------------------
def load_class_names(path='assets/class_names.txt'):
    """ Loads the class names created from the torch.datasets.ImageFolder 
        function. Removes uncessary punctuation and integers.
    
        Arguments:
            - file_path (str)
       
        Returns:
            - class_names (list)
    """
    with open('assets/class_names.txt', 'r') as f:
        class_list = ast.literal_eval(f.read())

    class_names = [item[4:].replace("_", " ") for item in class_list]
        
    return class_names


### ----------------------------------------------
def load_checkpoint(filepath='assets/model_transfer.pt'):
    """Load a model checkpoint and rebuild model architecture
    
       Arguments:
       - filepath (path to checkpoint.pth file)
       
       Output:
       - model (trained deep learning model)
       - class_dict (class_to_idx dictionary)
    
    """
    model = models.densenet121(pretrained=True)
    model.classifier = Classifier()
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    
    # Freeze the parameters so we dont backpropagate through them
    for param in model.parameters():
        param.requires_grad = False
        
    return model
### ----------------------------------------------