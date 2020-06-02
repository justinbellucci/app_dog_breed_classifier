# PROGRAMMER: Justin Bellucci 
# DATE CREATED: 06_01_2020                                  
# REVISED DATE: 

import ast
import torch
from torchvision import models
from classifier import Classifier

### ----------------------------------------------
def load_class_dict(class_dict_path):
    """Loads ImageNet class idx to labels text file.
    
       Input:
           - file_path (str)
       
       Output:
           - imagenet_class_dict (dict)
    """
    with open(class_dict_path) as f:
        imagenet_class_dict = ast.literal_eval(f.read())
        
    return imagenet_class_dict


### ----------------------------------------------
def load_checkpoint(filepath='model_transfer.pt'):
    """Load a model checkpoint and rebuild model architecture
    
       Arguments:
       - filepath (path to checkpoint.pth file)
       
       Output:
       - model (trained deep learning model)
       - class_dict (class_to_idx dictionary)
    
    """
    model = models.densenet121(pretrained=True)
    model.classifier = Classifier()
#     checkpoint = torch.load(filepath, map_location=("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint)
    
    # Freeze the parameters so we dont backpropagate through them
    for param in model.parameters():
        param.requires_grad = False
        
    return model
### ----------------------------------------------