# PROGRAMMER: Justin Bellucci 
# DATE CREATED: 06_01_2020                                  
# REVISED DATE: 

import torch
import cv2
from image_helper import process_image
from torchvision import models
from model_functions import load_checkpoint

### ----------------------------------------------
def predict_breed(img_path):
    """ Use trained model to obtain index corresponding to 
        predicted ImageNet class for image at specified path
    
        Arguments:
            - img_path (str)
        
        Returns:
            - Index corresponding to model's prediction
    """
    # model_transfer.load_state_dict(torch.load('model_transfer.pt'))
    model_transfer = load_checkpoint()
    # move model to correct device
    device = torch.device("cpu")
    model_transfer.to(device)
    model_transfer.eval()
    with torch.no_grad():
        # process image
        image = process_image(img_path)
        image = image.unsqueeze_(0)

        output = model_transfer(image)

        # convert output probabilities to predicted class
        _, class_tensor = torch.max(output, 1)

        # convert output tensor to numpy array
        class_pred = class_tensor.numpy()
        
    return  class_pred

### ----------------------------------------------
def VGG16_predict(img_path):
    """ Use pre-trained VGG-16 model to obtain index corresponding to 
        predicted ImageNet class for image at specified path
    
        Arguments:
            - img_path (str)
        
        Returns:
            - Index corresponding to VGG-16 model's prediction
    """
    # move model to correct device
    VGG16 = models.vgg16(pretrained=True)

    device = torch.device("cpu")
    VGG16.to(device)
    
    with torch.no_grad():
        # process image
        image = process_image(img_path)
        image = image.unsqueeze_(0)
        image.to(device)
        
        output = VGG16(image)
        
        # convert output probabilities to predicted class
        _, class_tensor = torch.max(output, 1)

        # convert output tensor to numpy array
        class_pred = class_tensor.numpy()
        
    return  class_pred