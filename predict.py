# PROGRAMMER: Justin Bellucci 
# DATE CREATED: 06_01_2020                                  
# REVISED DATE: 

import torch
import utility_functions
import model_functions
from get_input_args import get_input_args

def predict_breed_transfer(img_path):
    """
        Use trained model to obtain index corresponding to 
        predicted ImageNet class for image at specified path
    
        Args:
            img_path: path to an image
        
        Returns:
            Index corresponding to model's prediction
    """
    model_transfer.load_state_dict(torch.load('model_transfer.pt'))
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

# Import python modules
import cv2

### ----------------------------------------------
# returns "True" if face is detected in image stored at img_path
def face_detector_haar(img_path, detector_type):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector_type.detectMultiScale(gray)
    return len(faces) > 0

### ----------------------------------------------
def dog_detector(img_path):
    """Returns True if a dog is detected in the supplied image.
    
       Arguments:
           - image path (str)
       Returns:
           - Boolean
    """
    # run image through model
    class_pred = VGG16_predict(img_path)
    
    # Imagenet class dictionary keys 151-268, inclusive, correspond to dog names
    if ((class_pred >= 151) and (class_pred <= 268)):
        return True
    else:
        return False
