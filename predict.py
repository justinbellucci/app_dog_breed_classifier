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


def run_app(img_path):
    ## handle cases for a human face, dog, and neither
    detector_type = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    class_names = [item[4:].replace("_", " ") for item in train_data.classes]
    
    # run image through model
    is_dog = dog_detector(img_path)
    class_pred = predict_breed_transfer(img_path)
    dog_name = class_names[class_pred[0]]
    
    if is_dog == True:
        print('Yea! Dog detected - {}'.format(dog_name))
        show_img(img_path)
    else:
        faces = face_detector_haar(img_path, detector_type)
        if faces == True:
            print('Human detected! Similar to {}'.format(dog_name))
            show_img(img_path)
        else:
            print('Error - Neither Dog nor Human Detected.')
            show_img(img_path)