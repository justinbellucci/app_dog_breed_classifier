# PROGRAMMER: Justin Bellucci 
# DATE CREATED: 06_01_2020                                  
# REVISED DATE: 

import cv2 
from backend.predict import VGG16_predict

### ----------------------------------------------
# returns "True" if face is detected in image stored at img_path
def face_detector_haar(img_path):
    """ Returns True if a human face is detected in the image. Function uses
        the Haar Cascade face detector from OpenCV.

        Arguments:
            - image path (str)
        Returns:
            - Boolean value
    """
    detector_type = cv2.CascadeClassifier('assets/haarcascade_frontalface_alt.xml')
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector_type.detectMultiScale(gray)
    return len(faces) > 0

### ----------------------------------------------
def dog_detector(img_path):
    """ Returns True if a dog is detected in the supplied image. Function 
        uses the VGG16 CNN model to identify if a dog is present.
    
        Arguments:
            - image path (str)
        Returns:
            - Boolean
    """
    # run image through VGG16_predict function
    class_pred = VGG16_predict(img_path)
    
    # Imagenet class dictionary keys 151-268, inclusive, correspond to dog names
    if ((class_pred >= 151) and (class_pred <= 268)):
        return True
    else:
        return False
