# PROGRAMMER: Justin Bellucci 
# DATE CREATED: 06_01_2020                                  
# REVISED DATE: 

import detectors
import argparse
import torch
import cv2
from torchvision import models
from predict import predict_breed
from image_helper import show_img, process_image
from model_functions import load_class_names

### ----------------------------------------------
def get_input_args():
    
    default_path = 'backend/test_imgs/Labrador_retriever_01.jpg'

    # Creates Arguement Parser object named parser
    parser = argparse.ArgumentParser()
    # Argument 1: image path
    parser.add_argument('--img_path', type=str, default=default_path, help='Path to an image')

    in_args = parser.parse_args()
    return in_args

def run_app(img_path):
    ## handle cases for a human face, dog, and neither
    
    # run image through model
    is_dog = detectors.dog_detector(img_path)
    probs, classes = predict_breed(img_path)
    class_names = load_class_names()
    # dog_name = class_names[class_pred[0]]
    # print(class_pred[0])
    # print(probs)
    # print(classes)

    if is_dog == True:
        print("It's a dog! -> {:.3f}% probability it's a {}".format(probs[0]*100,
              class_names[classes[0]]))
        print("---------------------")
        print("Top 5 Class Probabilities")
        print("{}: {:.3f}%".format(class_names[classes[0]], probs[0]*100))
        print("{}: {:.3f}%".format(class_names[classes[1]], probs[1]*100))
        print("{}: {:.3f}%".format(class_names[classes[2]], probs[2]*100))
        print("{}: {:.3f}%".format(class_names[classes[3]], probs[3]*100))
        print("{}: {:.3f}%".format(class_names[classes[4]], probs[4]*100))
        print("/n")
        show_img(img_path)
    else:
        faces = detectors.face_detector_haar(img_path)
        if faces == True:
            print("Human detected! -> {:.3f}% probability they look similar to a {}".format(probs[0]*100,
                  class_names[classes[0]]))
            show_img(img_path)
        else:
            print('Error - Neither Dog nor Human Detected.')
            show_img(img_path)

def main():
    # Get command line arguments 
    in_arg = get_input_args()
    # Get image path for testing
    img_path = in_arg.img_path
    run_app(img_path)

# Call to main function to run the program
if __name__ == "__main__":
    main()


    