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
    
    default_path = '/Users/justinbellucci/GitHub/dog_breed_classifier_app/images/Curly-coated_retriever_03896.jpg'
    # default_path = '/Users/justinbellucci/GitHub/dog_breed_classifier_app/images/Rena_Sofer_0001.jpg'
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
# def main():
# # def predict( class_dict, topk=5):
#     ''' Predict the class (or classes) of an image using a trained deep learning model.
    
#         Arguments:
#         - image_path (str - path to image to process)
#         - model (trained neural network model)
#         - class_dict (class_to_idx dictionary from checkpoint)
#         - topk (int - number of top classes/probabilities to calculate)
    
#         Output:
#         - probs (list - top 5 predicted probabilities)
#         - classes (list - top 5 predicted classes)
#     '''
#     topk = 5 # Topk probabilities/classes
    
#     # Get command line arguments 
#     in_arg = get_input_args()
    
#     # Get image path for testing
#     image_path = in_arg.img_path
    
#     checkpoint = in_arg.chk_pt

#     # Switch to "cpu" mode 
#     device = torch.device("cpu")
    
#     # Load the checkpoint
#     model = model_functions.load_checkpoint(checkpoint)
   
#     model.to(device)
#     model.eval()
#     # Run image through the model
#     with torch.no_grad():
#         # Process image
#         image = utility_functions.process_image(image_path)
#         image = image.unsqueeze(0)
#         image.to(device)
        
#         # Run through model
#         log_pbs = model.forward(image)
#         ps = torch.exp(log_pbs)
#         top_p, top_c = ps.topk(topk, dim=1, largest=True, sorted=True)
#         probs = top_p.tolist()[0]
#         classes = top_c.tolist()[0]

#     # Print out the top 5 classes and corresponding flower names
    
#     # Invert the class dictionary
#     class_dict_invert = dict([[v,k] for k,v in model.class_to_idx.items()])
    
#     # Load the category to name json file
#     cat_to_name = utility_functions.import_cat_to_names(in_arg.cat_names)
    
#     flower_names = []
#     class_num = []
#     for c in classes:
#         class_num.append(class_dict_invert[c])
#         flower_names.append(cat_to_name[class_dict_invert[c]])
#     # Print functions ------------
#     print("/n")
#     print("Image: {}".format(image_path))
#     print("Top Flower Name: {}".format(flower_names[0].title()))
#     print("Top Probability: {:.3f}".format(probs[0]))
#     print("---------------------")
#     print("Top 5 Class:Probabilities")
#     print("{}: {:.3f}".format(flower_names[0].title(), probs[0]))
#     print("{}: {:.3f}".format(flower_names[1].title(), probs[1]))
#     print("{}: {:.3f}".format(flower_names[2].title(), probs[2]))
#     print("{}: {:.3f}".format(flower_names[3].title(), probs[3]))
#     print("{}: {:.3f}".format(flower_names[4].title(), probs[4]))
#     print("/n")  

def main():
    # Get command line arguments 
    in_arg = get_input_args()
    # Get image path for testing
    img_path = in_arg.img_path

    run_app(img_path)

# Call to main function to run the program
if __name__ == "__main__":
    main()

## ------------------------------------------------------

    