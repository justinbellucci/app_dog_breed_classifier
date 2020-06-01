# PROGRAMMER: Justin Bellucci 
# DATE CREATED: 06_01_2020                                  
# REVISED DATE: 


### ----------------------------------------------
def main():
# def predict( class_dict, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    
        Arguments:
        - image_path (str - path to image to process)
        - model (trained neural network model)
        - class_dict (class_to_idx dictionary from checkpoint)
        - topk (int - number of top classes/probabilities to calculate)
    
        Output:
        - probs (list - top 5 predicted probabilities)
        - classes (list - top 5 predicted classes)
    '''
    topk = 5 # Topk probabilities/classes
    
    # Get command line arguments 
    in_arg = get_input_args()
    
    # Get image path for testing
    image_path = in_arg.img_path
    
    checkpoint = in_arg.chk_pt

    # Switch to "cpu" mode 
    device = torch.device("cpu")
    
    # Load the checkpoint
    model = model_functions.load_checkpoint(checkpoint)
   
    model.to(device)
    model.eval()
    # Run image through the model
    with torch.no_grad():
        # Process image
        image = utility_functions.process_image(image_path)
        image = image.unsqueeze(0)
        image.to(device)
        
        # Run through model
        log_pbs = model.forward(image)
        ps = torch.exp(log_pbs)
        top_p, top_c = ps.topk(topk, dim=1, largest=True, sorted=True)
        probs = top_p.tolist()[0]
        classes = top_c.tolist()[0]

    # Print out the top 5 classes and corresponding flower names
    
    # Invert the class dictionary
    class_dict_invert = dict([[v,k] for k,v in model.class_to_idx.items()])
    
    # Load the category to name json file
    cat_to_name = utility_functions.import_cat_to_names(in_arg.cat_names)
    
    flower_names = []
    class_num = []
    for c in classes:
        class_num.append(class_dict_invert[c])
        flower_names.append(cat_to_name[class_dict_invert[c]])
    # Print functions ------------
    print("/n")
    print("Image: {}".format(image_path))
    print("Top Flower Name: {}".format(flower_names[0].title()))
    print("Top Probability: {:.3f}".format(probs[0]))
    print("---------------------")
    print("Top 5 Class:Probabilities")
    print("{}: {:.3f}".format(flower_names[0].title(), probs[0]))
    print("{}: {:.3f}".format(flower_names[1].title(), probs[1]))
    print("{}: {:.3f}".format(flower_names[2].title(), probs[2]))
    print("{}: {:.3f}".format(flower_names[3].title(), probs[3]))
    print("{}: {:.3f}".format(flower_names[4].title(), probs[4]))
    print("/n")  
    
# Call to main function to run the program
if __name__ == "__main__":
    main()


## ------------------------------------------------------

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