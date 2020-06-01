# PROGRAMMER: Justin Bellucci 
# DATE CREATED: 06_01_2020                                  
# REVISED DATE: 


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
def VGG16_predict(img_path):
    """
        Use pre-trained VGG-16 model to obtain index corresponding to 
        predicted ImageNet class for image at specified path
    
        Args:
            img_path: path to an image
        
        Returns:
            Index corresponding to VGG-16 model's prediction
    """
    # move model to correct device
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


### ----------------------------------------------
def load_checkpoint(filepath):
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
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    # Freeze the parameters so we dont backpropagate through them
    for param in model.parameters():
        param.requires_grad = False
        
    return model
### ----------------------------------------------
detector_type = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')