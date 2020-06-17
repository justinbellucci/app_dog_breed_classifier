# PROGRAMMER: Justin Bellucci 
# DATE CREATED: 06_16_2020                                  
# REVISED DATE: 

### ----------------------------------------------
# Use this to train a CNN on a dataset. This program
# uses features from a pretrained densenet121 CNN and 
# applies a custom classifier in order to classify 
# dog breeds. 
#
# 
### ----------------------------------------------
import argparse
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from classifier import Classifier
import torch.optim as optim
from loaders import data_loader

### ----------------------------------------------
def get_input_args():
    # directory to save the model checkpoint
    save_dir = 'backend/assets/model_checkpoint.pt'

    img_dir = 'dogImages'
    # Creates Arguement Parser object named parser
    parser = argparse.ArgumentParser()
    # Argument 1: image path
    parser.add_argument('--chkpt_dir', type=str, default=save_dir, help='Directory to save checkpoint')
    parser.add_argument('--epochs', type=int, default=40, help='Epochs to train')
    parser.add_argument('--img_dir', type=str, default=img_dir, help='Main directory of dog images')

    in_args = parser.parse_args()
    return in_args

def train(epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """ Train the CNN on a dataset to classify 120 dog breeds.
    
        Arguments:
            - epochs: int
            - loaders: Pytorch dataloaders
            - model: pretrained densenet121() 
            - optimizer: optim.SGD() used with model parameters
            - criterion: NLLLoss() loss function
            - use_cuda: boolean 
            - save_path: path to save model checkpoint

        Returns:
            - model
    """
    valid_loss_min = np.Inf
    # freeze parameters to prevent backprop in convolutional layers
    for param in model.parameters():
        param.requires_grade = False
        
    for e in range(epochs):
        train_loss = 0.0
        valid_loss = 0.0
        train_losses, valid_losses = [], []
        
        # -------- Training Pass ---------
        model.train()
        for batch_idx, (images, targets) in enumerate(loaders['train']):
            if use_cuda:
                images, targets = images.cuda(), targets.cuda()
            
            optimizer.zero_grad() # clear the gradients
            output = model(images)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))
        
        model.eval()
        for batch_idx, (images, targets) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                images, targets = images.cuda(), targets.cuda()
                
            ## update the average validation loss
            output = model(images)
            loss = criterion(output, targets)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.item() - valid_loss))
            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            e+1, 
            train_loss,
            valid_loss))
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        ## TODO: save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.3f} --> {:.3f}). Saving model ...'.format(
            valid_loss_min, valid_loss))
            
        valid_loss_min = valid_loss
        torch.save(model.state_dict(), save_path)
        
    return model

def main():
    # Get command line arguments 
    in_arg = get_input_args()
    save_path = in_arg.chkpt_dir
    epochs = in_arg.epochs
    img_dir = in_arg.img_dir

    # check to see if GPU is available
    use_cuda = torch.cuda.is_available()
    # import model
    model = models.densenet121(pretrained=True)
    # import loaders, classifier, loss, and optimizer fns
    loaders, train_data = data_loader(img_dir)
    model.classifier = Classifier()
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.7)
    # transfer model to GPU if available
    if use_cuda:
        model = model.cuda()

    # train the model
    train(epochs, loaders, model, optimizer, criterion, use_cuda, save_path)

# Call to main function to run the program
if __name__ == "__main__":
    main()