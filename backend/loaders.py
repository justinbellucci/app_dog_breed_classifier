# PROGRAMMER: Justin Bellucci 
# DATE CREATED: 06_16_2020                                  
# REVISED DATE: 
# 

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import datasets

def data_loader(img_dir):
    """ Apply transformations and load image data from a training, validation, 
        and test set. Normalizes image data according to ImageNet database.

        Arguments:
            - img_dir: main image folder
        
        Returns:
            - loaders_transfer: Pytorch dataloader dictionary
            - train_data.classes: class list 
    """
    train_dir = img_dir + '/train'
    valid_dir = img_dir + '/valid'
    test_dir = img_dir + '/test'
    loaders_transfer = {}
    
    batch_size = 20
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    
    # define transforms for the training, validation and testing sets
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.RandomRotation(90),
                                           transforms.RandomHorizontalFlip(p=0.4),
                                           transforms.ToTensor(),
                                           transforms.Normalize(img_mean, img_std)])
    
    valid_test_transforms = transforms.Compose([transforms.Resize(226),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize(img_mean, img_std)])
    
    # load the datasets with ImageFolder function and DataLoader
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=valid_test_transforms)
    
    loaders_transfer['train'] = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    loaders_transfer['valid'] = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    loaders_transfer['test'] = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return loaders_transfer, train_data

# ------------------------------------------------------

def test(loaders, model, criterion, use_cuda):
    """ Test function that prints how many correct images the 
        model classifies in the test set.

        Arguments:
            - loaders: Pytorch dataloader
            - model: model after loading state_dict 
            - criterion: NLLLoss()
            - use_cuda: boolean
    """
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))