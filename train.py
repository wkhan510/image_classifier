
import matplotlib.pyplot as plt

import argparse
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import json
import PIL
from PIL import Image


# checking for something...
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Define your transforms for the training, validation, and testing sets
def model_transformation(load_data):

    data_transforms = {
                       'train_transforms': transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])]),
                       'valid_transforms': transforms.Compose([transforms.Resize(225),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])]),
                       'test_transforms': transforms.Compose([transforms.Resize(225),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
                      }
    # Load the datasets with ImageFolder
    image_datasets = {
                      'train_data': datasets.ImageFolder(train_dir, transform=data_transforms['train_transforms']),
                      'valid_data': datasets.ImageFolder(valid_dir, transform=data_transforms['valid_transforms']),
                      'test_data': datasets.ImageFolder(test_dir, transform=data_transforms['test_transforms'])
                     }
    # Using the image datasets and the trainforms, define the dataloaders
    dataloader = {
                  'trainloader': torch.utils.data.DataLoader(image_datasets['train_data'], batch_size=64, shuffle=True),
                  'validloader': torch.utils.data.DataLoader(image_datasets['valid_data'], batch_size=64),
                  'testloader': torch.utils.data.DataLoader(image_datasets['test_data'], batch_size=64)
                  }

    return data_transforms, image_datasets, dataloader
# label mapping
def mapping(category_name):
    with open(category_name, 'r') as f:
        cat_to_name = json.load(f)    
    return cat_to_name
# Build and train your network

def classify_network(arch, m_dropout, lr):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model=models.vgg16(pretrained = True)

    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(nn.Linear(25088, 4096),
                           nn.ReLU(),
                           nn.Dropout(m_dropout),
                           nn.Linear(4096, 102),
                           nn.LogSoftmax(dim = 1))

# Now we neeed to attached this to our model.
    model.classifier = classifier
    #Define loss
    criterion = nn.NLLLoss()
    #Define optimizer with classifier and learning rate
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    model.to(device)
    
    return model, criterion, optimizer
    
def validation(model, valid_loader, criterion):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    accuracy = 0
    validation_loss = 0
    for images, labels in valid_loader:
        images, labels = images.to(device), labels.to(device)                
        logps = model.forward(images)
        batch_loss = criterion(logps, labels)                
        validation_loss += batch_loss.item()                
        # calculate accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim =1)                
        equality = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

    return validation_loss, accuracy

def train(device, model, dataloader, optimizer, criterion):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Do validation on the test set
    epochs =5
    # number of step we will take
    steps =0
    running_loss = 0
    print_every = 50
    
    # loop over epochs
    for epoch in range (epochs):
        model.to(device)
        #model.train()
        for images, labels in dataloader['trainloader']:
            steps += 1        
            # need to move images and labels to GPU or CPU whichever is avialable.
            images, labels = images.to(device), labels.to(device)        
            optimizer.zero_grad()        
            logps = model.forward(images)        
            loss = criterion(logps, labels)        
            # for backpropagation
            loss.backward()        
            optimizer.step()        
            # tracking loss as we fectching more data
            running_loss += loss.item()        
            # test our data for accuracy and validation
            if steps % print_every == 0:            
                #lets put our model in evaluation to test data for validation
                model.eval() 
                model.to(device)
                with torch.no_grad():
                    validation_loss, accuracy = validation(model, dataloader['validloader'], criterion)
                    
                print(f"Epochs {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {validation_loss/len(dataloader['validloader']):.3f}.. "
                      f"valid accuracy: {accuracy/len(dataloader['validloader']):.3f}")

                # now we need to set out training loss back to zero and model back to training 
                running_loss = 0
                model.train()
                
# TODO: Do validation on the test set
def network_test(device, model, dataloader, optimizer, criterion):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs =3
# number of step we will take
    steps =0
    running_loss = 0
    print_every = 50
# loop over epochs
    for epoch in range (epochs):
    
        for images, labels in dataloader['trainloader']:
            steps += 1        
        # need to move images and labels to GPU or CPU whichever is avialable.
            images, labels = images.to(device), labels.to(device)        
            optimizer.zero_grad()        
            logps = model.forward(images)        
            loss = criterion(logps, labels)        
        # for backpropagation
            loss.backward()        
            optimizer.step()        
        # tracking loss as we fectching more data
            running_loss += loss.item()        
        # test our data for accuracy and validation
            if steps % print_every == 0:            
            #lets put our model in evaluation to test data for validation
                model.eval() 
                model.to(device)
                test_loss = 0
                accuracy = 0
                with torch.no_grad():
                    for images, labels in dataloader['testloader']:
                        images, labels = images.to(device), labels.to(device)                
                        logps = model.forward(images)
                        batch_loss = criterion(logps, labels)                
                        test_loss += batch_loss.item()                
                    # calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim =1)                
                        equality = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
                    
                    print(f"Epochs {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"test_loss: {test_loss/len(dataloader['testloader']):.3f}.. "
                      f"Test accuracy: {accuracy/len(dataloader['testloader']):.3f}")

                # now we need to set out training loss back to zero and model back to training 
                    running_loss = 0
                    model.train()
# Save the checkpoint 
def checkpoint(model, image_datasets, optimizer, learning_rate):
    
    model.class_to_idx = image_datasets['train_data'].class_to_idx

    checkpoint = {'arch': 'vgg16',
                  'epochs':5,
                  'input_size': 25088,
                  'output_size': 102,
                  'hidden_layer_size': 4096,
                  'learning_rate': learning_rate,
                  'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'class_to_idx': image_datasets['train_data'].class_to_idx,
                  'optimizer_state_dict': optimizer.state_dict()}

    torch.save(checkpoint, 'checkpoint.pth')

Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['arch'] == "vgg16":
        model=models.vgg16(pretrained = True)
    #model.arch = checkpoint['arch']
    model.classifier = checkpoint['classifier']
    model.optimizer =  checkpoint['optimizer_state_dict']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model
