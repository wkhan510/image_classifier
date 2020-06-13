import argparse

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import json

def get_input_args():
    
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser(description = 'image classification')
    

    parser.add_argument('--data_dir', type = str, default="./flowers",
                                 help = 'path to the flowers images')
    parser.add_argument('--gpu', action = "store", default = 'gpu',
                                 help = 'selecting gpu / cpu')
    parser.add_argument('--arch',type= str, default = 'vgg16',
                                 help = 'path to architecture moduel')
    parser.add_argument('--save_dir', type = str, default='checkpoint.pth',
                                 help = ' path to save checkpoint')
    parser.add_argument('--epochs', type = int, default = 3,
                                 help='Number of Epochs')
    parser.add_argument('--input_size', type = int, default = 25088,
                                 help='Number of Epochs')
    parser.add_argument('--learning_rate', type = int, default = 0.003,
                                 help='learning rate')
    parser.add_argument('--dropout', type = int, default = 0.2,
                                 help='dropout')
    
    return parser.parse_args()

#data_dir = 'flowers'
#data_dir = in_arg.data_dir 
#train_dir = data_dir + '/train'
#valid_dir = data_dir + '/valid'
#test_dir = data_dir + '/test'

# Define your transforms for the training, validation, and testing sets
def model_transformation(load_data):
    
    train_dir = load_data + '/train'
    valid_dir = load_data + '/valid'
    test_dir = load_data + '/test'
   
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

# Build and train your network

def classify_network(arch, m_dropout, lr):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = 25088
        hidden_layer1 = 1024
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True)
        input_size = 25088
        hidden_layer1 = 1024
    elif arch == 'densenet':
        model = models.densenet121(pretrained=True)
        input_size = 1024
        hidden_layer1 = 256
    

   # model=models.vgg16(pretrained = True)

    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(nn.Linear(input_size, hidden_layer1),
                           nn.ReLU(),
                           nn.Dropout(m_dropout),
                           nn.Linear(hidden_layer1, 102),
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

def train(device, model, epoch_s, dataloader, optimizer, criterion):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Do validation on the test set
    epochs = epoch_s
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
    return model                
# TODO: Do validation on the test set
def network_test(device, model, dataloader, criterion):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_loss = 0
    accuracy = 0
    model.eval() 
    with torch.no_grad():
        for images, labels in dataloader['testloader']:
            images, labels = images.to(device), labels.to(device)                
            logps = model.forward(images)
            batch_loss = criterion(logps, labels)                
            test_loss += batch_loss.item()                
            #calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim =1)                
            equality = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
        print('network accuracy: %d %%' % (100 * accuracy / len(dataloader['testloader'])))

# Save the checkpoint 
def checkpoint(arch, model, image_datasets, optimizer, learning_rate, epoch_s, input_size):
    
    model.class_to_idx = image_datasets['train_data'].class_to_idx

    checkpoint = {'arch': arch,
                  'epochs': epoch_s,
                  'input_size': input_size,
                  'output_size': 102,
                  #'hidden_layer_size': 1024,
                  'learning_rate': learning_rate,
                  'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'class_to_idx': image_datasets['train_data'].class_to_idx,
                  'optimizer': optimizer}

    torch.save(checkpoint, 'checkpoint.pth')
   
    
    return None

    
    #'optimizer_state_dict': optimizer.state_dict()
# TODO: Write a function that loads a checkpoint and rebuilds the model
def main():
 
    
    in_arg = get_input_args()                                                                                   
    data_dir = in_arg.data_dir                                                                               
    train_dir = data_dir + '/train'
    test_dir = data_dir + '/test'
    valid_dir = data_dir + '/valid'
    
    
    
    arch = in_arg.arch
    device_model = in_arg.gpu
    lr = in_arg.learning_rate
    save_checkpoint = in_arg.save_dir
    #category_name = in_arg.cat_name
    epoch_s = in_arg.epochs
    input_size = in_arg.input_size
    dropout = in_arg.dropout
    

    #path = ('./flowers/test/34/image_06961.jpg')
    #path = ('./flowers/test/10/image_07090.jpg')
    #path = ('./flowers/test/34/image_06961.jpg')
    
    #Function that checks command line arguments using in_arg  
    
    data_transforms, image_datasets, dataloader = model_transformation(data_dir)
    model, criterion, optimizer = classify_network(arch, dropout, lr)
    #train(device_model, model, epoch_s, dataloader, optimizer, criterion)
    network_test(device_model, model, dataloader, criterion)
    checkpoint(arch, model, image_datasets, optimizer, lr, epoch_s, input_size)
 
    
    
if __name__ == "__main__":
    main()
