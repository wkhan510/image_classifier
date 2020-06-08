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
from get_input_parse import get_input_args
#import train
from train import model_transformation,  mapping, classify_network, validation, train, network_test, checkpoint, load_checkpoint
from predict import process_image, predict, load_checkpoint

def main():

    in_arg = get_input_args()
    data_dir = in_arg.data_dir
    train_dir = data_dir + '/train'
    test_dir = data_dir + '/test'
    valid_dir = data_dir + '/valid'
    
    #path = ('./flowers/test/34/image_06961.jpg')
    path = ('./flowers/test/10/image_07090.jpg')
    
    arch = in_arg.arch
    device_model = in_arg.gpu
    lr = in_arg.learning_rate
    save_checkpoint = in_arg.save_dir
    category_name = in_arg.cat_name
    epochs = in_arg.epochs
    dropout = in_arg.dropout
    

    
    #path = ('./flowers/test/34/image_06961.jpg')
    
    #Function that checks command line arguments using in_arg  
    
    data_transforms, image_datasets, dataloader = model_transformation(data_dir)
    mapping(category_name)
    model, criterion, optimizer = classify_network(arch, dropout, lr)
    train(device_model, model, dataloader, optimizer, criterion)
    network_test(device_model, model, dataloader, optimizer, criterion)
    checkpoint(model, image_datasets, optimizer, lr)
    #model = load_checkpoint(in_arg.save_dir)
    save_model = load_checkpoint(in_arg.save_dir)
    
    probs, classes = predict(path, save_model, topk=5)
    print(probs)
    print(classes)
    #print("flower_names", mapping(category_name))
 
    
    
if __name__ == "__main__":
    main()
