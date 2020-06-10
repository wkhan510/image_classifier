# Imports python modules
import argparse

def get_input_args():
    
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser(description = 'image classification')
    
    
    # command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('--test_image', type = str, default='./flowers/test/34/image_06961.jpg',
                                 help = 'path to the test image')
    parser.add_argument('--data_dir', type = str, default="./flowers/",
                                 help = 'path to the flowers images')
    parser.add_argument('--gpu', action = "store", default='gpu',
                                 help = 'selecting gpu / cpu')
    parser.add_argument('--arch',type= str, default = 'vgg16',
                                 help = 'path to architecture moduel')
    parser.add_argument('--save_dir', type = str, default='checkpoint.pth',
                                 help = ' path to save checkpoint')
    parser.add_argument('--cat_name', type = str, default='cat_to_name.json', 
                                 help='Path of Mapping flower Name')
    parser.add_argument('--epochs', type = int, default = 1,
                                 help='Number of Epochs')
    parser.add_argument('--learning_rate', type = int, default = 0.003,
                                 help='learning rate')
    parser.add_argument('--dropout', type = int, default = 0.2,
                                 help='dropout')                        
    
    
    return parser.parse_args()
