import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import json
import argparse

def get_input_args():
    
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser(description = 'image classification')
    
    
    # command line arguments as mentioned above using add_argument() from ArguementParser method
    #parser.add_argument('--test_image', type = str, 
                                 #help = 'path to the test image')
    parser.add_argument('--test_image', type = str, default='./flowers/test/20/image_04910.jpg',
                                 help = 'path to the test image')
    parser.add_argument('--gpu', action = "store", default='gpu',
                                 help = 'selecting gpu / cpu')
    parser.add_argument('--save_dir', type = str, default='checkpoint.pth',
                                 help = ' path to save checkpoint')
    parser.add_argument('--category_name', type = str, default='cat_to_name.json', 
                                 help='Path of Mapping flower Name')
    parser.add_argument('--top_k', type = int, default = 5, 
                         help = 'top k for classes')



    return parser.parse_args()

# label mapping
def mapping(category_name):
    with open(category_name, 'r') as f:
        cat_to_name = json.load(f)    
    return cat_to_name


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    if checkpoint['arch'] == "vgg16":
        model=models.vgg16(pretrained = True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.optimizer = checkpoint['optimizer']
    model.class_to_idx = checkpoint['class_to_idx']
    
    for param in model.parameters():
        param.requires_grad = False   
                          
    return model
   
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(image)
    width, height = image.size
    # print(image.size)
    if width > height:
        image.thumbnail(((width/height)*256, 256))
        
    elif height > width:
        image.thumbnail((256, (height/width)*256))
    
    new_width, new_height = image.size
    # cropping the image
    cropped_image = image.crop(((new_width-224)/2, (new_height-224)/2, (new_width+224)/2, (new_height+224)/2 ))
    np_image = np.array(cropped_image)/255
    
    
    # images to be normalized
    mean = [0.485, 0.456, 0.406]
    std_dev = [0.229, 0.224, 0.225]
    
    normalized_image = (np_image - mean) / std_dev
    
    new_image = torch.FloatTensor(normalized_image.transpose((2, 0, 1)))
    
    return new_image

def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device) 
    model.eval()
    
    
    image = process_image(image_path)
    image = image.numpy()

    image= torch.from_numpy(image).type(torch.FloatTensor)
 
    image = image.unsqueeze(0)
    image = image.to(device)
    
    with torch.no_grad():        
        
        logps = model.forward(image)
        ps = torch.exp(logps)
    
        top_ps, top_class = ps.topk(topk, dim = 1)

    top_pred = top_ps.tolist()[0] 
    class_to_idx = model.class_to_idx 
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_labels = [idx_to_class[x.item()] for x in top_class[0].data]

    return top_pred, top_labels

def main():

    in_arg = get_input_args()                                                                                   
   # data_dir = in_arg.data_dir
    cat_name = in_arg.category_name
    save_checkpoint = in_arg.save_dir
    test_image = in_arg.test_image
    top_k = in_arg.top_k
    
    #path = ('./flowers/test/34/image_06961.jpg')
    #path = ('./flowers/test/10/image_07090.jpg')  
    #path = ('./flowers/test/15/image_06351.jpg')
    #path = ('./flowers/test/19/image_06155.jpg')
     #path = ('./flowers/test/12/image_04023.jpg')
    #path = ('./flowers/test/20/image_04910.jpg')  
    cat_to_name = mapping(cat_name)
    save_model = load_checkpoint(save_checkpoint)
    
    probs, classes = predict(test_image, save_model, top_k)
    #print('flowers name:', mapping(cat_name))
    print(probs)
    print(classes)
    #print('Flower name is: {} Probability: {} ', format(cat_to_name[classes[0]].title(), str(probs[0])))
    #print("flower_names", mapping(cat_name))
 
    
    
if __name__ == "__main__":
    main()
