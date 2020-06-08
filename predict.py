import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
from train import load_checkpoint




#def load_checkpoint(filepath):
    #checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    #model=models.vgg16(pretrained = True)
    #model.arch = checkpoint['arch']
    #model.classifier = checkpoint['classifier']  
    #model.class_to_idx = checkpoint['class_to_idx']
    #model.optimizer = checkpoint['optimizer']
    #model.load_state_dict(checkpoint['state_dict'])
    
    #return model

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
    
    
    # print(image.size)
    # cropping the image
    cropped_image = image.crop(((new_width-224)/2, (new_height-224)/2, (new_width+224)/2, (new_height+224)/2 ))
    
    # print(cropped_image.size)
    # coverting image to numpy array, color channel encoded integers are 0-255, but we model need (0-1)
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
    # for using gpu we need to do the following change
    #image = torch.from_numpy(image).type(torch.cuda.FloatTensor)
    
    image = image.unsqueeze(0)
    image = image.to(device)
    
    with torch.no_grad():        
        
        logps = model.forward(image)
        ps = torch.exp(logps)
    
        top_ps, top_class = ps.topk(topk, dim = 1)
        #top_class = top_class.cpu().numpy() # testing
       # top_class = top_class.cpu().numpy()[0].tolist()
    #idx_to_class={val:key for key,val in model.class_to_idx.items()}
        
        #Converting the indices to class
        
        #top_pred = top_ps.data.numpy()[0]
    top_pred = top_ps.tolist()[0] # test 
        #pred_indexes = top_class.data.numpy()[0].tolist()
    # pred_indexes = top_class.cpu().numpy()[0].tolist() # test
    class_to_idx = model.class_to_idx  #test
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_labels = [idx_to_class[x.item()] for x in top_class[0].data]
    # pred_labels = [idx_to_class[x] for x in pred_indexes]
        
        
    #return top_ps, top_class
    return top_pred, top_labels
#path = ('./flowers/test/10/image_07090.jpg')
#path = ('./flowers/test/34/image_06961.jpg')
#path = "./flowers/test/102/image_08012.jpg"
#probs, classes = predict(path, load_checkpoint('checkpoint.pth'))
#probs, classes = predict(image_path, model)
#print(probs)
#print(classes)
