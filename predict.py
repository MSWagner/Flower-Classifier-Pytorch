# PROGRAMMER: Michael Wagner
# DATE CREATED: 08.04.2020                                  
# PURPOSE: Predicts a flower class.
#
# Use argparse Expected Call with <> indicating expected user input:
#      python predict.py <path to image> <path to checkpoint> --top_k <k most likely classes> --category_names <JSON path to map categories to real names> -g
#   python predict.py flowers/test/1/image_06764.jpg checkpoints_test/checkpoint_best_accuracy.pth
##

import torch
from torchvision import transforms
import torch.nn.functional as F

import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

import json

from helper import get_predict_input_args, load_checkpoint, build_model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    opened_img = Image.open(image)
    
    img_transforms =  transforms.Compose([
                      transforms.Resize(255),
                      transforms.CenterCrop(224),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], 
                                           [0.229, 0.224, 0.225])
    ])
    
    return img_transforms(opened_img)

def predict(image, model, topk, useGPU=True):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Use GPU if it's available
    device = torch.device("cuda" if torch.cuda.is_available() and useGPU else "cpu")
    print(f"Device: {device}")
    
    model.eval()
    model.to(device);
    
    image = image.unsqueeze_(0)
    
    with torch.no_grad():
        inputs = image.to(device)
        output = model.forward(inputs)
        probability = F.softmax(output.data,dim=1)
        
        return probability.topk(topk)

def main():
    input_args = get_predict_input_args()
    
    # Load checkpoint
    checkpoint, validation_accuracy = load_checkpoint(input_args.checkpoint_path)
        
    useGPU = input_args.gpu is not None
        
    # Build model
    model = build_model(checkpoint["arch"],
                        checkpoint["hidden_units_01"], 
                        checkpoint["hidden_units_02"], 
                        checkpoint)

    # Process image
    processed_image = process_image(input_args.image_path)
  
    # Predict topK
    topk = predict(processed_image, model, input_args.top_k, useGPU)
    
    # Show result
    with open(input_args.category_names_path, 'r') as f:
        cat_to_name = json.load(f)
    
    probs = topk[0][0].cpu().numpy()
    categories = [cat_to_name[str(category_index+1)] for category_index in topk[1][0].cpu().numpy()]
    
    for i in range(len(probs)):
        print("TopK {}, Probability: {}, Category: {}\n".format(i+1, probs[i], categories[i]))
    
if __name__ == '__main__':
    main()