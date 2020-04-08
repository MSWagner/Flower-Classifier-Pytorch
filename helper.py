import argparse
import torch
from torch import nn
from torchvision import models

def get_train_input_args():
    """
    Command Line Arguments:
      1. Data Folder as data_dir
      2. Save Folder for checkpoints as --save_dir with default value 'checkpoints'
      3. CNN Model Architecture as --arch with default value 'vgg16'
      4. Learning rate as --learning_rate with default value 0.001
      5. Traning epochs as --epochs with default value 1
      6. Hidden units of the first layer as --hidden_units_01 with default value 4096
      7. Hidden units of the second layer as --hidden_units_02 with default value 1024
      8. Path of a checkpoint as --checkpoint_path
      9. Use gpu if available as -g or --gpu

    This function returns these arguments as an ArgumentParser object.
    
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('data_dir', type = str, help = 'path to the folder of flower images')    
    parser.add_argument('--save_dir', type = str, default = 'checkpoints', help = 'Save folder for model checkpoints') 
    parser.add_argument('--arch', type = str, default = 'vgg16', choices=['vgg16', 'densenet121'], help = 'CNN Model Architecture') 
    parser.add_argument('-l', '--learning_rate', type = float, default = 0.001, help = 'Learning rate') 
    parser.add_argument('-e', '--epochs', type = int, default = 1, help = 'Epochs to train the model') 
    parser.add_argument('-h1', '--hidden_units_01', type = int, default = 4096, help = 'Hidden units of the first layer')
    parser.add_argument('-h2', '--hidden_units_02', type = int, default = 1024, help = 'Hidden units of the second layer')
    parser.add_argument('-cp', '--checkpoint_path', type = str, help = 'Path of a checkpoint')
    parser.add_argument('-g', '--gpu', action='store_true', required=False, help = 'Use gpu if available')

    in_args = parser.parse_args()
    
    if in_args is None:
        print("* Doesn't Check the Command Line Arguments because 'get_input_args' hasn't been defined.")
    else:
        print("Command Line Arguments:\n dir =", in_args.data_dir, "\n save_dir =", in_args.save_dir, "\n arch =", in_args.arch, "\n learning_rate =", in_args.learning_rate, "\n epochs =", in_args.epochs, "\n hidden_units_01 =", in_args.hidden_units_01, "\n hidden_units_02 =", in_args.hidden_units_02)
    
    if in_args.checkpoint_path is not None:
        print("\n checkpoint_path =", in_args.checkpoint_path)
        
    if in_args.gpu is not None:
        print("\n Use gpu if available")

    return in_args

def get_predict_input_args():
    """
    Command Line Arguments:
      1. Image path
      2. Checkpoint path with default value checkpoints/checkpoint_best_accuracy.pth
      3. Number of the top k most likely classes as --top_k with default value 5
      4. JSON file to map categories to real names as --category_names with default value cat_to_name.json
      5. Use gpu if available as -g or --gpu

    This function returns these arguments as an ArgumentParser object.
    
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('image_path', type = str, help = 'Path of the prediction image')    
    parser.add_argument('checkpoint_path', type = str, default = 'checkpoints/checkpoint_best_accuracy.pth', help = 'Checkpoint path of the prediction model') 
    parser.add_argument('-k', '--top_k', type = int, default = 1, help = 'Number of the top k most likely classes') 
    parser.add_argument('-json', '--category_names_path', type = str, default = "cat_to_name.json", help = 'JSON file to map categories to real names')
    parser.add_argument('-g', '--gpu', action='store_true', required=False, help = 'Use gpu if available')

    in_args = parser.parse_args()
    
    if in_args is None:
        print("* Doesn't Check the Command Line Arguments because 'get_input_args' hasn't been defined.")
    else:
        print("Command Line Arguments:\n image_path =", in_args.image_path, "\n checkpoint_path =", in_args.checkpoint_path, "\n top_k =", in_args.top_k, "\n category_names_path =", in_args.category_names_path)
        
    if in_args.gpu is not None:
        print("\n Use gpu if available")

    return in_args

def save_checkpoint(model, train_data, path, validation_accuracy, arch, hidden_units_01, hidden_units_02):
    model.class_to_idx = train_data.class_to_idx
    torch.save({'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx,
                'validation_accuracy': validation_accuracy,
                'arch': arch,
                'hidden_units_01': hidden_units_01,
                'hidden_units_02': hidden_units_02
               },
                path)
    
    print(f"-- Saved checkpoint: {path}\n")

def load_checkpoint(path):
    checkpoint = torch.load(path)
    
    return checkpoint, checkpoint['validation_accuracy']

def build_model(arch, hidden_units_01, hidden_units_02, checkpoint=None):
    pre_trained_archs = {
        "vgg16": 25088,
        "densenet121": 1024
    }

    model = eval("models.{}(pretrained=True)".format(arch))
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
        
    fc1_input = pre_trained_archs[arch]

    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                              ('fc_1', nn.Linear(fc1_input, hidden_units_01)),
                              ('relu_1', nn.ReLU()),
                              ('drop_1', nn.Dropout(p=0.5)),
                              ('fc_2', nn.Linear(hidden_units_01, hidden_units_02)),
                              ('relu_2', nn.ReLU()),
                              ('drop_2', nn.Dropout(p=0.5)),
                              ('fc_3', nn.Linear(hidden_units_02, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    
    if checkpoint is not None:
        model.class_to_idx = checkpoint['class_to_idx']
        model.load_state_dict(checkpoint['state_dict'])
        
    return model

def test_model(model, testloader, useGPU=True): 
    # Use GPU if it's available
    device = torch.device("cuda" if torch.cuda.is_available() and useGPU else "cpu")
    print(f"Device: {device}")
    
    model.to(device);
    model.eval()
    
    criterion = nn.NLLLoss()

    accuracy = 0
    test_loss = 0

    for inputs, labels in testloader:
        with torch.no_grad():
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test loss: {test_loss/len(testloader):.3f}.. "
          f"Test accuracy: {accuracy/len(testloader):.3f}")