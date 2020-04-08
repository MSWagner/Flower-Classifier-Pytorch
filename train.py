# PROGRAMMER: Michael Wagner
# DATE CREATED: 08.04.2020                                  
# PURPOSE: Trains a flower classifier using a pretrained CNN model.
#
# Use argparse Expected Call with <> indicating expected user input:
#      python train.py <directory with images> --save_dir <directory for checkpoints> -l <learning rate> -e <training epochs> -h1 <count of hidden units in layer1> -h2 <count of hidden units in layer2>  --arch <model> -g
#   
#   Example call:
#   python train.py flowers --save_dir checkpoints_test -l 0.001 -e 1 -h1 1024 -h2 512  --arch vgg16 -g
##

import torch
from torchvision import datasets, transforms
from torch import nn
from torch import optim
import torch.nn.functional as F

from helper import get_train_input_args, save_checkpoint, load_checkpoint, build_model, test_model

def main():
    input_args = get_train_input_args()
    
    # Create & adjust data
    train_dir = input_args.data_dir + '/train'
    valid_dir = input_args.data_dir + '/valid'
    test_dir = input_args.data_dir + '/test'
    
    print("\n\n Trainings folder: {}".format(train_dir))
    print(" Validation folder: {}".format(valid_dir))
    print(" Test folder: {}\n".format(test_dir))
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])]) 

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    
    # Load checkpoint
    checkpoint = None
    best_accuracy = 0
  
    if input_args.checkpoint_path is not None:
        checkpoint, best_accuracy = load_checkpoint(input_args.checkpoint_path)
        
    useGPU = input_args.gpu is not None
    
    arch = input_args.arch if checkpoint is None else checkpoint["arch"]
    hidden_units_01 = input_args.hidden_units_01 if checkpoint is None else checkpoint["hidden_units_01"]
    hidden_units_02 = input_args.hidden_units_02 if checkpoint is None else checkpoint["hidden_units_02"]
    
    # Build model
    model = build_model(arch,
                        hidden_units_01, 
                        hidden_units_02, 
                        checkpoint)

    # Train model
    print("\n\nStart Training...\n")
    
    if best_accuracy > 0:
        print("Last validation accuracy: {}".format(best_accuracy))
    
    epochs = input_args.epochs
    learning_rate = input_args.learning_rate
    steps = 0
    running_loss = 0
    print_every = 10

    train_losses, validation_losses = [], []
    
    # Use GPU if it's available and gpu is not None        
    device = torch.device("cuda" if torch.cuda.is_available() and useGPU else "cpu")
    print(f"Device: {device}")
    
    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    model.to(device);

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                validation_loss = 0
                accuracy = 0

                model.eval()
            
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        validation_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}, "
                      f"Steps {steps}\n"
                      f"Train loss: {running_loss/print_every:.3f}, "
                      f"Validation loss: {validation_loss/len(validloader):.3f}, "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}\n")

                train_losses.append(running_loss/print_every)
                validation_losses.append(validation_loss/len(validloader))

                if best_accuracy < accuracy/len(validloader) and accuracy/len(validloader) > 0.6:
                    best_accuracy = accuracy/len(validloader)
                    path = input_args.save_dir + "/checkpoint_best_accuracy.pth"

                    save_checkpoint(model, train_data, path, best_accuracy, input_args.arch, hidden_units_01, hidden_units_02)

                running_loss = 0
                
    print("\n\nEnd Training...\n")
    
    # Test trained model
    test_model(model, testloader)
    
if __name__ == '__main__':
    main()