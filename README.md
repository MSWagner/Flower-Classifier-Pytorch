# Flower image classifier with Pytorch

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

<img src="https://github.com/MSWagner/Flower-Classifier-Pytorch/blob/master/assets/prediction.png" width="400">

## Jupyter Notebook Files

- [Jupyter Notebook](https://github.com/MSWagner/Flower-Classifier-Pytorch/blob/master/Image%20Classifier%20Project.ipynb)
- [HTML](https://github.com/MSWagner/Flower-Classifier-Pytorch/blob/master/Image%20Classifier%20Project.html)

## Command line application

#### Train classifier:

```
python train.py <data_dir> --save_dir <checkpoint folder> -g
``` 

| Argument      | Short         | Default | Description  |
| ------------- |:-------------:| -------------:| -----:|
| data_dir      | | | Folder path for the flower images |
| --save_dir    | | checkpoints |Folder path to save the checkpoints |
| --arch | | vgg16 | CNN Model Architecture (vgg16 or densenet121) |
| --learning_rate | -l | 0.001 | Learning rate |
| --epochs | -e | 1 | Epochs to train the model |
| --hidden_units_01 | -h1 | 4096 | Hidden units of the first layer |
| --hidden_units_02 | -h2 | 1024 | Hidden units of the second layer |
| --checkpoint_path | -cp | None |  Path of a checkpoint you want to reuse |
| --gpu | -g | False |Use gpu if available |

#### Predict image:

```
python predict.py <image_path> <checkpoint_path> -g
``` 

| Argument      | Short         | Default | Description  |
| ------------- |:-------------:| -------------:| -----:|
| image_path      | | | Image path for the prediction |
| checkpoint_path | | checkpoints/checkpoint_best_accuracy.pth | Checkpoint path |
| --top_k | -k    | 1 | Number of the top k most likely classes |
| --json_path | -json | cat_to_name.json | JSON file path to map categories to real names |
| --gpu | -g | False |Use gpu if available |

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
