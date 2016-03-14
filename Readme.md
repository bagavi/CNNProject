# Image colorization

## Setup
After cloing the git repository. Download the following files in a 'Data' Folder just before the folder of this repo

## Structure
1. [TinyImageNet/train.lua](TinyImageNet/train.lua) can be used to train the neural network framework. Options to choose the input dataset, loading a pre-trained model, and other hyperparameters are included in the code itself. If running on a GPU, you may use train_AWS.lua
2. [TinyImageNet/test-checkpoint.ipynb](TinyImageNet/test-checkpoint.ipynb) can be used to view the results (or intermediate results) of the training procedure.
3. [TinyImageNet/Net3.lua](TinyImageNet/Net3.lua) Contains the Neural Network framework, which we used for majority of our experiments. Other frameworks are in Net1.lua, Net2.lua, Net4.lua, which we use for experimenting.
4. [TinyImageNet/VGG.lua](TinyImageNet/VGG.lua) Contains functions for loading, pruning the VGG net. Also contains VGG-specific utility functions including preprocessing data and extracting hypercolumns from VGG
5. [TinyImageNet/Utils.lua](TinyImageNet/Utils.lua) Contains some generic torch utility functions which we needed while working on the project.

## Data
1. TinyImageNet: TinyImageNet (will contain the training. test, val examples)= http://cs231n.stanford.edu/tiny-imagenet-200.zip (Unzip it)p

2. VGG_Caffe (contains the deploy text and trained model)	
   a. DeployText = https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/c3ba00e272d9f48594acef1f67e5fd12aff7a806/VGG_ILSVRC_16_layers_deploy.prototxt
   b. The Model  = http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel

# Links
1. Torch for Numpy users: https://github.com/torch/torch7/wiki/Torch-for-Numpy-users
2. Tensors in Torch: https://github.com/torch/tutorials/blob/master/2_Tensors.ipynb
3. 60min Deep neural network tutorial: https://github.com/soumith/cvpr2015/blob/master/Deep%20Learning%20with%20Torch.ipynb
4. Image package: https://github.com/torch/image
5. nn module documentation: https://github.com/torch/nn/blob/master/doc/overview.md#nn.overview.dok
6. https://github.com/torch/torch7 - The whole of Torch code.
7. torch-hdf5: https://github.com/deepmind/torch-hdf5
8. How to save weights and biases of a trained network: add the gitter link  
