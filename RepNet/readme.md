# RepNet

This is the code for RepNet (cite needed), designed to learn and create region representations for images.

## Getting Started

Guide to using RepNet:

Creating training and eval set:
1. Set a folder for your dataset. Have a folder called 'trainval_images' that holds your trainval images and a folder called 'trainval_GT' that holds a segmentation in the same manner as in BSDS500. For the images you want to have a representation generated for them, create a folder called 'all_images' and put them in it.
2. Copy createDataSet.m to the folder and run.
3. You'll have an output of the total number of segments in the trainval.
4. In resnet50_input.py change NUM_CLASSES put the total number of segments. Also set NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN to the trainval size. Also change the value of 'bsds_mean' to your dataset mean color.

Training the network:
1. Run resnet50_multi_gpu_train.py.
Make sure to setup:
-data_dir: Path to the dataset.
-weights: Path for pretrained weights in case of transfer learning.
-use_pretrained: Whether to use a pretrained weights.
-train_dir: Directory where to write event logs and checkpoint.

Creating evalutations:
1. Run resnet50_eval.py.
Make sure to setup:
-data_dir: Path to the dataset.
-num_examples: The number of images in 'all_images' you want to generate representation for.

### Prerequisites

Matlab for creating the dataset.

Python3.7 for training the network, with packages:

numpy==1.16.4
six==1.12.0
scikit_learn==0.21.3
tensorflow==2.0.0

### Pre-trained weights

For weights already trained on Pascal Context, please download 'fp_weights.npy' from the following link: XXXXXXXXX_TODO_XXXXXXXXXXXXXXX

## Authors

Oran Shayer (majority of contribution).

Or Isaacs.

