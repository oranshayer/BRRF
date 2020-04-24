# Enhancing Generic Segmentation with Learned Region Representations
This repository is the implementation of RepNet Learning and Boundaries and Region Representation Fusion, as reported in the paper "Enhancing Generic Segmentatoin with Learned Region Representations".

Authors: Or Isaacs*, Oran Shayer*, Michael Lindenbaum

(* - equal contribution)

## RepNet

The first part of the work is the representation learning algorithm.

### Getting Started

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

* For weights already trained on Pascal Context, please download 'fp_weights.npy' from the following link: TODO_WILL_BE_ADDED_SOON
* For weights already trained on BSDS500, please download from the following link: TODO_WILL_BE_ADDED_SOON
* Representations on BSDS500 images can be downloaded here: https://drive.google.com/open?id=18VetI01cEVgC0hTzoW4i7Z8nM5MJlUxF

## BRRF

The second part of this work is the BRRF algorithm, designed to segment images using both representation of RepNet and edge maps of COB.

### Getting Started

Guide to using BRRF:

General requirements:
1. Set in auxilaryFunc.py in the variable 'repDirUsed' the folder containing the representations of RepNet for your images named 'XXXX.npy' (XXX being the image name).
2. Set in auxilaryFunc.py in the variable 'cobResDir' the folder containing the COB CNN results for your images named 'XXXX.mat' (XXX being the image name). For all images you need to have a E1 variable in the mat file containing an edge map. For training images you need the .mat file to additionally hold a 'ucm' variable for the Ultrametric Contour Map for the edge map.
3. Set in auxilaryFunc.py in the variable 'databaseDir' the folder containing the data for your dataset. The algorithm is built around runnning for BSDS500 and might need changes for other datasets.

Creating training examples for BRRF:
1. Either run generateExamples.py or run main.py with the flag --generateXY (which will continue to the evaluation stage afterwards).
To change the thresholds on the training examples, change the 'edgeThresList' variable in 'auxilaryFunc.py'.

Evaluating the dataset:
1. Run 'main.py'. We recommend looking at the option available with -h.

### Prerequisites

Matlab for evaluating the segmentation.

Python3.7 for training the network, with packages:

numpy==1.16.4
scipy==1.3.0
scikit_image==0.15.0
matplotlib==3.1.0
Bottleneck==1.2.1
tqdm==4.32.1
matlab==0.1
scikit_learn==0.20.3
skimage==0.0

## Results
Can be downloaded from the following links:
* BSDS500: https://drive.google.com/open?id=1HSw01RcVlsbuRhxF2ziUQg-9_Ex3QIsm
* Pascal Context: https://drive.google.com/open?id=1h7ToFLTxSNe1OcZP2xwCddg4Ph4rflS9
* RCF edges for reproducability: https://drive.google.com/open?id=1Q56sVAEaBCHsKf-ZRqpRTWaKVxMoG9HD

## Citation

If you use our code for research, please cite our paper:
@article{isaacs2019enhancing,
    title={Enhancing Generic Segmentation with Learned Region Representations},
    author={Or Isaacs and Oran Shayer and Michael Lindenbaum},
    journal={arXiv preprint arXiv:1911.08564},
    year={2019}
}

