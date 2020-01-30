# RepNet

This is the code for the BRRF algorithm (cite needed), designed to segment images using both representation of RepNet and edge maps of COB (cite needed).
 
## Getting Started

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

## Authors

Or Isaacs.


