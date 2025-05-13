# ViC-for-Belle-2

Project Abstract:
Belle II in Tsukuba, Japan is an experiment where electrons and positrons are collided at relativistic speeds in a particle accelerator to study the differences between matter and antimatter. High-energy electron-positron collisions create other particles, which can be detected with a series of concentric detectors surrounding the collision point. One such detector is the calorimeter, and one such particle is the K-Long. K-Longs interact with the calorimeter’s detector material through the strong nuclear force, which the calorimeter reads as a blob of energy roughly centered on the K-Long’s position. For this project, we are taking a machine learning system originally created to find antineutrons traveling through the calorimeter of the BES III experiment in Bejing and adapting it to find the position and momentum of K-Longs in the Belle II calorimeter. This system projects the energy observed in the calorimeter’s individual crystal sensors onto a two-dimensional image, then uses an image recognition algorithm to find the most likely position and momentum of the K-Long. We compared this system to current methods of locating K-Longs and found that it gave a more accurate position estimate than current models when running on simulated data with no background, and that it could provide an estimate for the K-Long's momentum, which our current algorithms lack. ViC was further improved by altering the input images to include data from the inner layers of the K-Long and Muon Detector (KLM).

The original version of ViC can be found here: https://github.com/yuhongtian17/ViC
## Setup
First, clone this repository onto a Linux machine with access to one or more CUDA-capable Nvidia graphics cards:

```git clone https://github.com/smf0x10/ViC-for-Belle-2.git```

Enter the mmdetection directory:

```cd ViC-for-Belle-2/mmdetection```

Install the dependencies (If you happen to be using Iowa State's high-performance computing cluster for this, make sure you start an interactive session with a machine that has a GPU before doing this)

```python3 setup.py install```

## Training
To train the model, you will need some the JSON files containg the raw calorimeter and KLM data and annotations in the mmdetection/data/BELLE2/bbox_scale_10 directory. The exact paths to these files are specified in mmdetection/configs/belle_2/belle_2_datasets/belle_2_detection.py and belle_2_detection_klm.py. When running with KLM data, the expected formats of these JSON files is different from the expected formats when running without KLM data.

The belle_2_detection.py and belle_2_detection_klm.py scripts each contain a list named `train_ann_files` and two variables called `val_ann_file` and `test_ann_file`. When adding a new data file to mmdetection/data/BELLE2/bbox_scale_10, update these variables with the appropriate path to make the algorithm use them for training, evaluation, or testing.

Once the data files are present and the config files have been updated accordingly, run one of these scripts in the ViC-for-Belle-2 directory to start training (This part requires CUDA to be available):

```./train-belle-2-mmdet.sh``` to train without KLM data on ECL-only data files 

```./train-belle-2-klm-mmdet.sh``` to train on data files with KLM data included

## Testing
When the training process has finished, run one of the following scripts to test the model on a new set of data:

```./test-belle-2-mmdet.sh``` If running without KLM data

```./test-belle-2-klm-mmdet.sh``` If running with KLM data

## Visualization and Spreadsheet Generation
After training and testing the model, you can generate png files to visualize a subset of the data with the following scripts:

```./generate-visual-belle-2-mmdet path/to/work/directory```, where path/to/work/directory is the path to the directory with the testing results in it in work_dirs. If the testing scripts were unmodified, this will be `mmdetection/work_dirs/belle-2-hep2coco-mmt-simple/`. The png files will be created in a directory called "visual" inside this work directory.

```./generate-visual-belle-2-klm-mmdet path/to/work/directory``` is the same as above, but it includes KLM data. If the testing scripts were unmodified, the work directory should be `mmdetection/work_dirs/belle-2-hep2coco-mmt-klm-simple/`.

Excel spreadsheets containing the results can be created with the following scripts:

```./generate-excel-belle-2-mmdet path/to/work/directory```

```./generate-excel-belle-2-klm-mmdet path/to/work/directory```
