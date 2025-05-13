# This is set up to use 4 GPUs; to only use 1, change CUDA_VISIBLE_DEVICES to only include 0 and replace the 4 on the end with a 1
cd mmdetection
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh ./configs/belle_2/belle-2-klm-hep2coco-mmt-simple.py 4
