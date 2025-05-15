cd mmdetection
# To use 4 GPUS
#CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh ./configs/belle_2/belle-2-klm-hep2coco-mmt-simple.py 4
# To use 1 GPU
CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh ./configs/belle_2/belle-2-klm-hep2coco-mmt-simple.py 1
