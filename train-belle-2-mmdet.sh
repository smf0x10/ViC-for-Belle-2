cd mmdetection
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh ./configs/belle_2/belle-2-hep2coco-mmt-simple.py 4
