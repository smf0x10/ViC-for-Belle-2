cd mmdetection
#CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=33020 ./tools/dist_test.sh "./configs/belle_2/belle-2-klm-hep2coco-mmt-simple.py" "./work_dirs/belle-2-klm-hep2coco-mmt-simple/epoch_12.pth" 4 --out "./work_dirs/belle-2-klm-hep2coco-mmt-simple-with-sizes/results_ep12.pkl"

CUDA_VISIBLE_DEVICES=0 PORT=33020 ./tools/dist_test.sh "./configs/belle_2/belle-2-klm-hep2coco-mmt-simple.py" "./work_dirs/belle-2-klm-hep2coco-mmt-simple/epoch_12.pth" 1 --out "./work_dirs/belle-2-klm-hep2coco-mmt-simple/results_ep12.pkl"

