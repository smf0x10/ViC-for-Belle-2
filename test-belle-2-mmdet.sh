CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=33020 ./tools/dist_test.sh "./configs/belle_2/belle-2-hep2coco-mmt-simple.py" "./work_dirs/belle-2-hep2coco-mmt-simple/epoch_12.pth" 4 --out "./work_dirs/belle-2-hep2coco-mmt-simple-klm-compare/results_ep12.pkl"

