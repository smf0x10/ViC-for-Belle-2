mkdir $1/visual
python3 mmdetection/tools/analysis_tools/hep_eval.py --pkl $1/results_ep12.pkl --json mmdetection/data/BELLE2/bbox_scale_10/pgun_KL_sample_klm__b00000001__e00000047.json --output_dir $1/visual/ --visual_ind 0 --visual_end 40 --klm 1
