mkdir $1/excel
python3 mmdetection/tools/analysis_tools/hep_eval.py --pkl $1/results_ep12.pkl --json mmdetection/data/BELLE2/bbox_scale_10/pgun_KL_sample_klm__b00000001__e00000047.json --output_dir $1/excel/ --visual_ind -1 --klm 1
