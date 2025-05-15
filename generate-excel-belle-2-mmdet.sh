mkdir $1/excel
# The --json option should be changed to match the test_ann_file
python3 mmdetection/tools/analysis_tools/hep_eval.py --pkl $1/results_ep12.pkl --json mmdetection/data/BELLE2/bbox_scale_10/pgun_KL_sample__b00000001__e00000079.json --output_dir $1/excel/ --visual_ind -1
