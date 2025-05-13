mkdir $1/excel
python3 tools/analysis_tools/hep_eval.py --pkl $1/results_ep12.pkl --json data/BELLE2/bbox_scale_10/pgun_k_long_5__b00000001__e00009480.json --output_dir $1/excel/ --visual_ind -1
