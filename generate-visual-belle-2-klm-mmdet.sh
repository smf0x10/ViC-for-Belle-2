mkdir $1/visual
python3 tools/analysis_tools/hep_eval.py --pkl $1/results_ep12.pkl --json data/BELLE2/bbox_scale_10/with_klm_sizes/pgun_k_long_5__b00000001__e00009480.json --output_dir $1/visual/ --visual_ind 0 --visual_end 100 --klm 1
