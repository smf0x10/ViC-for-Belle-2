mkdir $1/visual
# Change the --json option to match the test_ann_file
# visual_end was originally set to 100, but it was reduced here because there anren't that many events in the sample file
python3 mmdetection/tools/analysis_tools/hep_eval.py --pkl $1/results_ep12.pkl --json mmdetection/data/BELLE2/bbox_scale_10/pgun_KL_sample__b00000001__e00000079.json --output_dir $1/visual/ --visual_ind 0 --visual_end 40
