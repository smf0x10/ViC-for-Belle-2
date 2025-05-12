# dataset settings
dataset_type = 'Belle2PGunDatasetKLM'
data_root = 'data/BELLE2/bbox_scale_10/'
# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromHEPengKLM', backend_args=backend_args,
         bg_version='black_randn', snr_db=10.0),
    dict(type='HEPLoadAnnotationsKLM', with_bbox=True, with_mmt=True),
    dict(type='Resize', scale=(960, 480), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='HEPPackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromHEPengKLM', backend_args=backend_args,
         bg_version='black_randn_seed', snr_db=10.0),
    dict(type='HEPLoadAnnotationsKLM', with_bbox=True, with_mmt=True),
    dict(type='Resize', scale=(960, 480), keep_ratio=True),
    dict(
        type='HEPPackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #
train_ann_files=[
    #'old/pgun_k_long_1__b00000001__e00009138.json',
    #'old/pgun_k_long_2__b00000001__e00009049.json',
    #'old/pgun_k_long_3__b00000001__e00009120.json',
    #'old/pgun_k_long_1__b00000001__e00010000.json',
    #'old/pgun_k_long_2__b00000001__e00010000.json',
    #'old/pgun_k_long_3__b00000001__e00010000.json',
    #'pgun_k_long_high_p_1__b00000001__e00016741.json',
    #'pgun_k_long_high_p_2__b00000001__e00016668.json',
    #'pgun_k_long_high_p_3__b00000001__e00016740.json',
    #'pgun_gamma_1__b00000001__e00019320.json',
    #'pgun_gamma_2__b00000001__e00019364.json',
    #'pgun_gamma_3__b00000001__e00019352.json',
    #'pgun_k_long_1__b00000001__e00016105.json',
    #'pgun_k_long_2__b00000001__e00016173.json',
    #'pgun_k_long_3__b00000001__e00016055.json',
    #'with_klm/pgun_klm_test.json'
    #'with_klm/pgun_k_long_1__b00000001__e00009348.json',
    #'with_klm/pgun_k_long_2__b00000001__e00009520.json',
    #'with_klm/pgun_k_long_3__b00000001__e00009340.json',
    'with_klm_sizes/pgun_k_long_1__b00000001__e00009348.json',
    'with_klm_sizes/pgun_k_long_2__b00000001__e00009520.json',
    'with_klm_sizes/pgun_k_long_3__b00000001__e00009340.json'
    ]

#val_ann_file='old/pgun_k_long_4__b00000001__e00009110.json'
#val_ann_file='old/pgun_k_long_4__b00000001__e00010000.json'
#val_ann_file='pgun_k_long_high_p_4__b00000001__e00016740.json'
#val_ann_file='pgun_gamma_4__b00000001__e00019347.json'
#val_ann_file='pgun_k_long_4__b00000001__e00015869.json'
#val_ann_file='with_klm/pgun_k_long_4__b00000001__e00009208.json'
val_ann_file='with_klm_sizes/pgun_k_long_4__b00000001__e00009208.json'
#test_ann_file='old/pgun_k_long_5__b00000001__e00009092.json'
#test_ann_file='old/pgun_k_long_5__b00000001__e00010000.json'
#test_ann_file='pgun_k_long_high_p_5__b00000001__e00016748.json'
#test_ann_file='pgun_gamma_5__b00000001__e00019313.json'
#test_ann_file='pgun_k_long_5__b00000001__e00009480.json'
#test_ann_file='with_klm/pgun_k_long_5__b00000001__e00009480.json'
test_ann_file='with_klm_sizes/pgun_k_long_5__b00000001__e00009480.json'


train_dataset_base = dict(
    type=dataset_type,
    data_root=data_root,
    # ann_file='annotations/instances_train2017.json',
    # data_prefix=dict(img='train2017/'),
    ann_file='',
    data_prefix=dict(img='./'),
    p_RM_thr=0.0,
    btr_eng_thr=0.0,
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline,
    backend_args=backend_args)
train_datasets = []

for i in range(0, len(train_ann_files)):
    temp = train_dataset_base.copy()
    temp['ann_file'] = train_ann_files[i]
    train_datasets.append(temp)

# print(train_datasets)

# ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        # https://github.com/open-mmlab/mmdetection/blob/main/mmdet/datasets/dataset_wrappers.py
        type='ConcatDataset',
        datasets=train_datasets))
val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # ann_file='annotations/instances_val2017.json',
        # data_prefix=dict(img='val2017/'),
        ann_file=val_ann_file,
        data_prefix=dict(img='./'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    # ann_file=data_root + 'annotations/instances_val2017.json',
    ann_file=data_root + val_ann_file,
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator

# inference on test dataset and
# format the output results for submission.
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + test_ann_file,
        data_prefix=dict(img='./'),
        test_mode=True,
        pipeline=test_pipeline))
test_evaluator = dict(
    type='CocoMetric',
    metric='bbox',
    format_only=True,
    ann_file=data_root + test_ann_file,
    outfile_prefix='./work_dirs/belle_2/test')
