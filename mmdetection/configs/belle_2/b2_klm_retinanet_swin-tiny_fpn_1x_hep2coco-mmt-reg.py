_base_ = [
    '../_base_/_hep2coco_models_/retinanet_swin-tiny_fpn.py',
    'belle_2_datasets/belle_2_detection_klm.py',
    '../_base_/_hep2coco_schedules_/schedule_1x_rst.py', '../_base_/default_runtime.py'
]

model = dict(
    bbox_head=dict(
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        loss_mmt_reg=dict(type='L1Loss', loss_weight=1.0)))
