_base_ = './drmm_r101_fpn_fullaug_3x_coco.py'
model = dict(
    backbone=dict(
        norm_cfg=dict(type='SyncBN', requires_grad=True), 
        norm_eval=False,
    ),
)
