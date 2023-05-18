#!/usr/bin/env bash

# -------- #
# TRAINING #
# -------- #

## D-RMM + Sparse R-CNN
bash ./tools/dist_train.sh \
    ./configs/drmm/drmm_r50_fpn_syncbn_fullaug_3x_coco.py \
    8

## Resume
#bash ./tools/dist_train.sh ./configs/drmm/drmm_r50_fpn_syncbn_fullaug_3x_coco.py 8 \
#    --resume ./work_dirs/drmm_r50_fpn_syncbn_fullaug_3x_coco/latest.pth



# multi-gpu training
# bash tools/dist_train.sh \
#    ${CONFIG_FILE} \
#    ${GPU_NUM}
#    --resume ${CHECKPOINT_FILE}
