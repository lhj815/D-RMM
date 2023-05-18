#!/usr/bin/env bash

# ---- #
# TEST #
# ---- #
bash ./tools/dist_test.sh ./configs/drmm/drmm_r101_fpn_syncbn_fullaug_3x_coco.py \
    ./work_dirs/sparse_mdod_r101_fpn_syncbn_fullaug_3x_coco/20220203_124211/epoch_36.pth \
    2 --eval bbox





# single-gpu testing
# python tools/test.py \
#    ${CONFIG_FILE} \
#    ${CHECKPOINT_FILE} \
#    [--out ${RESULT_FILE}] \
#    [--eval ${EVAL_METRICS}] \
#    [--show]
#    [--show-dir]

# multi-gpu testing
# bash tools/dist_test.sh \
#    ${CONFIG_FILE} \
#    ${CHECKPOINT_FILE} \
#    ${GPU_NUM} \
#    [--out ${RESULT_FILE}] \
#    [--eval ${EVAL_METRICS}]
