#!/bin/bash -ex
# 0 - 24

aug=("none" "dropN")
a1=$(( $2 % 2 ))
a2=$(( ($2 - $a1) / 2 ))

CUDA_VISIBLE_DEVICES=$5 python main_cl.py --dataset $1 --aug1 ${aug[$a1]} --aug_ratio1 0.1 --aug2 ${aug[$a2]} --aug_ratio2 0.1 --semi_split $4 --suffix $3
