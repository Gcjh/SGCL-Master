#!/bin/bash -ex
# 0 - 24

aug=("none" "dropN")
a1=$(( $2 % 2 ))
a2=$(( ($2 - $a1) / 2 ))

CUDA_VISIBLE_DEVICES=$3 python main.py --dataset $1 --aug1 ${aug[$a1]} --aug_ratio1 0.1 --aug2 ${aug[$a2]} --aug_ratio2 0.1 --suffix 0
CUDA_VISIBLE_DEVICES=$3 python main.py --dataset $1 --aug1 ${aug[$a1]} --aug_ratio1 0.1 --aug2 ${aug[$a2]} --aug_ratio2 0.1 --suffix 1
CUDA_VISIBLE_DEVICES=$3 python main.py --dataset $1 --aug1 ${aug[$a1]} --aug_ratio1 0.1 --aug2 ${aug[$a2]} --aug_ratio2 0.1 --suffix 2
CUDA_VISIBLE_DEVICES=$3 python main.py --dataset $1 --aug1 ${aug[$a1]} --aug_ratio1 0.1 --aug2 ${aug[$a2]} --aug_ratio2 0.1 --suffix 3
CUDA_VISIBLE_DEVICES=$3 python main.py --dataset $1 --aug1 ${aug[$a1]} --aug_ratio1 0.1 --aug2 ${aug[$a2]} --aug_ratio2 0.1 --suffix 4
