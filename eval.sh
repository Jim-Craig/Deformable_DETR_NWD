#!/bin/bash

GPU=2
CKPT=$1
OUTDIR=$2

CUDA_VISIBLE_DEVICES=$GPU python main.py \
  --eval \
  --resume $CKPT \
  --coco_path /data/coco \
  --output_dir $OUTDIR
