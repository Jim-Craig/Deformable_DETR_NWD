#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

torchrun \
  --standalone \
  --nproc_per_node=2 \
  main.py \
  --with_box_refine \
  --backbone resnet50 \
  --epochs 50 \
  --lr 2e-4 \
  --lr_backbone 1e-5 \
  --batch_size 2 \
  --num_queries 300 \
  --set_cost_class 2 \
  --set_cost_nwd 1 \
  --cls_loss_coef 2 \
  --bbox_loss_nwd 1 \
  --resume ./r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth \
  --output_dir ./outputs/nwd_deformable_detr
