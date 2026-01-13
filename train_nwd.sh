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
  --num_classes 11 \
  --num_queries 300 \
  --dataset_file visdrone \
  --set_cost_class 1 \
  --set_cost_nwd 2.5 \
  --set_cost_giou 1 \
  --cls_loss_coef 1 \
  --giou_loss_coef 1 \
  --bbox_loss_nwd 2.5 \
  --eval \
  --resume ./outputs/nwd_deformable_detr_visdrone/checkpoint0059.pth \
  --output_dir ./outputs/nwd_deformable_detr_visdrone_test
