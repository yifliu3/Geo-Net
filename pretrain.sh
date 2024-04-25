#!/bin/bash
CUDA_VISIBLE_DEVICES=7 python main.py --config cfgs/pretrain_teethseg3d.yaml --exp_name pretrain_teethseg --val_freq 10
