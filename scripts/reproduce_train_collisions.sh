#!/bin/bash
set -e

CUDA_DEVICE=0
DATASET="data/dataset/train/10/periodic/"
DT_UNDERSAMPLE=10
N_NEIGHBORS=1
DATA_FOLDER="data/processed_temp/train/dt1e-1_collisions/"
W_SIZE=5

python3 process_dataset.py --data_folder $DATASET \
                           --save_folder $DATA_FOLDER\
                           --var_target dvdt \
                           --dt_undersample $DT_UNDERSAMPLE \
                           --n_neighbors $N_NEIGHBORS \
                           --augment_x \
                           --augment_t \
                           --mode collisions \
                           --w_size $W_SIZE


MODEL_CFG="config/m_dt1e-1_default.yml"
MODEL_FOLDER="models/final/dt1e-1/collisions/0/"

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 train_gnn.py --data_folder $DATA_FOLDER \
                                                       --model_folder $MODEL_FOLDER \
                                                       --i_valid 9900 \
                                                       --train_steps 1000000 \
                                                       --loss_norm l2 \
                                                       --lr_scheduler \
                                                       --scale_targets \
                                                       --model_cfg $MODEL_CFG
