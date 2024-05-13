#!/bin/bash
set -e

CUDA_DEVICE=0

# dt = 10e-1
DATASET="data/dataset/train/10/periodic/"
DT_UNDERSAMPLE=10
N_NEIGHBORS=1
PATIENCE=10
DATA_FOLDER="data/processed_temp/train/dt1e-1/"

# augmentation
python3 process_dataset.py --data_folder $DATASET \
                           --save_folder $DATA_FOLDER\
                           --var_target dvdt \
                           --dt_undersample $DT_UNDERSAMPLE \
                           --n_neighbors $N_NEIGHBORS \
                           --augment_x \
                           --augment_t


for SEED in 0 1 2 3 4
do

  MODEL_CFG="config/m_dt1e-1_equivariant.yml"
  MODEL_FOLDER="models/final/dt1e-1/equivariant/${SEED}/"

  CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 train_gnn.py --data_folder $DATA_FOLDER \
                                                         --model_folder $MODEL_FOLDER \
                                                         --i_valid 9900 \
                                                         --train_steps 1000000 \
                                                         --loss_norm l2 \
                                                         --lr_scheduler \
                                                         --model_cfg $MODEL_CFG \
                                                         --random_seed $SEED

  MODEL_CFG="config/m_dt1e-1_nosent.yml"
  MODEL_FOLDER="models/final/dt1e-1/nosent/${SEED}/"

  CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 train_gnn.py --data_folder $DATA_FOLDER \
                                                        --model_folder $MODEL_FOLDER \
                                                        --i_valid 9900 \
                                                        --train_steps 1000000 \
                                                        --loss_norm l2 \
                                                        --lr_scheduler \
                                                        --model_cfg $MODEL_CFG \
                                                        --random_seed $SEED

  MODEL_CFG="config/m_dt1e-1_default.yml"
  MODEL_FOLDER="models/final/dt1e-1/${SEED}/"

  CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 train_gnn.py --data_folder $DATA_FOLDER \
                                                         --model_folder $MODEL_FOLDER \
                                                         --i_valid 9900 \
                                                         --train_steps 1000000 \
                                                         --loss_norm l2 \
                                                         --lr_scheduler \
                                                         --model_cfg $MODEL_CFG \
                                                         --random_seed $SEED

done
