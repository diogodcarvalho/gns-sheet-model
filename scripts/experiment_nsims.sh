#!/bin/bash
set -e

CUDA_DEVICE=0
DATASET="data/dataset/train/10/periodic/"

################################################################################
# dt = 1e-1
################################################################################
DT_UNDERSAMPLE=10
N_NEIGHBORS=1
PATIENCE=100
DATA_FOLDER="data/processed_temp/train/dt1e-1/"

for ARCHITECTURE in equivariant default
do

  MODEL_CFG="config/m_dt1e-1_${ARCHITECTURE}.yml"
  OUT_DIR="models/experiments/n_sim/${ARCHITECTURE}"

  ### NO AUGMENTATION
  python3 process_dataset.py --data_folder $DATASET \
                             --save_folder $DATA_FOLDER\
                             --var_target dvdt \
                             --dt_undersample $DT_UNDERSAMPLE \
                             --n_neighbors $N_NEIGHBORS

  for N_SIM in 1 10 100 1000 9900
  do
    MODEL_FOLDER="${OUT_DIR}/no_augment/dt1e-1/${N_SIM}/"
    echo $MODEL_FOLDER

    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 train_gnn.py --data_folder $DATA_FOLDER \
                                                --model_folder $MODEL_FOLDER \
                                                --i_train $N_SIM \
                                                --i_valid 9900 \
                                                --train_steps 2000000 \
                                                --patience $PATIENCE \
                                                --loss_norm l2 \
                                                --lr_scheduler \
                                                --model_cfg $MODEL_CFG
  done

  ### t augmentation
  python3 process_dataset.py --data_folder $DATASET \
                             --save_folder $DATA_FOLDER\
                             --var_target dvdt \
                             --dt_undersample $DT_UNDERSAMPLE \
                             --n_neighbors $N_NEIGHBORS \
                             --augment_t

   for N_SIM in 1 10 100 1000 9900
   do
    MODEL_FOLDER="${OUT_DIR}/augment_t/dt1e-1/${N_SIM}/"
    echo $MODEL_FOLDER

    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 train_gnn.py --data_folder $DATA_FOLDER \
                                                --model_folder $MODEL_FOLDER \
                                                --i_train $N_SIM \
                                                --i_valid 9900 \
                                                --train_steps 2000000 \
                                                --patience $PATIENCE \
                                                --loss_norm l2 \
                                                --lr_scheduler \
                                                --model_cfg $MODEL_CFG
  done

  ### x augmentation
  python3 process_dataset.py --data_folder $DATASET \
                             --save_folder $DATA_FOLDER\
                             --var_target dvdt \
                             --dt_undersample $DT_UNDERSAMPLE \
                             --n_neighbors $N_NEIGHBORS \
                             --augment_x

  for N_SIM in 1 10 100 1000 9900
  do
    MODEL_FOLDER="${OUT_DIR}/augment_x/dt1e-1/${N_SIM}/"
    echo $MODEL_FOLDER

    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 train_gnn.py --data_folder $DATA_FOLDER \
                                                --model_folder $MODEL_FOLDER \
                                                --i_train $N_SIM \
                                                --i_valid 9900 \
                                                --train_steps 2000000 \
                                                --patience $PATIENCE \
                                                --loss_norm l2 \
                                                --lr_scheduler \
                                                --model_cfg $MODEL_CFG
  done

  ### both augmentations
  python3 process_dataset.py --data_folder $DATASET \
                             --save_folder $DATA_FOLDER\
                             --var_target dvdt \
                             --dt_undersample $DT_UNDERSAMPLE \
                             --n_neighbors $N_NEIGHBORS \
                             --augment_x \
                             --augment_t

  for N_SIM in 1 10 100 1000 9900
  do
    MODEL_FOLDER="${OUT_DIR}/both/dt1e-1/${N_SIM}/"
    echo $MODEL_FOLDER

    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 train_gnn.py --data_folder $DATA_FOLDER \
                                                --model_folder $MODEL_FOLDER \
                                                --i_train $N_SIM \
                                                --i_valid 9900 \
                                                --train_steps 2000000 \
                                                --patience $PATIENCE \
                                                --loss_norm l2 \
                                                --lr_scheduler \
                                                --model_cfg $MODEL_CFG
  done
done
