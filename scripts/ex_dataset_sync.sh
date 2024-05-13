#!/bin/bash
echo Generating Simulations
python3 generate_dataset_sync.py --save_folder data/dataset/0_sync \
                                 --n_simulations 10 \
                                 --n_sheets 10 \
                                 --n_guards 3 \
                                 --dt 1e-2 \
                                 --t_max 10 \
                                 --boundary periodic \
                                 --n_it_crossings 2 \
                                 --track_sheets \
                                 --v_max 10 \
                                 --dx_max 0.2 \
                                 --dE_max 1e-5 \
                                 --random_seed 42

echo Generating graph/target pairs
python3 process_dataset.py --data_folder data/dataset/0_sync \
                           --save_folder data/processed/0_sync \
                           --var_target dvdt \
                           --dt_undersample 10 \
                           --n_neighbors 1 \
                           --augment_x \
                           --augment_t
