#!/bin/bash
echo Generating Simulations
python3 generate_dataset_async.py --save_folder data/dataset/0_async \
                                  --n_simulations 10 \
                                  --n_sheets 10 \
                                  --dt 1e-2 \
                                  --t_max 10 \
                                  --boundary periodic \
                                  --track_sheets \
                                  --v_max 10 \
                                  --dx_max 0.2 \
                                  --random_seed 42

echo Generating graph/target pairs
python3 process_dataset.py --data_folder data/dataset/0_async \
                           --save_folder data/processed/0_async \
                           --var_target dvdt \
                           --dt_undersample 10 \
                           --n_neighbors 1 \
                           --augment_x \
                           --augment_t