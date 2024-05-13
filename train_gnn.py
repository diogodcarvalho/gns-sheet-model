import yaml
import argparse

from generate_dataset_async import CustomFormatter
from gns.model import SMGNN
from gns.train import train


def parse_args():
    parser = argparse.ArgumentParser(
        description="""
        This script trains a GNN model on a dataset of sheet model simulations graph/target pairs.
        
        The following files will be generated:
            loss_i.txt - train loss at each gradient update step
            loss.txt - mean train + validation loss at the end of each epoch
            model_cfg.yml - GNN architecture hyperparameters
            params_best.pkl - GNN weights for epoch with min valid loss
            params_final.pkl - GNN weights for last epoch
            train_cfg.yml - argparse input parameters used
            train_data.yml - training dataset information
        """,
        formatter_class=CustomFormatter,
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        help="Path to folder containing pre-processed graphs/targets.",
    )
    parser.add_argument(
        "--model_folder",
        type=str,
        help="""Path where output files generated during training will be stored.
        We recomend using a subfolder of ./models/ .
        If chosen directory already exists, files will be over-written (not deleted).
        If parent directory does not exist, it will be automatically generated.""",
    )
    parser.add_argument(
        "--model_cfg",
        type=str,
        help="""Path to YAML config file with GNN model architecture hyperparameters.
        Examples are provided in ./config/*.yml""",
    )
    parser.add_argument(
        "--i_train",
        type=int,
        default=None,
        help="""Ending index of training set, i.e. will train on simulation [0; i_train[.
        Useful to train on smaller subset of data.
        If not set i_train = i_valid. (Optional, Default: i_valid)""",
    )
    parser.add_argument(
        "--i_valid",
        type=int,
        help="Starting index of validation set, i.e. will use simulations [i_valid; i_last] for validation.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="""Training batch size.
        Corresponds to the number of simulations per GNN weights update (1 graph = 1 simulation).""",
    )
    parser.add_argument(
        "--train_steps",
        type=int,
        default=2e6,
        help="Number of gradient updates (!= number of epochs)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        help="""Early stopping condition (number of epochs without valid loss improvement).
        (Optional, Default : None)""",
    )
    parser.add_argument(
        "--loss_norm", type=str, default="l2", help="Loss function, 'l1' or 'l2'."
    )
    parser.add_argument(
        "--lr_scheduler",
        action="store_true",
        help="Turn on lr scheduler (described in the paper).",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0,
        help="Adamw weight decay, if zero uses adam.",
    )
    parser.add_argument(
        "--scale_targets",
        action="store_true",
        help="""Scale targets by the dataset std value.
        Only useful for 'collisions' mode since targets range can be large.""",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed to use for GNN weights initialization.",
    )
    args = parser.parse_args()
    print(args)
    return args


def main(args):
    with open(args.model_cfg, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    args = vars(args)
    del args["model_cfg"]

    train(**args, net_fn=SMGNN, net_fn_params=model_cfg)


if __name__ == "__main__":
    main(parse_args())
