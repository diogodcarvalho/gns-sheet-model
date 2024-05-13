import jax
import yaml
import argparse

from generate_dataset_async import CustomFormatter
from gns.preprocess import build_train_dataset

# force jax to use CPU (faster for these calculations)
jax.config.update("jax_platform_name", "cpu")
# need higher precision to avoid problems with finite difference calculations
jax.config.update("jax_enable_x64", True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="""
        This script transforms a dataset of Sheet Model simulations into a training dataset 
        consisting of pairs of input graphs (one graph per simulation) and targets for a GNN model.
        
        Two directories will be generated:
            graphs/*.pkl
            targets/*.npy
        and the file names <i>.pkl / <i>.npy will match the corresponding simulation identifier.
            
        Additionally, a YAML file (info.yml) containing the original training dataset and preprocessing details
        will be generated.
        """,
        formatter_class=CustomFormatter,
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        help="Path to directory containing sheet model simulation dataset.",
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        help="""Path where to store output graphs and targets.
        Should be different from data_folder.
        We recommend using a subfolder of ./data/processed/ .
        If chosen directory already exists, files will be over-written (not deleted).
        If parent directory does not exist, it will be automatically generated.""",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="crossings",
        help="""Consider sheet interactions as 'crossings' or 'collisions' for graph/target generation.
        If dataset was generated with the --track_sheets flag enabled, either option can be selected.
        Otherwise, only 'collisons' can be used.
        GNN models trained with 'crossings' obtain significantly better results.
        """,
    )
    parser.add_argument(
        "--var_target",
        type=str,
        default="dvdt",
        help="""Output target, either 'dvdt' (finite difference acceleration) or 'dx' (displacement).
        Choosing 'dvdt' provides significantly better results.""",
    )
    parser.add_argument(
        "--dt_undersample",
        type=int,
        default=1,
        help="Undersampling factor to apply to the simulation data, i.e. dt = dt_data * dt_undersample",
    )
    parser.add_argument(
        "--w_size",
        type=int,
        default=1,
        help="""Input window size (number of past velocities to use in graph node representation).
        Only relevant for performance when mode = 'collisions'.""",
    )
    parser.add_argument(
        "--n_neighbors",
        type=int,
        default=1,
        help="""Number of neighboring nodes to connect.
        Higher connectivity should reduce the number of message passing steps required.""",
    )
    parser.add_argument(
        "--augment_t",
        action="store_true",
        help="Augment simulation with symmetric version around t-axis.",
    )
    parser.add_argument(
        "--augment_x",
        action="store_true",
        help="Augment simulation with symmetric version around x-axis.",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default=None,
        help="Config YAML file with graph/target params choice (overrides other args)",
    )
    args = parser.parse_args()
    print(args)
    return args


def main(args):
    if args.cfg is not None:
        print("Using YAML config:", args.cfg)

        with open(args.cfg, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        build_train_dataset(
            data_folder=args.data_folder, save_folder=args.save_folder, **cfg
        )
    else:
        cfg = vars(args)
        del cfg["cfg"]
        build_train_dataset(**cfg)


if __name__ == "__main__":
    main(parse_args())
