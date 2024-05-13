import sys
import yaml
import argparse
import numpy as np

from tqdm import tqdm
from pathlib import Path

from sheet_model.asynchronous import AsyncSheetModel
from sheet_model.utils import async2sync, get_dx_eq, get_x_eq


class CustomFormatter(argparse.RawDescriptionHelpFormatter):
    "Helper formatter for nicer argparse input arguments print"

    def _get_help_string(self, action):
        help_string = super()._get_help_string(action)
        if action.type is not None:
            help_string = f"[{action.type.__name__}] " + help_string
            if not action.default in [None, "==SUPPRESS=="]:
                help_string += f" (Optional, Default: {action.default})"
        else:
            help_string = "[flag] " + help_string
            if action.default:
                help_string += " (Optional, Default: Enabled)"
            else:
                help_string += " (Optional, Default: Disabled)"

        return help_string


def parse_args():
    parser = argparse.ArgumentParser(
        description="""
        This script allows you to generate a dataset of simulations using the Sheet Model Asynchronous algorithm.
        
        For each simulation, 3 (synchronous) time-series will be generated:
            x_<i>.py - (#timesteps, #sheets) positions [Units: L]
            v_<i>.py - (#timesteps, #sheets) velocities [Units: L * w_p]
            x_eq_<i>.py - (#timesteps, #sheets) equilibrium positions [Units: L]
        where <i> is a unique simulation identifier (integer).
        
        A YAML file (config.yml) containing the dataset details will be generated and stored in the same folder.
       
        Units:
            Distance - L (box size) or dx_eq := L/n_sheets (intersheet spacing in equilibrium)
            Time - 1/w_p (2*pi/w_p equals one plasma oscillation period)
        """,
        formatter_class=CustomFormatter,
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        help="""Path to directory where store simulation data.
        We recommend using a subfolder of ./data/dataset/.
        If chosen directory already exists, files will be over-written (not deleted).
        If parent directory does not exist, it will be automatically generated.""",
    )
    parser.add_argument(
        "--n_simulations",
        type=int,
        help="Number of simulations to produce.",
    )
    parser.add_argument(
        "--L",
        type=float,
        default=1,
        help="Simulation box length [Arb. Units].",
    )
    parser.add_argument(
        "--n_sheets",
        type=int,
        help="Number of sheets to use (same across simulations).",
    )
    parser.add_argument(
        "--boundary", type=str, help="Choose 'reflecting' or 'periodic'."
    )
    parser.add_argument(
        "--dt",
        type=float,
        help="""Simulation step to use for storage [Units: 1/w_p].
        Asynchronous time-series returned by the algorithm will be converted to synchronous time-series of this frequency.""",
    )
    parser.add_argument(
        "--t_max",
        type=float,
        help="Total simulation time [Units: 1/w_p].",
    )
    parser.add_argument(
        "--track_sheets",
        action="store_true",
        help="""If set, outputs trajectories considering sheet interactions as crossings.
        Otherwise considers interactions as collisions. 
        GNS performance is significantly better for datasets generated with this flag activated.""",
    )
    parser.add_argument(
        "--v_max",
        type=float,
        help="""Maximum initial velocity [Units: L/n_sheets * w_p].
        Initial sheet velocities will be sampled from a uniform distribution within the range [-v_max, v_max].""",
    )
    parser.add_argument(
        "--dx_max",
        type=float,
        default=0,
        help="""Maximum initial displacement from equilibrium [Units: L/n_sheets].
        Initial sheet displacements from equilibrium will be sampled from a uniform distribution within the range [-dx_max, dx_max].
        Value should be < 0.5""",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed to use for initialization of the sheet velocities and positions.",
    )
    args = parser.parse_args()
    print(args)
    return args


def main(args):
    dir_ = Path(args.save_folder)
    dir_.mkdir(parents=True, exist_ok=True)

    # save simulation parameters
    with open(dir_ / "info.yml", "w", encoding="utf-8") as f:
        aux = vars(args)
        aux["algorithm"] = "async"
        yaml.dump(aux, f)
        del aux

    # initialize sheet model
    sim = AsyncSheetModel(
        L=args.L, boundary=args.boundary, track_sheets=args.track_sheets
    )

    # equilibrium spacing
    dx_eq = get_dx_eq(args.n_sheets, args.L)
    # equilibrium positions
    x_eq = get_x_eq(args.n_sheets, args.L)
    # v_max in units of L
    v_max = args.v_max * dx_eq

    # generate training sims
    i = 0
    n_zeros = int(np.round(np.log10(args.n_simulations)))

    # seed for reproducibility
    np.random.seed(args.random_seed)

    with tqdm(total=args.n_simulations) as pbar:
        while i < args.n_simulations:
            # initial random positions
            x_0 = x_eq + args.dx_max * np.random.random(args.n_sheets) * dx_eq
            # initial random velocities
            v_0 = np.random.uniform(-v_max, v_max, args.n_sheets)

            try:
                T, X, V, X_eq, E = sim.run_simulation(
                    x_0=x_0,
                    v_0=v_0,
                    t_max=args.t_max,
                    return_inside_box=True,
                    verbose=False,
                )

                X, V, X_eq = async2sync(
                    T=T, X=X, V=V, X_eq=X_eq, dt=args.dt, t_max=args.t_max, L=args.L
                )

                # energy should always be ~ conserved !
                # if this does not happen there is a bug in the code :')
                if np.abs(E[-1] - E[0]) / E[0] < 1e-10:
                    np.save(dir_ / f"x_{i:0{n_zeros}d}.npy", X)
                    np.save(dir_ / f"v_{i:0{n_zeros}d}.npy", V)
                    np.save(dir_ / f"x_eq_{i:0{n_zeros}d}.npy", X_eq)

                    i += 1
                    pbar.update(1)

                else:
                    print("[Warning] Energy not conserved")
                    np.save(dir_ / f"px_{i:0{n_zeros}d}.npy", X)
                    np.save(dir_ / f"pv_{i:0{n_zeros}d}.npy", V)
                    np.save(dir_ / f"px_eq_{i:0{n_zeros}d}.npy", X_eq)

            except Exception as e:
                if e == KeyboardInterrupt:
                    sys.exit()

                else:
                    print(e)


if __name__ == "__main__":
    main(parse_args())
