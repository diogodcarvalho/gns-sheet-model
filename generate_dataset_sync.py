import sys
import yaml
import argparse
import numpy as np

from tqdm import tqdm
from pathlib import Path

from generate_dataset_async import CustomFormatter
from sheet_model.synchronous import SyncSheetModel
from sheet_model.utils import get_dx_eq, get_x_eq


def parse_args():
    parser = argparse.ArgumentParser(
        description="""
        This script allows you to generate a dataset of simulations using the Sheet Model Synchronous algorithm.
        
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
        "--boundary",
        type=str,
        help="Choose 'reflecting' or 'periodic'.",
    )
    parser.add_argument(
        "--n_guards",
        type=int,
        help="""Number of guard sheets to use (same value will be use for left/right boundary).
        For larger sheet velocities and/or time steps, a larger value of guards should be used.
        If a sheet crosses all guards, or all guards enter the simulation box, the simulation will abort and an error message will be shown.""",
    )
    parser.add_argument(
        "--n_it_crossings",
        type=int,
        default=2,
        help="""Number of iterations used to correct the estimate of the sheet crossing times.
        Setting n = 0 corresponds to not correcting for sheet crossings between t -> t + dt.
        Energy conservation does not change significantly for n > 2.""",
    )
    parser.add_argument(
        "--t_max",
        type=float,
        help="Total simulation time [Units: 1/w_p].",
    )
    parser.add_argument(
        "--dt",
        type=float,
        help="""Simulation time step to be used [Units: 1/w_p].
        Strongly affects energy conservation capabilities of the algorithm (DE/DT ~ dt^4).
        For larger sheet velocities, a smaller value can/should be used since this will not
        result in significant increases in run time (as less crossing corrections are applied per dt).
        Usual values are in the range [0.01, 0.1] 1/w_p.""",
    )
    parser.add_argument(
        "--dt_undersample",
        type=int,
        default=1,
        help="""Undersample factor for storage purposes,
        i.e. dt_storage = dt * dt_undersample.""",
    )
    parser.add_argument(
        "--track_sheets",
        action="store_true",
        help="""If set, outputs trajectories considering sheet interactions as crossings.
        Otherwise considers interactions as collisions.""",
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
        "--dE_max",
        type=float,
        default=1e-6,
        help="""Maximum relative energy loss allowed for storing simulation.
        If |E-E0|/E0 >= dE_max the simulation will be rejected. 
        For each rejected simulation, an extra one is performed to guarantee the desired dataset size.""",
    )
    parser.add_argument(
        "--accept_unsorted",
        action="store_true",
        help="""If set, stores simulations where x and x_eq might not be equally sorted in all time-steps
        (this can happen due to the sheet model crossing correction algorithm and is not a bug).
        Should always be disabled for generating training data for the GNN.
        For each rejected simulation, an extra one is performed to guarantee the desired dataset size.""",
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
    # seed for reproducibility
    np.random.seed(args.random_seed)

    dir_ = Path(args.save_folder)
    dir_.mkdir(parents=True, exist_ok=True)

    # save simulation parameters
    with open(dir_ / "info.yml", "w", encoding="utf-8") as f:
        aux = vars(args)
        aux["algorithm"] = "sync"
        yaml.dump(aux, f)
        del aux

    # initialize sheet model
    sim = SyncSheetModel(
        n_sheets=args.n_sheets,
        L=args.L,
        boundary=args.boundary,
        n_guards=args.n_guards,
        dt=args.dt,
        track_sheets=args.track_sheets,
        n_it_crossings=args.n_it_crossings,
    )

    # generate training sims
    i = 0
    n_zeros = int(np.round(np.log10(args.n_simulations)))

    with tqdm(total=args.n_simulations) as pbar:
        # equilibrium spacing
        dx_eq = get_dx_eq(args.n_sheets, args.L)

        while i < args.n_simulations:
            # initial positions of charged sheets
            x_0 = get_x_eq(args.n_sheets, args.L)
            x_0 += args.dx_max * np.random.random(args.n_sheets) * dx_eq

            # initial velocities
            v_max = args.v_max * dx_eq
            v_0 = np.random.uniform(-v_max, v_max, args.n_sheets)

            # ----------------------------------
            # run simulation
            # returns (for each time step):
            #   X - particle positions
            #   V - particle velocities
            #   X_eq - particle eq positions
            #   E - total energy
            try:
                X, V, X_eq, E = sim.run_simulation(
                    x_0=x_0,
                    v_0=v_0,
                    t_max=args.t_max,
                    dt_undersample=args.dt_undersample,
                    verbose=False,
                )

                # if X and X_eq not equally sorted, do not store
                if not args.accept_unsorted:
                    i_sort = np.argsort(X, axis=-1)
                    aux = np.take_along_axis(X_eq, i_sort, axis=-1)
                    if np.any(np.diff(aux, axis=-1) < 0):
                        print("[Warning] Not Sorted (x_i < x_j & x_eq_i > x_eq_j)")
                        continue

                # save only if energy is conserved
                if np.abs(E[-1] - E[0]) / E[0] < args.dE_max:
                    np.save(dir_ / f"x_{i:0{n_zeros}d}.npy", X)
                    np.save(dir_ / f"v_{i:0{n_zeros}d}.npy", V)
                    np.save(dir_ / f"x_eq_{i:0{n_zeros}d}.npy", X_eq)

                    i += 1
                    pbar.update(1)

                else:
                    print("[Warning] Energy not conserved")

            except Exception as e:
                if e == KeyboardInterrupt:
                    sys.exit()

                else:
                    print(e)


if __name__ == "__main__":
    main(parse_args())
