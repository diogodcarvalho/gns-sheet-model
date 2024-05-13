import jax
import glob
import yaml
import pickle

import numpy as np
import jax.numpy as jnp

from pathlib import Path
from tqdm import tqdm
from functools import partial
from typing import Tuple

from gns.graph import build_graphs
from gns.utils import dxdt
from gns.simulator import SMGNS
from sheet_model.synchronous import SyncSheetModel


def sliding_windows(X: jnp.ndarray, w_size: float) -> jnp.ndarray:
    """
    Returns an array of sliding windows of the input array

    Args:
        X - time series
        w_size - window size

    Returns:
        X_w - (X.shape[0] - w_size + 1, w_size, *X.shape[1:])
    """
    if w_size < 1:
        raise AssertionError(f"Make w_size >= 1: {w_size}")
    X_w = jnp.stack(
        [jnp.roll(X, w_size - i - 1, axis=0) for i in range(w_size)], axis=1
    )
    X_w = X_w[w_size - 1 :]
    return X_w


def init_XV_crossings(
    X: jnp.ndarray,
    V: jnp.ndarray,
    X_eq: jnp.ndarray,
    dt: float,
    boundary: str,
    L: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Returns initial conditions (x_0, v_0, x_eq_0) to use as input for the
    Graph Network Simulator based on the original Sheet Model time-series.

    Considers sheet interactions as crossings, and applies correction to the velocities
    of particles who interact with a reflecting boundary.

    Useful to compare predicted trajectories with ground truth data.

    Args:
        X, V, X_eq - time-series data from Sheet Model simulation.
        dt - time-series timestep.
        boundary - boundary condition ("periodic" or "reflecting").
        L - box length.

    Returns:
        x_0, v_0, x_eq_0 - initial conditions to use for Graph Network Simulator.
    """
    X_aux = X[:2]
    V_aux = V[:2]

    x_0 = X[1]
    x_eq_0 = X_eq[1]

    v_0 = jnp.squeeze(dxdt(X=X_aux, dt=dt, boundary=boundary, L=L))

    if boundary == "reflecting":
        # have to correct dxdt for collisions with wall
        # check for changes in velocity sign of particles who migh hit the wall
        possible_wall_collision = jnp.bitwise_or(
            X_aux[0] + V_aux[0] * dt < 0, X_aux[0] + V_aux[0] * dt > L
        )

        collision_flag = jnp.bitwise_and(
            V_aux[1] * V_aux[0] < 0, possible_wall_collision
        )

        # dxdt is computed as if particle crossed the wall instead of being reflected
        v_0_collision = (
            jnp.where(
                V_aux[0] > 0, jnp.sum(X_aux, axis=0) - 2 * L, jnp.sum(X_aux, axis=0)
            )
            / dt
        )

        v_0 = jnp.where(collision_flag, v_0_collision, v_0)

    i_sorted = jnp.argsort(x_0)

    x_0 = x_0[i_sorted]
    v_0 = v_0[i_sorted]
    x_eq_0 = x_eq_0[i_sorted]

    return x_0, v_0, x_eq_0


def init_XV_collisions(
    X: jnp.ndarray,
    X_eq: jnp.ndarray,
    w_size: jnp.ndarray,
    dt: float,
    boundary: str,
    L: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Similar to init_XV_crossings, but considers sheet interactions as collisions.

    Args:
        X, X_eq - time-series data from Sheet Model simulation.
        w_size - window size (#previous timesteps)
        dt - time-series timestep.
        boundary - boundary condition ("periodic" or "reflecting").
        L - box length.

    Returns:
        X_0, V_0, X_eq_0
            Initial conditions to use for Graph Network Simulator.
    """
    # v_t = (x_t - x_t-1) / dt
    V = dxdt(X=X, dt=dt, boundary=boundary, L=L)
    # have to remove for consistent size between X and V
    X = X[1:]
    X_eq = X_eq[1:]

    X_0 = sliding_windows(X[: w_size + 1], w_size=w_size)[0]
    V_0 = sliding_windows(V[: w_size + 1], w_size=w_size)[0]
    X_eq_0 = sliding_windows(X_eq[: w_size + 1], w_size=w_size)[0]

    return X_0, V_0, X_eq_0


def init_GNS_as_SM(
    x_0: jnp.array,
    v_0: jnp.array,
    x_eq_0: jnp.array,
    gns: SMGNS,
    dt_back: float = 1e-3,
    n_guards: int = 2,
) -> Tuple[jnp.array, jnp.array, jnp.array, float]:
    """
    Computes the initial conditions for the GNS based on the initial conditions
    for the provided Sheet Model simulation. This ensures that the exact same system
    will be modelled.

    This step is required since the Sheet Model uses the instantaneous velocities,
    while the GNS uses the finite difference velocities (which can become significantly
    different at large v or dt).

    Also guarantees we use the right value for the initial energy of the system for the
    energy conservation diagnostics.

    Args:
        x_0, v_0, x_eq_0 - sheet model initial conditions
        gns - SMGNS object
        dt_back - time-step to use for sheet model backward sim
        n_guards - number of guard sheets to used in backward Sheet Model sim.

    Returns:
        x_0_gns, v_0_gns, x_eq_0_gns, E0 - initial conditions to use for GNS
    """
    # compute one dt BACKWARDS with high-res sheet model
    # (equivalent to fliping velocities without changing the sign of dt)
    sim = SyncSheetModel(
        L=gns.L,
        n_sheets=x_0.shape[-1],
        boundary=gns.boundary,
        n_guards=n_guards,
        dt=-dt_back,
        n_it_crossings=2,
        track_sheets=True,
    )

    X, V, X_eq, E = sim.run_simulation(
        x_0=x_0,
        v_0=v_0,
        x_eq_0=x_eq_0,
        t_max=-gns.dt_train,
        dt_undersample=int(gns.dt_train / dt_back),
        verbose=False,
    )

    # compute finite difference velocities
    x_0_gns, v_0_gns, x_eq_0_gns = init_XV_crossings(
        X[::-1],
        V[::-1],
        X_eq[::-1],
        dt=gns.dt_train,
        boundary=gns.boundary,
        L=gns.L,
    )

    return x_0_gns, v_0_gns, x_eq_0_gns, E[0]


def crossings2collisions(
    X: jnp.ndarray, V: jnp.ndarray, X_eq: jnp.ndarray, L: float, np_=jnp
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Converts Sheet Model time-series data produced in "crossings" mode to
    "collisions" mode

    Args:
        X, V, X_eq - time-series data from Sheet Model simulation.
        L - box length.

    Returns:
        X_coll, V_coll, X_eq_coll - new time-series where sheet crossings are now
                                    treated as collisions.
    """
    i_sort = np_.argsort(X_eq % L, axis=-1)
    X_coll = np_.take_along_axis(X, i_sort, axis=-1)
    V_coll = np_.take_along_axis(V, i_sort, axis=-1)
    X_eq_coll = np_.take_along_axis(X_eq, i_sort, axis=-1)

    return X_coll, V_coll, X_eq_coll


def build_train_dataset(
    data_folder: str,
    mode: str = "crossings",
    var_target: str = "dvdt",
    w_size: int = 1,
    dt_undersample: int = 1,
    n_neighbors: int = 1,
    augment_t: bool = False,
    augment_x: bool = False,
    save_folder: str = ".",
) -> None:
    """
    Generates training dataset consisting of pairs of (graphs, targets) from dataset
    of Sheet Model simulations.

    Each graph + target array correspond to a full simulation.

    ONLY IMPLEMENTED FOR PERIODIC BOUNDARIES + SIMULATIONS GENERATED WITH CROSSINGS!

    Args:
        data_folder
            Path to dataset with sheet model simulations.

        mode
            Consider sheet interactions as "crossings" or "collisions".

        var_target
            Target value GNN should predict.
            "dvdt" (acceleration) OR "dx" (displacement)

        w_size
            Number of previous velocities to use in node representation.
            Only used for mode == "collisions".
            For mode == "crossings", w_size = 1.

        dt_undersample
            Undersample factor to apply to sheet model simulations.

        n_neighbors
            Number of neighbors to connect to each node.

        augment_t
            Add augmented data by fliping the time-series along the time-axis.
            Can be used in combination with augment_x.

        augment_x
            Add augmented data by fliping the box along the x-axis.
            Can be used in combination with augment_t.

        save_folder
            Folder where dataset of graphs and targets will be saved.
    """
    # load sheet model dataset parameters
    with open(Path(data_folder) / "info.yml", "r", encoding="utf-8") as f:
        opt = yaml.safe_load(f)

    if opt["boundary"] == "reflecting":
        raise NotImplementedError("Not implemented for reflecting boundaries")
    if not opt["track_sheets"]:
        raise NotImplementedError(
            "Not implemented for datasets generated with track_sheets=False"
        )
    if mode not in ["collisions", "crossings"]:
        raise ValueError(f"Invalid mode type: {mode}")

    # the following are needed to generate the graph/targets
    if "L" in opt:
        L = opt["L"]
    else:
        # by default older sheet model simulations were generated with L=1
        L = 1.0

    n_sheets = opt["n_sheets"]  # pylint: disable=possibly-unused-variable
    boundary = opt["boundary"]
    dt_simulator = opt["dt"]
    # async datasets don't have dt_undersample
    if "dt_undersample" in opt:
        dt_simulator *= opt["dt_undersample"]

    # make output dir
    save_folder = Path(save_folder)
    Path(save_folder / "graphs").mkdir(exist_ok=True, parents=True)
    Path(save_folder / "targets").mkdir(exist_ok=True, parents=True)

    # save original dataset + pre-process config parameters
    save_folder = str(save_folder)
    del opt, f
    args = dict(locals())

    with open(save_folder + "/info.yml", "w", encoding="utf-8") as f:
        yaml.dump(args, f)
        save_folder = Path(save_folder)
        del args

    get_graphs = partial(
        build_graphs, boundary=boundary, n_neighbors=n_neighbors, n_guards=0
    )

    @jax.jit
    def process_rollout(X, X_eq):
        if mode == "collisions":
            X, _, X_eq = crossings2collisions(X, jnp.zeros_like(X), X_eq, L)

        # shape - (1, #timesteps, #sheets)
        X = X[jnp.newaxis, ::dt_undersample]
        X_eq = X_eq[jnp.newaxis, ::dt_undersample]

        if augment_t:
            # flip along time-axis
            X = jnp.concatenate([X, jnp.flip(X, axis=-2)])
            X_eq = jnp.concatenate([X_eq, jnp.flip(X_eq, axis=-2)])

        if augment_x:
            # flip along x-axis
            X = jnp.concatenate([X, jnp.flip(L - X, axis=-1)])
            X_eq = jnp.concatenate([X_eq, jnp.flip(L - X_eq, axis=-1)])

        # finite difference velocity
        # shape - (1/2/4, #timesteps - 1, #sheets)
        V = dxdt(X=X, dt=dt_simulator * dt_undersample, boundary=boundary, L=L)
        X = X[:, 1:]
        X_eq = X_eq[:, 1:]

        # shape - (1/2/4, #timesteps - 1 - w_size - 1, w_size + 1, #sheets)
        X = jax.vmap(sliding_windows, in_axes=(0, None))(X, w_size + 1)
        V = jax.vmap(sliding_windows, in_axes=(0, None))(V, w_size + 1)
        X_eq = jax.vmap(sliding_windows, in_axes=(0, None))(X_eq, w_size + 1)

        # shape - (1/2/4, #timesteps - 1 - wsize - 1, 1, #sheets)
        if var_target == "dx":
            # target_t = x_t+1 - x_t
            targets = jnp.copy(V[:, :, -1]) * (dt_simulator * dt_undersample)
        elif var_target == "dvdt":
            # target_t = (v_t+1 - v_t) / dt
            targets = jnp.diff(V[:, :, -2:], axis=2) / (dt_simulator * dt_undersample)
        else:
            raise ValueError("Invalid var_target")

        # remove target timesteps from arrays
        # (1/2/4, #timesteps - 1 - w_size - 1, w_size, #sheets)
        X = X[:, :, :-1]
        V = V[:, :, :-1]
        X_eq = X_eq[:, :, :-1]

        # flatten first dim
        # (1/2/4 * (#timesteps - 1 - w_size - 1), w_size, #sheets)
        X = X.reshape(-1, *X.shape[2:])
        V = V.reshape(-1, *V.shape[2:])
        X_eq = X_eq.reshape(-1, *X_eq.shape[2:])
        # (1/2/4 * (#timesteps - 1 - w_size - 1), 1, #sheets)
        targets = targets.reshape(-1, *targets.shape[2:])

        # ensure that sheets are ordered by their position
        # required for proper graph construction (nearest neighbors connected)
        i_sorted = jnp.argsort(X[:, -1:], axis=-1)
        X = jnp.take_along_axis(X, i_sorted, axis=-1)
        V = jnp.take_along_axis(V, i_sorted, axis=-1)
        X_eq = jnp.take_along_axis(X_eq, i_sorted, axis=-1)
        targets = jnp.take_along_axis(targets, i_sorted, axis=-1)

        # full simulation in 1 graph
        graphs = get_graphs(X, V, X_eq)
        # targets are also flattened
        targets = targets.reshape(-1, 1)

        return graphs, targets

    # loop over rollout folders
    X_files = sorted(glob.glob(data_folder + "/x_[0-9]*.npy"))
    X_eq_files = sorted(glob.glob(data_folder + "/x_eq_*.npy"))

    if not X_eq_files:
        raise NotImplementedError("Pre-processing not possible without x_eq files")

    n_files = len(X_files)
    n_zeros = int(np.round(np.log10(n_files)))

    for i, (xf, xeqf) in tqdm(enumerate(zip(X_files, X_eq_files)), total=n_files):
        X = jnp.load(xf)
        X_eq = jnp.load(xeqf)

        graphs, targets = process_rollout(X, X_eq)

        with open(save_folder / f"graphs/{i:0{n_zeros}d}.pkl", "wb") as f:
            pickle.dump(graphs, f)

        jnp.save(save_folder / f"targets/{i:0{n_zeros}d}.npy", targets)
