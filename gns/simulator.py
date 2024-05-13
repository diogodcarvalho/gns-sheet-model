import jax
import jraph
import pickle
import yaml

import haiku as hk
import jax.numpy as jnp

from functools import partial
from tqdm.auto import tqdm
from typing import Optional, Union
from pathlib import Path

from gns.model import SMGNN
from gns.graph import build_graphs
from gns.utils import get_dx_eq, get_E


class SMGNS(object):
    """
    Sheet Model Graph Network Simulator as described in Carvalho et al. [1].

    Simulator is inspired on the Graph Network Simulator framework proposed by
    Sanchez-Gonzalez et al. in "Learning to Simulate Complex Physics with Graph
    Networks" [2].

    [1] https://arxiv.org/abs/2310.17646
    [2] https://arxiv.org/abs/2002.09405

    Args:
        model_folder
            Path to folder containing trained model.
            Should contain model_cfg.yml, params_best.pkl and train_data.yml.

        boundary
            Boundary conditions to use.
            Does not need to match trained boundary conditions.

        verbose
            Print simulator training information.

        L
            Simulation box size to use.
            Does not need to match box length during training.

    Params:
        model_folder, boundary, L
            Same as in args.

        net_fn_params
            GNN architecture parameters loaded from model_folder/model_cfg.yml

        net_params
            GNN weights loaded from model_folder/params_best.pkl

        stats
            Mean/std used to scale training targets.
            Will be None if scaling was not used.

        train_data_info
            Training data information loaded model_folder/train_data.yml

        dt_train
            Simulation time-step used during training.
            Equals train_data_info['dt'] * train_data_info['dt_undersample']
    """

    def __init__(
        self,
        model_folder: str,
        boundary: str = None,
        verbose: bool = True,
        L: float = 1,
    ):
        self.model_folder = Path(model_folder)

        with open(self.model_folder / "model_cfg.yml", "r", encoding="utf-8") as f:
            self.net_fn_params = yaml.safe_load(f)

        with open(self.model_folder / "params_best.pkl", "rb") as f:
            self.net_params = pickle.load(f)

        if Path(self.model_folder / "stats.pkl").exists():
            with open(self.model_folder / "stats.pkl", "rb") as f:
                self.stats = pickle.load(f)
        else:
            self.stats = None

        with open(self.model_folder / "train_data.yml", "r", encoding="utf-8") as f:
            self.train_data_info = yaml.safe_load(f)
            # compute dt used to train gnn model
            self.dt_train = (
                self.train_data_info["dt_undersample"]
                * self.train_data_info["dt_simulator"]
            )

        # box size
        self.L = L

        # override boundary if user defined
        if boundary is not None:
            self.boundary = boundary
        else:
            self.boundary = self.train_data_info["boundary"]

        # initialize gnn model
        net = SMGNN(**self.net_fn_params)
        self._gnn = hk.without_apply_rng(hk.transform(net))

        if verbose:
            print("Simulator Info")
            print("   GNN model:", self.model_folder)
            print("   boundary:", self.boundary)
            print("   L:", self.L)
            print("   dt_train:", self.dt_train)

            print("   Training Dataset")
            for k, v in self.train_data_info.items():
                print(f"      {k}: {v}")

    def _build_graph(
        self, x: jnp.ndarray, v: jnp.ndarray, x_eq: jnp.ndarray, n_guards: int
    ) -> jraph.GraphsTuple:
        """
        Generates graph representation of the system.
        """
        return build_graphs(
            x,
            v,
            x_eq,
            n_guards=n_guards,
            boundary=self.boundary,
            n_neighbors=self.train_data_info["n_neighbors"],
            L=self.L,
        )

    def _handle_boundary(
        self,
        x: jnp.ndarray,
        v: jnp.ndarray,
        x_eq: jnp.ndarray,
        labels,
        track_sheets: bool,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Resolves boundary crossings by re-injecting sheets that left the box
        and updating the equilibrium positions of all sheets.
        """
        # check which sheets crossed boundary
        exit_left = x < 0
        exit_right = x >= self.L

        if self.boundary == "reflecting":
            x = jnp.where(exit_left, -x, x)
            x = jnp.where(exit_right, 2 * self.L - x, x)
            v = jnp.where(jnp.logical_or(exit_left, exit_right), -v, v)

        elif self.boundary == "periodic":
            # number of boundary crossings
            # accounts for sheets who go "over" the box multiple times
            # in a single timestep
            aux = x // self.L
            n_left = jnp.sum(jnp.where(aux < 0, -aux, 0), dtype=int)
            n_right = jnp.sum(jnp.where(aux > 0, aux, 0), dtype=int)
            # enforce periodicity
            x = jnp.where(exit_left, x - aux * self.L, x)
            x = jnp.where(exit_right, x - aux * self.L, x)
            # update equilibrium positions
            # considering n crossings on both sides
            x_eq += (n_left - n_right) * get_dx_eq(x.shape[-1], self.L)

            if not track_sheets:
                labels = jnp.roll(labels, n_right - n_left)

        return x, v, x_eq, labels

    def _update_xv(
        self,
        x: jnp.ndarray,
        v: jnp.ndarray,
        x_eq: jnp.ndarray,
        n_guards: int,
        dt: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Predicts the accelerations using the GNN + applies the ODE integrator.
        """
        graph = self._build_graph(x, v, x_eq, n_guards)
        preds = jnp.squeeze(self._gnn.apply(self.net_params, graph))

        if self.stats is not None:
            preds *= self.stats["y_std"]

        if self.boundary == "reflecting":
            # remove guards
            preds = preds[n_guards:-n_guards]

        if self.train_data_info["var_target"] == "dx":
            v = preds / dt
            x += preds

        elif self.train_data_info["var_target"] == "dvdt":
            v += preds * dt
            x += v * dt

        return x, v

    @partial(jax.jit, static_argnums=(0, 4))
    def _sort(
        self, x: jnp.ndarray, v: jnp.ndarray, labels: jnp.ndarray, track_sheets: bool
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Sorts arrays according to the sheets current positions.
        """
        i_sorted = jnp.argsort(x)
        x = x[i_sorted]
        v = v[i_sorted]
        if track_sheets:
            labels = labels[i_sorted]

        return x, v, labels, i_sorted

    @partial(jax.jit, static_argnums=(0,))
    def _label_sort(
        self, x: jnp.ndarray, v: jnp.ndarray, x_eq: jnp.ndarray, labels: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Sorts arrays according to the sheets labels.
        """
        i_sorted = jnp.argsort(labels)
        return x[i_sorted], v[i_sorted], x_eq[i_sorted]

    @partial(jax.jit, static_argnums=(0, 5, 6, 7))
    def pred(
        self,
        x: jnp.ndarray,
        v: jnp.ndarray,
        x_eq: jnp.ndarray,
        labels: jnp.ndarray,
        n_guards: int,
        track_sheets: bool,
        dt: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Performs a single step update of the GNS.
        """
        x, v = self._update_xv(x, v, x_eq, n_guards, dt)
        x, v, x_eq, labels = self._handle_boundary(x, v, x_eq, labels, track_sheets)

        return x, v, x_eq, labels

    def pred_rollout(
        self,
        x_0: jnp.ndarray,
        v_0: jnp.ndarray,
        x_eq_0: jnp.ndarray,
        t_max: float,
        dt: Optional[float] = None,
        dt_undersample: int = 1,
        n_guards: int = 0,
        track_sheets: bool = True,
        verbose: bool = True,
        np_=jnp,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Performs a rollout prediction using the GNS.

        Args:
            x_0, v_0, x_eq_0
                Initial conditions. Ensure that sheets are ordered!

            t_max
                Rollout duration.

            dt
                Simulation step to use, if not set use the training value.
                For correct functioning of the GNS this value should not be provided.

            dt_undersample
                Undersampling factor to apply to output data.

            n_guards
                Number of guard sheets to use in graph representation.
                Only used for reflecting boundaries.

            track_sheets
                If True, output arrays represent individual sheets trajectories.
                If False, output arrays represent collisional-like trajectories.

            verbose
                If True, print progress bar.

            np_
                Library to use for output array (i.e. numpy or jax.numpy)
                Using numpy is faster but simulation is not differentiable.
                Using jax.numpy is slower but simulation is differentiable.

        Returns:
            X_roll, V_roll, X_eq_roll, E_roll
                Rollout arrays with shape (timesteps, #sheets) for trajectories
                and (timesteps,) for energy.
        """
        x = x_0.copy()
        v = v_0.copy()
        x_eq = x_eq_0.copy()
        labels = jnp.arange(x_0.shape[-1])

        X_roll = [x_0]
        V_roll = [v_0]
        X_eq_roll = [x_eq_0]
        E_roll = [sum(get_E(x_0, v_0, x_eq_0))]

        if dt is None:
            dt = self.dt_train

        for i in tqdm(range(int(t_max / dt)), disable=not verbose):
            x, v, x_eq, labels = self.pred(
                x, v, x_eq, labels, n_guards, track_sheets, dt
            )

            x, v, labels, _ = self._sort(x, v, labels, track_sheets)

            E = sum(jax.jit(get_E)(x, v, x_eq))

            if (i + 1) % dt_undersample == 0:
                x_aux, v_aux, x_eq_aux = self._label_sort(x, v, x_eq, labels)

                X_roll.append(x_aux)
                V_roll.append(v_aux)
                X_eq_roll.append(x_eq_aux)
                E_roll.append(E)

        return (
            np_.array(X_roll),
            np_.array(V_roll),
            np_.array(X_eq_roll),
            np_.array(E_roll),
        )

    def __hash__(self):
        return hash((self.model_folder, self.boundary, self.L))

    def __eq__(self, other):
        """
        Defined to avoid unecessary recompilation of class methods
        when using self as static.
        https://jax.readthedocs.io/en/latest/faq.html#how-to-use-jit-with-methods
        """
        return (
            type(self) == type(other)
            and self.model_folder == other.model_folder
            and self.boundary == other.boundary
            and self.L == other.L
        )


class SMGNS_MW(SMGNS):
    """
    Sheet Model Graph Network Simulator version which receives moving window inputs.
    This is the version used for the collisions example in Appendix.

    Args:
        model_folder
            Path to folder containing trained model.
            Should contain model_cfg.yml, params_best.pkl and train_data.yml.

        boundary
            Boundary conditions to use.
            Do not need to match trained boundary conditions.

        verbose
            Print simulator training information.

        L
            Simulation box size to use.

    Params:
        model_folder, boundary, L
            Same as in args.

        net_fn_params
            GNN architecture parameters loaded from model_folder/model_cfg.yml

        net_params
            GNN weights loaded from model_folder/params_best.pkl

        stats
            Mean/std used to scale training targets.
            Will be None if scalling was not used.

        train_data_info
            Training data information loaded from model_folder/train_data.yml

        dt_train
            Simulation time-step used during training.
            Equals train_data_info['dt'] * train_data_info['dt_undersample']

        w_size
            Number of past time-steps velocities to use as input.
            Loaded from train_data_info.
    """

    def __init__(
        self,
        model_folder: str,
        boundary: str = None,
        verbose: bool = True,
        L: float = 1,
    ):
        super().__init__(
            model_folder=model_folder, boundary=boundary, L=L, verbose=verbose
        )

        self.w_size = self.train_data_info["w_size"]

    def _update_xv_moving_window(
        self,
        X: jnp.ndarray,
        V: jnp.ndarray,
        X_eq: jnp.ndarray,
        n_guards: int,
        dt: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Predicts the accelerations using the GNN + applies the ODE integrator.
        """
        graph = self._build_graph(X, V, X_eq, n_guards)
        preds = jnp.squeeze(self._gnn.apply(self.net_params, graph))
        preds *= self.stats["y_std"]

        if self.boundary == "reflecting":
            # remove guards
            preds = preds[n_guards:-n_guards]

        if self.train_data_info["var_target"] == "dx":
            x = X[-1] + preds
            v = preds / dt

        elif self.train_data_info["var_target"] == "dvdt":
            v = V[-1] + preds * dt
            x = X[-1] + v * dt

        return x, v

    @partial(jax.jit, static_argnums=(0, 8))
    def _sort_moving_window(
        self,
        x: jnp.ndarray,
        v: jnp.ndarray,
        x_eq: jnp.ndarray,
        X_mw: jnp.ndarray,
        V_mw: jnp.ndarray,
        X_eq_mw: jnp.ndarray,
        labels,
        track_sheets: bool,
    ):
        """
        Sorts moving window arrays according to the last time-step sheet positions.
        """

        def update_sliding_window(a, A_mw):
            return jnp.concatenate([A_mw[1:], a[jnp.newaxis, ...]], axis=0)

        x, v, labels, i_sorted = self._sort(x, v, labels, track_sheets)

        i_sorted = i_sorted[jnp.newaxis]

        X_mw = jnp.take_along_axis(X_mw, i_sorted, axis=-1)
        V_mw = jnp.take_along_axis(V_mw, i_sorted, axis=-1)
        X_eq_mw = jnp.take_along_axis(X_eq_mw, i_sorted, axis=-1)

        X_mw = update_sliding_window(x, X_mw)
        V_mw = update_sliding_window(v, V_mw)
        X_eq_mw = update_sliding_window(x_eq, X_eq_mw)

        return x, v, X_mw, V_mw, X_eq_mw, labels

    @partial(jax.jit, static_argnums=(0, 5, 6, 7))
    def pred_moving_window(
        self,
        X: jnp.ndarray,
        V: jnp.ndarray,
        X_eq: jnp.ndarray,
        labels: jnp.ndarray,
        n_guards: int,
        track_sheets: bool,
        dt: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Performs a single step update of the GNS.
        """
        x, v = self._update_xv_moving_window(X, V, X_eq, n_guards, dt)
        x, v, x_eq, labels = self._handle_boundary(x, v, X_eq[-1], labels, track_sheets)
        return x, v, x_eq, labels

    def pred_rollout(
        self,
        x_0: jnp.ndarray,
        v_0: jnp.ndarray,
        x_eq_0: jnp.ndarray,
        t_max: float,
        dt: Optional[float] = None,
        dt_undersample: int = 1,
        n_guards: int = 0,
        track_sheets: bool = True,
        verbose: bool = True,
        np_=jnp,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Performs a rollout prediction using the GNS.

        Args:
            x_0, v_0, x_eq_0
                Initial conditions. Ensure that sheets are ordered!
                Arrays of shape (#timesteps, n_sheets)

            t_max
                Rollout duration.

            dt
                Simulation step to use, if not set use the training value.
                For correct functioning of the GNS this value should not be provided.

            dt_undersample
                Undersampling factor to apply to output data.

            n_guards
                Number of guard sheets to use in graph representation.
                Only used for reflecting boundaries.

            track_sheets
                If True, output arrays represent individual sheets trajectories.
                If False, output arrays represent collisional-like trajectories.

            verbose
                If True, print progress bar.

            np_
                Library to use for output array (i.e. numpy or jax.numpy)
                Using numpy is faster but simulation is not differentiable.
                Using jax.numpy is slower but simulation is differentiable.

        Returns:
            X_roll, V_roll, X_eq_roll, E_roll
                Rollout arrays with shape (timesteps, #sheets) for trajectories
                and (timesteps,) for energy.
        """
        x = x_0.copy()
        v = v_0.copy()
        x_eq = x_eq_0.copy()
        labels = jnp.arange(x_0.shape[-1])

        if dt is None:
            dt = self.dt_train

        X_roll = [x_0[-1]]
        V_roll = [v_0[-1]]
        X_eq_roll = [x_eq_0[-1]]
        E_roll = [sum(get_E(x_0[-1], v_0[-1], x_eq_0[-1]))]

        X_mw = x_0.copy()
        V_mw = v_0.copy()
        X_eq_mw = x_eq_0.copy()

        for i in tqdm(range(int(t_max / dt)), disable=not verbose):
            x, v, x_eq, labels = self.pred_moving_window(
                X_mw, V_mw, X_eq_mw, labels, n_guards, track_sheets, dt
            )

            x, v, X_mw, V_mw, X_eq_mw, labels = self._sort_moving_window(
                x, v, x_eq, X_mw, V_mw, X_eq_mw, labels, track_sheets
            )

            E = sum(jax.jit(get_E)(x, v, x_eq))

            if (i + 1) % dt_undersample == 0:
                x_aux, v_aux, x_eq_aux = self._label_sort(x, v, x_eq, labels)

                X_roll.append(x_aux)
                V_roll.append(v_aux)
                X_eq_roll.append(x_eq_aux)
                E_roll.append(E)

        return (
            np_.array(X_roll),
            np_.array(V_roll),
            np_.array(X_eq_roll),
            np_.array(E_roll),
        )


GNS = Union[SMGNS, SMGNS_MW]
