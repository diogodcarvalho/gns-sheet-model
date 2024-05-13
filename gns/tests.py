import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from tqdm.auto import tqdm
from typing import Optional

from gns.utils import abs_d, abs_periodic_d, get_dx_eq, wasserstein_d
from gns.preprocess import init_XV_crossings, init_XV_collisions, crossings2collisions
from gns.simulator import GNS


def plot_trajectory_comparison(
    X_pred: np.ndarray,
    X_true: np.ndarray,
    boundary: str,
    dt: float,
    L: float = 1,
    save_file: Optional[str] = None,
    rasterized: bool = True,
    ms_pred: float = 3,
    ms_true: float = 15,
    normalize_dx_eq: bool = False,
):
    """
    Plot comparison between GNS and Ground Truth (Sheet Model) simulations.
    Includes trajectories and rollout error metrics.

    Args:
        X_pred - GNS trajectories
        X_true - Ground Truth trajectories
        boundary - boundary type
        dt - simulation time-step
        L - simulation box size
        save_file -  file name where to save plot
        rasterized - if True, rasterizes scatter plot
        ms_pred - size of GNS markers
        ms_true - size of Ground Truth markers
        normalize_dx_eq - if True, normalize distances to dx_eq
    """
    _, ax = plt.subplots(2, 1, figsize=(15, 7), gridspec_kw={"height_ratios": [2, 1]})

    t = np.arange(len(X_pred)) * dt
    dx_eq = get_dx_eq(X_pred.shape[-1], L)

    if normalize_dx_eq:
        X_p_norm = X_pred / dx_eq
        X_t_norm = X_true / dx_eq
    else:
        X_p_norm = X_pred / L
        X_t_norm = X_true / L

    ##### position
    colors = plt.get_cmap(plt.cm.tab10)(np.arange(X_pred.shape[1]) % 11)

    for i in range(X_pred.shape[1]):
        ax[0].plot(
            t, X_p_norm[:, i], ".", ms=ms_pred, color=colors[i], rasterized=rasterized
        )
        ax[0].plot(
            t[: len(X_true)],
            X_t_norm[:, i],
            ".",
            alpha=min(50.0 / len(X_pred), 0.1),
            ms=ms_true,
            color=colors[i],
            rasterized=rasterized,
        )

    if normalize_dx_eq:
        ax[0].set_ylabel(r"$x$ [$\delta$]")
        ax[0].set_ylim([0, X_pred.shape[-1]])
    else:
        ax[0].set_ylabel("$x$ [$L$]")
        ax[0].set_ylim([0, 1])

    legend_elements = [
        Line2D([], [], marker="o", ms=15, ls="", color="tab:blue", label="GNS"),
        Line2D(
            [],
            [],
            marker="o",
            ms=15,
            ls="",
            color="tab:blue",
            alpha=0.2,
            label="Ground Truth",
        ),
    ]

    ax[0].legend(handles=legend_elements, loc="upper right", framealpha=0.9)

    ##### MAE
    if boundary == "periodic":
        d = abs_periodic_d(X_pred[: len(X_true)], X_true, L)
    else:
        d = abs_d(X_pred[: len(X_true)], X_true)

    # average over sheets
    d = jnp.mean(d, axis=-1)

    print(f"Rollout MAE: {np.mean(d)/L:.4f} [L], {np.mean(d)/dx_eq:.4f} [dx_eq]")

    if normalize_dx_eq:
        d /= dx_eq
        ax[1].set_ylabel(r"Error [$\delta$]")
    else:
        d /= L
        ax[1].set_ylabel(r"Error [$L$]")

    ax[1].plot(t, d, linewidth=4, label="MAE")
    ax[1].set_yscale("log")
    ax[1].set_xlabel(r"$t$ [$\omega_p^{-1}$]")
    ax[1].get_yaxis().get_major_formatter().labelOnlyBase = False
    ax[1].get_yaxis().set_ticks(10.0 ** np.arange(-12, 12, 2))
    ax[1].set_ylim(np.min(d[1:]) / 2, np.max(d) * 2)

    ##### EMD
    d = [wasserstein_d(X_pred[j], X_true[j], boundary, L) for j in range(len(X_true))]
    d = np.array(d)

    print(f"Rollout EMD: {np.mean(d)/L:.4f} [L], {np.mean(d)/dx_eq:.4f} [dx_eq]")

    if normalize_dx_eq:
        d /= dx_eq
    else:
        d /= L

    ax[1].plot(t, d, ":", color="tab:red", linewidth=4, label="EMD")

    for i, a in enumerate(ax):
        a.set_xlim([t[0], t[-1]])
        if i < 1:
            a.set_xticklabels([])

    ax[1].legend(loc="lower right")

    if save_file:
        plt.tight_layout()
        plt.savefig(save_file)

    plt.show()


def rollout_error_collisions(
    X: np.ndarray, V: np.ndarray, X_eq: np.ndarray, sim: GNS, n_guards: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes rollout error metrics for trained GNS.
    Considers COLLISIONAL dynamics.

    Args:
        X, V, X_eq - arrays with dataset of simulations (#simulations, #timesteps, #sheets)
        sim - GNS to test
        n_guards - number of guards to use

    Returns:
        e_x - MAE (#simulations,)
        e_w - Wasserstein Distance (#simulations,)
    """
    e_x = []
    e_w = []

    X, V, X_eq = crossings2collisions(X, V, X_eq, L=sim.L)

    i_sorted = jnp.argsort(X[:, sim.w_size], axis=-1)
    i_sorted = i_sorted[:, jnp.newaxis]

    X = jnp.take_along_axis(X, i_sorted, axis=-1)
    V = jnp.take_along_axis(V, i_sorted, axis=-1)
    X_eq = jnp.take_along_axis(X_eq, i_sorted, axis=-1)

    for i in tqdm(range(len(X))):
        X0, V0, X_eq0 = init_XV_collisions(
            X[i],
            X_eq[i],
            w_size=sim.w_size,
            dt=sim.dt_train,
            boundary=sim.boundary,
            L=sim.L,
        )

        X_pred, _, _, _ = sim.pred_rollout(
            X0,
            V0,
            X_eq0,
            t_max=sim.dt_train * (len(X[i]) - sim.w_size - 1),
            n_guards=0 if sim.boundary == "periodic" else n_guards,
            track_sheets=False,
            verbose=False,
        )

        if sim.boundary == "periodic":
            d = abs_periodic_d(X_pred, X[i, sim.w_size :], sim.L)
        else:
            d = abs_d(X_pred, X[i, sim.w_size :])

        e_x.append(np.mean(d))
        e_w.append(
            np.mean(
                [
                    wasserstein_d(
                        X_pred[j], X[i, j + sim.w_size], boundary=sim.boundary, L=sim.L
                    )
                    for j in range(len(X_pred))
                ]
            )
        )

    return np.array(e_x), np.array(e_w)


def rollout_error_crossings(
    X: np.ndarray, V: np.ndarray, X_eq: np.ndarray, sim: GNS, n_guards: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes rollout error metrics for trained GNS.
    Considers CROSSING (NON-COLLISIONAL) dynamics.

    Args:
        X, V, X_eq - arrays with dataset of simulations (#simulations, #timesteps, #sheets)
        sim - GNS to test
        n_guards - number of guards to use

    Returns:
        e_x - MAE (#simulations,)
        e_w - Wasserstein Distance (#simulations,)
    """
    e_x = []
    e_w = []

    i_sorted = jnp.argsort(X[:, 1], axis=-1)
    i_sorted = i_sorted[:, jnp.newaxis]

    X = jnp.take_along_axis(X, i_sorted, axis=-1)
    V = jnp.take_along_axis(V, i_sorted, axis=-1)
    X_eq = jnp.take_along_axis(X_eq, i_sorted, axis=-1)

    for i in tqdm(range(len(X))):
        X0, V0, X_eq0 = init_XV_crossings(
            X[i], V[i], X_eq[i], dt=sim.dt_train, boundary=sim.boundary, L=sim.L
        )

        X_pred, _, _, _ = sim.pred_rollout(
            X0,
            V0,
            X_eq0,
            t_max=sim.dt_train * (len(X[i]) - 2),
            dt=sim.dt_train,
            n_guards=0 if sim.boundary == "periodic" else n_guards,
            track_sheets=True,
            verbose=False,
        )

        if sim.boundary == "periodic":
            d = abs_periodic_d(X_pred, X[i, 1:], sim.L)
        else:
            d = abs_d(X_pred, X[i, 1:])

        e_x.append(np.mean(d))
        e_w.append(
            np.mean(
                [
                    wasserstein_d(
                        X_pred[j], X[i, j + 1], boundary=sim.boundary, L=sim.L
                    )
                    for j in range(len(X_pred))
                ]
            )
        )

    return np.array(e_x), np.array(e_w)
