import numpy as np
import jax.numpy as jnp

from scipy.optimize import linear_sum_assignment


def get_dx_eq(n_sheets: int, L: float = 1.0) -> float:
    """
    Computes the intersheet equilibrium distance.

    Args:
        n_sheets - number of sheets
        L - box size

    Returns:
        dx_eq - intersheet spacing
    """
    return L / n_sheets


def get_x_eq(n_sheets: int, L: float = 1.0) -> jnp.array:
    """
    Computes the sheet equilibrium positions.

    Args:
        n_sheets - number of sheets
        L - box size

    Returns:
        x_eq - equilibrium positions
    """
    dx = get_dx_eq(n_sheets, L)
    x_eq = (jnp.arange(n_sheets) + 1 / 2.0) * dx

    return x_eq


def get_E(
    X: jnp.ndarray, V: jnp.ndarray, X_eq: jnp.ndarray
) -> tuple[jnp.array, jnp.array]:
    """
    Computes the kinetic and potential energy of the system.
        E_kin = 1/2 * m * v^2
        E_pot = 1/2 * m * w_p * (x - x_eq)^2

    Note: In our units, m = 1 and w_p = 1

    Args:
        X, V - time series data
        X_eq - equilibrium positions (time-series or single time-step)

    Returns:
        E_kin, E_pot - kinetic and potential energy (units depend on X,V units)
    """
    dX = abs_d(X, X_eq)
    E_kin = jnp.sum(jnp.square(V), axis=-1) / 2.0
    E_pot = jnp.sum(jnp.square(dX), axis=-1) / 2.0

    return E_kin, E_pot


def abs_d(x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the absolute distance through the box.

    Args:
        x1, x2 - position arrays

    Returns:
        d - absolute distance
    """
    return jnp.abs(x1 - x2)


def abs_periodic_d(x1: jnp.ndarray, x2: jnp.ndarray, L: float = 1.0) -> jnp.ndarray:
    """
    Computes the absolute distance considering a periodic box (always d <= L/2).

    Value  is given by the minimum between the distance throught the box OR
    through the walls.

    Args:
        x1, x2 - position arrays (same units a L)
        L - box length

    Returns:
        d - periodic absolute distance
    """
    d = abs_d(x1, x2)
    d = jnp.where(d < L / 2.0, d, L - d)
    return d


def periodic_d(x1: jnp.ndarray, x2: jnp.ndarray, L: float = 1.0) -> jnp.ndarray:
    """
    Computes relative distance considering a periodic box (always -L/2 <= d <= L/2).

    Args:
        x1, x2 - position arrays (same units a L)
        L - box length

    Returns:
        d - periodic distance
    """
    d = x1 - x2
    d = jnp.where(d > L / 2.0, d - L, d)
    d = jnp.where(d < -L / 2.0, d + L, d)
    return d


def wasserstein_d(
    x1: jnp.ndarray, x2: jnp.ndarray, boundary: str, L: float = 1.0
) -> np.ndarray:
    """
    Computes wasserstein distance (Earth Mover's Distance) for both reflecting
    and periodic boundaries.

    Args:
        x1, x2 - position arrays (same units a L)
        boundary - "periodic" or "reflecting"
        L - box length

    Returns:
        w_d - wasserstein distance
    """
    if boundary == "reflecting":
        d_matrix = abs_d(x1.reshape(1, -1), x2.reshape(-1, 1))

    elif boundary == "periodic":
        d_matrix = abs_periodic_d(x1.reshape(1, -1), x2.reshape(-1, 1), L)

    # NOT JITTABLE
    ix1, ix2 = linear_sum_assignment(d_matrix)
    w_d = np.mean(d_matrix[ix1, ix2])
    return w_d


def dxdt(X: jnp.ndarray, dt: float, boundary: str, L: float = 1.0) -> jnp.ndarray:
    """
    Computes the finite difference velocity v_t = (x_t - x_t-1)/dt.

    WARNING: For periodic boundaries, if the sheet traveled a distance larger
    than L/2 in a single time step, this calculation is incorrect.

    Args:
        X - time-series with shape (*, #time-steps, #sheets) (same units a L)
        dt - time-series time step [1/w_p]
        boundary - boundary condition
        L - box length

    Returns:
        V - finite difference velocity [a.u. w_p]
    """
    V = jnp.diff(X, axis=-2)

    if boundary == "periodic":
        V = jnp.where(V > L / 2.0, V - L, V)
        V = jnp.where(V < -L / 2.0, V + L, V)

    V /= dt
    return V


def get_stats(x, np_=np) -> dict:
    """
    Get stats for buffer

    Args:
        x - array over which to compute stats
        np_ - array type to return, i.e. numpy or jax.numpy

    Returns:
        stats - dict with statistics
    """
    return {
        "mean": np_.mean(x),
        "std": np_.std(x),
        "min": np_.min(x),
        "max": np_.max(x),
        "argmin": np_.argmin(x),
        "argmax": np_.argmax(x),
        "argmean": np_.argmin(np_.abs(np_.mean(x) - x)),
    }
