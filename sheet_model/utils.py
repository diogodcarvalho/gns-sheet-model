import numpy as np
import matplotlib.pyplot as plt

from typing import Union, Optional
from scipy.optimize import root
from scipy.special import wofz, gammaln
from scipy.constants import k as kb


def get_E(x: np.ndarray, v: np.ndarray, x_eq: np.ndarray) -> tuple[float, float]:
    """
    Computes the kinetic and potential energy of the system.
        e_kin = 1/2 * m * v^2
        e_pot = 1/2 * m * w_p * (x - x_eq)^2

    Note: In our units, m = 1 and w_p = 1

    Args:
        x, v, x_eq - arrays with system state (..., n_sheets)

    Returns:
        e_kin, e_pot - kinetic and potential energy (..., 1)
    """
    e_kin = np.sum(np.square(v), axis=-1) / 2.0
    e_pot = np.sum(np.square(x - x_eq), axis=-1) / 2.0

    return np.squeeze(e_kin), np.squeeze(e_pot)


def get_dx_eq(n_sheets: int, L: float = 1) -> float:
    """
    Computes the intersheet equilibrium distance.

    Args:
        n_sheets - number of sheets
        L - box size

    Returns:
        dx_eq - intersheet spacing
    """
    return L / n_sheets


def get_x_eq(n_sheets: int, L: float = 1) -> np.ndarray:
    """
    Computes the sheet equilibrium positions.

    Args:
        n_sheets - number of sheets
        L - box size

    Returns:
        x_eq - equilibrium positions
    """
    dx = get_dx_eq(n_sheets, L)
    x_eq = (np.arange(n_sheets) + 1 / 2.0) * dx
    return x_eq


def get_relaxation_time(lambda_d: float) -> float:
    """
    Computes estimate of system relaxation time as in Dawson (1962)
    https://pubs.aip.org/aip/pfl/article/5/4/445/803511/One-Dimensional-Plasma-Model

    Args:
        lambda_d - Debye length [dx_eq]

    Returns:
        t [w_p^-1]
    """
    return np.sqrt(2 * np.pi) * lambda_d


def get_roots_dispersion_relation(kvth: float) -> tuple[np.cdouble, np.cdouble]:
    """
    Finds numerically the roots of the plasma dispersion relation for a given k
        https://farside.ph.utexas.edu/teaching/plasma/Plasma/node111.html

    Args:
        kvth - angular wave number * debye length

    Returns:
        w_num - Numerical solution to the dispersion relation
                Equals None if the algorithm did not converge.

        w_approx - Approximate solution to the dispersion relation J. Jackson (1962)
                    https://iopscience.iop.org/article/10.1088/0368-3281/1/4/301/pdf
                   see equations (6.2) and (6.3)
    """

    def f(x):
        x_c = x[0] + 1j * x[1]
        z = 1j * np.sqrt(np.pi) * wofz(x_c)
        return 1 + kvth**2 + x_c * z

    # initial guess is approximate solution
    w_r = np.sqrt(1.0 + 3 * kvth**2)
    w_i = -np.sqrt(np.pi / 8) * np.exp(-1 / (2 * kvth**2) - 1.5) / (kvth**3)

    # zeta = w / (sqrt(2) * k * v_th) and v_th = lambda_D * omega_p
    zeta_0 = [w_r / (np.sqrt(2) * kvth), w_i / (np.sqrt(2) * kvth)]
    # find zeros of the dispersion relation
    zeta = root(
        lambda x: [np.real(f(x)), np.imag(f(x))],
        zeta_0,
        options={"maxfev": 100, "xtol": 1e-10},
    )

    if zeta.success:
        # numerical solution
        w_num = zeta.x * np.sqrt(2) * kvth
        w_num = w_num[0] + 1j * w_num[1]
    else:
        w_num = None

    # approx_solution
    w_approx = w_r + 1j * w_i

    return w_num, w_approx


def get_mode_info(
    mode: int, n_sheets: int, v_th: float, L: float = 1, num: bool = False
) -> dict:
    """
    Computes mode wavelength, phase velocity, damping rate, etc. for a given
    plasma thermal velocity.

    Args:
        mode - Mode number
        n_sheets - Number of sheets
        v_th - Thermal velocity [dx_eq w_p]
        L - Simulation box length
        num - If true solve numerically the dispersion relation.
              Otherwise use same approximation as Dawson (1962)

    Returns:
        info - Mode information, contains:
            m - mode number
            lambda - mode wavelength [dx_eq]
            k - mode wave number [dx_eq^-1]
            w - mode wave frequency (numeric if num=True) [w_p]
            gamma - mode damping factor (numeric if num=True) [w_p^-1]
            v_ph - phase velocity (numeric if num=True) [dx_eq w_p]
            w_approx - mode wave frequency using Dawson's approximation [w_p]
            gamma_approx -  mode damping factor using Dawson's approximation [w_p^-1]
            v_ph - phase velocity using Dawson's approximation [dx_eq w_p]
            n_sheets - number of sheets
            L - box length
    """
    dx_eq = get_dx_eq(n_sheets, L)
    lambda_ = 2 * L / mode / dx_eq  # [dx_eq]
    k = 2 * np.pi / lambda_  # [dx_eq^-1]

    if num:
        w_num, w_approx = get_roots_dispersion_relation(k * v_th)

        gamma_approx = np.imag(w_approx)
        w_approx = np.real(w_approx)

        if w_num is None:
            w = np.real(w_approx)
            gamma = np.imag(w_approx)
        else:
            w = np.real(w_num)
            gamma = np.imag(w_num)

        v_ph = w / k

    else:
        w = np.sqrt(1 + 3 * k**2 * v_th**2)  # [w_p]
        v_ph = w / k  # [dx_eq w_p]
        v_g = 3 * v_th**2 / v_ph  # [dx_eq w_p]
        dfdv_vph = (
            -1
            / np.sqrt(2 * np.pi)
            * v_ph
            / v_th**3
            * np.exp(-(v_ph**2) / (2 * v_th**2))
        )
        gamma = np.pi / 2 * w / k**2 * (1 - v_g / v_ph) * dfdv_vph  # [w_p]
        # gamma_jackson_12 = np.sqrt(np.pi/8) / k**3 / v_th**3 * np.exp(-1/(2*k**2 * v_th**2))
        # gamma_jackson_63 = np.sqrt(np.pi/8) / k**3 / v_th**3 * np.exp(-w**2/(2*k**2 * v_th**2))
        w_approx = w
        gamma_approx = gamma

    info = {
        "m": mode,
        "lambda": lambda_,
        "k": k,
        "w": w,
        "gamma": gamma,
        "v_ph": v_ph,
        "w_approx": w_approx,
        "gamma_approx": gamma_approx,
        "v_ph_approx": w_approx / k,
        "n_sheets": n_sheets,
        "L": L,
    }

    return info


def get_mode_initial_conditions(
    dx_eq_nmax: float, mode_info: dict, phase: float = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes initial sheet positions and velocities required to excite the mode
    (traveling wave).

    Args:
        dx_eq_nmax - max density perturbation
        mode_info - obtained with get_mode_info()
        phase - initial phase of wave [radians]

    Returns:
        x0, v0, x_eq0 - initial conditions for traveling wave
    """
    L = mode_info["L"]

    dx_eq = get_dx_eq(mode_info["n_sheets"], L)
    x_eq = get_x_eq(mode_info["n_sheets"], L)
    dx_eq_xmax = dx_eq_nmax / mode_info["k"]

    x0 = dx_eq_xmax * np.sin(mode_info["k"] / dx_eq * x_eq + phase) * dx_eq + x_eq
    v0 = -dx_eq_xmax * np.cos(mode_info["k"] / dx_eq * x_eq + phase) * dx_eq
    x_eq0 = np.copy(x_eq)

    # account for particles leaving through the boundaries
    i_r = x0 > L
    i_l = x0 < 0

    if np.any(i_r) and np.any(i_l):
        raise AssertionError(
            "Particles left through both boundaries! Chose smaller dx_eq_nmax"
        )

    elif np.any(i_r):
        x0[i_r] -= L
        x_eq0[i_r] -= L

        x0 = np.roll(x0, np.sum(i_r))
        v0 = np.roll(v0, np.sum(i_r))
        x_eq0 = np.roll(x_eq0, np.sum(i_r))

    elif np.any(i_l):
        x0[i_l] += L
        x_eq0[i_l] += L

        x0 = np.roll(x0, -np.sum(i_l))
        v0 = np.roll(v0, -np.sum(i_l))
        x_eq0 = np.roll(x_eq0, -np.sum(i_l))

    return x0, v0, x_eq0


def get_mode_amplitude(m: int, x: np.ndarray, x_eq: np.ndarray) -> float:
    """
    Computes mode amplitude.

    Args:
        m - mode number
        x, x_eq - sheet positions + equilibrium positions

    Returns:
        Am - mode amplitude (same units as x)
    """
    N = x.shape[-1]
    Am = np.sum(
        (x - x_eq) * np.sin(np.pi / N * m * (np.arange(N) + 1 / 2)),
        axis=-1,
    )
    Am *= 2 / N

    return Am


def get_mode_evolution(
    m: int, x: np.ndarray, x_eq: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes mode amplitude + amplitude rate of change.

    Args:
        m - mode number
        x, x_eq - sheet positions + equilibrium positions
        dt - simulation time step [w_p^-1]

    Returns:
        Am - mode amplitude (same units as x)
        dAm - mode amplitude rate of change
    """
    Am = get_mode_amplitude(m, x, x_eq)
    dAm = np.diff(Am, axis=0) / dt
    return Am[1:], dAm


def get_Efield(
    x: np.ndarray, nx: int, dx_eq: float, L: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute electric field inside the box.

    Args:
        x - sheet positions
        nx - number of grid points in which to compute the electric field
        dx_eq - intersheet equilibrium distance
        L - box size

    Returns:
        grid - points in which electric field as calculated [dx_eq]
        Ef - electric field [m_e e^-1 dx_eq w_p^2]
    """
    grid = np.linspace(0.0, L, nx)
    Ef = np.copy(grid)

    j = 0
    x_sorted = np.sort(x)

    for i, x_grid in enumerate(grid[1:]):
        if j < len(x):
            while x_sorted[j] <= x_grid:
                j += 1
                if j == len(x):
                    break

        Ef[i + 1] -= j * dx_eq

    Ef /= dx_eq

    return grid / dx_eq, Ef


def get_Efield_vectorized(
    x: np.ndarray, nx: int, dx_eq: float, L: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute electric field inside the box.

    Args:
        x - sheet positions
        nx - number of grid points in which to compute the electric field
        dx_eq - intersheet equilibrium distance
        L - box size

    Returns:
        grid - points in which electric field as calculated [dx_eq]
        Ef - electric field [m_e e^-1 dx_eq w_p^2]
    """
    grid = np.linspace(0.0, L, nx)  # (nx)
    dx = grid[1] - grid[0]  # grid resolution [L]

    # index of the grid point to the right of the sheet
    i_x = (x // dx).astype(np.int32) + 1

    # count number of sheets contributing to a given grid point
    counts = np.apply_along_axis(
        lambda x: np.histogram(x, bins=np.arange(0, nx + 1))[0],
        axis=1,
        arr=i_x,
    )

    # contribution from sheets
    Ef = -counts * dx_eq
    Ef = np.cumsum(Ef, axis=-1)

    # contribution from ions
    Ef += grid

    return grid / dx_eq, Ef / dx_eq


def get_kEfield(
    x: np.ndarray, nx: int, dx_eq: float, L: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes electric field amplitude spectrum

    Args:
        x - sheet positions
        nx - number of grid points in which to compute the electric field
        dx_eq - intersheet equilibrium distance
        L - box size

    Returns:
        k - wave number [dx_eq^-1]
        kEf - electric field spectrum amplitude [m_e e^-1 dx_eq^2 w_p^2]
    """
    grid, Ef = get_Efield(x, nx, dx_eq, L)
    # already in units of dx_eq
    dx_Efield = grid[1] - grid[0]
    # units of Ef * dx_eq
    kEf = np.abs(np.fft.fft(Ef)) / len(grid)

    # units of [dx_eq^-1]
    k = 2 * np.pi * np.fft.fftfreq(len(grid), dx_Efield)

    # remove duplicate frequencies
    k = k[: len(grid) // 2]
    kEf = kEf[: len(grid) // 2]

    return k, kEf


def get_fdist(
    x: np.ndarray,
    x_eq: np.ndarray,
    v: np.ndarray,
    x_range: tuple[float, float],
    v_range: tuple[float, float],
    bins: int = 40,
    normalize: bool = True,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray], tuple[float, float]]:
    """
    Computes distribution function f(x-x_eq, v).

    Args:
        x, x_eq, v - 1D arrays with system state
        xrange - x-x_eq range to consider for 2D histogram
        vrange - v range to consider for 2D histogram
        bins - number of bins per axis
        normalize - if False returns counts,if true returns density function
                    such that sum(f)*dx*dv = 1 (not sum(f) = 1)

    Returns:
        fdist - distribution function
        (x_edges, v_edges) - histogram edges
        (dx, dv) - bin sizes
    """
    fdist, x_edges, v_edges = np.histogram2d(
        (x - x_eq), v, bins=bins, range=[x_range, v_range]
    )
    dx = x_edges[1] - x_edges[0]
    dv = v_edges[1] - v_edges[0]
    if normalize:
        fdist /= len(x) * dx * dv
    return fdist, (x_edges, v_edges), (dx, dv)


def get_H(fdist: np.ndarray, dx: float, dv: float) -> float:
    """
    Compute H value according to classical H-theorem.
        H = integral_xv(f log(f) dx dv)
        https://en.wikipedia.org/wiki/H-theorem

    Args:
        fdist, dx, dv - obtained from get_fdist()

    Returns:
        H - H-value
    """
    H = np.sum(fdist * np.log(fdist, where=fdist > 0)) * dx * dv
    return H


def get_S(fdist: np.ndarray) -> float:
    """
    Computes combinatorial Boltzmann entropy.

    Args:
        fdist - obtained from get_fdist()

    Returns:
        S - entropy
    """
    N = np.sum(fdist)
    S = kb * (gammaln(N) - np.sum(gammaln(fdist[fdist > 0])))
    return S


def get_Sxv(fdist: np.ndarray) -> dict:
    """
    Computes combinatorial Boltzmann entropy with separate values for positions and
    velocities. Inspired on diagnostic from H. Hiang et. al. (2019)
        https://arxiv.org/pdf/1902.02733.pdf

    Args:
        fdist - obtained from get_fdist()

    Returns:
        S - dictionary containing:
            x - Entropy associated with positions
            v - "" with velocities
            total - total entropy
    """
    Nx = np.sum(fdist, axis=1)
    Sx = get_S(Nx)
    Sv = kb * (np.sum(gammaln(Nx[Nx > 0])) - np.sum(gammaln(fdist[fdist > 0])))
    S = get_S(fdist)
    return {"x": Sx, "v": Sv, "total": S}


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """
    Computes 1D moving average.

    Args:
        x - time series
        w - window size
    """
    return np.convolve(x, np.ones(w), "valid") / w


def get_max_dE(
    E: np.ndarray,
    dt: float,
    w_ma: float = 2 * np.pi,
    t_skip: float = 0,
    E0: float = None,
) -> float:
    """
    Computes maximum relative deviation from initial energy: (E-E0)/E0

    Args:
        E - energy over time steps
        dt - simulation step
        w_ma - moving window size to use (set to dt if you do not want to use it)
        t_skip - initial time interval to skip for moving average calculation
        E0 - initial energy to consider (if None uses E[0])

    Returns:
        dE - maximum relative deviation
    """
    if E0 is None:
        E0 = E[0]

    dE = np.abs(moving_average((E - E0) / E0, int(w_ma / dt)))
    dE = np.max(dE[int(t_skip / dt) :])

    return dE


def async2sync(
    T: np.ndarray,
    X: np.ndarray,
    V: np.ndarray,
    X_eq: np.ndarray,
    dt: float,
    t_max: float,
    L: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts the outputs of the asynchronous sheet model to synchronous time-series.

    Args:
        T, X, V, X_eq - arrays obtained from async sheet model simulation
        dt - desired time-step [w_p^-1]
        t_max - max time [w_p^-1]
        L - box size

    Returns:
        X_s, V_s, X_eq_s - synchronous time-series
    """

    def eom(x, v, x_eq, dt):
        xi = x - x_eq
        vf = v * np.cos(dt) - xi * np.sin(dt)
        xf = x + v * np.sin(dt) - xi * (1 - np.cos(dt))
        return xf, vf

    i = 1
    t_cross_last = T[0]
    t_cross_next = T[1]

    X_s = [X[0].copy()]
    V_s = [V[0].copy()]
    X_eq_s = [X_eq[0].copy()]

    for t in np.arange(dt, t_max + dt, dt):
        # find first crossing that happens after t
        while t_cross_next < t:
            t_cross_last = t_cross_next
            i += 1
            if i == len(T):
                t_cross_next = np.inf
            else:
                t_cross_next = T[i]

        # advance from t_cross_last -> t
        dt_s = t - t_cross_last
        x, v = eom(X[i - 1], V[i - 1], X_eq[i - 1], dt_s)
        X_s.append(x.copy())
        V_s.append(v.copy())
        X_eq_s.append(X_eq[i - 1].copy())

    X_s = np.array(X_s)
    V_s = np.array(V_s)
    X_eq_s = np.array(X_eq_s)

    # ensure sheets are inside the box
    n = X_s // L
    X_s -= n * L
    X_eq_s -= n * L

    return X_s, V_s, X_eq_s


def plot_X(
    X: np.ndarray,
    T: Union[float, np.ndarray],
    L: float = 1.0,
    figsize: tuple[float, float] = (15, 7),
    save_file: Optional[str] = None,
):
    """
    Plots sheet trajectories.

    Args:
        X - array with sheet trajectories
        T - simulation time step (dt for sync) or array (async)
        L - box size
        figsize - figure size
        save_file - file name where to save plot
    """
    _, ax = plt.subplots(1, 1, figsize=figsize)

    if not isinstance(T, np.ndarray):
        T = np.arange(X.shape[0]) * T

    colors = plt.get_cmap(plt.cm.tab10)(np.arange(X.shape[1]) % 11)

    for i in range(X.shape[1]):
        ax.plot(T, X[:, i] / L, ".", color=colors[i], rasterized=True)
    ax.set_ylabel(r"$x$ [$L$]")
    ax.set_xlabel(r"$t$ [$\omega_p^{-1}$]")
    ax.set_ylim([0, 1])
    ax.set_xlim([0, T[-1]])

    if save_file is not None:
        plt.tight_layout()
        plt.savefig(save_file, dpi=300)

    plt.show()


def plot_XE(
    X: np.ndarray,
    E: np.ndarray,
    T: Union[float, np.ndarray],
    L: float = 1,
    figsize: tuple[float, float] = (15, 7),
    save_file: Optional[str] = None,
):
    """
    Plots sheet trajectories and energy variation.

    Args:
        X - array with sheet trajectories
        T - simulation time step (dt for sync) or array (async)
        L - box size
        figsize - figure size
        save_file - file name where to save plot
    """
    _, ax = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": [2, 1]})

    if not isinstance(T, np.ndarray):
        T = np.arange(X.shape[0]) * T

    ##### position
    colors = plt.get_cmap(plt.cm.tab10)(np.arange(X.shape[1]) % 11)

    for i in range(X.shape[1]):
        ax[0].plot(T, X[:, i] / L, ".", color=colors[i])
    ax[0].set_ylabel("$x$ [$L$]")
    ax[0].set_ylim([0, 1])

    ##### energy
    ax[1].plot(T, (E - E[0]) / E[0])
    ax[1].set_ylabel(r"$\Delta\epsilon/\epsilon_0$")
    ax[1].set_xlabel(r"$t$ [$\omega_p^{-1}$]")
    ax[1].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    for i, a in enumerate(ax):
        a.set_xlim([T[0], T[-1]])

        if i < 1:
            a.set_xticklabels([])

    if save_file:
        plt.tight_layout()
        plt.savefig(save_file, dpi=300)

    plt.show()


def plot_XVE(
    X: np.ndarray,
    V: np.ndarray,
    E: np.ndarray,
    T: Union[float, np.ndarray],
    L: float = 1,
    figsize: tuple[int, int] = (15, 10),
    save_file: Optional[str] = None,
):
    """
    Plot sheet model simulation.

    Args:
        X, V, E - arrays with simulation output
        T - simulation time step (dt for sync) or array (async)
        L - box size
        figsize - figure size
        save_file - file name where to save plot
    """
    _, ax = plt.subplots(
        3, 1, figsize=figsize, gridspec_kw={"height_ratios": [2, 1, 1]}
    )

    dx = get_dx_eq(X.shape[-1], L)

    if not isinstance(T, np.ndarray):
        T = np.arange(X.shape[0]) * T

    ax[0].plot(T, X / dx, ".")
    ax[0].set_ylabel(r"$x$ [$\delta$]")
    ax[0].set_ylim(0, X.shape[-1])

    ax[1].plot(T, V / dx, ".")
    ax[1].set_ylabel(r"$v$ [$\delta\!\cdot\!\omega_p$]")

    ax[2].plot(T, (E - E[0]) / E[0])
    ax[2].set_ylabel(r"$\Delta\epsilon/\epsilon_0$")

    for i, a in enumerate(ax):
        a.set_xlim([T[0], T[-1]])
        if i < 2:
            a.set_xticklabels([])

    ax[2].set_xlabel(r"$t$ [$\omega_p^{-1}$]")

    if save_file is not None:
        plt.tight_layout()
        plt.savefig(save_file, dpi=300)

    plt.show()
