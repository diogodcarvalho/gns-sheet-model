import numpy as np

from typing import Optional
from tqdm.auto import tqdm

from sheet_model.utils import get_dx_eq, get_x_eq, get_E


class SyncSheetModel(object):
    """
    Implementation of Dawson's single-specie synchronous sheet model algorithm [1]
    and slight modifications described in [2].

    [1] J. Dawson, "The Electrostatic Sheet Model for a Plasma and its Modification
        to Finite-Size Particles", Methods in Computational Physics vol. 9
        Academic Press Inc., U.S. (1970)

    [2] D. Carvalho et al., https://arxiv.org/abs/2310.17646

    Args:
        n_sheets - number of particles
        L - box size
        boundary - 'reflecting' or 'periodic'
        n_guards - number of guard sheets
        dt - simulation step [1/w_p]
        track_sheets - use 'True' for crossing dynamics, False for collisional dynamics.
        n_it_crossings - number of iterations to use for crossing time estimation
        new_dx_max - if True, overwrites original dx_max criteria from [1] and uses the one described in [2].

    Params:
        n_sheets
        L
        boundary
        n_guards
        dt
        track_sheets
        n_it_crossings
        new_dx_max
        dx_eq - intersheet spacing in equilibrium [L]
    """

    def __init__(
        self,
        n_sheets: int,
        L: float,
        boundary: str,
        n_guards: int,
        dt: float,
        track_sheets: bool,
        n_it_crossings: int,
        new_dx_max: bool = True,
    ):
        self.n_sheets = n_sheets
        self.L = L
        self.boundary = boundary

        if n_guards < 1:
            raise AssertionError("n_guards must be >= 1")
        elif n_guards > n_sheets:
            raise NotImplementedError("Can't have more guards than sheets")
        else:
            self.n_guards = n_guards

        self.dt = dt
        self.track_sheets = track_sheets
        self.n_it_crossings = n_it_crossings
        self.new_dx_max = new_dx_max

        self.dx_eq = get_dx_eq(self.n_sheets, self.L)

    def _check_inputs(
        self, x_0: np.ndarray, v_0: np.ndarray, x_eq_0: Optional[np.ndarray]
    ):
        """
        Check if initial conditions are valid.
        """
        if x_0.shape[-1] != self.n_sheets:
            raise ValueError("Number of sheets does not match x_0 array dimensions")

        if not np.all(np.logical_and(x_0 > 0, x_0 < self.L)):
            raise ValueError(
                "Initial positions contain values outside the simulation box"
            )

        if not np.all(x_0[1:] - x_0[:-1] > 0):
            raise ValueError("Initial positions are not sorted")

        if x_eq_0 is not None:
            if not np.all(x_eq_0[1:] - x_eq_0[:-1] > 0):
                raise ValueError("Initial equilibrium positions are not sorted")
            if len(x_eq_0) != len(x_0):
                raise ValueError("Length of x_0 != x_eq_0")

        if len(v_0) != len(x_0):
            raise ValueError("Length of x_0 != v_0")

    def _check_guards(self, x: np.ndarray):
        """
        Check if any sheet crossed all guards or all guards entered the simulation box.
        """
        n_g = self.n_guards
        # check if any sheet crossed all guards
        if np.min(x[:n_g]) > np.min(x[n_g:-n_g]) or np.max(x[-n_g:]) < np.max(
            x[n_g:-n_g]
        ):
            raise AssertionError(
                "Sheet crossed all guards - Increase n_guards or decrease dt"
            )
        # in periodic check if all guards on one of the sides entered the box
        # can happen if more particles than n_guards cross the same boundary
        if self.boundary == "periodic":
            if np.all(x[:n_g] >= 0) or np.all(x[-n_g:] < self.L):
                raise AssertionError(
                    "All guards on one side entered the box - Increase n_guards or decrease dt"
                )

    def _add_guards(
        self, x: np.ndarray, v: np.ndarray, x_eq: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Append guard sheets to x, v, x_eq buffers.
        """
        L = self.L
        n_g = self.n_guards

        if self.boundary == "periodic":
            x_l = x[-n_g:] - L
            v_l = v[-n_g:]
            x_eq_l = x_eq[-n_g:] - L

            x_r = x[:n_g] + L
            v_r = v[:n_g]
            x_eq_r = x_eq[:n_g] + L

        elif self.boundary == "reflecting":
            x_l = -np.flip(x[:n_g])
            v_l = -np.flip(v[:n_g])
            x_eq_l = -np.flip(x_eq[:n_g])

            x_r = L + (L - np.flip(x[-n_g:]))
            v_r = -np.flip(v[-n_g:])
            x_eq_r = L + (L - np.flip(x_eq[-n_g:]))

        x_all = np.concatenate([x_l, x, x_r])
        v_all = np.concatenate([v_l, v, v_r])
        x_eq_all = np.concatenate([x_eq_l, x_eq, x_eq_r])

        return x_all, v_all, x_eq_all

    def _remove_guards(
        self, x: np.ndarray, v: np.ndarray, x_eq: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Remove guard sheets from x, v, x_eq buffers.
        """
        return (
            x[self.n_guards : -self.n_guards],
            v[self.n_guards : -self.n_guards],
            x_eq[self.n_guards : -self.n_guards],
        )

    def _initialize_labels(self) -> np.ndarray:
        """
        Initialize label buffer.
        Guard sheets labels are related to associated 'real' sheet.
        """
        n_s = self.n_sheets
        n_g = self.n_guards

        if self.boundary == "reflecting":
            # -n_g, ..., -1, 1, ..., N, N+N, ..., N+N-n_g
            labels = np.concatenate(
                [np.arange(-n_g, 0, 1), np.arange(n_s) + 1, 2 * n_s - np.arange(n_g)]
            )

        elif self.boundary == "periodic":
            # -N+n_g, ..., -N, 1, ..., N, N+1, ..., N+n_ng
            labels = np.concatenate(
                [
                    -n_s - 1 + np.arange(n_g, 0, -1),
                    np.arange(n_s) + 1,
                    n_s + 1 + np.arange(n_g),
                ]
            )

        return labels

    @staticmethod
    def _advance_eom(
        x_i: np.ndarray, v_i: np.ndarray, x_eq_i: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Advances x,v from t -> t+dt according to analytical equation of motion.
        """
        xi_i = x_i - x_eq_i
        v_f = v_i * np.cos(dt) - xi_i * np.sin(dt)
        x_f = x_i + v_i * np.sin(dt) - xi_i * (1 - np.cos(dt))

        return x_f, v_f

    def _get_crossing_time(
        self,
        x_i: np.ndarray,
        v_i: np.ndarray,
        x_eq_i: np.ndarray,
        x_f: np.ndarray,
    ) -> float:
        """
        Estimates crossing time between 2 sheets using iterative method.
        Auxiliary to _handle_crossings.
        """
        t_c = self.dt
        x_c = x_f.copy()
        dx_i = x_i[1] - x_i[0]

        for _ in range(self.n_it_crossings):
            dx_c = x_c[1] - x_c[0]
            t_c *= dx_i / (dx_i - dx_c)
            x_c, _ = self._advance_eom(x_i, v_i, x_eq_i, t_c)

        return t_c

    def _correct_trajectory(
        self,
        x_i: np.float64,
        v_i: np.float64,
        x_eq_i: np.float64,
        t_cross: list[float],
        r_cross: list[int],
    ) -> tuple[np.float64, np.float64, np.float64]:
        """
        Corrects individual sheet trajectory based on estimated crossings.
        Auxiliary to _handle_crossings.
        """
        # sort crossings by time in which they occur
        i_cross = np.argsort(t_cross)
        t_cross = np.array(t_cross)[i_cross]
        r_cross = np.array(r_cross)[i_cross]

        t = 0
        t_c = 0
        x_c = x_i.copy()
        v_c = v_i.copy()
        x_eq_c = x_eq_i.copy()

        for t_c, r_c in zip(t_cross, r_cross):
            x_c, v_c = self._advance_eom(x_c, v_c, x_eq_c, t_c - t)
            x_eq_c += r_c * self.dx_eq
            t = t_c

        x_f, v_f = self._advance_eom(x_c, v_c, x_eq_c, self.dt - t_c)
        x_eq_f = x_eq_c

        return x_f, v_f, x_eq_f

    def _handle_crossings(
        self,
        x_i: np.ndarray,
        v_i: np.ndarray,
        x_f: np.ndarray,
        v_f: np.ndarray,
        x_eq_i: np.ndarray,
        labels_i: np.ndarray,
        v_neg_max: float,
        xi_pos_max: Optional[float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Detects crossings between t -> t+dt after equation of motion update.
        Corrects sheet trajectories based on estimated crossing times.
        """
        # stores crossing times for each sheet
        TCROSS: list[list[float]] = [[] for _ in range(len(x_f))]
        # stores rank of crossings (+1 particles from left, -1 sheet from right)
        RCROSS: list[list[int]] = [[] for _ in range(len(x_f))]
        # max distance sheet j > i could be and still cross in this timestep
        DX_MAX = (v_neg_max + v_i) * self.dt
        if self.new_dx_max:
            DX_MAX += (xi_pos_max - x_i + x_eq_i) * self.dt**2 / 2  # (1 - self.dt)

        # estimate crossings times (from left to right)
        for i in range(len(x_f) - 1):
            for j in range(i + 1, len(x_f)):
                # if crossing with j-th sheet not possible
                # move on to the next i-th sheet
                if x_i[j] - x_i[i] > DX_MAX[i]:
                    break
                # check if crossing occured
                if x_f[j] < x_f[i]:
                    # estimate crossing time
                    tc = self._get_crossing_time(
                        x_i[[i, j]],
                        v_i[[i, j]],
                        x_eq_i[[i, j]],
                        x_f[[i, j]],
                    )

                    TCROSS[i].append(tc)
                    TCROSS[j].append(tc)
                    RCROSS[i].append(1)
                    RCROSS[j].append(-1)

        # rank change for each sheet
        NCROSS = np.array([np.sum(r) for r in RCROSS], dtype=np.int64)
        assert np.sum(NCROSS) == 0, "#left crossings != #right crossings"

        i_sortcross = np.arange(len(x_f))
        i_sortcross += NCROSS

        # all sheet must have a unique array index
        assert len(i_sortcross) == len(
            np.unique(i_sortcross)
        ), "found non-unique sheet indices"

        x_eq_f = x_eq_i.copy()
        labels_f = labels_i.copy()

        # correct trajectories of sheets who underwent crossings
        for i in range(len(x_f)):
            if TCROSS[i]:
                x_f[i], v_f[i], x_eq_f[i] = self._correct_trajectory(
                    x_i[i], v_i[i], x_eq_i[i], TCROSS[i], RCROSS[i]
                )

        # re-arrange buffers to account for crossings
        # this is not a full sort!
        i_sort = np.argsort(i_sortcross)
        x_f = x_f[i_sort]
        v_f = v_f[i_sort]
        x_eq_f = x_eq_f[i_sort]

        if self.track_sheets:
            labels_f = labels_f[i_sort]

        return x_f, v_f, x_eq_f, labels_f

    def _reflecting_boundary(
        self, x: np.ndarray, v: np.ndarray, x_eq: np.ndarray, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Updates reflecting boundary guards.
        Auxiliary to _handle_boundary.
        """
        L = self.L
        n_g = self.n_guards

        x_new = np.copy(x)
        v_new = np.copy(v)
        labels_new = np.copy(labels)

        # update all guards
        x_new[:n_g] = -np.flip(x[n_g : 2 * n_g])
        v_new[:n_g] = -np.flip(v[n_g : 2 * n_g])
        x_new[-n_g:] = L + (L - np.flip(x[-2 * n_g : -n_g]))
        v_new[-n_g:] = -np.flip(v[-2 * n_g : -n_g])

        # switch labels between guards and real sheets that crossed
        for j in range(n_g):
            # guard entered on left
            if labels_new[n_g + j] < 1:
                labels_new[n_g + j] *= -1
            # guard entered on right
            if labels_new[-2 * n_g + j] > self.n_sheets:
                labels_new[-2 * n_g + j] -= self.n_sheets

        labels_new[:n_g] = -np.flip(labels_new[n_g : 2 * n_g])
        labels_new[-n_g:] = np.flip(labels_new[-2 * n_g : -n_g]) + self.n_sheets

        return x_new, v_new, x_eq, labels_new

    def _periodic_boundary(
        self, x: np.ndarray, v: np.ndarray, x_eq: np.ndarray, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Updates periodic boundary guards.
        Auxiliary to _handle_boundary.
        """
        L = self.L
        n_g = self.n_guards

        x_new = np.copy(x)
        v_new = np.copy(v)
        x_eq_new = np.copy(x_eq)
        labels_new = np.copy(labels)

        # check if rolling neeeded to keep sheets inside the box
        # in the center of the arrays
        i_roll = -np.argmax(x[: 2 * n_g] > 0) + n_g

        if i_roll != 0:
            x_new = np.roll(x_new, i_roll)
            v_new = np.roll(v_new, i_roll)
            x_eq_new -= i_roll * self.dx_eq

        if self.track_sheets:
            if i_roll != 0:
                labels_new = np.roll(labels_new, i_roll)
            # check for guards inside the box
            for j in range(n_g):
                # guard entered on the left
                if labels_new[n_g + j] < 1:
                    labels_new[n_g + j] *= -1
                # guard entered on the right
                if labels_new[-2 * n_g + j] > self.n_sheets:
                    labels_new[-2 * n_g + j] -= self.n_sheets

        elif i_roll != 0:
            # roll labels of sheets inside the box
            labels_new[n_g:-n_g] = np.roll(labels_new[n_g:-n_g], i_roll)

        # update guards
        # has to be done because of crossings + rolling
        x_new[-n_g:] = x_new[n_g : 2 * n_g] + L
        v_new[-n_g:] = v_new[n_g : 2 * n_g]
        x_new[:n_g] = x_new[-2 * n_g : -n_g] - L
        v_new[:n_g] = v_new[-2 * n_g : -n_g]
        labels_new[:n_g] = -labels_new[-2 * n_g : -n_g]
        labels_new[-n_g:] = labels_new[n_g : 2 * n_g] + self.n_sheets

        return x_new, v_new, x_eq_new, labels_new

    def _handle_boundary(
        self, x: np.ndarray, v: np.ndarray, x_eq: np.ndarray, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Updates guards and enforces boundary conditions after correcting for crossings.
        """
        if self.boundary == "periodic":
            x, v, x_eq, labels = self._periodic_boundary(x, v, x_eq, labels)

        elif self.boundary == "reflecting":
            x, v, x_eq, labels = self._reflecting_boundary(x, v, x_eq, labels)

        return x, v, x_eq, labels

    def _single_step(
        self,
        x_i: np.ndarray,
        v_i: np.ndarray,
        x_eq_i: np.ndarray,
        labels_i: np.ndarray,
        v_neg_max: float,
        xi_pos_max: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs a single step t -> t+dt of the synchronous Sheet Model algorithm.
        """
        # ODE update (not accounting for crossings yet)
        x_f, v_f = self._advance_eom(x_i, v_i, x_eq_i, self.dt)
        x_eq_f = x_eq_i.copy()
        labels_f = labels_i.copy()

        # check if a sheet crossed all guards
        # or all guards from one boundary entered the simulation box
        self._check_guards(x_f)

        # correct crossings
        if self.n_it_crossings > 0:
            x_f, v_f, x_eq_f, labels_f = self._handle_crossings(
                x_i, v_i, x_f, v_f, x_eq_i, labels_i, v_neg_max, xi_pos_max
            )
        else:
            i_sort = np.argsort(x_f)
            x_f = x_f[i_sort]
            v_f = v_f[i_sort]
            if self.track_sheets:
                labels_f = labels_f[i_sort]

        # update guards
        x_f, v_f, x_eq_f, labels_f = self._handle_boundary(x_f, v_f, x_eq_f, labels_f)

        return x_f, v_f, x_eq_f, labels_f

    def run_simulation(
        self,
        x_0: np.ndarray,
        v_0: np.ndarray,
        t_max: float,
        x_eq_0: Optional[np.ndarray] = None,
        dt_undersample: int = 1,
        return_guards: bool = False,
        verbose: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs a simulation rollout.

        Args:
            x_0 - initial positions [L]
            v_0 - initial velocites [L w_p]
            t_max - maximum rollout time.
            x_eq_0 - initial equilibrium position (defaults to standard values)
            dt_undersample - undersample factor to use for storage
            return_guards - include guard cells in output trajectories
            verbose - print progress bar

        Returns
            X, V, X_eq, E - rollout buffers + energy (#timesteps, #sheets or 1)
        """
        # check if inputs are valid
        self._check_inputs(x_0, v_0, x_eq_0)

        # if x_eq_0 not provided, use default
        if x_eq_0 is None:
            x_eq_0 = get_x_eq(self.n_sheets, self.L)

        # arrays that store info at t
        x, v, x_eq = self._add_guards(x_0, v_0, x_eq_0)
        # array to keep track of sheet ids
        labels = self._initialize_labels()

        # initialize output buffers
        if return_guards:
            X = [x[np.argsort(labels)]]
            V = [v[np.argsort(labels)]]
            X_eq = [x_eq[np.argsort(labels)]]
        else:
            aux = self._remove_guards(x, v, x_eq)
            X = [aux[0]]
            V = [aux[1]]
            X_eq = [aux[2]]

        E = [np.sum(get_E(*self._remove_guards(x, v, x_eq)))]

        for i in tqdm(range(int(t_max / self.dt)), disable=not verbose):
            if self.new_dx_max:
                # always update maximum negative velocity and displacement from equilibrium
                v_neg_max = np.max(-v, initial=0)
                xi_pos_max = np.max(x - x_eq, initial=0)
            elif i % int(1.0 / self.dt) == 0:
                # if a plasma period has passed, update maximum negative velocity
                v_neg_max = np.max(-v, initial=0)
                xi_pos_max = None

            x, v, x_eq, labels = self._single_step(
                x, v, x_eq, labels, v_neg_max, xi_pos_max
            )

            if (i + 1) % dt_undersample == 0:
                labelsort = np.argsort(labels)
                x_aux = x[labelsort].copy()
                v_aux = v[labelsort].copy()
                x_eq_aux = x_eq[labelsort].copy()

                aux = self._remove_guards(x_aux, v_aux, x_eq_aux)

                if return_guards:
                    X.append(x_aux)
                    V.append(v_aux)
                    X_eq.append(x_eq_aux)

                else:
                    X.append(aux[0].copy())
                    V.append(aux[1].copy())
                    X_eq.append(aux[2].copy())

                E.append(np.sum(get_E(*aux)))

        X = np.asarray(X)
        V = np.array(V)
        X_eq = np.array(X_eq)
        E = np.array(E)

        return X, V, X_eq, E
