import numpy as np
from typing import Tuple, Optional

from sheet_model.synchronous import SyncSheetModel


class ModSyncSheetModel(SyncSheetModel):
    """
    Modified implementation of Dawson's synchronous sheet model [1] as described
    in [2].

    It introduces two alterations to the crossing correction routine:

        #1. Resolves crossings only up to the Nth neigbor.
        #2. A full sort is performed (by position) after correcting for crossings.

    #2 is required to ensure that sheets have unique positions in the final array.

    These alterations are done solely for additional comparisons with the GNS.
    This model should not be used in detriment of the (A)SyncSheetModel!

    [1] J. Dawson, "The Electrostatic Sheet Model for a Plasma and its Modification
        to Finite-Size Particles", Methods in Computational Physics vol. 9
        Academic Press Inc., U.S. (1970)

    [2] https://arxiv.org/abs/2310.17646

    Args:
        n_sheets - number of particles
        L - box size
        boundary - 'reflecting' or 'periodic'
        n_guards - number of guard sheets
        dt - simulation step [1/w_p]
        track_sheets - use 'True' for crossing dynamics, False for collisional dynamics.
        n_it_crossings - number of iterations to use for crossing time estimation
        n_max_neighbor - maximum neigbor to check for crossing
        new_dx_max - if True, overwrites original dx_max criteria from [1] and uses the one described in [2].

    Params:
        n_sheets
        L
        boundary
        n_guards
        dt
        track_sheets
        n_it_crossings
        n_max_neighbor
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
        n_max_neighbor: int,
        new_dx_max: bool = True,
    ):
        super().__init__(
            n_sheets=n_sheets,
            L=L,
            boundary=boundary,
            n_guards=n_guards,
            dt=dt,
            track_sheets=track_sheets,
            n_it_crossings=n_it_crossings,
            new_dx_max=new_dx_max,
        )

        self.n_max_neighbor = n_max_neighbor

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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Corrects sheet trajectories by accounting for crossings.
        Follows algorithm described in [1].
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
            for j in range(i + 1, min(len(x_f), i + 1 + self.n_max_neighbor)):
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
        assert np.sum(NCROSS) == 0

        # correct trajectories of sheets who underwent crossings
        for i in range(len(x_f)):
            if TCROSS[i]:
                x_f[i], v_f[i], _ = self._correct_trajectory(
                    x_i[i], v_i[i], x_eq_i[i], TCROSS[i], RCROSS[i]
                )

        # sort sheets
        i_sort = np.argsort(x_f)
        x_f = x_f[i_sort]
        v_f = v_f[i_sort]
        x_eq_f = x_eq_i.copy()
        labels_f = labels_i.copy()

        if self.track_sheets:
            labels_f = labels_f[i_sort]

        return x_f, v_f, x_eq_f, labels_f
