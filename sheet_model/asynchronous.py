import itertools
import numpy as np

from tqdm.auto import tqdm
from heapq import heappush, heappop
from typing import Optional, Union
from dataclasses import dataclass

from sheet_model.utils import get_dx_eq, get_x_eq, get_E


@dataclass
class HeapEntry:
    """Class which represent an item stored in the heap."""

    t_cross: float
    count: int
    tag: str

    def __lt__(self, other):
        """Allow for comparion between entries."""
        if self.t_cross != other.t_cross:
            return self.t_cross < other.t_cross
        else:
            return self.count < other.count


class AsyncSheetModel(object):
    """
    Adaptation of Dawson's single-specie asynchronous sheet model algorithm [1]
    using a standard priority queue [2] instead of the original table method.
    Additionally, we do not limit the maximum crossing time added to the queue.

    [1] J. Dawson, "The Electrostatic Sheet Model for a Plasma and its Modification
        to Finite-Size Particles", Methods in Computational Physics vol. 9
        Academic Press Inc., U.S. (1970)

    [2] https://docs.python.org/3/library/heapq.html#priority-queue-implementation-notes

    Args:
        L - box size
        boundary - 'reflecting' or 'periodic'
        track_sheets - use 'True' for crossing dynamics, False for collisional dynamics.

    Params:
        L
        boundary
        track_sheets
        crossing_pq - priority queue with predicted crossing times
        entry_finder - mapping of tags 'i-j' to heap entries
        counter - unique sequence count, serves as tiebreaker for pq
        REMOVED - placeholder for a removed crossing
        x - all sheet positions at the current time (t=t_cross)
        v - same for velocities
        x_eq - same for equilibrium positions
        labels - array with identifier of each sheet
        n_sheet - number of sheets
        dx_eq - intersheet spacing in equilibrium [L]
    """

    def __init__(self, L: float, boundary: str, track_sheets: bool):
        self.L = L
        self.boundary = boundary
        self.track_sheets = track_sheets

        self.crossing_pq: list[HeapEntry] = []
        self.entry_finder: dict[str, HeapEntry] = dict()
        self.counter = itertools.count()
        self.REMOVED = "<removed>"

        # to be filled after initial conditions are provided
        self.x = np.empty([])
        self.v = np.empty([])
        self.x_eq = np.empty([])
        self.labels = np.empty([])
        self.n_sheets = -1
        self.dx_eq = np.nan

    @staticmethod
    def _id2tag(i: int, j: int) -> str:
        """
        Generate tag from sheet indices.
        """
        return f"{i}-{j}"

    @staticmethod
    def _tag2id(tag: str) -> tuple[int, int]:
        """
        Recover indices from tag.
        """
        i, j = tag.split("-")
        return int(i), int(j)

    def _check_inputs(
        self, x_0: np.ndarray, v_0: np.ndarray, x_eq_0: Optional[np.ndarray]
    ):
        """
        Check if initial conditions are valid.
        """
        if len(x_0) < 2:
            raise ValueError("Must have > 1 sheet")

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

    def _analytical_crossing_time(
        self,
        x: np.ndarray,
        v: np.ndarray,
        x_eq: np.ndarray,
        force_non_zero: bool = False,
    ) -> np.ndarray:
        """
        Computes analytical solution to crossing time, i.e. roots of:

            a*sin(t) + b*cos(t) + c = 0
            a = v_j - v_i
            b = (x_j - x_i) - (x_eq_j - x_eq_i)
            c = x_eq_j - x_eq_i

        If sheets do not cross, np.inf is returned.

        Args:
            x, v, x_eq - arrays of shape (..., n_sheets)
            force_non_zero - if True, forces solution to be > 0, i.e. removes
                             trivial solution when sheets just crossed

        Returns:
            t - crossing time in ]0, 2*pi] (..., n_sheets-1)
        """
        a = np.atleast_1d(np.diff(v))
        b = np.atleast_1d(np.diff(x) - np.diff(x_eq))
        c = np.atleast_1d(np.diff(x_eq))

        # by default, crossing time = inf
        # usually there exist 2 solutions within [0, 2*pi]
        tp = np.full_like(a, np.inf, dtype=np.float64)
        tn = np.full_like(a, np.inf, dtype=np.float64)

        # real solutions condition
        aux = a * a + b * b - c * c
        i1 = aux >= 0
        # special cases
        # 1e-5*dx_eq factor introduced for numerical stability
        # b = c
        i2 = np.abs(b - c) < 1e-5 * self.dx_eq
        # a = 0
        i3 = np.abs(a) < 1e-5 * self.dx_eq

        # case where b = c and a = 0
        i23 = np.logical_and(i2, i3)
        if np.any(i23):
            tp[i23] = np.pi

        # case where b = c and a != 0
        i2n3 = np.logical_and(i2, ~i3)
        if np.any(i2n3):
            # between [0, 2*pi]
            tp[i2n3] = (2 * np.arctan(-b[i2n3] / a[i2n3])) % (2 * np.pi)

        # remaining real solutions (aux > 0 and b != c)
        i1n2 = np.logical_and(i1, ~i2)
        if np.any(i1n2):
            aux = np.sqrt(aux[i1n2])
            if force_non_zero:
                tp[i1n2] = 2 * np.arctan(
                    np.sign(a[i1n2]) * (np.abs(a[i1n2]) + aux) / (b[i1n2] - c[i1n2])
                )
                # force [0, 2*pi]
                tp[i1n2] = tp[i1n2] % (2 * np.pi)
            else:
                tp[i1n2] = 2 * np.arctan((a[i1n2] + aux) / (b[i1n2] - c[i1n2]))
                tn[i1n2] = 2 * np.arctan((a[i1n2] - aux) / (b[i1n2] - c[i1n2]))
                # force [0, 2*pi]
                tp[i1n2] = tp[i1n2] % (2 * np.pi)
                tn[i1n2] = tn[i1n2] % (2 * np.pi)

        # pick minimum (per pair of sheets)
        t = np.squeeze(np.min([tp, tn], axis=0))

        return t

    def _add_crossing(self, tag: str, t_cross: float = 0):
        """
        Adds a new crossing or updates the entry of an existing crossing.

        Args:
            tag - format "i-j", where i,j are unique sheet IDs (index in arrays)
        """
        self._remove_crossing(tag)
        count = int(tag.split("-")[-1])
        entry = HeapEntry(t_cross, count, tag)
        self.entry_finder[tag] = entry
        heappush(self.crossing_pq, entry)

    def _remove_crossing(self, tag: str):
        """
        Mark an existing tag as REMOVED. Crossing will still be present in the
        priority queue until it is updated of popped.

        Args:
            tag - format "i-j", where i,j are unique sheet IDs (index in arrays)
        """
        if tag in self.entry_finder:
            entry = self.entry_finder.pop(tag)
            entry.tag = self.REMOVED

    def _pop_crossing(self) -> HeapEntry:
        """
        Remove and return the lowest t_cross entry which is not marked as REMOVED.
        Raise KeyError if empty.

        Returns:
            t_cross - crossing time
            tag - format "i-j" where i,j are unique sheet IDs (index in arrays)
        """
        while self.crossing_pq:
            entry = heappop(self.crossing_pq)
            if entry.tag is not self.REMOVED:
                del self.entry_finder[entry.tag]
                return entry
        raise KeyError("Empty priority queue")

    def _initialize_arrays(
        self, x: np.ndarray, v: np.ndarray, x_eq: Optional[np.ndarray] = None
    ):
        """
        Initializes self.x,v,etc arrays with input values.
        Guard sheets are appended to the beginning/end of the arrays.

        Guards labels equal -labels of the corresponding 'real' sheets.
        """
        self.n_sheets = len(x)
        self.dx_eq = get_dx_eq(self.n_sheets, self.L)
        # if x_eq not provided, use default
        if x_eq is None:
            x_eq = get_x_eq(self.n_sheets, self.L)

        # append guard sheets
        if self.boundary == "reflecting":
            # 2 guards (left & right)
            x = np.insert(x, [0, len(x)], [-x[0], 2 * self.L - x[-1]])
            v = np.insert(v, [0, len(v)], [-v[0], -v[-1]])
            x_eq = np.insert(x_eq, [0, len(x_eq)], [-x_eq[0], 2 * self.L - x_eq[-1]])
            labels = np.array(
                [-1] + list(range(1, self.n_sheets + 1)) + [-self.n_sheets]
            )
        elif self.boundary == "periodic":
            # 1 guard (left)
            x = np.insert(x, 0, x[-1] - self.L)
            v = np.insert(v, 0, v[-1])
            x_eq = np.insert(x_eq, 0, x_eq[-1] - self.L)
            labels = np.array([-self.n_sheets] + list(range(1, self.n_sheets + 1)))

        self.x = x
        self.v = v
        self.x_eq = x_eq
        self.labels = labels

    def _initialize_pq(self):
        """
        Fill priority queue with crossing times for the initial conditions.
        """
        t_cross = self._analytical_crossing_time(self.x, self.v, self.x_eq)
        for i, t in enumerate(t_cross):
            if t != np.inf:
                self._add_crossing(self._id2tag(i, i + 1), t)

    def _advance_eom(self, dt: float):
        """
        Advance sheet positions/velocities according to equation of motion.

        Args:
            dt - step size [w_p^-1]
        """
        xi = self.x - self.x_eq
        v_new = self.v * np.cos(dt) - xi * np.sin(dt)
        x_new = self.x + self.v * np.sin(dt) - xi * (1 - np.cos(dt))

        self.v = v_new
        self.x = x_new

    def _single_step(self, t: float) -> Union[None, float]:
        """
        Advance system until the next crossing (t -> t_cross).

        Args:
            t - current time [w_p^-1]
        Returns:
            t_cross - crossing time [w_p^-1]
        """
        try:
            entry = self._pop_crossing()
        except KeyError:
            # reached the end of the pq
            return None

        # advance all sheets from t->t_cross
        self._advance_eom(entry.t_cross - t)

        # get indices of sheets that crossed
        i, j = self._tag2id(entry.tag)

        # switch relative ordering
        # x_eq is not swapped!
        self.x[[i, j]] = self.x[[j, i]]
        self.v[[i, j]] = self.v[[j, i]]
        # if sheets do not include guards, switch their labels
        # other cases will be handled together with guard re-assignment
        if i != 0 and j != self.n_sheets + 1:
            self.labels[[i, j]] = self.labels[[j, i]]

        # pairs that require new crossing times
        pairs = [[i, j]]
        if i != 0:
            pairs.append([i - 1, i])
        if j != len(self.x) - 1:
            pairs.append([j, j + 1])
        if self.boundary == "periodic":
            if i == 0:
                # last sheet in the box will change to mirror new left guard
                pairs.append([self.n_sheets - 1, self.n_sheets])
            if j == self.n_sheets:
                # left guard will change
                pairs.append([0, 1])

        # re-assign guards if necessary
        if self.boundary == "reflecting":
            # left sheet was the first real sheet
            if i == 1:
                # update left guard
                self.x[0] = -self.x[1]
                self.v[0] = -self.v[1]
                self.labels[0] = -self.labels[1]
            # right sheet was the last real sheet
            if j == self.n_sheets:
                # update right guard
                self.x[-1] = 2 * self.L - self.x[-2]
                self.v[-1] = -self.v[-2]
                self.labels[-1] = -self.labels[-2]
        elif self.boundary == "periodic":
            # left sheet was the left guard
            if i == 0:
                # switch labels of sheets that crossed
                self.labels[[0, 1]] = -self.labels[[1, 0]]
                # update last real sheet (equivalent to new guard)
                self.x[-1] = self.x[0] + self.L
                self.v[-1] = self.v[0]
                self.x_eq[-1] = self.x_eq[0] + self.L
                self.labels[-1] = -self.labels[0]
            # right sheet was the last real sheet
            elif j == self.n_sheets:
                # update left guard
                self.x[0] = self.x[-1] - self.L
                self.v[0] = self.v[-1]
                self.labels[0] = -self.labels[-1]

        # update crossing times in heap
        for p in pairs:
            tcp = self._analytical_crossing_time(
                self.x[p], self.v[p], self.x_eq[p], force_non_zero=(p[0] == i)
            )

            if tcp != np.inf:
                # inserts/updates crossing time
                self._add_crossing(self._id2tag(*p), entry.t_cross + tcp)
            else:
                # must be done because (p1,p2) could have been in the heap!
                self._remove_crossing(self._id2tag(*p))

        return entry.t_cross

    def run_simulation(
        self,
        x_0: np.ndarray,
        v_0: np.ndarray,
        t_max: float,
        x_eq_0: Optional[np.ndarray] = None,
        return_inside_box: bool = True,
        dt_sampling: Optional[float] = None,
        verbose: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs a simulation rollout.
        The output arrays entries are not equally spaced (temporally)!

        Args:
            x_0 - initial positions [L]
            v_0 - initial velocites [L w_p]
            t_max - maximum rollout time.
            x_eq_0 - initial equilibrium position (defaults to standard values)
            return_inside_box - force final sheet positions to be inside the box
            dt_sampling - asynchronous data is only appeded to output buffers every dt_sampling.
                if set, output can not be later transformed to synchronous format.
            verbose - print progress bar

        Returns
            T - array with crossing timesteps [w_p^-1]
            X - sheet positions at each crossing time (#t_cross, n_sheets) [L]
            V - same for velocites [L w_p]
            X_eq - same for equilibrium positions [L]
            E - same for the energy of the system (will be constant unless there is a bug ...)
        """
        self._check_inputs(x_0, v_0, x_eq_0)
        self._initialize_arrays(x_0, v_0, x_eq_0)
        self._initialize_pq()

        # only store 'real' sheets in buffers
        i_real = self.labels > 0
        T = [0.0]
        X = [self.x[i_real]]
        V = [self.v[i_real]]
        X_eq = [self.x_eq[i_real]]
        E = [np.sum(get_E(X[0], V[0], X_eq[0]))]

        if verbose:
            pbar = tqdm(total=100)
            dtbar = t_max / 100

        # run simulation
        t = 0.0
        while t < t_max:
            t_cross = self._single_step(t)

            if t_cross is None or t_cross > t_max:
                break

            else:
                assert t_cross >= t, "[Bug] Should never happen"

                # remove guard(s)
                i_real = self.labels > 0
                x = self.x[i_real].copy()
                v = self.v[i_real].copy()
                x_eq = self.x_eq[i_real].copy()

                if self.track_sheets:
                    i_sort = np.argsort(self.labels[i_real])
                    x = x[i_sort]
                    v = v[i_sort]
                    x_eq = x_eq[i_sort]

            if (dt_sampling is None) or (t_cross // dt_sampling > t // dt_sampling):
                T.append(t_cross)
                E.append(np.sum(get_E(x, v, x_eq)))
                X.append(x)
                V.append(v)
                X_eq.append(x_eq)

            if verbose and t_cross // dtbar > t // dtbar:
                td = min(t_max, t_cross)
                pbar.update(td // dtbar - t // dtbar)

            t = t_cross

            if not np.all(np.diff(self.x) >= -self.dx_eq * 1e-3):
                print("t:", t_cross, "dx min", np.min(np.diff(self.x)))
                print("dx[<0]", np.diff(self.x)[np.diff(self.x) < 0])
                print("i[dx<0]", np.argwhere(np.diff(self.x) < 0))
                raise RuntimeError("[Bug] Sheets Not Ordered")

        if verbose:
            pbar.update(t_max // dtbar - t // dtbar + 1)
            pbar.close()

        T = np.array(T)
        X = np.array(X)
        V = np.array(V)
        X_eq = np.array(X_eq)
        E = np.array(E)

        if return_inside_box:
            if self.boundary == "periodic":
                n = X // self.L
            elif self.boundary == "reflecting":
                # use X_eq in this case to avoid problems when X ~ L
                n = X_eq // self.L
            X -= n * self.L
            X_eq -= n * self.L

        return T, X, V, X_eq, E
