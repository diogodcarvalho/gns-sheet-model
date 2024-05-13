import jax
import jraph
import jax.numpy as jnp

from functools import partial

from gns.utils import get_dx_eq


def add_guards(
    x: jnp.ndarray,
    v: jnp.ndarray,
    x_eq: jnp.ndarray,
    dx_eq: jnp.ndarray,
    L: float,
    n_guards: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.array]:
    """
    Appends mirrored guard sheets to original data arrays.

    Used for reflecting boundaries.

    Args:
        x, v, x_eq - 2D arrays (#sheets, w_size)
        dx_eq - equilibrium distance between sheets
        L - simulation box size
        n_guards - number of guard sheets

    Returns
        x, v, x_eq - updated arrays with guards
    """

    l_guard, r_guard = get_guards(x, v, x_eq, dx_eq, L, n_guards)

    x = jnp.concatenate([l_guard["x"].reshape(-1, 1), x, r_guard["x"].reshape(-1, 1)])

    v = jnp.concatenate(
        [
            l_guard["v"].reshape(-1, v.shape[-1]),
            v,
            r_guard["v"].reshape(-1, v.shape[-1]),
        ]
    )

    x_eq = jnp.concatenate(
        [l_guard["x_eq"].reshape(-1, 1), x_eq, r_guard["x_eq"].reshape(-1, 1)]
    )

    return x, v, x_eq


def get_guards(
    x: jnp.ndarray,
    v: jnp.ndarray,
    x_eq: jnp.ndarray,
    dx_eq: jnp.ndarray,
    L: float,
    n_guards: float,
) -> tuple[dict, dict]:
    """
    Computes guard sheets (left and right) for reflecting boundaries.

    Args:
        x, v, x_eq - ND arrays (#sheets, [...])
        dx_eq - equilibrium distance between sheets
        L - simulation box size
        n_guards - number of guard sheets

    Returns:
        l_guard, r_guard - left and right guard sheets
    """
    n_sheets = x.shape[0]
    l_guard = dict()
    r_guard = dict()

    l_guard["x_eq"] = x_eq[0] - dx_eq * jnp.arange(n_guards, 0, -1)
    r_guard["x_eq"] = x_eq[-1] + dx_eq * jnp.arange(1, n_guards + 1)

    if n_guards <= n_sheets:
        l_guard["x"] = jnp.flip(-x[:n_guards], axis=0)
        l_guard["v"] = jnp.flip(-v[:n_guards], axis=0)

        r_guard["x"] = jnp.flip(2 * L - x[-n_guards:], axis=0)
        r_guard["v"] = jnp.flip(-v[-n_guards:], axis=0)

    else:
        n_loops = n_guards // n_sheets
        n_rest = n_guards % n_sheets

        l_guard["x"] = jnp.flip(-x, axis=0)
        l_guard["v"] = jnp.flip(-v, axis=0)

        r_guard["x"] = jnp.flip(2 * L - x, axis=0)
        r_guard["v"] = jnp.flip(-v, axis=0)

        aux_l = jnp.flip(x, axis=0)
        aux_r = jnp.flip(L - x, axis=0)
        aux_v = jnp.flip(v, axis=0)

        for i in range(1, n_loops):
            aux_l = jnp.flip(L - aux_l, axis=0)
            aux_r = jnp.flip(L - aux_r, axis=0)
            aux_v = jnp.flip(-aux_v, axis=0)

            l_guard["x"] = jnp.concatenate([-L * i - aux_l, l_guard["x"]])
            l_guard["v"] = jnp.concatenate([aux_v, l_guard["v"]])

            r_guard["x"] = jnp.concatenate([r_guard["x"], L * (i + 1) + aux_r])
            r_guard["v"] = jnp.concatenate([r_guard["v"], aux_v])

        if n_rest != 0:
            aux_l = jnp.flip(L - aux_l, axis=0)
            aux_r = jnp.flip(L - aux_r, axis=0)
            aux_v = jnp.flip(-aux_v, axis=0)

            l_guard["x"] = jnp.concatenate(
                [-L * n_loops - aux_l[-n_rest:], l_guard["x"]]
            )
            l_guard["v"] = jnp.concatenate([aux_v[-n_rest:], l_guard["v"]])

            r_guard["x"] = jnp.concatenate(
                [r_guard["x"], L * (n_loops + 1) + aux_r[:n_rest]]
            )
            r_guard["v"] = jnp.concatenate([r_guard["v"], aux_v[:n_rest]])

    return l_guard, r_guard


def get_nodes(x: jnp.ndarray, v: jnp.ndarray, x_eq: jnp.ndarray, dx_eq: float) -> dict:
    """
    Computes node representation.

    Each node contains:
        x - sheet position [dx_eq]
        v - sheet velocity [dx_eq w_p]
        x_eq - sheet equilibrium position [dx_eq]

    Note:
        All distances/velocities are normalized to dx_eq.
    """
    nodes = {
        "x": x / dx_eq,
        "v": v / dx_eq,
        "x_eq": x_eq / dx_eq,
    }

    return nodes


def get_edges(
    x: jnp.ndarray, dx_eq: float, L: float, boundary: str, n_neighbors: int
) -> dict:
    """
    Computes edge representation.
    Each connected pair of sheets is associated with 2 edges (i->j, i<-j).

    Each edge contains:
        dx - relative distance between sheets [dx_eq]

    Output order is:
        [i->j 1st neighbors, ..., i->j Nth neighbors,
         i<-j 1st neighbors, ..., i<-j Nth neighbors]

    Note:
        dx_i->j = (x_j - x_i)
        All distances/velocities are normalized to dx_eq.
    """
    if boundary == "reflecting":
        dx = jnp.diff(x, axis=0)
        # compute n-neighbors distance
        for n in range(2, n_neighbors + 1):
            dx = jnp.concatenate([dx, x[n:] - x[:-n]])

    elif boundary == "periodic":
        n_sheets = x.shape[0]
        # n-neigbor distances
        dx = jnp.concatenate(
            [jnp.roll(x, -(n + 1), axis=0) - x for n in range(n_neighbors)]
        )
        # d < 0 can happen with periodic boundary
        dx = jnp.where(dx < 0, dx + L, dx)
        # fix distances when n_neighbors >= n_sheets (adds +L at each loop)
        dx += (
            jnp.repeat(jnp.arange(n_neighbors) + 1, n_sheets).reshape(-1, 1)
            // n_sheets
            * L
        )

    edges = {"dx": jnp.concatenate([-dx, dx]) / dx_eq}

    return edges


def get_senders_receivers(
    n_node: int, boundary: str, n_neighbors: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes senders and receivers arrays corresponding to the edges obtained
    with get_edges().
    """
    if n_node > 1:
        nodes = jnp.arange(n_node)

        if boundary == "periodic":
            senders = jnp.concatenate(
                [
                    jnp.concatenate([nodes for _ in range(n_neighbors)]),
                    jnp.concatenate(
                        [jnp.roll(nodes, -(n + 1)) for n in range(n_neighbors)]
                    ),
                ]
            )

            receivers = jnp.concatenate(
                [
                    jnp.concatenate(
                        [jnp.roll(nodes, -(n + 1)) for n in range(n_neighbors)]
                    ),
                    jnp.concatenate([nodes for _ in range(n_neighbors)]),
                ]
            )

        elif boundary == "reflecting":
            senders = jnp.concatenate(
                [
                    jnp.concatenate([nodes[:-n] for n in range(1, n_neighbors + 1)]),
                    jnp.concatenate([nodes[n:] for n in range(1, n_neighbors + 1)]),
                ]
            )

            receivers = jnp.concatenate(
                [
                    jnp.concatenate([nodes[n:] for n in range(1, n_neighbors + 1)]),
                    jnp.concatenate([nodes[:-n] for n in range(1, n_neighbors + 1)]),
                ]
            )

    else:
        raise AssertionError("Graph must contain more than 1 node")

    return senders, receivers


def build_graphs(
    X: jnp.ndarray,
    V: jnp.ndarray,
    X_eq: jnp.ndarray,
    boundary: str,
    n_neighbors: int = 1,
    n_guards: int = 0,
    L: float = 1,
) -> jraph.GraphsTuple:
    """
    Build graph for Sheet Model Graph Neural Network based on input time-series.
    Each node in the graph corresponds to a sheet, while edges represent connections
    between neighboring sheets.

    IMPORTANT: Sheets MUST be ordered according to their position (at each time-step)
               for proper graph construction.

    Args:
        X, V, X_eq - Time-series data of 3 possible shapes:
            1D - (#sheets)
            2D - (w_size, #sheets)
            3D - (#timesteps, w_size, #sheets)

        boundary - Boundary conditions ("periodic" or "reflecting").
        n_neighbors - Number of n-closest neighbors to connect to each node.
        n_guards - Number of guard sheets to use. Only required for reflecting boundaries.
        L - Simulation box length.

    Returns
        graphs - Single graph containing the full time-series data.
            If multiple time steps were provided, the graph is given by multiple
            non-connected subgraphs.
    """
    globals_ = {
        "L": L,
        "n_sheets": X.shape[-1],
        "dx_eq": get_dx_eq(X.shape[-1], L),
        "n_guards": n_guards,
    }

    # ensure axis ordering is correct for node/edge computation functions
    # ([timesteps,] n_sheets, w_size)
    if X.ndim == 1:
        # (n_sheets, 1)
        X = X[:, jnp.newaxis]
        X_eq = X_eq[:, jnp.newaxis]
        V = V[:, jnp.newaxis]

    elif X.ndim == 2:
        # (n_sheets, 1)
        X = X[-1:].T
        X_eq = X_eq[-1:].T
        # (n_sheets, w_size)
        V = V.T

    elif X.ndim == 3:
        # (timesteps, n_sheets, w_size)
        X = jnp.swapaxes(X, 1, 2)
        V = jnp.swapaxes(V, 1, 2)
        X_eq = jnp.swapaxes(X_eq, 1, 2)
        # only the last timestep matters for X and X_eq
        X = X[:, :, -1:]
        X_eq = X_eq[:, :, -1:]

    # add guards
    if boundary == "reflecting":
        if n_guards < 1:
            raise AssertionError("Set n_guards >= 1 for reflecting boundaries")
        elif 2 * n_guards + globals_["n_sheets"] <= n_neighbors:
            raise AssertionError("Ensure that 2*n_guards + n_sheets > n_neighbors")
        else:
            guards_fn = partial(
                add_guards, dx_eq=globals_["dx_eq"], L=L, n_guards=n_guards
            )
            if X.ndim == 2:
                X, V, X_eq = guards_fn(X, V, X_eq)
            elif X.ndim == 3:
                X, V, X_eq = jax.vmap(guards_fn, in_axes=(0, 0, 0))(X, V, X_eq)

    elif boundary == "periodic" and n_guards != 0:
        globals_["n_guards"] = 0
        print("n_guards != 0 ignored for periodic boundaries")

    nodes_fn = partial(get_nodes, dx_eq=globals_["dx_eq"])

    edges_fn = partial(
        get_edges,
        boundary=boundary,
        n_neighbors=n_neighbors,
        dx_eq=globals_["dx_eq"],
        L=L,
    )

    # single time step case
    if X.ndim == 2:
        nodes = nodes_fn(X, V, X_eq)
        edges = edges_fn(X)

        n_node = jnp.array([nodes["x"].shape[0]])
        n_edge = jnp.array([edges["dx"].shape[0]])

        senders, receivers = get_senders_receivers(
            n_node=nodes["x"].shape[0], boundary=boundary, n_neighbors=n_neighbors
        )

        graphs = jraph.GraphsTuple(
            nodes=nodes,
            edges=edges,
            senders=senders,
            receivers=receivers,
            n_node=n_node,
            n_edge=n_edge,
            globals={k: jnp.asarray([v]) for k, v in globals_.items()},
        )

    # multiple time steps case
    elif X.ndim == 3:
        nodes = jax.vmap(nodes_fn, in_axes=(0, 0, 0))(X, V, X_eq)
        edges = jax.vmap(edges_fn, in_axes=(0,))(X)

        n_graphs = nodes["x"].shape[0]
        n_node = nodes["x"].shape[1] * jnp.ones(n_graphs, dtype=int)
        n_edge = edges["dx"].shape[1] * jnp.ones(n_graphs, dtype=int)

        n_node_static = nodes["x"].shape[1]
        n_edge_static = edges["dx"].shape[1]

        # all graphs have the same senders and receivers
        senders, receivers = get_senders_receivers(
            n_node=n_node_static, boundary=boundary, n_neighbors=n_neighbors
        )

        senders = senders * jnp.ones((n_graphs, n_edge_static), dtype=int)
        receivers = receivers * jnp.ones((n_graphs, n_edge_static), dtype=int)

        senders = senders.flatten()
        receivers = receivers.flatten()

        # senders / receivers offset for GraphsTuple
        offsets = jnp.roll(jnp.cumsum(n_node, dtype=int), 1)
        offsets = offsets.at[0].set(0)
        offsets = jnp.repeat(offsets, n_edge_static)

        senders = senders + offsets
        receivers = receivers + offsets

        # flatten dicts
        nodes = {k: v.reshape(-1, v.shape[-1]) for k, v in nodes.items()}
        edges = {k: v.reshape(-1, v.shape[-1]) for k, v in edges.items()}

        graphs = jraph.GraphsTuple(
            nodes=nodes,
            edges=edges,
            senders=senders,
            receivers=receivers,
            n_node=n_node,
            n_edge=n_edge,
            globals={k: jnp.asarray([v]) for k, v in globals_.items()},
        )

    return graphs
