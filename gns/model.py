import jax
import jraph
import haiku as hk
import jax.numpy as jnp
import jax.tree_util as tree

from typing import Callable


def SMGNN(
    n_messages: int = 1,
    linear_enconder: bool = True,
    linear_decoder: bool = True,
    h_dim: int = 128,
    mlp_layers: int = 2,
    mlp_activation: Callable = jax.nn.relu,
    mlp_activate_final: bool = False,
    residual_connection: bool = False,
    node_update_use_sent_messages: bool = False,
    reflection_equivariance: bool = True,
) -> Callable[[jraph.GraphsTuple], jnp.ndarray]:
    """
    Returns a method that applies the GNN which enforces the invariances & symmetries
    of J. Dawnson's One Dimensional Electrostatic Sheet Model [1] as described in
    Carvalho et al. [2] + some additional options.

    Adapts the Encode-Process-Decode architecture from Battaglia et al. [3] and
    the GraphNetwork block implemented in Jraph [4].

    Work was motivated by Sanchez-Gonzalez et al. "Learning to Simulate Complex
    Physics with Graph Networks" [5].

    [1] https://pubs.aip.org/aip/pfl/article/5/4/445/803511/One-Dimensional-Plasma-Model
    [2] https://arxiv.org/abs/2310.17646
    [3] https://arxiv.org/abs/1806.01261
    [4] https://github.com/google-deepmind/jraph/blob/master/jraph/_src/models.py
    [5] https://arxiv.org/abs/2002.09405

    Args:
        n_messages - Number of message passing steps.
        linear_enconder - If true use linear encoder. If false use MLP.
        linear_decoder - If true use linear decoder. If false use MLP.
        h_dim - Number of hidden units (same for linear or MLP).
        mlp_layers - Number of layers to use in each MLP (hidden + output).
        mlp_activation - Activation function to use for MLP layers.
        mlp_activate_final - Use activation function on MLP output layer.
        residual_connection - If true include residual connection between message passing steps.
        node_update_use_sent_messages - If true use sent messages as input to node update function.
            Similar to Jraph GraphNetwork implementation [4].
        reflection_equivariance - Enforce reflection equivariance.
            If false message passing architecture is equivalent to a standard GraphNetwork [3] (with no globals).
            Only implemented for node_update_use_sent_messages = False.
    """
    if node_update_use_sent_messages and reflection_equivariance:
        raise NotImplementedError(
            "Can't set both node_update_use_sent_messages & reflection_equivariance to True"
        )

    def encoder_node_fn(node: jnp.ndarray, s_i: jnp.ndarray) -> jnp.ndarray:
        xi = node["x"] - node["x_eq"]

        if reflection_equivariance:
            n = jnp.concatenate([jnp.abs(xi), node["v"] * s_i], axis=-1)
        else:
            n = jnp.concatenate([xi, node["v"]], axis=-1)

        if linear_enconder:
            v = hk.Linear(h_dim)(n)
        else:
            v = hk.nets.MLP(
                [h_dim for _ in range(mlp_layers)],
                activation=mlp_activation,
                activate_final=mlp_activate_final,
            )(n)

        return v

    def encoder_edge_fn(edge: jnp.ndarray, s_i: jnp.ndarray) -> jnp.ndarray:
        if reflection_equivariance:
            e = edge["dx"] * s_i
        else:
            e = edge["dx"]

        if linear_enconder:
            e = hk.Linear(h_dim)(e)
        else:
            e = hk.nets.MLP(
                [h_dim for _ in range(mlp_layers)],
                activation=mlp_activation,
                activate_final=mlp_activate_final,
            )(e)

        return e

    def decoder_node_fn(v: jnp.ndarray) -> jnp.ndarray:
        if linear_decoder:
            y = hk.Linear(1)(v)
        else:
            y = hk.nets.MLP(
                [h_dim for _ in range(mlp_layers - 1)] + [1],
                activation=mlp_activation,
                activate_final=False,
            )(v)

        return y

    @jraph.concatenated_args
    def update_edge_fn(x: jnp.ndarray) -> jnp.ndarray:
        return hk.nets.MLP(
            [h_dim for _ in range(mlp_layers)],
            activation=mlp_activation,
            activate_final=mlp_activate_final,
        )(x)

    @jraph.concatenated_args
    def update_node_fn(x: jnp.ndarray) -> jnp.ndarray:
        return hk.nets.MLP(
            [h_dim for _ in range(mlp_layers)],
            activation=mlp_activation,
            activate_final=mlp_activate_final,
        )(x)

    def message_passing_fn(
        v: jnp.ndarray,
        e: jnp.ndarray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
        s_ij: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        sum_n_node = tree.tree_leaves(v)[0].shape[0]

        v_sender = v[senders]
        v_receiver = v[receivers]

        if reflection_equivariance:
            # pylint: disable-next=too-many-function-args
            e = update_edge_fn(s_ij * v_sender, v_receiver, e)
        else:
            # pylint: disable-next=too-many-function-args
            e = update_edge_fn(v_sender, v_receiver, e)

        e_received = jraph.segment_sum(e, receivers, sum_n_node)

        if node_update_use_sent_messages:
            e_sent = jraph.segment_sum(e, senders, sum_n_node)
            # pylint: disable-next=too-many-function-args
            v = update_edge_fn(v, e_sent, e_received)
        else:
            # pylint: disable-next=too-many-function-args
            v = update_edge_fn(v, e_received)

        return v, e

    def applyGNN(graph: jraph.GraphsTuple) -> jnp.ndarray:
        nodes, edges, receivers, senders, _, _, _ = graph

        if reflection_equivariance:
            xi = nodes["x"] - nodes["x_eq"]
            s_i = jnp.where(
                jnp.abs(xi) < 1e-10,
                jnp.where(jnp.abs(nodes["v"]) < 1e-10, 1, jnp.sign(nodes["v"])),
                jnp.sign(xi),
            )
            s_ij = s_i[receivers] * s_i[senders]
        else:
            s_i = jnp.zeros_like(nodes["x"])
            s_ij = jnp.zeros_like(edges["dx"])

        v = encoder_node_fn(graph.nodes, s_i)
        e = encoder_edge_fn(graph.edges, s_i[receivers])

        for _ in range(n_messages):
            if residual_connection:
                dv, de = message_passing_fn(v, e, senders, receivers, s_ij)
                v += dv
                e += de
            else:
                v, e = message_passing_fn(v, e, senders, receivers, s_ij)

        if reflection_equivariance:
            y = decoder_node_fn(v) * s_i * graph.globals["dx_eq"]
        else:
            y = decoder_node_fn(v) * graph.globals["dx_eq"]

        return y

    return applyGNN
