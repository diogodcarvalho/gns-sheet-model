import jax
import jraph
import optax
import functools
import pickle
import yaml

import haiku as hk
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm
from typing import Callable, List, Optional, Iterable


class DataLoader(object):
    """
    Simple DataLoader for sheet model pre-processed graph datasets.
    Facilitates custom spliting of training/validation set.

    Args:
        data_folder
            Path to folder with graph + targets dataset.

        i_train
            Max index for training data (i.e use files 0 - i_train).
            If not provided, considers i_train=i_valid.

        i_valid
            Starting index for validation data (i.e. use i_valid - len(dataset)).

        scale_targets
            If True, divides targets by std.

    Params:
        data_folder
            Path to folder with graph + targets dataset.

        n_files
            Number of graph + target pairs in the dataset.

        n_zeros
            Max number of trailing zeros to expect on file name.

        n_train
            Number of training instances (i_train).

        n_valid.
            Number of validation instances (n_files - i_valid).

        scale_targets
            If True, divides targets by std.

        y_mean, y_std
            Training dataset target statistics.
    """

    def __init__(
        self,
        data_folder: str,
        i_train: int = None,
        i_valid: int = 0,
        scale_targets: bool = False,
    ):
        self.data_folder = Path(data_folder)
        self.n_files = len(list((self.data_folder / "graphs").iterdir()))
        self.n_zeros = int(np.round(np.log10(self.n_files)))

        if i_train is None:
            self.n_train = i_valid
        else:
            assert i_train <= i_valid, "Set i_train <= i_valid"
            self.n_train = i_train

        self.i_valid = i_valid
        self.n_valid = self.n_files - i_valid

        self.scale_targets = scale_targets
        self.y_mean, self.y_std = self._compute_stats()

    def _compute_stats(self):
        # stats are only calculated on the training data
        if self.i_valid < 1e4:
            aux = jnp.concatenate([self.get_target(i) for i in range(self.i_valid)])
            y_mean = jnp.mean(aux, axis=0)
            y_std = jnp.std(aux, axis=0)

        else:
            y_mean = jnp.mean(self.get_target(0), axis=0) / self.i_valid
            y_std = jnp.zeros(y_mean.shape)

            for i in tqdm(range(1, self.i_valid)):
                y_mean += jnp.mean(self.get_target(i), axis=0) / self.i_valid

            for i in range(self.i_valid):
                y_std += (
                    jnp.mean(jnp.square(self.get_target(i) - y_mean), axis=0)
                    / self.i_valid
                )

            y_std = jnp.sqrt(y_std)

        return y_mean, y_std

    def _scale(self, target: float) -> float:
        return target / self.y_std

    def get_graph(self, i: int) -> jraph.GraphsTuple:
        with open(self.data_folder / f"graphs/{i:0{self.n_zeros}d}.pkl", "rb") as f:
            graphs = pickle.load(f)
        return graphs

    def get_target(self, i: int) -> jnp.ndarray:
        return jnp.load(self.data_folder / f"targets/{i:0{self.n_zeros}d}.npy")

    def get(self, i: int) -> tuple[jraph.GraphsTuple, jnp.ndarray]:
        assert i >= 0 and i < self.n_files
        graph, target = self.get_graph(i), self.get_target(i)
        if self.scale_targets:
            target = self._scale(target)
        return graph, target

    def get_train(self, i: int) -> tuple[jraph.GraphsTuple, jnp.ndarray]:
        graph, target = self.get(i % self.n_train)
        return graph, target

    def get_train_batch(
        self, i_range: Iterable[int]
    ) -> tuple[jraph.GraphsTuple, jnp.ndarray]:
        if len(i_range) == 1:
            graphs, targets = self.get_train(i_range[0])

        else:
            graphs = []
            targets = []

            for i in i_range:
                g, t = self.get_train(i)
                graphs.append(g)
                targets.append(t)

            graphs = jraph.batch(graphs)
            graphs = jraph.GraphsTuple(
                nodes=graphs.nodes,
                edges=graphs.edges,
                n_node=graphs.n_node,
                n_edge=graphs.n_edge,
                senders=graphs.senders,
                receivers=graphs.receivers,
                globals={k: v[0] for k, v in graphs.globals.items()},
            )

            targets = jnp.concatenate(targets, axis=0)

        return graphs, targets

    def get_valid(self, i: int) -> tuple[jraph.GraphsTuple, jnp.ndarray]:
        graph, target = self.get(i + self.i_valid)
        return graph, target


def compute_loss(
    params: hk.Params,
    graph: jraph.GraphsTuple,
    targets: jnp.ndarray,
    net: jraph.GraphsTuple,
    loss_norm: str,
) -> jnp.ndarray:
    """
    Computes loss function (l2 or l1)
    Note: Loss units is dx_eq
    """
    preds = net.apply(params, graph)

    if loss_norm == "l1":
        loss = jnp.abs((preds - targets) / graph.globals["dx_eq"])
    elif loss_norm == "l2":
        loss = jnp.square((preds - targets) / graph.globals["dx_eq"])

    return jnp.mean(loss)


def scheduler(
    step_number: int,
    lr_start: float = 1e-4,
    lr_final: float = 1e-6,
    decay_rate: float = 0.1,
    transition_steps: int = 10**6,
) -> float:
    """
    Learning rate scheduler used in Sanchez-Gonzalez et al. (2020)
        https://arxiv.org/abs/2002.09405
    """
    lr = lr_final + (lr_start - lr_final) * decay_rate ** (
        step_number / transition_steps
    )
    return lr


def train(
    data_folder: str,
    model_folder: str,
    net_fn: Callable[[jraph.GraphsTuple], jnp.ndarray],
    net_fn_params: dict,
    train_steps: int,
    i_train: int,
    i_valid: int,
    batch_size: int = 1,
    loss_norm: str = "l2",
    lr_scheduler: bool = False,
    patience: Optional[int] = None,
    weight_decay: float = 0,
    scale_targets: bool = False,
    random_seed: int = 42,
) -> hk.Params:
    """
    Main training loop

    Args:
        data_folder
            Path to folder with graph + target dataset.

        model_folder
            Path to folder where trained model will be saved.

        net_fn
            Function representing NN model.

        net_fn_params
            Dictionary containing net_fn input parameters.

        train_steps
            Number of gradient updates (not epochs).

        i_train
            Ending index of training data. Useful for training in subset.
            If None, the Dataloader sets i_train = i_valid.

        i_valid
            Starting index of validation data.

        batch_size
            Batch size to use.

        loss_norm
            Which norm to use in loss function ("l1" or "l2").

        lr_scheduler
            If true use learning rate scheduler as described in Carvalho et al (2023)

        patience
            Number of epochs allowed without improvement in validation loss

        weight_decay
            Weigth decay value to use with optax.adamw()

        scale_targets
            Scale targets by the training dataset std value.
            Should be used for collisional dynamics.

        random_seed
            Seed to use for model parameter initialization.


    Generated files:
        model_cfg.yml
            Config which stores the input net_fn_params (model architecture).

        train_data.yml
            Stores information about training dataset used.

        loss.txt
            Train + validation loss at the end of each epoch.

        loss_i.txt
            Train loss at each gradient update.

        params_best.pkl
            Network parameters that achieved lowest validation loss.

        params_final.pkl
            Network parameters at the last gradient update.

        stats.yml
            Only generated if scale_targets = True.
            Stores training dataset targets mean and std value.
    """
    train_params = dict(locals())

    # create output dir
    model_folder = Path(model_folder)
    model_folder.mkdir(parents=True, exist_ok=True)

    # save training parameters
    with open(model_folder / "train_cfg.yml", "w", encoding="utf-8") as f:
        yaml.dump(train_params, f)

    # save model config
    with open(model_folder / "model_cfg.yml", "w", encoding="utf-8") as f:
        yaml.dump(net_fn_params, f)

    # copy train dataset config to model folder
    data_folder = Path(data_folder)
    with open(data_folder / "info.yml", "r", encoding="utf-8") as f:
        info = yaml.safe_load(f)

    with open(model_folder / "train_data.yml", "w", encoding="utf-8") as f:
        yaml.dump(info, f)
        del info

    # transform impure `net_fn` to pure functions with hk.transform.
    if net_fn_params is not None:
        net = net_fn(**net_fn_params)
        net = hk.without_apply_rng(hk.transform(net))
    else:
        net = hk.without_apply_rng(hk.transform(net_fn))

    # initialize dataloader
    data = DataLoader(data_folder, i_train, i_valid, scale_targets)
    # if scale_targets = True, save training data stats
    with open(model_folder / "stats.pkl", "wb") as f:
        stats = {"y_mean": float(data.y_mean), "y_std": float(data.y_std)}
        # yaml.dump(stats, f)
        pickle.dump(stats, f)
        print(stats)

    # Get a candidate graph and target to initialize the network.
    graph, _ = data.get(0)
    # Initialize the network.
    params = net.init(jax.random.PRNGKey(random_seed), graph)

    if lr_scheduler:
        # initialize learning rate scheduler
        lr = functools.partial(
            scheduler,
            lr_start=1e-4,
            lr_final=1e-6,
            decay_rate=0.1,
            transition_steps=10**6,
        )
    else:
        lr = 1e-4

    # initialize the optimizer
    if weight_decay != 0:
        opt_init, opt_update = optax.adamw(lr, weight_decay=weight_decay)
    else:
        opt_init, opt_update = optax.adam(lr)

    opt_state = opt_init(params)

    # initializes compute_loss with net architecture
    compute_loss_fn = functools.partial(compute_loss, net=net, loss_norm=loss_norm)
    # jit loss function computation
    compute_loss_train = jax.value_and_grad(compute_loss_fn)
    # similar for valid without gradient calculation
    compute_loss_valid = jax.jit(compute_loss_fn)

    # files where training results will be saved
    model_folder = Path(model_folder)
    log = model_folder / "loss.txt"
    log_raw = model_folder / "loss_i.txt"
    best_val_loss = jnp.inf
    best_params_pkl = Path(model_folder) / "params_best.pkl"
    final_params_pkl = Path(model_folder) / "params_final.pkl"

    with open(log, "w", encoding="utf-8") as f:
        f.write("train valid\n")

    with open(log_raw, "w", encoding="utf-8") as f:
        f.write("train\n")

    # util functions for train / valid iterations
    @jax.jit
    def train_step(graph, target, params, opt_state):
        loss, grad = compute_loss_train(params, graph, target)
        updates, opt_state = opt_update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state

    def valid_step(params):
        val_loss = 0
        for i in range(data.n_valid):
            graph, target = data.get_valid(i)
            val_loss += compute_loss_valid(params, graph, target)
        val_loss /= data.n_valid
        return val_loss

    loss_acc = 0
    no_loss_improvement = 0

    # training loop
    for idx in range(int(train_steps)):
        graph, target = data.get_train_batch(
            np.arange(idx * batch_size, (idx + 1) * batch_size)
        )

        loss, params, opt_state = train_step(graph, target, params, opt_state)
        loss_acc = loss_acc + loss

        # compute validation after going through the full training set
        if (idx * batch_size) % data.n_train >= data.n_train - batch_size:
            loss_acc = loss_acc / data.n_train
            val_loss = valid_step(params)

            with open(log, "a", encoding="utf-8") as f:
                f.write(f"{loss_acc} {val_loss}\n")

            # save best valid iteration
            if val_loss < best_val_loss:
                with open(best_params_pkl, "wb") as f:
                    pickle.dump(params, f)

                best_val_loss = val_loss
                no_loss_improvement = 0

            # early stop condition
            elif patience is not None:
                if no_loss_improvement + 1 == patience:
                    break
                else:
                    no_loss_improvement += 1

            print(f"step: {idx}, loss: {loss_acc:.4f}, val_loss: {val_loss:.4f}")

            loss_acc = 0

        else:
            # uncomment for printing error per gradient update
            print(f"step: {idx}, loss: {loss:.4f}")
            # pass

        with open(log_raw, "a", encoding="utf-8") as f:
            f.write(f"{loss}\n")

    with open(final_params_pkl, "wb") as f:
        pickle.dump(params, f)

    print("Training finished")

    return params


def evaluate(
    data_folder: str,
    model_folder: str,
    net_fn: Callable[[jraph.GraphsTuple], jraph.GraphsTuple],
    loss_norm: str,
    i_range: Iterable,
) -> float:
    """
    Compute loss function of pre-trained model in a dataset of choice.

    Args:
        data_folder
            Path to folder with graph + target dataset.

        model_folder
            Path to folder with pre-trained model.

        net_fn
            Function representing NN model.

        loss_norm
            Which norm to use for the loss function ("l1" or "l2").

        i_range
            List of dataset instance indices to use.

    Returns:
        loss
            Average loss.
    """
    model_folder = Path(model_folder)

    with open(model_folder / "model_cfg.yml", "r", encoding="utf-8") as f:
        net_fn_params = yaml.safe_load(f)

    with open(model_folder / "params_best.pkl", "rb") as f:
        net_params = pickle.load(f)

    net = net_fn(**net_fn_params)
    net = hk.without_apply_rng(hk.transform(net))

    compute_loss_fn = functools.partial(compute_loss, net=net, loss_norm=loss_norm)

    loss = 0
    data = DataLoader(data_folder)

    for i in i_range:
        graph, target = data.get(i)
        loss += jax.jit(compute_loss_fn)(net_params, graph, target)

    loss /= len(i_range)

    return loss


def plot_train(path: str) -> None:
    """
    Plots train + validation loss (one point per epoch).
    """
    loss = pd.read_csv(Path(path) / "loss.txt", delimiter=" ")

    print(
        f"Min Train: {np.min(loss['train'][1:])} , Min Valid: {np.min(loss['valid'])}"
    )

    plt.figure(figsize=(15, 7))
    plt.plot(loss["train"], label="train")
    plt.plot(loss["valid"], label="valid")

    aux = np.arange(np.argmin(loss["valid"]) + 1)
    plt.plot(aux, np.min(loss["valid"]) * np.ones(len(aux)), "k--")
    aux = np.arange(np.argmin(loss["train"][1:]) + 2)
    plt.plot(aux, np.min(loss["train"][1:]) * np.ones(len(aux)), "k--")

    plt.yscale("log")

    y_min = np.floor(np.log10(np.max([loss[["train", "valid"]].min().min(), 1e-8])))
    y_max = np.floor(np.log10(loss[["train", "valid"]].max().max())) + 1
    plt.ylim([10**y_min, 10**y_max])
    plt.xlim([0, len(loss["train"]) - 1])
    plt.legend()
    plt.xlabel(r"#epochs")
    plt.ylabel("Loss")

    plt.show()


def plot_train_multi(paths: List[str]) -> None:
    """
    Plots validation loss for multiple models in a single figure.
    """
    plt.figure(figsize=(15, 7))
    x_max = 0
    y_max = 0
    y_min = np.inf

    for path in paths:
        loss = pd.read_csv(Path(path) / "loss.txt", delimiter=" ")

        plt.plot(loss["valid"], label=path)
        x_max = max(x_max, len(loss))
        y_max = max(y_max, loss["valid"].max())
        y_min = min(y_min, loss["valid"].min())

    y_min = np.floor(np.log10(np.max([y_min, 1e-8])))
    y_max = np.floor(np.log10(y_max)) + 1

    plt.yscale("log")
    plt.ylim([10**y_min, 10**y_max])
    plt.xlim([0, x_max - 1])
    plt.legend()
    plt.xlabel(r"\#epochs")
    plt.ylabel("Loss")
    plt.show()


def plot_train_raw(path: str) -> None:
    """
    Plots training loss at each gradient update step.
    """
    loss = pd.read_csv(Path(path) / "loss_i.txt", delimiter=" ")
    plt.figure(figsize=(15, 7))
    plt.plot(loss["train"], label="train")

    plt.yscale("log")
    y_min = np.floor(np.log10(np.max([np.min(loss["train"]), 1e-8])))
    y_max = np.floor(np.log10(np.max(loss["train"]))) + 1
    plt.ylim([10**y_min, 10**y_max])
    plt.xlim([0, len(loss["train"])])
    plt.legend()
    plt.xlabel(r"\#gradient updates")
    plt.ylabel("Loss")
    plt.show()
