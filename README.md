# gns-sheet-model


This repository contains the Graph Network Simulator for the 1D Plasma Electrostatic Sheet Model developed in D. Carvalho _et al._ ["Learning the dynamics of a one-dimensional plasma model with graph neural networks"](https://arxiv.org/abs/2310.17646).

The Graph Network Simulator is inspired in the work of A. Sanchez-Gonzalez _et al._ 
["Learning to Simulate Complex Physics Systems with Graph Networks"](https://arxiv.org/abs/2002.09405) and is implemented using [JAX](https://github.com/google/jax), [Optax](https://github.com/google-deepmind/optax), [Haiku](https://github.com/google-deepmind/dm-haiku), and [Jraph](https://github.com/deepmind/jraph).

The physics simulator used to generate the training data is based on the 1D Electrostatic Sheet Model proposed by J. Dawson in 
["One-Dimensional Plasma Model"](https://aip.scitation.org/doi/pdf/10.1063/1.1706638) (1962) which is here re-implemented in Python. We include in this repository the synchronous and asynchronous version of the algorithm as described in D. Carvalho _et al._ (2023) [Appendix A](https://arxiv.org/abs/2310.17646).


![Hello](notebooks/img/twostream_comparison.gif)
*Example of simulator capabilities: Two-stream instability in the cold-beam regime using the Sheet Model (SM) and the Graph Network Simulation (GNS).*

**Contents**:
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Available Datasets and Pretrained Models](#available-datasets-and-pretrained-models)
- [Quickstart](#quickstart)
  - [Tutorials](#tutorials)
  - [Reproducing Paper Results](#reproducing-paper-results)
  - [Training New Models](#training-new-models)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)
---

## Directory Structure

- ```config/``` - contains configuration files for GNN architecture parameters
- ```data/``` - directory where datasets are stored by default
- ```models/``` - directory where trained models are stored by default
- ```gns/``` - code relative to the Graph Network Simulator
- ```sheet_model/``` - code relative to the different Sheet Model implementations (synchronous, modified, and asynchronous)
- ```scripts/``` - example scripts on how to create datasets + train different types of models (equivalent to those shown in the paper)
- ```notebooks/``` - Jupyter Notebooks containing tutorials, benchmarks, and the study of different plasma kinetic phenomena


## Installation

To use the code, you must first clone the git repository.

```
git clone https://github.com/diogodcarvalho/dawgnnson-sheet-model.git
```

We recommend creating a new virtual environment to avoid issues with previously installed Python packages.

This can be done for e.g. with:

```
cd dawgnnson-sheet-model/
python -m venv .venv/gns
source .venv/gns/bin/activate
```

Installing the code + mandatory dependencies will require ~ 2GB of disk space.  

#### Installing JAX

An older version of JAX and jaxlib (0.3.25) must be installed to use the pre-trained models. 

For the CPU version, this can be done using:

```
pip install --upgrade pip
pip install jax==0.3.25 -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install jaxlib==0.3.25 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

For GPU support please refer to the [official documentation](https://jax.readthedocs.io/en/latest/installation.html).
Using GPUs significantly decreases the code run-time.

If you do not wish to run the pre-trained models, the latest releases can be used.

#### Installing other dependencies

This step should only be done **after** installing the desired JAX version since otherwise, some dependencies will automatically use the latest version.

The majority of the project dependencies can be quickly installed with:

```
pip install -r requirements.txt
```

**Optional**

To produce figures using LaTeX, one needs to install the base version as well as some additional packages.

```
sudo apt-get install dvipng texlive-latex-extra texlive-fonts-recommended cm-super
```

## Available Datasets and Pretrained Models

The training and test data plus model weights produced in our work are available in a [Zenodo repository](https://doi.org/10.5281/zenodo.10440186). They occupy approximately ~7GB of disk space once extracted.

They can be automatically downloaded and unzipped using an auxiliary script we provide:

```
chmod +x scripts/download_files.sh
./scripts/download_files.sh
```

If instead, you download the files manually, we recommend storing the extracted folders in the ```data/``` and ```models/``` directories such that the provided notebooks and scripts can be run without requiring any changes.

## Quickstart

### Tutorials

After the installation (+ downloading available datasets / pre-trained models if desired) we recommend an initial look at the:
- [Tutorial Notebooks](notebooks/tutorials) which contain examples of how to perform simulations with the Sheet Model and pre-trained GNS models.
- [Auxiliary Scripts](scripts) which show how to generate small datasets and train equivalent models to those shown in the original paper.


### Reproducing Paper Results

After downloading the available datasets and pre-trained models, you should be able to replicate all the results present in the main body of the paper by running the notebooks provided in [notebooks/benchmarks](notebooks/benchmarks) and [notebooks/examples](notebooks/examples). Run-time differences might be observed depending on the hardware + software version.

To reproduce the training pipelines we also provide several [scripts](scripts) that train new (equivalent) models on the training dataset provided.

Any of these scripts can be run using:
```
chmod +x scripts/reproduce_{chosen file}.sh
./scripts/reproduce_{chosen file}.sh
```
However, keep in mind that each model will require several hours to train (and each script trains several serially).



### Training New Models

To train a new GNN model the subsequent steps must be followed:

1. Generate a dataset of simulations with the a/synchronous sheet model (use --help for detailed options):

```
$ python3 generate_dataset_sync.py [options]
```
```
$ python3 generate_dataset_async.py [options]
```

2. Generate pairs of graphs/targets from a dataset of simulations 

```
$ python3 process_dataset.py [options]
```

3. Train GNN

```
$ python3 train_gnn.py [options]
```

4. Examples of how to quickly use the new GNN are presented in the companion [notebooks](notebooks).

---

### Acknowledgments

This work was supported by the FCT - Fundação para a Ciência e Tecnologia, I.P. under the Project No 2022.02230.PTDC (X-MASER) and PhD Fellowship Grant 2022.13261.BD and has received funding from the European Union’s H2020 programme through the project IMPULSE (grant agreement No 871161).

We would also like to thank [@OsAmaro](https://github.com/OsAmaro) for beta-testing the code and welcome further contributions and feedback from the community!

### Citation

If you use the code, consider citing our [paper](https://arxiv.org/abs/2310.17646):

```
@article{carvalho2023graph,
  title={Learning the dynamics of a one-dimensional plasma model with graph neural networks},
  author={Carvalho, Diogo D and Ferreira, Diogo R and Silva, Luis O},
  journal={arXiv preprint arXiv:2310.17646},
  year={2023}
}
```
