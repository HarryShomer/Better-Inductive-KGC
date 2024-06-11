# Installation

We detail the installation process using Conda on Linux. Note that all environments used can be found in the `envs` directory.


## 1. Install Conda
```
curl -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

## 2. Setup Code and Packages for Most Experiments

The system and CUDA version used for the majority of experiments are:
- Ubuntu 20.04.6 LTS
- CUDA 11.6
- Python 3.9.13

We first clone the repository.
```
git clone git@github.com:HarryShomer/KGC-Inductive-PPR.git
cd KGC-Inductive-PPR/
```

The package requirements can be found in the `envs/std_env.yml` file. Installing this will also create an environment for the project, `std_env`. 
```
# Install environment requirements
conda env create -f envs/std_env.yml   

# Activate environment
conda activate std_env
```

## 2. Setup Code and Packages for Rest of Experiments

ULTRA and Nodepiece require different environments. We note that the only difference from the previous section is the python version and the package versions.

Nodepiece requires python 3.8.19. The full package requirements can be found in `nodepiece_env.yaml`.

ULTRA requires python 3.9.19. The full package requirements can be found in `ultra_env.yaml`.

These environments can be set up as detailed in the previous section.

We note that the correct CUDA version can be installed multiple ways. For manual installation, please see [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html). On a HPCC system, the correct version can be ativated via SLURM. Please see [here](https://hpcf.umbc.edu/gpu/how-to-run-on-the-gpus/) for more details.
