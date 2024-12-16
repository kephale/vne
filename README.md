[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![vne](https://github.com/quantumjot/vne/actions/workflows/test.yml/badge.svg)](https://github.com/quantumjot/vne/actions/workflows/test.yml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

# vne

von Neumman's Elephant

"With four parameters I can fit an elephant, and with five I can make him wiggle his trunk."

### Installation

```sh
conda create -n vne python=3.8
conda activate vne
git clone https://github.com/quantumjot/vne.git
cd vne
```

#### Inference only
```sh
pip install -e .
```

#### Training/Development
```sh
pip install -e ".[dev]"
```

Launch mlflow server:

```
mlflow server --host 127.0.0.1 --port 5000
```

ddp training
```
torchrun --nproc_per_node=4 examples/train_mlc_ddp.py --num_nodes 1 --devices 4 --strategy ddp
```