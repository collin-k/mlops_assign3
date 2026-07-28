# MLOps Assignment 3 - AutoML

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place `athletes.csv` under `data/` if it is not already there.

## Reproduce

```bash
python preprocess.py
jupyter lab notebooks/01_tpot_automl.ipynb
jupyter lab notebooks/02_h2o_automl.ipynb
```
