#!/usr/bin/bash

# project directory
export HOMER_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# modules
ml purge
ml cuda/12.2 cudnn/11.6 anaconda/3.0

# conda
eval "$(conda shell.bash hook)"
conda activate homer

# auto completion
eval "$(python main.py -sc install=bash)"

# resolve multithreading conflict
export MKL_THREADING_LAYER=GNU
