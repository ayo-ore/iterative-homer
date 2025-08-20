# HOMER

A re-implementation of the HOMER method for reweighting hadronization histories [[2410.06342]](https://arxiv.org/abs/2410.06342). The original codebase, which includes data generation scripts can be found at the [MLHad GitLab](https://gitlab.com/uchep/mlhad/-/tree/master/HOMER?ref_type=heads).

## Getting started
- Clone the repository
- Create the conda environment:
```
conda env create -f env.yaml
```

## Basic usage
This project uses [Hydra](https://hydra.cc/docs/intro/) to configure experiments. The default settings are given in `config/default.yaml` and each can be overridden via the command line.

The script `homer.py` is used to run experiments, which typically consist of training/evaluating a model and making plots. The all-in-one `IterativeExperiment` runs steps 1 and 2 of the HOMER method in iterations. To launch the experiment with default settings, simply use:
```
python homer.py -cn iterative
```
In the above, `-cn` is short for `--config_name`.

Experiment settings can be adjusted from the command line. For example, to set the number of iterations to 3, use:
```
python homer.py -cn iterative iterations=3
```
Each step of Homer can also be run in isolation:
```
python homer.py -cn step_one/hl
```
The second step requires you to specify the location of a step-one classifier:
```
python homer.py -cn step_two/deterministic w_class_path=/path/to/step/one/exp
```

<!-- The following is a description of the two main experiments:
| Experiment name | Description | Compatible networks |
| :-------- | :------- | :------- |
| `step_one` | Trains a network in a self-supervised manner in order to learn informative lightcone summaries. | `vit` |
| `step_two` | Trains a network to predict simulation parameters given a light cone. The model performance is measured in terms of a normalized -->

## Running on a cluster
Submission options for `slurm` and `pbs` schedulers are integrated. As an example, one can submit a job to slurm (the default option) using
```
python homer.py ... submit=True
```
Most important options can be specified:
```
python homer.py ... cluster.queue=gshort cluster.time=2:0:0 cluster.mem=8G
```

## Continuing an experiment
One often needs to re-run a previous experiment. This can be achieved simply from the command line. Common examples include:

- Continuing training from a saved checkpoint:
```
python homer.py prev_exp_dir=/path/to/prev/exp training.warm_start=True  
```
- Repeating evaluation and/or plotting using a saved model:
```
python3 homer.py prev_exp_dir=/path/to/prev/exp train=False
python3 homer.py prev_exp_dir=/path/to/prev/exp train=False evaluate=False 
```
The specific configuration will be loaded from the previous experiment. Command line overrides are also applied.

## Further settings
The following is a description of parameters in `config/default.yaml` that may not be self explanatory or can affect performance.
| Parameter name | Description |
| :------- | :------- |
| `use_tf32` | Whether to use the TensorFloat32 format where possible. Setting to `True` may increase performance.|
| `training.use_amp` | Whether to use automatic mixed precision during training. Setting to `True` may increase performance.|
| ... | |
