# SpecRepair / A Robust Optimisation Perspective on Counterexample-Guided Repair of Neural Networks
SpecRepair is a tool for neural network repair. 
It aims at creating neural networks with mathematical safety guarantees.
Safety guarantees are mandatory for applying neural networks in safety critical domains.
To obtain such guarantees, 
SpecRepair uses the [ERAN verifier](https://github.com/eth-sri/eran/).
Repair is based on constrained optimisation using the L1 penalty function method,
leveraging neural network training algorithms.
For more information on SpecRepair, see
[SPIN 2022](https://doi.org/10.1007/978-3-031-15077-7_5)
or [arXiv](https://arxiv.org/abs/2106.01917).

This repository contains the source code for the publication 
["A Robust Optimisation Perspective on Counterexample-Guided Repair of Neural Networks"](https://arxiv.org/abs/2301.11342)
by David Boetius, Stefan Leue and Tobias Sutter, accepted at ICML 2023.
You can download the experimental data presented in this paper
from [Zenodo](https://doi.org/10.5281/zenodo.7938547).
```bibtex
@Article{BoetiusLeueSutter2023,
  author = {David Boetius and Stefan Leue and Tobias Sutter},
  title = {A Robust Optimisation Perspective on Counterexample-Guided Repair of Neural Networks},
  journal = {CoRR},
  volume = {abs/2301.11342},
  year = {2023},
  doi = {10.48550/arXiv.2301.11342},
  note = {accepted for ICML 2023}
}
```

**Naming:** For historical reasons, SpecAttack is called *DeepOpt* in this 
repository and SpecRepair is called *L1 penalty repair*.
For more information on SpecRepair and SpecAttack, see 
[SPIN 2022](https://doi.org/10.1007/978-3-031-15077-7_5)
or [arXiv](https://arxiv.org/abs/2106.01917).

## Prerequisites
This project is tested on Ubuntu 2022.04 LTS.
To get started, 
you need an installation of [Ubuntu 2022.04](https://ubuntu.com/#download).
If you want to make use of a CUDA device (GPU), you need to install the 
NVIDIA drivers.
Check that the `nvidia-smi` command line tool is available.

Later on, you also need a [Gurobi license](https://www.gurobi.com/).
Academic users can obtain a free academic license from www.gurobi.com.
The license needs to be placed in `$HOME/gurobi.lic`.
More on this below.

## Installation
In Ubuntu 2022.04, start a terminal, navigate to the folder where
this README is placed and run
```shell
./setup.sh
```
This automatically installs system dependencies and
creates a new python environment in `./env-nn-repair`. 
The script also install the ERAN verifier. 
To pull dependencies, the script requires an internet connection.
Also, the script queries you for sudo rights to install system dependencies.
Alternatively, you may go through the script and execute each command manually.

The `setup.sh` script checks whether `nvidia-smi` is available, to determine
whether a CUDA device is available.
When it finds `nvidia-smi`, it automatically installs the GPU version
of ERAN and PyTorch. 

### Obtaining a Gurobi License
For running the ERAN verifier, you need a Gurobi license.
First, register with Gurobi and request a new academic license (if you are eligible) 
as described [here](https://www.gurobi.com/features/academic-named-user-license/).
The Gurobi site will display you a license key, that you can use to
download a license using `grbgetkey`.
The `setup.sh` script installed the `grbgetkey` script as part of ERAN.
To use it, do the following
```shell
cd eran/gurobi*/linux64/  # navigate to the gurobi installation
# the Gurobi installation is owned by the root user currently
export NOT_ROOT_USER="$USER"
sudo chown -R "$NOT_ROOT_USER" .
cd bin/
# Now you can run grbgetkey
# keep the defaults, so that the license is installed in $HOME/gurobi.lic. 
# setup.sh assumes that location.
./grbgetkey YOUR_LICESE_KEY  
```

## Running the Experiments
Activate the new virtual environment, navigate to the `experiments/` directory
and run one of the shell scripts.
```shell
source ./env-nn-repair/bin/activate
cd experiments/
./mnist_early_exit_1.sh  # or another experiment
```
There are several experiments in the `experiments/` directory:
 * acasxu_early_exit_1.sh - Runs the optimal vs. early-exit and falsifier experiments using ACAS Xu
 * collision_detection_early_exit_3.sh - Runs the optimal vs. early-exit experiments using CollisionDetection
 * mnist_early_exit_1.sh - Runs the optimal vs. early-exit and falsifier experiments using MNIST
 * ouroboros_rmi_early_exit_1.sh - Runs the optimal vs. early-exit experiments using the first stage RMI networks
 * ouroboros_rmi_compare_1.sh - Runs the linear regression repair experiments using SpecRepair and Ouroboros

All experiments place their output in `output/`. 
The networks before repair are contained in `resources/`.
The `scipts/` directory contains code for training networks,
for interpreting the experiments and the Quadratic Programming RMI experiment
 * ouroboros_rmi_second_stage_repair_quadratic_programming.sh - Runs the linear regression repair experiments using Quadratic Programming

Each experiment generates a logfile, counterexample, and network checkpoints
and potentially a repaired network. 
Most information is contained in the logfile.
To extract and aggregate this information, run the `collec_experiment_results.py`
script in `scripts/`.
For experiments that perform multiple repetitions, in particular MNIST, 
CollisionDetection, and repairing the first-stage RMI networks,
you also need to aggregate the repetitions.
For example,
```shell
# We are currently in experiments/ and mnist_early_exit_1.sh has finished running
# mnist_early_exit_1.sh has created many directories in output/mnist_repair_1
# that each have a timestamp as name
cd ../scripts
# The python virtual environment is assumed to be still active
# This command aggregates the several repetitions of each experiment
# For ACAS Xu, you can skip this step
python select_by_median_runtime.py \
  --experiment_dirs ../output/mnist_repair_1/* \  # we use globs to include all experiment output directories
  --group_by falsifiers verifier_exit_mode \  # group runs by these command line arguments of the experiments/mnist_repair_1.py script
  --output_path ../output/mnist/repair_1/ \  # where to put the aggregated data
  --output_prefix collected  # a name prefix for directories containing the aggregated data
# This prints some statistics about the experiments and emits warnings.
# The number of warnings are summarised in the last output lines.
# Check the warnings to see whether your experiments ran alright (some variation is expected).
# Now we use collect_experiment_results.py to extract data from the aggregation directories.
python collect_experiment_results.py \
  --counterexample_violation_at_generation \  # useful to compare falsifiers/verifiers
  --count_introduced_counterexamples \  # also nice to know, there are more options what to extract
  --detailed_runtimes \  # these three options are necessary for later analysis
  ../output/mnist_repair_1/collected_optimal/  # one of the aggregation directories created before
# run the above command for the other aggregation directories.
# For ACAS Xu, directly run the above command on the time-stamped output directories
# created by experiments/acasxu_early_exit_1.sh.
# With the results, collected, we can plot the data and interpret the results
python early_exit_cactus_plots.py \  # does more than just cactus plots
  --optimal ../output/mnist_repair_1/collected_optimal \
  --early_exit ../output/mnist_repair_1/collected_early_exit \
  ...
# This script can also export the data for visualisation using
# the --export and --output_dir command line options.
```

To analyse the ACAS Xu results, `scripts/` contains a few more scripts
that are used to analyse experiments:
 * acasxu_calculate_fidelity.py - computes accuracy fidelity and MAE fidelity
 * make_acasxu_table.py - create a LaTeX table containing listing the results of an ACAS Xu experiment
 * plot_acasxu_fidelity.py - create plots of fidelity results, similarly to some plots of early_exit_cactus_plots

For `experiments/ouroboros_compare_1.sh`, 
don't run `scripts/collect_experiment_results.py`.
Instead, use the much faster shell script `scripts/collect_broad.sh`.
This is the only script you need to run to collect the results of this experiment.

## Source Code
Most source code is in `nn_repair/`. 
SpecAttack is found in the `deep-opt/` directory.
The modified ERAN verifier is in `eran/`

## Troubleshooting
If an experiments fails when running ERAN with an error from
multiprocessing that pickling a 'PyCapsule' is not supported, 
something similar to this:
```
<multiprocessing.pool.ExceptionWithTraceback object at 0x123456789abc>'. 
Reason: 'TypeError("cannot pickle 'PyCapsule' object")
```
then this is most likely because the GRB_LICENSE_FILE environment variable isn't set. 
In this case, Gurobi could not find a license, but the precise
error message is eaten up by multiprocessing.
To fix the issue, set the GRB_LICENSE_FILE or provide a Gurobi license
in `$HOME/gurobi.lic`.
The `setup.sh` script sets GRB_LICENSE_FILE to that location.

Experiments may fail due to a "verification problem" when they run close to the
timeout. 
In that case, ERAN reports an error, but doesn't provide an error message.
It's likely in such a situation that ERAN was interrupted by the timeout
mechanism in `experiments/experiment_base`, which uses an exception.
Because PyTorch data loaders may also eat up these exceptions, there is a 
second separate timeout mechanism in the experiments.

## Licensing
The code contained in this repository is licensed at the terms of the 
Apache-2.0 license (see LICENSE file), unless stated otherwise. 

Some original networks (ACAS Xu, MNIST, CIFAR10) come from different repositories. 
The repaired networks may constitute derived works of these networks. 
The directories (or a super-directory) which contain such networks contain a 
COPYRIGHT file that names the original copyright holder of the network
and an attribution template. 

All networks prefixed with 'eran_' are original networks from the 
[ERAN repository](https://github.com/eth-sri/eran).
The copyright holder of these networks is the Secure, Reliable, and 
Intelligent Systems Lab (SRI), Department of Computer Science ETH Zurich 
(Copyright 2020). 

All original ACAS Xu networks are copies of the networks used for Reluplex. 
(https://arxiv.org/abs/1702.01135, https://github.com/guykatzz/ReluplexCav2017): 

   G. Katz, C. Barrett, D. Dill, K. Julian and M. Kochenderfer. 
   Reluplex: An Efficient SMT Solver for Verifying
   Deep Neural Networks. Proc. 29th Int. Conf. on Computer Aided
   Verification (CAV). Heidelberg, Germany, July 2017. 

These networks are licensed under the Creative Commons Attribution 4.0 
International License.
The copyright holders of these networks are the authors listed 
above and Stanford University (2016-2017).
Repaired networks may constitute derived works of these networks.

Before reusing any repaired network, 
check the experiment code to find out about the original network 
that is being repaired. 
The repaired network may constitute derived works of networks from 
different sources such as ERAN, that need to be attributed. 
