import re
import signal
from typing import Sequence, Tuple, Callable, Final, Optional, Union, Dict

import logging
import logging.handlers
from logging import info, warning
import sys
import os
import platform

import psutil
import GPUtil
from datetime import datetime
from pathlib import Path
import dill

import torch
import torch.utils.tensorboard
import tensorflow as tf
import numpy as np
import random
import argparse
from collections import OrderedDict

import ray
import multiprocessing as mp

from deep_opt import NeuralNetwork, Property, CounterexampleAmount
from nn_repair import repair_network, RepairNetworkDelegate, CounterexampleGenerator
from nn_repair.falsifiers import DeepOpt, FastGradientSignMethod, ProjectedGradientDescentAttack, \
    DifferentialEvolutionPGDAttack
from nn_repair.training import TrainingLoop, TensorboardLossPlot, Checkpointing
from nn_repair.training import TrainingLoopHook, TrainingLoopHookCallLocation
from nn_repair.utils.timing import LogExecutionTime
from nn_repair.verifiers import ERAN
from nn_repair.verifiers.linear_regression_verifier import LinearRegressionVerifier

try:
    from nn_repair.verifiers import Marabou
except ImportError:
    pass

QUADRATIC_PENALTY_KEY: Final[str] = 'quadratic_penalty'
L1_PENALTY_KEY: Final[str] = 'l1_penalty'
L1_PLUS_QUADRATIC_PENALTY_KEY: Final[str] = 'l1_plus_quadratic_penalty'
AUGMENTED_LAGRANGIAN_KEY: Final[str] = 'augmented_lagrangian'
LN_BARRIER_KEY: Final[str] = 'ln_barrier'
RECIPROCAL_BARRIER_KEY: Final[str] = 'reciprocal_barrier'
L1_PENALTY_RECIPROCAL_BARRIER_KEY: Final[str] = 'l1_penalty_and_reciprocal_barrier'
DATASET_AUGMENTATION_KEY: Final[str] = 'dataset_augmentation'
FINE_TUNING_KEY: Final[str] = 'fine_tuning'
LP_MINIMAL_MODIFICATION_KEY: Final[str] = 'linear_programming_minimal_modification'
MARABOU_MINIMAL_MODIFICATION_KEY: Final[str] = 'marabou_minimal_modification'
DL2_KEY: Final[str] = 'dl2'
NEURON_FIXATION_KEY: Final[str] = 'neuron_fixation'
LINEAR_REGRESSION_DATASET_AUGMENTATION_KEY: Final[str] = 'linear_regression_dataset_augmentation'

BACKEND_OPTIONS = (
    QUADRATIC_PENALTY_KEY, L1_PENALTY_KEY, L1_PLUS_QUADRATIC_PENALTY_KEY, AUGMENTED_LAGRANGIAN_KEY, LN_BARRIER_KEY,
    RECIPROCAL_BARRIER_KEY, L1_PENALTY_RECIPROCAL_BARRIER_KEY, DATASET_AUGMENTATION_KEY, FINE_TUNING_KEY,
    LP_MINIMAL_MODIFICATION_KEY, MARABOU_MINIMAL_MODIFICATION_KEY, DL2_KEY, NEURON_FIXATION_KEY,
    LINEAR_REGRESSION_DATASET_AUGMENTATION_KEY,
)


def seed_rngs(seed=0):
    """
    Seed the Random Number Generators (RNGs) of pytorch, tensorflow, numpy, and random.

    This method uses the passed seed for pytorch, the seed + 1 for tensorflow,
    seed + 2 for numpy and seed + 3 for random.

    :param seed: The seed to use for the RNGs.
      The seed is incremented by small numbers for different libraries.
    """
    torch.manual_seed(seed)
    tf.random.set_seed(seed + 1)
    np.random.seed((seed + 2) % 2 ** 32)
    random.seed((seed + 3) % 2 ** 32)


class ExperimentBase:
    """
    The skeleton for many repair experiments.
    """

    @staticmethod
    def argparse_add_arguments(parser: argparse.ArgumentParser, supported_repair_backends: Sequence[str],
                               default_falsifier_cascade='DeepOpt', default_verifier='ERAN',
                               checkpoint_options: bool = True, restore_options: bool = True,
                               logging_options: bool = True, repair_network_options: bool = True,
                               experiment_execution_options: bool = True, backend_choice: bool = True):
        """
        Adds the command line arguments supported by experiment_base to the given argparse ArgumentParser

        The various xyz_options arguments allow omitting certain argument groups from the command line arguments.
        """
        if checkpoint_options:
            checkpoints_group = parser.add_argument_group('checkpoints')
            checkpoints_group.add_argument('--save_checkpoints', action='store_true', dest='save_checkpoints',
                                           help='Whether to save checkpoints '
                                                '(intermediate modified networks and counterexamples) '
                                                'on repair_network level.')
            checkpoints_group.add_argument('--save_training_checkpoints', dest='save_training_checkpoints',
                                           action='store_true',
                                           help='Whether to store intermediate model states during training.')
            # Not implemented:
            # checkpoints_group.add_argument('--save_backend_checkpoints', dest='save_backend_checkpoints',
            #                                action='store_true',
            #                                help='Whether to store checkpoints of intermediate backend states.'
            #                                     'This is only available for some backends. ')

        if restore_options:
            restore_group = parser.add_argument_group('restoring from a checkpoint')
            restore_group.add_argument('--restore_from', dest='restore_timestamp', default=None,
                                       help='Restore a checkpoint (counterexamples and/or modified network) '
                                            'from the run at this time. This is the name of the output directory.')
            restore_group.add_argument('--restore_iteration', dest='restore_iteration', default=0, type=int,
                                       help='The iteration of which the checkpoint should be restored. ')
            restore_group.add_argument('--restore_only_counterexamples', action='store_true', dest='restore_only_cx',
                                       help='Whether to restore the modified network and the counterexamples from the '
                                            'checkpoint or only the counterexamples.')

        if logging_options:
            log_group = parser.add_argument_group('logging')
            log_group.add_argument('--log_level', dest='log_level', default='DEBUG', type=str,
                                   help='Control the amount of log messages displayed and stored in the log file. '
                                        'Possible values: DEBUG (most messages), INFO, WARNING, ERROR '
                                        '(fewest messages). Alternatively you may also use an int value '
                                        'to specify the log level. Check the logging module for more information.')
            log_group.add_argument('--tensorboard', action='store_true',
                                   help='Visualise the training loss using tensorboard.')
            log_group.add_argument('--preliminary', action='store_true',
                                   help='Put the outputs of this experiment in an extra prelim directory inside '
                                        'the experiment output directory.')

        if repair_network_options:
            repair_network_group = parser.add_argument_group('repair_network')
            repair_network_group.add_argument('--max_repair_steps', dest='max_repair_steps', default=None, type=int,
                                              help='The maximum number of repair steps to perform. '
                                                   'If this option is not present, no upper limit is enforced.')
            repair_network_group.add_argument('--log_execution_times', dest='log_execution_times',
                                              action='store_true',
                                              help='Whether to measure and log the execution times of falsifiers, '
                                                   'verifiers and the repair backend')
            repair_network_group.add_argument('--falsifiers', dest='falsifiers', type=str,
                                              help='Specify a falsifier cascade by providing a list of falsifier names,'
                                                   'separated by semicolon (";") or periods ("."). '
                                                   'A semicolon express that falsification will be continued even if '
                                                   'the previous falsifier found a counterexample. '
                                                   'Periods express that falsification will stop in such a case.\n'
                                                   'Options:\n'
                                                   '  - DeepOpt\n'
                                                   '  - DeepOpt_single (generates only a single counterexample)\n'
                                                   '  - DeepOpt_cutoff (applies cutoff inside DeepOpt)\n'
                                                   '  - DeepOpt_cutoff_single (both of the above)\n'
                                                   '  - FGSM\n'
                                                   '  - PGD[SDG,RESTARTS] (here RESTARTS has to be replaced by the '
                                                   'number of restarts to perform)\n'
                                                   '  - PGD[RMSprop,RESTARTS] (see above)\n'
                                                   '  - PGD[Adam,RESTARTS] (see above)\n'
                                                   '  - PGD_single[...] (options in square brackets like for PGD)\n'
                                                   '  - DEA[SGD,POPULATION_SIZE,ITERATIONS] '
                                                   '(replace POPULATION_SIZE and ITERATIONS with '
                                                   'the respective values you want to use for these parameters in '
                                                   'the differential evolution PGD attack.\n'
                                                   '  - DEA[RMSprop,POPULATION_SIZE,ITERATIONS] (see above)\n'
                                                   '  - DEA[Adam,POPULATION_SIZE,ITERATIONS] (see above)\n'
                                                   'Example: "--falsifiers FGSM;PGD[Adam,10].DeepOpt" will create a '
                                                   'cascade of falsifiers starting with applying FGSM. '
                                                   'Regardless of whether FGSM finds counterexamples, '
                                                   'PGD with 10 restarts and Adam as optimiser is also applied. '
                                                   'Only if FGSM and PGD have not found any counterexamples, DeepOpt is'
                                                   'applied.', default=default_falsifier_cascade,)
            repair_network_group.add_argument('--verifier', dest='verifier', type=str, default=default_verifier,
                                              help='Specify the verifier used for network repair.\n'
                                                   'Options:\n'
                                                   '  - ERAN (acasxu style ERAN verification)\n'
                                                   '  - ERAN_single (generates only a single counterexample)\n'
                                                   '  - ERAN_plain (plain style ERAN verification)\n'
                                                   '  - ERAN_plain_single (plain style single counterexample '
                                                   'generation)\n'
                                                   '  - Marabou\n'
                                                   '  - LinearRegressionVerifier\n'
                                                   '  - LinearRegressionVerifier_single (generates only a single counterexamle)\n'
                                                   '  - None\n'
                                                   'Specify "--verifier None" to disable verification. Please note'
                                                   'that the default value may be overwritten by the experiment.')
            repair_network_group.add_argument('--verifier_exit_mode', dest="verifier_exit_mode", type=str,
                                              default="early_exit",
                                              choices=[
                                                  "early_exit", "optimal",
                                                  "runtime_threshold",
                                                  "runtime_threshold_decrease",
                                                  "switch1", "switch3", "switch5",
                                              ],
                                              help="When to exit the verifier. Options are 'optimal', "
                                                   "'early_exit' and 'runtime_threshold'. "
                                                   "When the mode is 'optimal', the verifier is setup "
                                                   "to find the most-violating counterexample. "
                                                   "For 'early-exit' it is setup to return any counterexample. "
                                                   "When the mode is set to 'runtime_threshold', the time the "
                                                   "verifier can search for a counterexample is capped. "
                                                   "When exceeding the runtime budget, the verifier returns "
                                                   "the current most-violating counterexample. "
                                                   "It continues search until it finds the first counterexample, "
                                                   "when no counterexample was found within the permitted runtime. "
                                                   "In the 'runtime_threshold_decrease' mode, the runtime is confined "
                                                   "as in the 'runtime_threshold' mode, but here the runtime budget "
                                                   "starts at a high value and is decreased for later repair steps. "
                                                   "In the switch1, switch3 and switch5 modes, the verifier starts in "
                                                   "optimal mode and switches to early-exit mode after 1, 3 or 5 "
                                                   "repair steps. "
                                                   "The following verifiers support exit modes: ERAN.")

        if experiment_execution_options:
            experiment_execution_group = parser.add_argument_group('experiment execution')
            experiment_execution_group.add_argument(
                '--timeout', dest='timeout', default=None, type=float,
                help='The timeout in hours (fractions are permitted) '
                     'for each repair case of the experiment. '
                     'By default timeout is disabled.'
            )
            experiment_execution_group.add_argument(
                '--timestamp', dest='timestamp', default=None, type=str,
                help="Set the timestamp of the experiment to use for "
                     "determining the output directory. "
                     "When omitted the current time is used. "
                     "Use this argument when using bash's timeout for "
                     "better enforcing timeouts for individual experiment "
                     "cases. "
            )

        if backend_choice:
            backend_group = parser.add_mutually_exclusive_group(required=True)
            if QUADRATIC_PENALTY_KEY in supported_repair_backends:
                backend_group.add_argument('--quadratic_penalty', action='store_true', dest=QUADRATIC_PENALTY_KEY,
                                           help='Use the PenaltyFunctionRepairDelegate with a '
                                                'quadratic penalty function.')
            if L1_PENALTY_KEY in supported_repair_backends:
                backend_group.add_argument('--l1_penalty', action='store_true', dest=L1_PENALTY_KEY,
                                           help='Use the PenaltyFunctionRepairDelegate with a '
                                                'l1 exact penalty function.')
            if L1_PLUS_QUADRATIC_PENALTY_KEY in supported_repair_backends:
                backend_group.add_argument('--l1_plus_quadratic_penalty', action='store_true',
                                           dest=L1_PLUS_QUADRATIC_PENALTY_KEY,
                                           help='Use the PenaltyFunctionRepairDelegate with a '
                                                'l1 + quadratic penalty function.')
            if AUGMENTED_LAGRANGIAN_KEY in supported_repair_backends:
                backend_group.add_argument('--augmented_lagrangian', action='store_true',
                                           dest=AUGMENTED_LAGRANGIAN_KEY,
                                           help='Use the AugmentedLagrangianRepairDelegate.')
            if LN_BARRIER_KEY in supported_repair_backends:
                backend_group.add_argument('--ln_barrier', action='store_true', dest=LN_BARRIER_KEY,
                                           help='Use the BarrierFunctionRepairDelegate with a '
                                                'logarithmic (natural logarithm) barrier function.')
            if RECIPROCAL_BARRIER_KEY in supported_repair_backends:
                backend_group.add_argument('--reciprocal_barrier', action='store_true', dest=RECIPROCAL_BARRIER_KEY,
                                           help='Use the BarrierFunctionRepairDelegate with a '
                                                'reciprocal (1/x) barrier function.')
            if L1_PENALTY_RECIPROCAL_BARRIER_KEY in supported_repair_backends:
                backend_group.add_argument('--l1_penalty_and_reciprocal_barrier', action='store_true',
                                           dest=L1_PENALTY_RECIPROCAL_BARRIER_KEY,
                                           help='Use the BarrierAndPenaltyFunctionRepairDelegate with a '
                                                'l1 penalty function and a reciprocal barrier function.')
            if DATASET_AUGMENTATION_KEY in supported_repair_backends:
                backend_group.add_argument('--dataset_augmentation', action='store_true',
                                           dest=DATASET_AUGMENTATION_KEY,
                                           help='Use the DatasetAugmentationRepairDelegate.')
            if FINE_TUNING_KEY in supported_repair_backends:
                backend_group.add_argument('--fine_tuning', action='store_true', dest=FINE_TUNING_KEY,
                                           help='Use the FineTuningRepairDelegate')
            if LP_MINIMAL_MODIFICATION_KEY in supported_repair_backends:
                backend_group.add_argument('--lp_minimal_modification', action='store_true',
                                           dest=LP_MINIMAL_MODIFICATION_KEY,
                                           help='Use the LinearProgrammingMinimalModificationRepairDelegate')
            if MARABOU_MINIMAL_MODIFICATION_KEY in supported_repair_backends:
                backend_group.add_argument('--marabou_minimal_modification', action='store_true',
                                           dest=MARABOU_MINIMAL_MODIFICATION_KEY,
                                           help='Use the MarabouMinimalModificationRepairDelegate')
            if DL2_KEY in supported_repair_backends:
                backend_group.add_argument('--dl2', action='store_true', dest=DL2_KEY,
                                           help='Use the DL2RepairDelegate')
            if NEURON_FIXATION_KEY in supported_repair_backends:
                backend_group.add_argument('--neuron_fixation', action='store_true', dest=NEURON_FIXATION_KEY,
                                           help='Use the NeuronFixationRepairDelegate')
            if LINEAR_REGRESSION_DATASET_AUGMENTATION_KEY in supported_repair_backends:
                backend_group.add_argument('--augment_lin_reg', action='store_true', dest=LINEAR_REGRESSION_DATASET_AUGMENTATION_KEY,
                                           help='Use the LinearModelDatasetAugmentationRepairDelegate.')

    @staticmethod
    def _get_cx_gen_from_command_line_name(name: str) -> Optional[CounterexampleGenerator]:
        name = name.upper()
        if name == 'DEEPOPT':
            return DeepOpt(how_many_counterexamples=DeepOpt.DEFAULT_COUNTEREXAMPLE_AMOUNT,
                           it_split_sampling=0, cutoff=None)
        elif name == 'DEEPOPT_SINGLE':
            return DeepOpt(how_many_counterexamples=CounterexampleAmount.SINGLE,
                           it_split_sampling=0, cutoff=None)
        elif name == 'DEEPOPT_CUTOFF':
            return DeepOpt(how_many_counterexamples=DeepOpt.DEFAULT_COUNTEREXAMPLE_AMOUNT, it_split_sampling=0)
        elif name == 'DEEPOPT_CUTOFF_SINGLE':
            return DeepOpt(how_many_counterexamples=CounterexampleAmount.SINGLE, it_split_sampling=0)
        elif name == 'FGSM':
            return FastGradientSignMethod()
        elif name.startswith('PGD'):
            pgd_single_cx = name.startswith('PGD_SINGLE')
            pgd_optimiser = name[name.index('[')+1 : name.index(',')]
            if pgd_optimiser == 'RMSPROP':
                pgd_optimiser = 'RMSprop'
            elif pgd_optimiser == 'ADAM':
                pgd_optimiser = 'Adam'
            pgd_restarts = int(name[name.index(',')+1:-1].strip())
            return ProjectedGradientDescentAttack(optimizer=pgd_optimiser, num_restarts=pgd_restarts,
                                                  single_counterexample=pgd_single_cx)
        elif name.startswith('DEA'):
            # dea_single_cx = name.startswith('DEA_SINGLE')
            dea_optimizer = name[name.index('[')+1:name.index(',')]
            if dea_optimizer == 'RMSPROP':
                dea_optimizer = 'RMSprop'
            elif dea_optimizer == 'ADAM':
                dea_optimizer = 'Adam'
            dea_population_size, dea_iterations = [
                int(param_value.strip())
                for param_value in name[name.index(',')+1:-1].split(',')
            ]
            return DifferentialEvolutionPGDAttack(dea_optimizer,
                                                  population_size=dea_population_size, iterations=dea_iterations)
        elif name == 'ERAN':
            return ERAN()
        elif name == 'ERAN_SINGLE':
            return ERAN(single_counterexample=True)
        elif name == 'ERAN_PLAIN':
            return ERAN(use_acasxu_style=False, use_milp=True)
        elif name == 'ERAN_PLAIN_SINGLE':
            return ERAN(use_acasxu_style=False, use_milp=True, single_counterexample=True)
        elif name == 'MARABOU':
            return Marabou()
        elif name == 'LINEARREGRESSIONVERIFIER':
            return LinearRegressionVerifier()
        elif name == 'LINEARREGRESSIONVERIFIER_SINGLE':
            return LinearRegressionVerifier(single_counterexample=True)
        elif name == 'NONE':
            return None
        else:
            raise ValueError(f"Unknown falsifier/verifier: {name}.")

    @staticmethod
    def _parse_falsifier_cascade(falsifier_spec):
        falsifier_cascade = []
        carry_on_falsifiers = set()

        continue_pattern = re.compile(r'[^;.]+;')
        stop_pattern = re.compile(r'[^;.]+\.')
        while len(falsifier_spec) > 0:
            falsifier_spec = falsifier_spec.strip()

            match = continue_pattern.match(falsifier_spec)
            if match:
                falsifier_name = falsifier_spec[:falsifier_spec.index(';')]
                falsifier = ExperimentBase._get_cx_gen_from_command_line_name(falsifier_name)
                assert falsifier is not None
                falsifier_cascade.append(falsifier)
                carry_on_falsifiers.add(falsifier)
                falsifier_spec = falsifier_spec[falsifier_spec.index(';')+1:]
                continue

            match = stop_pattern.match(falsifier_spec)
            if match:
                falsifier_name = falsifier_spec[:falsifier_spec.index('.')]
                falsifier = ExperimentBase._get_cx_gen_from_command_line_name(falsifier_name)
                assert falsifier is not None
                falsifier_cascade.append(falsifier)
                falsifier_spec = falsifier_spec[falsifier_spec.index('.')+1:]
                continue

            # end of cascade string
            falsifier = ExperimentBase._get_cx_gen_from_command_line_name(falsifier_spec)
            if falsifier is not None:
                falsifier_cascade.append(falsifier)
            falsifier_spec = ""
        return falsifier_cascade, carry_on_falsifiers

    def __init__(self, experiment_dir: str,
                 command_line_args: Optional[argparse.Namespace],
                 initial_verifier_runtime_threshold: float = 1.0,
                 verifier_runtime_threshold_update: Callable[[int, float], float] = lambda i, prev: prev * 2,
                 initial_verifier_runtime_threshold_decrease: float = 10.0,
                 verifier_runtime_threshold_update_decrease: Callable[[int, float], float] = lambda i, prev: prev / 2):
        """
        Initializes the ExperimentBase class.

        Configures the logger. Call this at the beginning of your main method.
        Make sure to call if before casting any log messages.

        Records the execution time that is used to name all files.

        :param experiment_dir: The name of the subdirectory in the outputs directory in which
         to save log files, repaired networks, etc.
        :param command_line_args: Further arguments passed on the command line and parsed with argparse.
         The parser needs to be set up using the static argparse_add_arguments method.
        """
        self.experiment_dir = experiment_dir
        self.command_line_args = command_line_args
        self.initial_verifier_runtime_threshold = initial_verifier_runtime_threshold
        self.verifier_runtime_threshold_update = verifier_runtime_threshold_update
        self.initial_verifier_runtime_threshold_decrease = initial_verifier_runtime_threshold_decrease
        self.verifier_runtime_threshold_update_decrease = verifier_runtime_threshold_update_decrease

        if command_line_args.timestamp is None:
            self.time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        else:
            self.time = command_line_args.timestamp
        output_dir = Path('..', 'output', experiment_dir)
        if command_line_args.preliminary:
            output_dir = Path(output_dir, 'prelim')
        self.output_dir = Path(output_dir, self.time)
        os.makedirs(self.output_dir, exist_ok=True)

        self.checkpointing_training_hook: Optional[Checkpointing] = None

        # initialize the logger
        # configure the logger to print
        loglevel_string = command_line_args.log_level
        if loglevel_string.upper() == 'DEBUG':
            loglevel = logging.DEBUG
        elif loglevel_string.upper() == 'INFO':
            loglevel = logging.INFO
        elif loglevel_string.upper() == 'WARNING':
            loglevel = logging.WARNING
        elif loglevel_string.upper() == 'ERROR':
            loglevel = logging.ERROR
        else:
            loglevel = int(loglevel_string)
        logging.basicConfig(level=loglevel, format="%(levelname)s: %(message)s", stream=sys.stdout)
        mp.get_logger()  # otherwise messages logged by processes are lost

    def register_hooks(self, training_loop: TrainingLoop, training_checkpointing_frequency=10):
        """
        Registers loggers and other hooks to a training loop as instructed by the command line arguments.

        Registers the following hooks:
         - A TensorboardLossPlot if instructed so in the command line arguments (post iteration hook)
        """
        if self.command_line_args.tensorboard:
            tensorboard_writer = torch.utils.tensorboard.SummaryWriter(log_dir='../tensorboard', flush_secs=10)
            TensorboardLossPlot(
                tensorboard_writer, frequency=1, average_training_loss=False,
                training_loss_tag=f'{self.time}/training_loss'
            ).register(training_loop)

        if self.command_line_args.save_training_checkpoints:
            self.checkpointing_training_hook = Checkpointing(
                output_directory=None, frequency=training_checkpointing_frequency,
                save_model_state=True, save_optimizer_state=False
            )
            self.checkpointing_training_hook.register(training_loop)

    def execute(self, network: NeuralNetwork, specification: Sequence[Property],
                repair_backends: Dict[str, Callable[[], RepairNetworkDelegate]],
                experiment_name: Optional[str] = None,
                losses: Sequence[Tuple[str, Callable[[], torch.Tensor]]] = (),
                save_original_network=False,
                repair_network_further_kwargs=None,
                raise_exceptions=True):
        """
        Provides a default skeleton for running a repair experiment

        The repair_backends parameter allows providing a number of different repair backends as a dictionary.
        What backend is used is determined via the command line arguments.
        Possible keys of the dictionary:
         * quadratic_penalty: PenaltyFunctionRepair with a quadratic penalty function
         * l1_penalty: PenaltyFunctionRepairDelegate with a l1 penalty function
         * l1_plus_quadratic_penalty: PenaltyFunctionRepairDelegate with a l1 plus quadratic penalty function
         * ln_barrier: BarrierFunctionRepairDelegate with a logarithmic barrier function (natural logarithm)
         * reciprocal_barrier: BarrierFunctionRepairDelegate with a reciprocal barrier function (:math:`1/(g(x))`)
         * l1_penalty_and_reciprocal_barrier: A BarrierPenaltyFunctionRepairDelegate with l1 penalty
           and reciprocal barrier function.
         * augmented_lagrangian: AugmentedLagrangianRepairDelegate
         * dataset_augmentation: DatasetAugmentationRepairDelegate
        These keys are also defined as string constants in the experiment_base file
        The value of the dictionary needs to be an instance of the corresponding class
        (a subclass of ``RepairNetworkDelegate``).

        :param experiment_name: If running multiple experiments as part of one larger experiment,
          this parameter should be used to supply the name of the sub-experiment.
          All output files will be placed in another subdirectory of the overall experiment output directory
          that is specified at initialisation of ExperimentBase.
          If experiment_name is None, the output files will be placed in the experiment output directory directly.
        :param network: The network to repair.
          May be replaced by a restored checkpoint.
        :param specification: The specification to repair.
        :param repair_backends: The repair backends for repairing the network using ``repair_network``.
        :param losses: Losses to log at the beginning and the end.
          First element: name, second element: callable to calculate the loss.
        :param save_original_network: Whether to save the original network in the logging directory.
          This option is useful for synthesis/training.
        :param repair_network_further_kwargs: Further arguments for `repair_network`.
          Falsifier cascade and verifier are provided by experiment_base
          (specified via command line arguments with customizable defaults).
        :param raise_exceptions: Whether to raise exceptions (after logging) or to consume them.
         This option is helpful if you want to run further experiments, regardless if one fails.
        """
        if repair_network_further_kwargs is None:
            repair_network_further_kwargs = {}

        run_output_dir: Path
        if experiment_name is None:
            run_output_dir = self.output_dir
        else:
            run_output_dir = Path(self.output_dir, experiment_name)
            os.makedirs(run_output_dir, exist_ok=False)

        logfile_handler = logging.FileHandler(Path(run_output_dir, f'log.txt'), encoding='utf-8')
        logging.root.addHandler(logfile_handler)

        # make sure the repaired network will be able to be saved in the end
        repaired_network_file_path = Path(run_output_dir, f"repaired_network.pyt")
        assert not repaired_network_file_path.exists(), 'Repaired network output file already exists'
        # for the original network, we do not have to check, as it is stored
        # at the beginning of the repair anyway
        original_network_file_path = Path(run_output_dir, f"original_network.pyt")

        if self.checkpointing_training_hook is not None:
            training_checkpoints_output_dir = Path(run_output_dir, 'training_checkpoints')
            os.mkdir(training_checkpoints_output_dir)
            self.checkpointing_training_hook.set_output_directory(training_checkpoints_output_dir)

        cmd_line_args_log_string = '\n'.join(f'    {k}: {v}' for k, v in vars(self.command_line_args).items())
        info(f'Running experiment: {self.experiment_dir}'
             f'{": " + experiment_name if experiment_name is not None else ""}.\n'
             f'Commandline arguments: {{\n' + cmd_line_args_log_string + '\n}\n'
             f'Machine details: {{\n' + self.get_machine_stats() + '\n}\n'
             f'Network:\n{network}')

        ray.init(ignore_reinit_error=True)

        backend: RepairNetworkDelegate
        for backend_key in BACKEND_OPTIONS:
            if getattr(self.command_line_args, backend_key, False):
                backend = repair_backends[backend_key]()
                break
        else:
            raise NotImplementedError("Unknown repair backend")

        # parse the falsifier cascade
        falsifier_cascade, carry_on_falsifiers = self._parse_falsifier_cascade(self.command_line_args.falsifiers)
        verifier = self._get_cx_gen_from_command_line_name(self.command_line_args.verifier)

        if isinstance(verifier, ERAN):
            exit_mode = self.command_line_args.verifier_exit_mode
            info(f"Setting up ERAN verifier with exit mode {exit_mode}.")
            if exit_mode == "runtime_threshold":
                verifier.exit_mode = self.initial_verifier_runtime_threshold
                info(f"Initial verifier runtime threshold is {verifier.exit_mode}.")

                def update_verifier_runtime_threshold(repair_step):
                    verifier.exit_mode = self.verifier_runtime_threshold_update(
                        repair_step, verifier.exit_mode
                    )
                    info(f"Increasing verifier runtime threshold to {verifier.exit_mode}.")
            elif exit_mode == "runtime_threshold_decrease":
                verifier.exit_mode = self.initial_verifier_runtime_threshold_decrease
                info(f"Initial verifier runtime threshold is {verifier.exit_mode}.")

                def update_verifier_runtime_threshold(repair_step):
                    verifier.exit_mode = self.verifier_runtime_threshold_update_decrease(
                        repair_step, verifier.exit_mode
                    )
                    info(f"Decreasing verifier runtime threshold to {verifier.exit_mode}.")
            elif exit_mode in ("switch1", "switch3", "switch5"):
                switch_repair_step = int(exit_mode[-1])

                verifier.exit_mode = "optimal"
                info(f"Verifier starts in optimal mode. "
                     f"Switching to early-exit in repair step {switch_repair_step}.")
                # mode is switched at the end of the repair step
                switch_repair_step = switch_repair_step - 1

                def update_verifier_runtime_threshold(repair_step):
                    if repair_step == switch_repair_step:
                        verifier.exit_mode = "early_exit"
                        info(f"Switching verifier to early-exit mode.")
            else:
                verifier.exit_mode = exit_mode

                def update_verifier_runtime_threshold(repair_step):
                    pass
        else:
            def update_verifier_runtime_threshold(repair_step):
                pass

        def post_repair_step_hook(repair_step):
            update_verifier_runtime_threshold(repair_step)

        checkpoint_handler: Callable
        if self.command_line_args.save_checkpoints:
            checkpoints_dir = Path(run_output_dir, 'checkpoints')
            counterexamples_dir = Path(run_output_dir, 'counterexamples')
            os.mkdir(checkpoints_dir)
            os.mkdir(counterexamples_dir)

            def save_checkpoint(iteration, cx, net: NeuralNetwork):
                checkpoint_file_path = Path(checkpoints_dir, f"repair_step_{iteration}.pyt")
                checkpoint_file_path.touch(exist_ok=False)
                info(f"Saving checkpoint network in file: {checkpoint_file_path}")
                torch.save(net, checkpoint_file_path)

                cx_checkpoint_file_path = Path(counterexamples_dir, f"repair_step_{iteration}.dill")
                cx_checkpoint_file_path.touch(exist_ok=False)
                info(f"Saving counterexamples in file: {cx_checkpoint_file_path}")
                with open(cx_checkpoint_file_path, 'w+b') as cx_file:
                    dill.dump(cx, cx_file)
            checkpoint_handler = save_checkpoint
        else:
            def do_nothing(*args):
                pass
            checkpoint_handler = do_nothing

        counterexamples_checkpoint = None
        if self.command_line_args.restore_timestamp is not None:
            checkpoint_timestamp = self.command_line_args.restore_timestamp
            checkpoint_iteration = self.command_line_args.restore_iteration
            restore_only_counterexamples = self.command_line_args.restore_only_cx

            saved_checkpoint_dir = Path('..', 'output', self.experiment_dir, str(checkpoint_timestamp))
            assert saved_checkpoint_dir.exists(), \
                f"Run directory to restore from does not exist. " \
                f"Please check the timestamp for typos: {self.command_line_args.restore_timestamp}."

            experiment_error_log_name = self.experiment_dir  # used to display warning messages
            if experiment_name is not None:
                saved_checkpoint_dir = Path(saved_checkpoint_dir, experiment_name)
                experiment_error_log_name += '/' + experiment_name

            saved_counterexamples_file = Path(saved_checkpoint_dir, 'counterexamples',
                                              f"repair_step_{checkpoint_iteration}.dill")
            try:
                with open(saved_counterexamples_file, 'rb') as counterexamples_checkpoint_file:
                    counterexamples_checkpoint = dill.load(counterexamples_checkpoint_file)
            except FileNotFoundError:
                # this may happen especially if not all sub-experiments could be executed
                # for the execution we are trying to restore from
                warning(f"No saved counterexamples checkpoint found for experiment: {experiment_error_log_name}. "
                        f"Looked for file: {saved_counterexamples_file}.\n"
                        f"Continuing without counterexamples checkpoint.")

            if not restore_only_counterexamples:  # also restore a network
                saved_network_file = Path(saved_checkpoint_dir, 'checkpoints',
                                          f"repair_step_{checkpoint_iteration}.pyt")
                try:
                    network = torch.load(saved_network_file)
                except FileNotFoundError:
                    warning(f"No saved checkpoint network found for experiment: {experiment_error_log_name}. "
                            f"Looked for file: {saved_network_file}.\n"
                            f"Continuing without original network instead of checkpoint.")

        try:
            if save_original_network:
                original_network_file_path.touch(exist_ok=False)
                info(f"Saving original network in file: {original_network_file_path}")
                torch.save(network, original_network_file_path)

            initial_loss_values = dict([(name, calc_loss()) for name, calc_loss in losses])
            info(f"Starting repair.\nInitial loss: " +
                 "; ".join(f"{initial_loss_values[name]:.4f} ({name})" for name in initial_loss_values))

            # this context manager will throw an ExperimentTimeoutError when time is up
            with ExperimentBase.WithTimeout(self.command_line_args.timeout):
                with LogExecutionTime(
                    "overall repair", enable=self.command_line_args.log_execution_times
                ):
                    status, repaired_network = repair_network(
                        network, specification, backend,
                        falsifier_cascade, lambda fls: fls in carry_on_falsifiers,
                        verifier,
                        checkpoint_handler=checkpoint_handler,
                        initial_counterexamples=counterexamples_checkpoint,
                        max_iterations=self.command_line_args.max_repair_steps,
                        measure_execution_times=self.command_line_args.log_execution_times,
                        post_repair_step_hook=post_repair_step_hook,
                        **repair_network_further_kwargs
                    )
            info(f"Repair finished: {status}")

            final_loss_values = dict([(name, calc_loss()) for name, calc_loss in losses])
            info("Initial loss: " +
                 "; ".join(f"{initial_loss_values[name]:.4f} ({name})" for name in initial_loss_values) + "\n" +
                 "Final loss:   " +
                 "; ".join(f"{final_loss_values[name]:.4f} ({name})" for name in final_loss_values) + "\n" +
                 "Difference:   " +
                 "; ".join(f"{abs(initial_loss_values[name] - final_loss_values[name]):.4f} ({name})"
                           for name in initial_loss_values)
                 )

            repaired_network_file_path.touch(exist_ok=False)
            info(f"Saving repaired network in file: {repaired_network_file_path}")
            torch.save(repaired_network, repaired_network_file_path)

        except ExperimentBase.ExperimentTimeoutError:
            warning("Experiment timed out")
        # purposely catching everything, including e.g. KeyboardInterrupt
        # allows aborting one experiment and still running the remaining ones.
        except:
            logging.exception('Repair failed with an error.')
            if raise_exceptions:
                raise
        finally:
            if self.checkpointing_training_hook is not None:
                info('Waiting for all training checkpoints to be written to disk.')
                self.checkpointing_training_hook.close()

            logging.root.removeHandler(logfile_handler)

    class ExperimentTimeoutError(Exception):
        pass

    class WithTimeout:
        def __init__(self, timeout: Optional[float]):
            self.timeout = timeout

            def timeout_handler(signum, frame):
                raise ExperimentBase.ExperimentTimeoutError()
            if timeout is not None:
                signal.signal(signal.SIGALRM, timeout_handler)

        def __enter__(self):
            if self.timeout is not None:
                signal.alarm(int(self.timeout * 3600))  # timeout is in hours, alarm want's seconds
            return self

        def __exit__(self, exc_type, exc_value, exc_traceback):
            signal.alarm(0)  # cancel the previously set alarm

    @staticmethod
    def get_machine_stats() -> str:
        """
        Collects a bunch of relevant machine statistics:
         * platform details (includes operation system name)
         * processor name
         * current system load
         * installed memory
         * current memory usage
         * installed swap
         * current swap usage
         * number of GPus
         * for each GPU: name, load, memory, memory usage.
        """
        stats = (
            f'    platform: {platform.platform(aliased=True)}\n'
            f'    CPU: {platform.processor()}\n'
            f'    CPU Cores: {psutil.cpu_count(logical=False)} (physical) '
            f'{psutil.cpu_count(logical=True)} (logical)\n'
            f'    CPU load: {psutil.cpu_percent(interval=2)}%\n'  # test for two seconds
        )
        memory_stats = psutil.virtual_memory()
        stats += (
            f'    Total Memory: {memory_stats.total / (1024 ** 3):.2f} GB\n'
            f'    Memory used: {memory_stats.percent}%\n'
        )
        swap_stats = psutil.swap_memory()
        stats += (
            f'    Total Swap: {swap_stats.total / (1024 ** 3):.3f} GB\n'
            f'    Swap used: {swap_stats.percent}%\n'
        )
        gpus = GPUtil.getGPUs()
        if len(gpus) == 0:
            stats += '    GPUs: None'
        else:
            stats += f'    GPUs: {len(gpus)}'
            for gpu in gpus:
                stats += (
                    f'    GPU {gpu.id}: \n'
                    f'        Name: {gpu.name}\n'
                    f'        GPU load: {gpu.load}\n'
                    f'        Total GPU Memory: {gpu.memoryTotal} MB\n'
                    f'        GPU Memory used: {100*gpu.memoryUsed/gpu.memoryTotal:.1f}%\n'
                )
        return stats


class TrackingLossFunction:
    """
    A wrapper for other loss functions that records the calculated loss values
    and potentially other, related losses for logging purposes.

    Such a wrapper is useful for logging task losses that are calculated using batches
    and that can not be easily recreated.
    """
    def __init__(self, loss_function: Callable[[], Union[torch.Tensor, Tuple[torch.Tensor, ...]]],
                 log_names: Union[str, Sequence[str]] = ('training loss', )):
        """
        Initializes a TrackingLossFunction.

        :param loss_function: The loss function to wrap. This function may return one or multiple values.
         In any case only the first value returned is passed on as the actual loss. The other return values
         are only used for logging.
        :param log_names: The names or keys of the loss values. Names need to be supplied in the same order as
         the return values of the loss function.
        """
        if isinstance(log_names, str):
            log_names = (log_names, )
        self.log_names = log_names
        self.loss_function = loss_function

        self.last_loss_values: OrderedDict[str, Optional[torch.Tensor]] = \
            OrderedDict(zip(log_names, [None] * len(log_names)))

    def get_additional_losses(self, average: Union[bool, Sequence[bool]] = True) \
            -> Tuple[Tuple[str, Callable, bool], ...]:
        """
        Gets the tracked losses in the format such that it can be used as additional_losses for
        a logging hook (see nn_repair.training.logging).

        :param average: Whether to average the individual losses over multiple iterations
            or only calculate the losses in the iterations where a logging is performed.
            If only one value is passed, this value is used for all losses.
            Passing a sequence allows fine-grained control. The values of the sequence
            needs to have the same order as the log_names passed at initialization.
            See ``nn_repair.training.logging.LogLoss`` for more details.
        :return: A tuple of tuples that can be used as an additional_losses argument.
        """
        if isinstance(average, bool):
            average = [average] * len(self.log_names)

        return tuple((name, lambda: self.last_loss_values[name], avg) for name, avg in zip(self.log_names, average))

    def __call__(self) -> torch.Tensor:
        losses = self.loss_function()
        if isinstance(losses, torch.Tensor):
            losses = (losses, )
        for value, name in zip(losses, self.log_names):
            self.last_loss_values[name] = value.item()
        return losses[0]

    class ResetLosses(TrainingLoopHook):
        def __init__(self, tracking_loss_function):
            self.tracking_loss_function = tracking_loss_function

        def __call__(self, loop: 'TrainingLoop', call_location: TrainingLoopHookCallLocation, *args, **kwargs):
            self.tracking_loss_function.reset_losses()

        def register(self, loop: 'TrainingLoop'):
            loop.add_post_training_hook(self)

    def register_loss_resetting_hook(self, training_loop: TrainingLoop):
        """
        Registers a post training hook to the training_loop which
        resets the recorded loss values.
        """
        TrackingLossFunction.ResetLosses(self).register(training_loop)

    def reset_losses(self):
        """
        Resets the stored loss values.
        """
        self.last_loss_values = OrderedDict(zip(self.log_names, [None] * len(self.log_names)))
