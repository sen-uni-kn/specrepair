from typing import Sequence, Tuple, Optional, Callable, Deque, List, Union
from abc import ABC, abstractmethod
from enum import Enum, auto
from dataclasses import dataclass
from builtins import property

from logging import info, warning


import numpy as np
from collections import deque

import torch

from deep_opt import NeuralNetwork, Property
from nn_repair.counterexamples import CounterexampleGenerator, Counterexample
from nn_repair.falsifiers import DeepOpt
from nn_repair.training import TrainingLoop
from nn_repair.utils.timing import LogExecutionTime
from nn_repair.utils.shift_property import shift_property


class RepairStatus(Enum):
    SUCCESS = auto()
    FAILURE = auto()
    ERROR = auto()

    def __str__(self):
        if self is RepairStatus.SUCCESS:
            return "success"
        elif self is RepairStatus.FAILURE:
            return "failure"
        elif self is RepairStatus.ERROR:
            return "error"
        else:
            raise NotImplementedError("Unknown RepairStatus in RepairStatus.__str__")


class RepairNetworkDelegate(ABC):
    """
    Abstract base class for backends for repair_network that actually implement
    methods to repair a network by fixing counterexamples.
    """

    @abstractmethod
    def with_specification(self, specification: Sequence[Property]):
        raise NotImplementedError()

    @abstractmethod
    def register_strengthened_property(self, strengthened_property: Property, original_property: Property):
        raise NotImplementedError()

    @property
    @abstractmethod
    def network(self):
        raise NotImplementedError()

    @network.setter
    @abstractmethod
    def network(self, value: NeuralNetwork):
        raise NotImplementedError()

    @abstractmethod
    def new_counterexample(self, prop: Property, network_inputs: torch.Tensor):
        """
        Updates the counterexamples used for repair with the new counterexample for the given property.
        This method is called repeatedly before repair is invoked.

        :param prop: The property for which a new counterexample was found.
        :param network_inputs: The input values for which the current network violates the specification.
        """
        raise NotImplementedError()

    @abstractmethod
    def property_satisfied(self, prop: Property):
        """
        Calls of this method inform a RepairNetworkDelegate that a specification is satisfied with the
        current network. This may be useful to clean up counterexamples if this is required by your method.

        :param prop: The property that is satisfied by the network.
        """
        raise NotImplementedError()

    @abstractmethod
    def repair(self) -> RepairStatus:
        """
        Repair the network to no longer violate the specification for the previously registered counterexamples.

        :return: The final status of the repair.
          Return SUCCESS if all counterexamples no longer violate the specification.
          Return FAILURE if this could not be archived.
          Return ERROR if any other error occurred from which recovery is possible, otherwise raise
          an exception.
        """
        raise NotImplementedError()

    def violation_stats(self) -> str:
        """
        Create a log message that gives details on the violation of the properties with the counterexamples
        as applicable for the implementation.
        The default implementation returns the empty string,
        which effectively disables the log message.

        :return: A text giving statistics about the violation of the specification with the counterexamples
          that will be used in the next invocation of repair.
        """
        return ''


def repair_network(
    initial_network: NeuralNetwork,
    specification: Sequence[Property],
    repair_backend: RepairNetworkDelegate,
    falsifier_cascade: Sequence[CounterexampleGenerator] = (
        DeepOpt(
            how_many_counterexamples=DeepOpt.DEFAULT_COUNTEREXAMPLE_AMOUNT, it_split_sampling=0, cutoff=None
        ),
    ),
    continue_falsification: Callable[[CounterexampleGenerator], bool] = lambda args: False,
    verifier: Optional[CounterexampleGenerator] = None,
    training_mode: bool = False,
    training_mode_training_loop: Optional[TrainingLoop] = None,
    checkpoint_handler: Callable[[int, Sequence[Tuple[Property, np.ndarray]], NeuralNetwork], None] = lambda *args: None,
    initial_counterexamples: Sequence[Tuple[Property, np.ndarray]] = None,
    max_iterations: Optional[int] = None,
    do_not_skip_properties: bool = False,
    abort_on_backend_error: bool = False,
    measure_execution_times: bool = False,
    post_repair_step_hook: Optional[Callable[[int], None]] = None,
) -> Tuple[RepairStatus, NeuralNetwork]:
    """
    Tries to repair a neural network such that it fulfils all the supplied properties (specification).

    ``repair_network`` alternates between finding counterexamples using a set of falsifiers and potentially
    a verifier and repairing the counterexamples using a RepairNetworkDelegate (backend).
    One alternation is called a *repair step*.

    :param initial_network: The networks that should be repaired.
           Typically, this object is directly modified during
           repair, but this solely depends on the behaviour of the repair_backend.
    :param specification: The specification against which the network should be repaired.
    :param repair_backend: The RepairNetworkDelegate that will do the heavy lifting during the repair.
    :param falsifier_cascade: The falsifiers that will be used for finding counterexamples.
           The falsifiers are executed one after another until one of the falsifiers finds a counterexample.
           Once a falsifier has found a counterexample, ``continue_falsification`` is invoked to determine
           whether the next falsifier should also be executed.
           This may make sense to generate more counterexamples
           or if some falsifiers are relatively cheap. By default, once a counterexample has been found,
           the following falsifiers are no longer executed.
           If a falsifier does not find a counterexample
           at some point, it is disabled, meaning that it will not be skipped in the next falsification run.
           If all falsifiers fail to find a counterexample, the verifier is executed, if provided.
           By default, only DeepOpt is contained in the falsifier cascade.
    :param continue_falsification: Determine whether the further falsifiers in the cascade should be executed.
           after a counterexample has been found. The default parameter always returns False,
           which means that the falsifier cascade is left as soon as a counterexample is found.
           The argument to this callable is the CounterexampleGenerator that discovered the counterexample.
    :param verifier: An verifier that can prove that a specific property is satisfied or returns
           one or multiple counterexamples otherwise. This verifier is only used if all falsifiers can not
           find further counterexamples.
           Pass None (default value) to not use a verifier.
    :param training_mode: In training mode at least one training run is performed, not matter
           whether counterexamples are found or not.
           In this mode, falsifiers aren't disabled in the first iteration if they don't find counterexamples.
           In later iterations, they may still be disabled.
    :param training_mode_training_loop: The training loop to use in training mode when no counterexample is
           found in the first iteration.
    :param checkpoint_handler: This parameter can be used to save or log intermediate networks and computed
           counterexamples that are obtained after one repair step and the counterexamples that were used
           for the repair. The first argument to this callable gives the current iteration,
           the second argument gives the counterexamples and the third argument is the network
           after the repair step in that iteration.
           A checkpoint can be restored by passing the repaired network as `initial_network`. The state before the repair
           can be reconstructed by setting initial_network to the network before the repair and
           initial_counterexamples to the computed counterexamples.
    :param initial_counterexamples: Define the counterexamples that are used for repair in the first iteration. If this
           parameter is not set to None, falsification and verification will be skipped in the first iteration and only
           the given counterexamples will be used. Pass None to disable this.
    :param max_iterations: The maximum number of repair steps to perform.
           Pass None to continue until repair is archived.
    :param do_not_skip_properties: By default, the repair procedure skips properties that have been verified already
           from successive verification even if the network changes to reduce the computation time.
           If this parameter is passed as True, then all properties are always checked after every retraining step.
           If it is discovered that a property was once repaired but got broken again through more repairs,
           then for this property, this behavior is enabled automatically.
    :param abort_on_backend_error: Whether to abort repair when the backend fails to repair a set of counterexamples.
           Some backends may be able to recover from such a failure if more counterexamples are provided.
           In this case, it makes sense to continue the repair even if the repair backend failed.
           This behaviour is enabled by default. Pass True to stop the repair when the backend fails.
    :param measure_execution_times: Whether to measure and log the execution times of the repair backend
           and the falsifiers and verifier.
    :param post_repair_step_hook: A hook that is run at the end of each repair step.
           Accepts the current repair step number as argument.
           Intended for updating repair-step dependent verifier options (for example the
           runtime threshold).
    :return: The status of the repair when stopping and the final network.
             If the status is SUCCESS, the returned network is a repaired version of the initial network,
             otherwise the returned network is an intermediate result.
    """
    assert len(falsifier_cascade) > 0 or verifier is not None, "No way to generate counterexamples provided"

    @dataclass
    class PropertyStatus:
        # records for each falsifier, whether this falsifier
        # did fail to find a counterexample in it's last execution
        falsifier_exhaustion: List[bool]
        # records failures of the verifiers
        falsifier_error: List[bool]
        # whether the verifier proved the property to be save previously
        verified: bool = False
        # whether the verifier failed with an error for the property the last time it was executed.
        unable_to_verify: bool = False
        # whether to always run falsification and verification for this property.
        always_check: bool = False
        # indicates if the status is up to date for the current network. (True means up-to-date)
        fresh_results: bool = False

        def needs_check(self):
            # a property needs to be checked if
            #  1) is always check is True
            #  2) if some falsifier is not yet exhausted
            #  3) if verified is False (only if a verifier is present)
            return self.always_check or \
                   not all(self.falsifier_exhaustion) or \
                   (not self.verified and verifier is not None)

    # the currently obtained results and options for each of the properties
    property_status = dict(
        (prop, PropertyStatus(
            falsifier_exhaustion=[False] * len(falsifier_cascade),
            falsifier_error=[False] * len(falsifier_cascade),
            always_check=do_not_skip_properties)
         ) for prop in specification
    )

    repair_backend.with_specification(specification)
    repair_backend.network = initial_network

    i = 0
    if initial_counterexamples is not None:
        info(f"Using provided initial counterexamples in first iteration "
             f"(no falsification/verification performed)")
        for prop, counterexample in initial_counterexamples:
            repair_backend.new_counterexample(prop, torch.tensor(counterexample))
        info(f'Using {len(initial_counterexamples)} provided initial specification violations:\n'
             f'{repair_backend.violation_stats()}')

        with LogExecutionTime('repair backend', enable=measure_execution_times):
            backend_status = repair_backend.repair()
        checkpoint_handler(i, tuple(initial_counterexamples), repair_backend.network)

        if backend_status == RepairStatus.FAILURE:
            warning("Backend could not repair counterexamples. " +
                    ("Aborting repair." if abort_on_backend_error else "Continuing anyway."))
            if abort_on_backend_error:
                return RepairStatus.FAILURE, repair_backend.network
        if backend_status == RepairStatus.ERROR:
            return RepairStatus.ERROR, repair_backend.network
        i += 1

    while max_iterations is None or i < max_iterations:
        info(f"repair_network iteration {i}")

        counterexamples_for_checkpoint: Deque[Tuple[Property, np.ndarray]] = deque()
        # for the training mode to determine whether to run the backend or use the training loop
        found_counterexamples = False
        for prop in specification:
            if not property_status[prop].needs_check():
                info(f"Skipping property: {prop.property_name} (did previously hold).")
                continue

            counterexamples: Optional[List[Counterexample]] = None  # None indicates failure to verify

            for fi, falsifier in enumerate(falsifier_cascade):
                # Do not invoke a falsifiers again if it did not find counterexamples last time
                if property_status[prop].falsifier_exhaustion[fi] or property_status[prop].falsifier_error[fi]:
                    continue

                info(f'Falsifying property {prop.property_name} with {falsifier.name}')
                with LogExecutionTime(f'{falsifier.name} for property {prop.property_name}',
                                      enable=measure_execution_times):
                    cxs, status = falsifier.find_counterexample(repair_backend.network, prop)

                property_status[prop].fresh_results = True
                if cxs is None:  # None indicates error here
                    warning(f"Error while running {falsifier.name} for property {prop.property_name}. "
                            f"Falsifier status: {status}")
                    if not training_mode or i > 0:
                        info(f"Not using {falsifier.name} for property {prop.property_name} from now on.")
                        property_status[prop].falsifier_exhaustion[fi] = True
                        property_status[prop].falsifier_error[fi] = True
                elif len(cxs) == 0:
                    if not training_mode or i > 0:
                        info(f"Could not find any counterexamples with {falsifier.name} "
                             f"for property {prop.property_name}. Disabling.")
                        property_status[prop].falsifier_exhaustion[fi] = True
                    if counterexamples is None:
                        counterexamples = []
                else:
                    info(f"Found {len(cxs)} counterexamples with {falsifier.name} "
                         f"for property {prop.property_name}.")
                    if counterexamples is None:
                        counterexamples = list(cxs)
                    else:
                        counterexamples.extend(cxs)
                    if not continue_falsification(falsifier):
                        break

            # run verifier if falsifiers did not find counterexamples
            # and we did not previously verified the property (unless always_check is set)
            if verifier is not None and (counterexamples is None or len(counterexamples) == 0) \
                    and (not property_status[prop].verified or property_status[prop].always_check):
                info(f"Verifying property {prop.property_name} using {verifier.name}")

                with LogExecutionTime(f'{verifier.name} for property {prop.property_name}',
                                      enable=measure_execution_times):
                    cxs, status = verifier.find_counterexample(repair_backend.network, prop)

                property_status[prop].fresh_results = True
                if cxs is None:
                    if not training_mode or i > 0:
                        warning(f"Error while verifying property: {prop.property_name}: Skipping property. "
                                f"Verifier status: {status}.")
                        property_status[prop].unable_to_verify = True
                    else:  # i == 0 and training_mode
                        warning(f"Error while verifying property: {prop.property_name}: "
                                f"Retrying in second iteration (training mode). "
                                f"Verifier status: {status}.")
                elif len(cxs) == 0:
                    if not training_mode or i > 0:
                        info(f"Property verified: {prop.property_name}")
                        property_status[prop].verified = True
                        property_status[prop].unable_to_verify = False
                    else:
                        info(f"Property verified in first iteration (training mode): {prop.property_name}. "
                             f"Checking again in second iteration.")
                    assert counterexamples is None or len(counterexamples) == 0, 'Falsifiers and verifier disagreed.'
                    counterexamples = []
                else:
                    info(f"Found {len(cxs)} counterexamples using verifier.")
                    property_status[prop].verified = False
                    property_status[prop].unable_to_verify = False
                    if counterexamples is None:
                        counterexamples = list(cxs)
                    else:
                        counterexamples.extend(cxs)

            if counterexamples is not None:
                if len(counterexamples) == 0:
                    repair_backend.property_satisfied(prop)
                else:
                    found_counterexamples = True
                    for counterexample in counterexamples:
                        violation = counterexample.property_satisfaction
                        info(f'Counterexample violation of new counterexample: {violation:.10f}')
                        # if counterexample.property != prop:
                        #     info(f"Property strengthened by counterexample generator for a counterexample.")
                        #     repair_backend.register_strengthened_property(counterexample.property, prop)
                        if not prop.property_satisfied_from_internal(
                                counterexample.inputs_as_tensor().unsqueeze(0),
                                counterexample.network_outputs,
                        ):
                            # regular counterexample, all fine
                            repair_backend.new_counterexample(prop, counterexample.inputs_as_tensor())
                            counterexamples_for_checkpoint.append((prop, counterexample.inputs))
                        else:
                            # counterexample does not violate the original property (spurious counterexample)
                            # check if the strengthened version in counterexample.property works out
                            # in case that is present
                            if counterexample.property != prop:
                                info(f'Handling spurious counterexample by strengthened property.')
                                if counterexample.property.property_satisfied_from_internal(
                                    counterexample.inputs_as_tensor().unsqueeze(0),
                                    counterexample.network_outputs,
                                ):
                                    # also strengthened version is spurious
                                    # this may be due to differences in the neural network representation
                                    # inside the cx generator and the repair method
                                    # caused e.g. by serialisation to pass the model to the cx generator
                                    warning(
                                        'Spurious counterexample returned by verifier or falsifier. '
                                        'Counterexample is also spurious for strengthened property. '
                                        f'Counterexample: {counterexample}, '
                                        f'Strengthened output constraint: {counterexample.property.output_constraint}'
                                    )
                                    # still try to handle the counterexample
                                repair_backend.register_strengthened_property(counterexample.property, prop)
                            else:
                                warning(
                                    'Spurious counterexample returned by verifier or falsifier.'
                                    f': {counterexample}'
                                )
                                # still try to fix it using other handling mechanisms

                            if counterexample.property.property_satisfied_from_internal(
                                    counterexample.inputs_as_tensor().unsqueeze(0),
                                    counterexample.network_outputs) \
                                    and counterexample.internal_violation is not None:
                                assert counterexample.internal_violation >= 0, \
                                    f"Non-positive internal violation for counterexample: {counterexample}"
                                info(f'Handling spurious counterexample by internal violation.')
                                # make the violation function of the property positive by shifting it such
                                # that it matches the internal violation
                                # violation_here = counterexample.property.satisfaction_function(
                                #     counterexample.network_outputs_as_tensor().unsqueeze(0),
                                #     counterexample.inputs_as_tensor().unsqueeze(0)
                                # )
                                violation_here = counterexample.property_satisfaction
                                violation_distance = counterexample.internal_violation - violation_here
                                shifted_property = shift_property(counterexample.property, violation_distance)
                                # TODO: in some situations registering a new property twice here
                                #       (strengthened + internal violation)
                                repair_backend.register_strengthened_property(shifted_property, prop)
                                repair_backend.new_counterexample(shifted_property, counterexample.inputs_as_tensor())
                                counterexamples_for_checkpoint.append((shifted_property, counterexample.inputs))
                            else:
                                repair_backend.new_counterexample(counterexample.property,
                                                                  counterexample.inputs_as_tensor())
                                counterexamples_for_checkpoint.append((counterexample.property, counterexample.inputs))

        # check whether all properties are verified
        # (or no counterexamples found using falsifiers if there is no verifier present)
        # Note: falsifier errors also set falsifier_exhaustion to True
        if verifier is not None:
            all_properties_done = all(st.verified or st.unable_to_verify for st in property_status.values())
        else:
            all_properties_done = all(all(st.falsifier_exhaustion) for st in property_status.values())
        # do not perform the final checks in the training mode in the first iteration
        if all_properties_done and (not training_mode or i > 0):
            info(f"No further counterexamples found. Checking skipped properties...")
            for prop in (p for p, st in property_status.items() if not st.fresh_results):
                # rerun the verifier for all properties where we do not have results for the latest changes
                # to the network. If we do not have a verifier, run the last falsifier
                checker: CounterexampleGenerator
                if verifier is not None:
                    info(f"Verifying skipped property {prop.property_name} for latest network using {verifier.name}.")
                    checker = verifier
                else:
                    checker = falsifier_cascade[-1]
                    info(f"Falsifying skipped property {prop.property_name} for latest network using {checker.name}.")

                with LogExecutionTime(f'{checker.name} for property {prop.property_name}',
                                      enable=measure_execution_times):
                    counterexamples, status = checker.find_counterexample(repair_backend.network, prop)

                if counterexamples is None:  # None indicates error here
                    warning(f"Error while checking skipped property: {prop.property_name}: Status: {status}.")
                    if verifier is not None:
                        property_status[prop].unable_to_verify = True
                    else:
                        property_status[prop].falsifier_exhaustion[-1] = True
                        property_status[prop].falsifier_error[-1] = True
                elif len(counterexamples) > 0:
                    info(f"Property {prop.property_name} does not hold, although it did previously hold."
                         f"Resuming repair and always checking this property from now on.")
                    found_counterexamples = True
                    for counterexample in counterexamples:
                        # violation = -prop.satisfaction_function(
                        #     counterexample.network_outputs_as_tensor().unsqueeze(0),
                        #     counterexample.inputs_as_tensor().unsqueeze(0)
                        # ).item()
                        violation = -counterexample.property_satisfaction
                        info(f'Counterexample violation of new counterexample: {violation:.10f}')
                        repair_backend.new_counterexample(prop, counterexample.inputs_as_tensor())
                        counterexamples_for_checkpoint.append((prop, counterexample.inputs))
                    property_status[prop].verified = False
                    # also give all falsifiers another try
                    property_status[prop].falsifier_exhaustion = [False] * len(falsifier_cascade)
                    property_status[prop].falsifier_error = [False] * len(falsifier_cascade)
                    property_status[prop].unable_to_verify = False
                    property_status[prop].always_check = True

            if all(st.verified or st.unable_to_verify for st in property_status.values())\
                    or verifier is None and all(all(st.falsifier_exhaustion) for st in property_status.values()):
                break  # repair achieved!!!

        # ********************************************************
        # enough verification, falsification; let the repair begin

        if training_mode and i == 0 and not found_counterexamples:
            info("No violations found in first training mode iteration. Performing regular training.")
            with LogExecutionTime('training loop', enable=measure_execution_times):
                training_mode_training_loop.execute()
            backend_status = RepairStatus.SUCCESS
        else:
            num_violated_properties = sum(  # True is interpreted as 1
                not all(st.falsifier_exhaustion) or (verifier is not None and not st.verified)
                for st in property_status.values()
            )
            info(f'Found {num_violated_properties} specification violations:\n'
                 f'{repair_backend.violation_stats()}')

            with LogExecutionTime('repair backend', enable=measure_execution_times):
                backend_status = repair_backend.repair()

        checkpoint_handler(i, tuple(counterexamples_for_checkpoint), repair_backend.network)
        # the network was modified. All falsification/verification results are no longer fresh
        for prop in property_status:
            property_status[prop].fresh_results = False

        if backend_status == RepairStatus.FAILURE:
            warning("Backend could not repair counterexamples. " +
                    ("Aborting repair." if abort_on_backend_error else "Continuing anyway."))
            if abort_on_backend_error:
                return RepairStatus.FAILURE, repair_backend.network
        if backend_status == RepairStatus.ERROR:
            return RepairStatus.ERROR, repair_backend.network

        if post_repair_step_hook is not None:
            post_repair_step_hook(i)

        i += 1
    else:
        # loop was not ended with break
        warning("Repair failed: Maximum number of iterations exhausted.")
        return RepairStatus.FAILURE, repair_backend.network

    if any(any(st.falsifier_error) or st.unable_to_verify for st in property_status.values()):
        # log a tabular summary of the errors
        property_names = [prop.property_name for prop in specification]
        property_name_max_length = max(map(len, property_names))
        property_name_max_length = max(property_name_max_length, len('Property'))
        summary_table = '(*) indicates an error.\n\n'
        table_header = 'Property' + (' ' * (property_name_max_length - len('Property'))) + ' '
        if verifier is not None:
            table_header += ' | ' + verifier.name
        for falsifier in falsifier_cascade:
            table_header += ' | ' + falsifier.name
        summary_table += table_header + '\n'
        summary_table += ('-' * len(table_header)) + '\n'
        for prop in specification:
            status = property_status[prop]
            summary_table += prop.property_name + (' ' * (property_name_max_length - len(prop.property_name))) + ' '
            if verifier is not None:
                summary_table += ' | ' + ('*' if status.unable_to_verify else ' ') + (' ' * (len(verifier.name) - 1))
            for fi, falsifier in enumerate(falsifier_cascade):
                summary_table += ' | ' + ('*' if status.falsifier_error[fi] else ' ') \
                                 + (' ' * (len(falsifier.name) - 1))
            summary_table += '\n'
        info("A counterexample generation problem was detected during the repair.\n" + summary_table)

    # verification of some properties might have failed
    if verifier is not None and any(st.unable_to_verify for st in property_status.values()):
        warning(f"The following properties could not be verified due to errors: "
                f"{[prop.property_name for prop, st in property_status.items() if st.unable_to_verify]}")
        return RepairStatus.ERROR, repair_backend.network
    else:
        info("All properties verified! Repair successful.")
        return RepairStatus.SUCCESS, repair_backend.network
