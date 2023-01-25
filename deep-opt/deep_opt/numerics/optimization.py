from copy import deepcopy
from typing import List, Union, Tuple, Callable, Iterable, Optional, Sequence

from logging import info, error
from enum import Flag, auto

import random
import numpy as np
import torch
import ray
from scipy import optimize
from scipy.optimize import OptimizeResult
import scipy.spatial.qhull
from tqdm import tqdm

from deep_opt.models.neural_network import NeuralNetwork
from deep_opt.models.property import Property
from deep_opt.utils.ray_utils import ray_map

# NOTE: floating point precision
# because single precision / float32 is standard in torch
# and double precision is not supported by ONNX
# for compatibility np.float32 is used as floating point type throughout this file


class CounterexampleAmount(Flag):
    """
    Control the number of generated counterexamples by specifying places
    where multiple counterexamples may be generated.
     * SINGLE: only generate a single counterexample. This counterexample will be the one with the largest
       violation in the input bound that is processed first. This maybe somewhat non-deterministic due to
       parallel execution on multiple input bounds.
     * ALL_MINIMA: return all minima for an input region that constitute counterexamples. This only adds a little
       additional runtime as it only involves checking property satisfaction for a larger number of input samples.
       Multiple minima are calculated either way.
     * FOR_ALL_INPUT_BOUNDS: return the counterexample(s) found for all input bounds, do not stop with the
       first obtained violation. This may significantly increase runtime as there might be a large number
       of input regions in later iterations.
     * ALL_ITERATIONS: Always run up to the maximum number of iterations and return all counterexamples found
       during that process.
    In any case, the overall runtime should be comparable with the time optimize_by_property takes if no counterexamples
    can be found. Only the ALL_MINIMA option in combination with all other options may lead to an increase beyond
    that mark, but this should only be relevant if there is a huge number of minima.
    """
    SINGLE = 0
    ALL_MINIMUMS = auto()
    FOR_ALL_INPUT_BOUNDS = auto()
    ALL_ITERATIONS = auto()


def optimize_by_property(network: Union[str, NeuralNetwork], prop: Property, title: str = None,
                         how_many_counterexamples: CounterexampleAmount = CounterexampleAmount.SINGLE,
                         it_simple_upsampling: int = 2, it_peak_sampling: int = 1, it_split_sampling: int = 1,
                         n=None, half_bounds_per_dim=False, cutoff: Optional[float] = 0.001,
                         use_gradient_information=True) \
        -> Tuple[str, Union[None, np.ndarray, List[np.ndarray]], int]:
    """
    DeepOpt property optimization.

    :param title: display title.
    :param network: file path to a .nnet file or a network to falsify.
    :param prop: Input constraints and output property tuple
           (bounds, function, output constraint).
    :param how_many_counterexamples: Controls how many counterexamples are generated.
     See `CounterexampleAmount`s documentation for more information.
    :param it_simple_upsampling Iteration count for simple up-sampling.
    :param it_peak_sampling     Iteration count for peak bound based up-sampling.
    :param it_split_sampling    Iteration count for bound splitting based up-sampling.
    :param n                    (Optional) sampling count.
    :param half_bounds_per_dim  (Optional) split sampling splits at every dimension.
    :param cutoff: (Optional) sufficiently small objective value to terminate search.
      Pass None to search for the global optimum.
    :param use_gradient_information: Whether to use a local optimiser than utilises gradient information for SHGO
     or not. The optimiser using gradient optimisation is SLSQP and the gradient-free one is COBYLA.
     This option only applies for the counterexample search. Input region refinement is always performed using COBYLA.
    :return: SAT/UNSAT/ERROR, Counter example(s)/None, minimums count.
     If ``how_many_counterexamples`` is set to ``SINGLE`` (default value),
     then the second return values is a single np.array if a counterexample were found.
     For other values of ``how_many_counterexamples`` a list of counterexamples is returned
     if counterexamples were found.
    """

    if isinstance(network, str):
        network = NeuralNetwork.load_from_nnet(network)
    else:
        # need to place the network on CPU for ray
        network = deepcopy(network)
        network.cpu()

    it_total = it_simple_upsampling + it_peak_sampling + it_split_sampling

    # 1 Optimize.

    # 1.1 Get bounds (basic input constraints).
    bounds: Tuple[Tuple[float, float]] = prop.input_bounds(network)
    current_bounds = [bounds]

    # 1.2 Get optimizable function (output property).
    # start with the original network (approximate later)
    objective_function = lambda x_: prop.satisfaction_function(x_, network)
    if prop.input_constraint is None:
        constraint = None
    else:
        constraint = prop.input_constraint_satisfaction_function

    minimizer_point_failed = False
    # record total number of function evaluations
    sum_nfev = 0
    sum_nlfev = 0

    if title:
        info("------------------" + "-"*len(title) + "\n" 
             f"Local minimums ({title}):\n"
             "------------------" + "-"*len(title))

    # 2 Check property.
    local_minima: List[np.ndarray] = []
    counterexamples: List[np.ndarray] = []
    try:
        for i in range(it_total):
            progress = None
            # first up-sampling step: increase sampling density inside SHGO
            if i < it_simple_upsampling:
                sample_dim = len(bounds)
                if n is None:
                    local_n = min(50, (int(np.log(sample_dim)) + i) * 16)
                    local_n = max(2, local_n)
                else:
                    local_n = n + i * 16
                info(f"({i}): Simple up-sampling (n={local_n})")
                res = _shgo(objective_function, bounds, constraint, local_n, use_gradient_information, cutoff)
                # make it a generator for iteration
                shgo_results = (res for _ in range(1))
            elif i < it_simple_upsampling + it_peak_sampling and not minimizer_point_failed:
                new_bounds = []
                results = ray_map(
                    _get_bounds_by_peaks, current_bounds, 1,
                    objective_function, constraint=constraint, n=10, iters=1, use_gradient=use_gradient_information
                )
                for result in results:
                    bounds_by_peaks, res = result
                    sum_nfev += res.nfev
                    sum_nlfev += res.nlfev
                    new_bounds.append(bounds_by_peaks)
                    if res.success is False:
                        minimizer_point_failed = True
                current_bounds = _flatten(new_bounds)

                local_n = 100 if n is None else n
                shgo_results = ray_map(
                    _shgo_remote, current_bounds, 1,
                    objective_function, constraint=constraint, n=local_n,
                    use_gradient=use_gradient_information, cutoff=cutoff
                )
                info(f"({i}): Peak up-sampling (count={len(current_bounds)}, n={local_n})")
                progress = tqdm(desc=f'🥑 Processing peak bounds', total=len(current_bounds))
            else:
                current_bounds = _flatten([_half_bounds(bound) if not half_bounds_per_dim else _half_bounds_per_dim(bound)
                                           for bound in current_bounds])

                local_n = 100 if n is None else n
                shgo_results = ray_map(
                    _shgo_remote, current_bounds, 1,
                    objective_function, constraint=constraint, n=local_n,
                    use_gradient=use_gradient_information, cutoff=cutoff
                )
                info(f"({i}): Split up-sampling (count={len(current_bounds)}, n={local_n})")
                progress = tqdm(desc=f'🥑 Processing split bound spaces', total=len(current_bounds))

            for res in shgo_results:
                if progress is not None:
                    progress.update()
                sum_nfev += res.nfev
                sum_nlfev += res.nlfev

                if how_many_counterexamples & CounterexampleAmount.ALL_MINIMUMS and 'xl' in res.keys():
                    xl = res.xl
                else:
                    xl = [res.x]

                for x in xl:
                    x_tensor = torch.tensor(x).unsqueeze(0)
                    y = prop.calc_network_outputs_tensor(x_tensor, network)
                    if isinstance(y, Iterable):  # multi variable properties return iterables, not single tensors
                        y = torch.stack([y_elem for y_elem in y])
                    # y may have multiple dimensions. x is calculated here, so it is flat for sure
                    point = np.array([*x, *y.flatten().detach().cpu().numpy()])

                    local_minima.append(point)
                    if not prop.property_satisfied(x_tensor, network):
                        info("❌ Counterexample found! {}".format(tuple(map(lambda x_: round(x_, 4), point))))
                        counterexamples.append(point)

                        if not how_many_counterexamples:  # => SINGLE
                            shgo_results.close()
                            if progress is not None:
                                progress.close()
                            info(f"DeepOpt finished. nfev={sum_nfev}, nlfev={sum_nlfev}")
                            return 'SAT', point, -1  # num local minima not available

                if len(counterexamples) > 0 and not how_many_counterexamples & CounterexampleAmount.FOR_ALL_INPUT_BOUNDS:
                    # stop here, no need to look at the other input bounds
                    shgo_results.close()
                    if progress is not None:
                        progress.close()
                    info(f"DeepOpt finished. nfev={sum_nfev}, nlfev={sum_nlfev}")
                    return 'SAT', counterexamples, len(local_minima)

            if len(counterexamples) > 0 and not how_many_counterexamples & CounterexampleAmount.ALL_ITERATIONS:
                info(f"DeepOpt finished. nfev={sum_nfev}, nlfev={sum_nlfev}")
                return 'SAT', counterexamples, len(local_minima)

        if len(counterexamples) > 0:
            info(f"DeepOpt finished. nfev={sum_nfev}, nlfev={sum_nlfev}")
            return 'SAT', counterexamples, len(local_minima)
        else:
            info("⭕️ Global minimum does not violate specification.")
            info(f"DeepOpt finished. nfev={sum_nfev}, nlfev={sum_nlfev}")
            return 'UNSAT', None, len(local_minima)
    except Exception as ex:
        error(f"Error in DeepOpt: {ex}")
        return 'ERROR', None, -1


@ray.remote
def _get_bounds_by_peaks(objective_function: Callable,
                         bounds: Tuple[Tuple[np.float32, np.float32]], constraint: Optional[Callable],
                         n: int, iters: int, use_gradient: bool) \
        -> Tuple[List[Tuple[Tuple[np.float32, np.float32]]], Optional[OptimizeResult]]:
    """
    Return new bounds that go around the function's peaks.

    :param objective_function: An optimizable function.
    :param bounds: Function bounds.
    :param constraint: A constraint on the input.
    :param: n: Number of sampling points.
    :param iters: Number of SHGO iterations.
    :param use_gradient: Whether to use SLSQP (use_gradient=True) or COBYLA.
    :return: New bounds that go around the function's peaks.
    """
    def flipped_objective(_x):
        return -objective_function(_x)

    res = _shgo(flipped_objective, bounds, constraint, n, use_gradient, cutoff=None)
    current_nfev = res.nfev
    current_nlfev = res.nlfev

    _objective, _gradient = _get_objective_and_gradient(flipped_objective, cutoff=None)
    if constraint is not None:
        _constraint, _constraint_gradient = _get_objective_and_gradient(constraint, cutoff=None)

    if constraint is not None:
        constr = {
            'type': 'ineq',
            'fun': _constraint
        }
    else:
        constr = None
    sampling_method = 'sobol'
    minimizer_kwargs = {}
    if use_gradient:
        minimizer_kwargs['method'] = 'SLSQP'
        minimizer_kwargs['jac'] = _gradient
        # options = {'jac': True}
        if constr is not None:
            constr['jac'] = _constraint_gradient
    else:
        minimizer_kwargs['method'] = 'COBYLA'

    i = 0
    d = (max([upper for (_, upper) in bounds]) - min([lower for (lower, _) in bounds])) / 10000
    while not res.success and 'Failed to find a feasible minimizer point' in res.message:
        try:
            slightly_altered_bounds = [(lower - random.uniform(0, d), upper + random.uniform(0, d))
                                       for (lower, upper) in bounds]

            info('Local minimization...')
            res = optimize.shgo(_objective,
                                bounds=slightly_altered_bounds,
                                constraints=constr,
                                n=n,
                                iters=iters,
                                sampling_method=sampling_method,
                                minimizer_kwargs=minimizer_kwargs)

            current_nfev += res.nfev
            if 'nlfev' in res.keys():
                current_nlfev += res.nlfev
        except scipy.spatial.qhull.QhullError:
            info('📦️ QhullError')
            return [bounds], res
        d *= 2
        i += 1

        if i == 2:
            info(f'No minimizer point for {i} iterations.')

            res.nfev = current_nfev
            res.nlfev = current_nlfev
            # return [bounds], res
            break

    bounds_by_peaks = []
    fraction = 2

    if 'xl' in res.keys():  # this occurs if no minimiser could be found.
        info(f'Found {len(res.xl)} peaks...')
    elif 'x' in res.keys():
        res.xl = [res.x]
    else:
        info('Found no peak.')
        return [bounds], res

    for x in res.xl:
        bound = []
        for j in range(len(bounds)):
            lower, upper = bounds[j]
            difference = upper - lower
            x_bound: Tuple[np.float32, np.float32] = (
                max(lower, x[j] - difference / fraction),
                min(upper, x[j] + difference / fraction)
            )
            # Don't move out of the original bounds
            if x_bound[0] < lower:
                x_bound = (lower, x_bound[1])
            if x_bound[1] > upper:
                x_bound = (x_bound[0], upper)

            if x_bound[0] > x_bound[1]:
                x_bound = (lower, upper)
            bound.append(x_bound)
        bounds_by_peaks.append(tuple(bound))

    res.nfev = current_nfev
    res.nlfev = current_nlfev
    return bounds_by_peaks, res


@ray.remote
def _shgo_remote(*args, **kwargs):
    return _shgo(*args, **kwargs)


def _shgo(objective_function: Callable, bounds: Sequence[Tuple[float, float]], constraint: Optional[Callable],
          n: int, use_gradient: bool = True, cutoff: Optional[float] = 0.001):
    """
    :param objective_function function to optimize.
    :param bounds input constraints.
    :param constraint: constraint function on the input.
    :param n sampling point count.
    :param use_gradient: Whether to use SLSQP (use_gradient=True) or COBYLA.
    :param cutoff: Cutoff parameter to terminate SHGO early. Pass None to disable cutoff.
      With cutoff active, this routine will only search for a solution with an objective function value at most as
      large as the cutoff. Without cutoff it will search for the actual global minimum.
    """
    _objective, _gradient = _get_objective_and_gradient(objective_function, cutoff)
    if constraint is not None:
        _constraint, _constraint_gradient = _get_objective_and_gradient(constraint, cutoff=None)

    if constraint is not None:
        constr = {
            'type': 'ineq',
            'fun': _constraint
        }
    else:
        constr = None
    sampling_method = 'sobol'
    minimizer_kwargs = {}
    iters: int
    options = {'minimize_every_iter': True}
    if cutoff is not None:
        options['f_min'] = cutoff
    if use_gradient:
        minimizer_kwargs['method'] = 'SLSQP'
        minimizer_kwargs['jac'] = _gradient
        # options = {'jac': True}
        if constr is not None:
            constr['jac'] = _constraint_gradient
        iters = 3
    else:
        minimizer_kwargs['method'] = 'COBYLA'
        iters = 1

    try:
        res = optimize.shgo(_objective, bounds=bounds, constraints=constr,
                            n=n, iters=iters, sampling_method=sampling_method,
                            minimizer_kwargs=minimizer_kwargs, options=options)
    except scipy.spatial.qhull.QhullError:
        slightly_altered_bounds = [(lower - random.uniform(0, max(50, np.abs(upper - lower) / 100)),
                                    upper + random.uniform(0, max(50, np.abs(upper - lower) / 100)))
                                   for (lower, upper) in bounds]

        res = optimize.shgo(_objective, bounds=slightly_altered_bounds, constraints=constr,
                            n=n, iters=3, sampling_method=sampling_method,
                            minimizer_kwargs=minimizer_kwargs, options=options)

        if res.success is False:
            _objective, _gradient = _get_objective_and_gradient(objective_function, cutoff=None)
            if 'jac' in minimizer_kwargs:
                minimizer_kwargs['jac'] = _gradient
            res = optimize.shgo(_objective, bounds=slightly_altered_bounds, constraints=constr,
                                n=n, iters=3,
                                sampling_method=sampling_method, minimizer_kwargs=minimizer_kwargs, options=options)
    return res


def _get_objective_and_gradient(objective_function, cutoff):
    # FIXME: scipy should actually support calculating the gradient alongside the objective function
    #        An issue was filled: https://github.com/scipy/scipy/issues/13547
    # For now we can just use multiple functions (but this requires more function evaluations)
    if cutoff is not None:
        def _objective_with_cutoff(x: torch.Tensor) -> torch.Tensor:
            obj_value = objective_function(x)
            obj_value = torch.maximum(torch.tensor(-cutoff), obj_value)
            return obj_value
        _tensor_objective = _objective_with_cutoff
    else:
        _tensor_objective = objective_function

    def _grad(x: np.array):
        inputs = torch.as_tensor(x, dtype=torch.float).unsqueeze(0).requires_grad_(True)
        obj_value = _tensor_objective(inputs)
        dy_dx = torch.autograd.grad(obj_value, inputs)[0]
        return dy_dx.detach().cpu().numpy()

    @torch.no_grad()
    def _obj(x: np.array) -> torch.Tensor:
        inputs = torch.as_tensor(x, dtype=torch.float).unsqueeze(0)
        obj_value = _tensor_objective(inputs)
        return obj_value.item()
    return _obj, _grad


def _flatten(list_of_lists: Iterable[List]) -> List:
    """
    Flatten a list of lists.
    :param list_of_lists: list of lists.
    :return: Flattened list.
    """
    return [item for sublist in list_of_lists for item in sublist]


def _half_bounds(bounds: Tuple[Tuple[np.float32, np.float32]]) -> List[Tuple[Tuple[np.float32, np.float32]]]:
    """
    Take bounds and return two half-spaces.
    :param bounds:
    :return:
    """
    output_bounds = []

    lower_bounds = []
    upper_bounds = []

    # split bounds by dimension
    for bound in bounds:
        split_boundary: np.float32 = bound[0] + (bound[1] - bound[0]) / np.float32(2.0)

        lower_bounds.append((bound[0], split_boundary))
        upper_bounds.append((split_boundary, bound[1]))

    # build all case combinations (exponential increase)
    num_cases = pow(2, len(bounds))

    for case_idx in range(num_cases):
        binary = '{0:b}'.format(case_idx).zfill(len(bounds))

        current_bound_case = []

        for i in range(len(binary)):
            if binary[i] == '0':
                current_bound_case.append(lower_bounds[i])
            else:
                current_bound_case.append(upper_bounds[i])

        output_bounds.append(tuple(current_bound_case))

    return output_bounds


def _half_bounds_per_dim(bounds: Tuple[Tuple[np.float32, np.float32]]) -> List[Tuple[Tuple[np.float32, np.float32]]]:
    """
    Take bounds and return two half-spaces per dimension.

    :param bounds: original bounds.
    :return: list of new bounds.
    """
    output_bounds = []

    lower_bounds = []
    upper_bounds = []

    # split bounds by dimension
    for bound in bounds:
        split_boundary = bound[0] + (bound[1] - bound[0]) / 2.0

        lower_bounds.append((bound[0], split_boundary))
        upper_bounds.append((split_boundary, bound[1]))

    # build all case combinations (exponential increase)
    num_cases = pow(2, len(bounds))
    for case_idx in range(num_cases):
        binary = '{0:b}'.format(case_idx).zfill(len(bounds))

        current_bound_case = []

        for i in range(len(binary)):
            if binary[i] == '0':
                current_bound_case.append(lower_bounds[i])
            else:
                current_bound_case.append(upper_bounds[i])
        output_bounds.append(tuple(current_bound_case))
    return output_bounds
