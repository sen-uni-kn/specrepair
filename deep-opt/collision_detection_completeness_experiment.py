import platform
from datetime import datetime
from pathlib import Path
import logging
from logging import basicConfig, info
import sys

import torch
import ray
import pickle

from deep_opt.numerics.optimization import optimize_by_property

# If deep-opt is complete for continuously differentiable networks,
# then it's results in one iteration need to coince exactly with
# the results of ERAN in complete mode.
# To test this we run deep-opt for 100 robustness properties (each one for a single sample)
# and compare the results with the ERAN results.
# To make sure that the test is not too simple, we also run this experiment with
# a non continuously differentiable network (ReLU based), for which deep-opt is incomplete


if __name__ == '__main__':
    log_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = Path(f"output/collision_detection_completeness_experiment/results_{log_time}.yaml")
    log_file_path.touch(exist_ok=False)
    basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", handlers=[
        logging.FileHandler(f"output/collision_detection_completeness_experiment/log_{log_time}.txt"),
        logging.StreamHandler(sys.stdout)
    ])

    ray.init()

    relu_network = torch.load('resources/collision-detection/ReLU_network.pyt')
    tanh_network = torch.load('resources/collision-detection/tanh_network.pyt')
    info(f"ReLU network: {relu_network}")
    info(f"tanh network: {tanh_network}")

    # IMPORTANT: these results are most likely not valid (12.04.2021)
    with open('resources/collision-detection/ReLU_robustness_verification_results.yaml') as f:
        relu_verification_results = pickle.load(f)
    with open('resources/collision-detection/tanh_robustness_verification_results.yaml') as f:
        tanh_verification_results = pickle.load(f)


    with log_file_path.open(mode='a') as log_file:
        log_file.write(f"# DeepOpt CollisionDetection Completeness Experiment Logs\n"
                       f"run_specification:\n"
                       f"    - time: \"{log_time}\"\n"
                       f"    - system_info: {{name: {platform.uname().node}, os: {platform.uname().system}}}\n"
                       f"    - branch: 2021-03-25_completeness_experiment\n"
                       f"relu_results:\n")
        # test if ReLU results are replicated first, to have a baseline
        relu_deviated_from_eran = []
        relu_error_encountered = False
        i = 0
        for res_dict in relu_verification_results:
            prop = res_dict['property']
            does_hold = res_dict['verified']
            status, _, _ = optimize_by_property(relu_network, prop, title=f"ReLU network robustness #{i}")
            log_file.write(f"    - {status}\n")
            if status == 'ERROR':
                relu_error_encountered = True
            else:
                deep_opt_does_hold = status == 'UNSAT'
                relu_deviated_from_eran.append(does_hold != deep_opt_does_hold)
            info(f"ReLU network robustness #{i}: does_hold: {does_hold}; deep_opt_does_hold: {deep_opt_does_hold}")
            i += 1
        relu_num_deviations = sum(relu_deviated_from_eran)
        info(f"ReLU: DeepOpt result deviated from the ground truth (ERAN) in {relu_num_deviations} cases.\n"
             f"Error encountered: {relu_error_encountered}")

        # now run the same for the tanh networks
        log_file.write(f"tanh_results:\n")
        tanh_deviated_from_eran = []
        tanh_error_encountered = False
        i = 0
        for res_dict in tanh_verification_results:
            prop = res_dict['property']
            does_hold = res_dict['verified']
            status, _, _ = optimize_by_property(tanh_network, prop, title=f"tanh network robustness #{i}")
            log_file.write(f"    - {status}\n")
            if status == 'ERROR':
                tanh_error_encountered = True
            else:
                deep_opt_does_hold = status == 'UNSAT'
                tanh_deviated_from_eran.append(does_hold != deep_opt_does_hold)
            info(f"tanh network robustness #{i}: does_hold: {does_hold}; deep_opt_does_hold: {deep_opt_does_hold}")
            i += 1
        tanh_num_deviations = sum(tanh_deviated_from_eran)
        info(f"tanh: DeepOpt result deviated from the ground truth (ERAN) in {tanh_num_deviations} cases.\n"
             f"Error encountered: {tanh_error_encountered}")

        log_file.write(f"overall_result:\n"
                       f"    - relu_num_deviations_from_ground_truth: {relu_num_deviations}\n"
                       f"    - tanh_num_deviations_from_ground_truth: {tanh_num_deviations}\n"
                       f"    - relu_deep_opt_error_encountered: {relu_error_encountered}\n"
                       f"    - tanh_deep_opt_error_encountered: {tanh_error_encountered}\n")
