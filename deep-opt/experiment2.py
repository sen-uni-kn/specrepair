import platform
import time
from datetime import datetime
from pathlib import Path
import logging
import sys

import ray

from deep_opt.numerics.optimization import optimize_by_property
from properties import get_properties

ray.init()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:\n%(message)s", stream=sys.stdout)
    for (prop, number) in get_properties():
        if number in ["1", "2"]:
            log_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_file_path = Path(f"./output/experiment2/result_prop{number}_deep_opt_{log_time}.log")
            log_file_path.touch(exist_ok=False)

            with log_file_path.open(mode='a') as log_file:
                log_file.write(
                    ("# DeepOpt logs\n"
                     "specification:\n"
                     "    - time: \"{}\"\n"
                     "    - property: {}\n"
                     "    - system_info: {{name: {}, os: {}}}\n"
                     "results:\n"
                     ).format(log_time, number, platform.uname().node, platform.uname().system)
                )

                network_ranges = prop.network_ranges

                mean_execution_time = 0
                num_networks = 0

                for hu in [25, 50, 100, 250, 500, 1000]:
                    start_time_over_approx = time.time()

                    # inputs file
                    network_file = './resources/hcas/HCAS_rect_v6_pra0_tau00_{}HU.nnet'.format(hu)

                    title = f'{hu}'

                    result_over_approx = optimize_by_property(network_file, prop, title)

                    time_over_approx = time.time() - start_time_over_approx

                    log_file.write(
                        ("    - network: {}\n"
                         "        - optimize_by_property:\n"
                         "          result: {}\n"
                         "          counter example: {}\n"
                         "          number of local extrema: {{min: {}}}\n"
                         "          execution_time: {}\n"
                         ).format(title,
                                  result_over_approx[0],
                                  result_over_approx[1],
                                  result_over_approx[2],
                                  time_over_approx)
                    )

                    mean_execution_time += time_over_approx
                    num_networks += 1
                    log_file.flush()

                    if num_networks > 0:
                        mean_execution_time = mean_execution_time / num_networks

                log_file.write("Mean execution time: {}".format(mean_execution_time))
