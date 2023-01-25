# generates a README file summarising the contents of an output directory
# in a human commenting friendly fashion.
# New summaries are appended to the end of the file only if the name of the subdirectory
# is not yet present in the file.
# This script builds on experiment results being already collected, i.e. summary or overview files being
# present in each subdirectory.
# The prelim/ subdirectory is ignored
import argparse
import os
from pathlib import Path
import ruamel.yaml
import time
from datetime import datetime


def time_from_timestamp(timestamp):
    try:
        time1 = datetime.strptime(timestamp, "%Y-%m-%d_%H-%M-%S")
        return time.mktime(time1.timetuple())
    except ValueError:
        return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate/Complement output directory README')
    parser.add_argument('output_dir', type=str,
                        help='The top level experiment output directory containing the subdirectories'
                             'of the individual runs.')
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    experiment_run_dirs = []
    for dir_element in os.listdir(output_dir):
        dir_element_path = Path(output_dir, dir_element)
        if dir_element_path.is_dir() and dir_element != 'prelim':
            experiment_run_dirs.append(dir_element_path)
    experiment_run_dirs = sorted(experiment_run_dirs, key=lambda path: time_from_timestamp(path.name))
    print(f'Found {len(experiment_run_dirs)} experiment run output directories.')

    readme_path = Path(output_dir, 'README.md')
    if not readme_path.exists():
        print('No README found, creating new one.')
        with open(readme_path, 'wt') as readme_file:
            # transform the directory name into a better looking headline
            experiment_name = ' '.join(word.capitalize() for word in output_dir.name.split('_'))
            readme_file.write(f'# {experiment_name}\n'
                              f'EXPERIMENT DESCRIPTION MISSING\n\n')

    # get the whole file contents
    with open(readme_path, 'rt') as readme_file:
        current_readme_contents = readme_file.read()

    yaml = ruamel.yaml.YAML(typ='safe')
    # now append short descriptions for all runs that are not already included in the
    # README file
    with open(readme_path, 'at') as readme_file:
        for run_dir in experiment_run_dirs:
            if run_dir.name not in current_readme_contents:
                # look for the summary.yaml file or overview.yaml file
                # if there is only one case in the subdirectory
                run_summary_path = Path(run_dir, 'results.yaml')
                backends = []
                networks = []
                cases = None
                if run_summary_path.exists():
                    with open(run_summary_path, 'rt') as summary_file:
                        yaml_results = yaml.load(summary_file)
                    summary = yaml_results['summary']
                    backends = summary['repair_backends']
                    networks = summary['networks'] if 'network' in yaml_results else None
                    cases = list(yaml_results['cases'].keys())
                else:
                    print(f'Run directory does not contain collected results: {run_dir}. '
                          f'Run collected_experiment_results before running this script.')
                    continue  # Nothing to write, so don't try

                readme_file.write(f'## {run_dir.name}\n')
                if len(backends) == 1:
                    readme_file.write(f'Backend: {backends[0]}\n')
                else:
                    readme_file.write(f'Backends: {backends}\n')
                if networks is not None and len(networks) == 1:
                    readme_file.write(f'Network: {networks[0]}\n')
                elif networks is not None:
                    readme_file.write(f'Networks: {networks}\n')
                if cases is not None:
                    readme_file.write(f'{len(cases)} cases: {cases}\n')
                readme_file.write('\n')



