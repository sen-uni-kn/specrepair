from time import time
from logging import info


class LogExecutionTime:
    """
    A context manager to log the execution time of falsifiers, verifiers,
    and the repair backend.
    """
    def __init__(self, operation_name, enable=True):
        self.operation_name = operation_name
        self.start = None
        self.measure_execution_times = enable

    def __enter__(self):
        if self.measure_execution_times:
            self.start = time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.measure_execution_times:
            duration = time() - self.start
            # do not log more than milliseconds
            info(f'Executing {self.operation_name} took: {duration:.3f} seconds.')
