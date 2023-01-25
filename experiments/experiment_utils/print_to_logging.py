# solution from https://stackoverflow.com/a/2216517/10550998
# by blokeley (modified)
# licensed under CC BY-SA 2.5
# Due to CC BY-SA this file is licensed also under CC BY-SA as a modification of blokeleys answer
import sys
import logging


class LoggerAsFile:
    """File-like object to log text using the `logging` module."""

    def __init__(self, name=None, level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.level = level

    def write(self, msg, level=None):
        if level is None:
            level = self.level
        self.logger.log(level, msg)

    def flush(self):
        for handler in self.logger.handlers:
            handler.flush()


class PrintToLogging:
    """
    A context manager for temporarily redirecting printing to stdout and stderr to
    the root logger
    """

    def __init__(self):
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def __enter__(self):
        sys.stdout = LoggerAsFile()
        sys.stderr = LoggerAsFile(level=logging.ERROR)

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout
        sys.stderr = self.stderr