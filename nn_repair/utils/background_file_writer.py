from typing import Union, Callable, Any

import torch
import pickle
import dill

import threading
from queue import Queue


class BackgroundFileWriter(threading.Thread):
    """
    A thread that writes objects that arrive on a queue to files.

    Since writing too files is not CPU-heavy, this class allows to save e.g. checkpoints
    in a non-blocking way.

    The thead can be terminated by sending None to the object_queue.
    """
    def __init__(self, object_queue: Queue, writing_function: Union[str, Callable]):
        """
        Constructs a BackgroundFileWriter.

        :param object_queue: The queue to which (object, filename) tuples will be sent.
         The BackgroundFileWriter will then write the given objects to the file with the specified name using
         the writing_function. If None is retrieved from this queue, the thread terminates.

         The objects and filenames need to be such that the writing_function can handle.
         Otherwise there are no restrictions.
        :param writing_function: The function used to write objects to files.
         You can either use a default option by providing a string. Permitted values are ``'torch'``,
         ``'pickle'`` and ``'dill'``, which will store objects using ``torch.save``,
         ``pickle.dump`` and ``dill.dump`` respectively. To dump using pickle or dill, the given file will
         be opened in binary write mode ('wb').

         Alternatively you can directly provide a callable, which accepts the object to be written and
         the filename as arguments. If providing a custom callable, the queue may also contain different elements
         but tuples of objects and filenames. All elements of the queue will be passed to the write function
         as arguments using the star operator (``writing_function(*element)``).
        """
        super().__init__()
        self.queue = object_queue
        if writing_function == 'torch':
            def torch_write(obj, filename):
                with open(filename, 'wb') as file:
                    torch.save(obj, file)
            self.writing_fn = torch_write
        elif writing_function == 'pickle':
            def pickle_write(obj, filename):
                with open(filename, 'wb') as file:
                    pickle.dump(obj, file)
            self.writing_fn = pickle_write
        elif writing_function == 'dill':
            def dill_write(obj, filename):
                with open(filename, 'wb') as file:
                    dill.dump(obj, file)
            self.writing_fn = dill_write
        else:
            self.writing_fn = writing_function

    def run(self) -> None:
        while True:
            queue_element = self.queue.get()
            if queue_element is None:
                self.queue.task_done()
                break
            self.writing_fn(*queue_element)
            self.queue.task_done()


