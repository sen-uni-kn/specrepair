from typing import Iterable, Generator, TypeVar
import ray
from ray.remote_function import RemoteFunction

T = TypeVar('T')
V = TypeVar('V')


def ray_map(func: RemoteFunction, source: Iterable[T], source_param_at: int = 0, *args, **kwargs) \
        -> Generator[V, None, None]:
    """
    Applies a `@ray.remote` function (func) to the elements of an iterable (source).<br>
    The computation results are returned as an iterator that waits for the respective
    computation is done. The computation will usually not be done when this method returns.
    The iterator will only yield new results as they arrive.

    :param func: The function to apply. Needs to accept an object from source as first argument.
    Further arguments that should be applied in all calls all the same can be given via args and kwargs.<br>
    This function calls func.remove to start the asynchronous computation, so func should be annotated
    with `@ray.remote`.
    :param source: The elements on which func should be applied.
    :param source_param_at: Which place does the argument from the iterated collection have in funcs parameter list?
    Zero indexed.
    :param args: Further positional arguments to func that are the same for all calls.
      All arguments that go before the source parameter need to be given as positional arguments!
    :param kwargs: Further keyword arguments to func that are the same for all calls.
      All arguments that go before the source parameter need to be given as positional arguments!
    :return: An iterator that successively yields the results of applying func to the elements,
    in the order the elements are given in source.
    """

    result_ids = []
    for obj in source:
        args2 = [*args]
        args2.insert(source_param_at, obj)
        result_ids.append(func.remote(*args2, **kwargs))

    # Check results and stops the pool
    try:
        while result_ids:
            done, result_ids = ray.wait(result_ids)
            yield ray.get(done[0])
    except GeneratorExit:
        # this is raised if .close is called on this generator
        # shutdown the pool
        for result_id in result_ids:
            ray.cancel(result_id)
