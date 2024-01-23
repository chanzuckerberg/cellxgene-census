import cProfile
import os
import tempfile
import time
from collections import defaultdict
from functools import wraps
from typing import Dict

fn_cum_time: Dict[str, float] = defaultdict(lambda: 0)
fn_calls: Dict[str, int] = defaultdict(lambda: 0)


def cprofile(func):  # type: ignore[no-untyped-def]
    """
    Wraps a function with cProfile. Returns the result of the function and the path a file containing a dump of cProfile Stats object file, as 2-tuple.
    The stats are returned as a file to allow for aggregation of stats from multiple child processes.
    """

    @wraps(func)
    def cprofile_wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
        with cProfile.Profile() as prof:
            result = func(*args, **kwargs)

            f = tempfile.mkstemp()[1]
            prof.dump_stats(f)

            return result, f

    return cprofile_wrapper


def timeit_report(func):  # type: ignore[no-untyped-def]
    """
    Decorator to calls to all nested functions that are decorated with @timeit (including the function being decorated by this decorator,
    if it is also decorated with @timeit).
    Prints a report of the cumulative and average time spent in each function, along with the number of calls to each function.
    The profiling is simpler than cProfile-based profiling, but prints immediate output to the command line that is easier to read.
    """

    @wraps(func)
    def timeit_report_wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
        result = func(*args, **kwargs)

        sorted_fn_names = [k for k, _ in sorted(fn_cum_time.items(), key=lambda i: i[1], reverse=True)]
        for fn_name in sorted_fn_names:
            print(
                f"[timing {os.getpid()}] {fn_name}: "
                f"cum_time={fn_cum_time[fn_name]} sec; avg_time={(fn_cum_time[fn_name] / fn_calls[fn_name]):.3f}; "
                f"calls={fn_calls[fn_name]}"
            )

        return result

    return timeit_report_wrapper


def timeit(func):  # type: ignore[no-untyped-def]
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        exec_time = end_time - start_time

        fn_name = func.__name__
        fn_cum_time[fn_name] += exec_time
        fn_calls[fn_name] += 1
        # print(f'[timing] {fn_name}: exec time={exec_time:.3f} sec; '
        #       f'cum_time={fn_cum_time[fn_name]} sec; avg_time={(fn_cum_time[fn_name] / fn_calls[fn_name]):.3f}; '
        #       f'calls={fn_calls[fn_name]}')

        return result

    return timeit_wrapper
