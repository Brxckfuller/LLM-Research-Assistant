import time
from contextlib import contextmanager


@contextmanager
def timed_stage(name: str, timings: dict):
    start = time.perf_counter()
    yield
    timings[name] = time.perf_counter() - start