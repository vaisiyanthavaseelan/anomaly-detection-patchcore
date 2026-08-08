import os
import sys

_REQUIRED_ENV = {
    # torch and faiss each bundle their own OpenMP runtime; on macOS having both
    # loaded in one process crashes (or silently misbehaves) unless duplicates
    # are allowed and parallelism is disabled. Must be set before the process's
    # first import of torch/faiss/numpy, so a plain os.environ update is too
    # late here -- re-exec the interpreter with the vars already in place.
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "OMP_NUM_THREADS": "1",
}


def ensure_openmp_env():
    if all(os.environ.get(k) == v for k, v in _REQUIRED_ENV.items()):
        return
    os.environ.update(_REQUIRED_ENV)
    os.execve(sys.executable, [sys.executable] + sys.argv, os.environ)
