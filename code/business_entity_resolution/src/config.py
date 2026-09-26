"""Paths and global parameters."""
import os

ROOT = os.environ.get("ER_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
DATA_DIR = os.environ.get("ER_DATA", os.path.join(ROOT, "dataset"))
WORK_DIR = os.environ.get("ER_WORK", os.path.join(ROOT, "work"))
OUT_DIR = os.environ.get("ER_OUT", os.path.join(ROOT, "output"))
N_JOBS = int(os.environ.get("ER_JOBS", os.cpu_count() or 4))
SEED = int(os.environ.get("ER_SEED", 42))   # E9 seed-variance runs override via ER_SEED

os.makedirs(WORK_DIR, exist_ok=True)


def src_path(split, k):
    return os.path.join(DATA_DIR, split, f"{split}_source{k}.tsv")


def gt_path():
    return os.path.join(DATA_DIR, "train", "train_ground_truth.tsv")


def work(*p):
    path = os.path.join(WORK_DIR, *p)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _init_store(module, attr, shared):
    import importlib
    getattr(importlib.import_module(module), attr).update(shared)


def parallel_map(func, items, store_attr, shared):
    """Map `func` over `items` in a process pool; `shared` is placed into the
    module-level dict named `store_attr` in func's module.

    Uses fork (zero-copy globals) where available; otherwise ships `shared`
    to each worker through the pool initializer (portable, e.g. Windows).
    """
    import importlib
    import multiprocessing as mp
    store = getattr(importlib.import_module(func.__module__), store_attr)
    if N_JOBS <= 1 or len(items) <= 1:
        store.update(shared)
        try:
            return [func(x) for x in items]
        finally:
            store.clear()
    if "fork" in mp.get_all_start_methods():
        store.update(shared)
        try:
            with mp.get_context("fork").Pool(N_JOBS) as pool:
                return pool.map(func, items, chunksize=1)
        finally:
            store.clear()
    with mp.Pool(N_JOBS, initializer=_init_store, initargs=(func.__module__, store_attr, shared)) as pool:
        return pool.map(func, items, chunksize=1)
