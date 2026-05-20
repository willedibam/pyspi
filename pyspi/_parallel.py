"""Parallel SPI execution backend for Calculator.compute().

Design:
- Dataset bytes are placed in a single multiprocessing.shared_memory block.
  Workers attach read-only; no per-worker pickling of the array.
- SPIs are bucketed by their ``_cache_namespace`` class attribute. Each bucket
  becomes one task assigned to a single worker, so a cache populated lazily on
  the Data object (e.g. data.spectral_bv, data.covariance) is reused across
  every variant in that bucket. SPIs without a tag run as single-SPI tasks.
- Per-SPI failures yield a NaN matrix; the rest of the run continues.
- If ``checkpoint_dir`` is set, each finished SPI is atomically written to
  ``<dir>/<identifier>.npy`` (and ``<identifier>.error`` on failure). A
  subsequent run with ``resume=True`` will load these and skip the SPIs.
"""

from __future__ import annotations

import multiprocessing as mp
import multiprocessing.shared_memory as shm
import concurrent.futures as cf
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

# Worker-local state populated by _worker_init. Module globals are safe here
# because each worker process has its own independent copy.
_WORKER_STATE: dict = {}


def _attach_data(shm_name, shape, dtype_str, procnames, name):
    """Build a Data object that views an existing shared-memory block.

    The shared array is the parent's already-normalised/detrended ``_dataset._data``,
    so we bypass Data.__init__ to avoid re-applying those transforms.
    """
    from pyspi.data import Data

    shared = shm.SharedMemory(name=shm_name)
    arr = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shared.buf)
    data = Data.__new__(Data)
    data.normalise = False
    data.detrend = False
    data._data = arr
    data.data_type = arr.dtype.type
    data.n_processes = arr.shape[0]
    data.n_observations = arr.shape[1]
    data.n_replications = arr.shape[2]
    data._procnames = list(procnames)
    data._name = name or "N/A"
    return data, shared


def _worker_init(shm_name, shape, dtype_str, procnames, ds_name, configfile):
    """ProcessPoolExecutor initializer. Runs once per worker.

    Re-instantiates SPIs from the configfile (some SPI classes use closures in
    ``__init__`` that aren't picklable, so we can't ship instances across the
    process boundary).
    """
    try:
        from threadpoolctl import threadpool_limits
        # Pin BLAS to 1 thread per worker: n_jobs workers * full BLAS = oversubscribe.
        threadpool_limits(limits=1, user_api="blas")
    except ImportError:
        pass

    data, shared = _attach_data(shm_name, shape, dtype_str, procnames, ds_name)

    # Direct call to the shared loader — no throwaway Calculator instantiation,
    # no stdout suppression needed.
    from pyspi.calculator import load_spis_from_yaml, Calculator
    if Calculator._optional_dependencies is None:
        from pyspi.utils import check_optional_deps
        Calculator._optional_dependencies = check_optional_deps()
    spis, _ = load_spis_from_yaml(
        configfile,
        optional_dependencies=Calculator._optional_dependencies,
        verbose=False,
    )

    _WORKER_STATE["data"] = data
    _WORKER_STATE["shm"] = shared
    _WORKER_STATE["spis"] = spis


def _run_task(spi_keys, checkpoint_dir):
    """Compute a bucket of SPIs sequentially in this worker.

    Returns a list of (key, matrix, error_str_or_None, elapsed).
    """
    import warnings

    data = _WORKER_STATE["data"]
    spis = _WORKER_STATE["spis"]
    M = data.n_processes
    out = []
    for key in spi_keys:
        t0 = time.perf_counter()
        err = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                S = spis[key].multivariate(data)
            S = np.array(S, dtype=float, copy=True)
            if S.shape != (M, M):
                raise ValueError(f"SPI returned shape {S.shape}, expected ({M},{M})")
            np.fill_diagonal(S, np.nan)
        except Exception as e:
            S = np.full((M, M), np.nan)
            err = f"{type(e).__name__}: {e}"
        elapsed = time.perf_counter() - t0
        if checkpoint_dir is not None:
            _atomic_npy_write(Path(checkpoint_dir) / f"{key}.npy", S)
            err_path = Path(checkpoint_dir) / f"{key}.error"
            if err is not None:
                err_path.write_text(err)
            elif err_path.exists():
                err_path.unlink()
        out.append((key, S, err, elapsed))
    return out


def _atomic_npy_write(path: Path, arr: np.ndarray) -> None:
    """Atomic write: numpy.save then os.replace. POSIX rename is atomic.

    Note: ``np.save(path, arr)`` auto-appends ``.npy`` if absent — that
    rewrites our ``.npy.tmp`` to ``.npy.tmp.npy`` and breaks the rename.
    Passing a file handle bypasses that behaviour.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, arr)
    os.replace(tmp, path)


def build_tasks(spi_keys, spis) -> list[list[str]]:
    """Bucket SPI keys by ``_cache_namespace``.

    Tagged SPIs sharing a namespace form one multi-SPI task (cache shared
    on the worker's Data). Untagged SPIs become single-SPI tasks.
    """
    cacheless: list[list[str]] = []
    grouped: dict[str, list[str]] = defaultdict(list)
    for key in spi_keys:
        ns = getattr(type(spis[key]), "_cache_namespace", None)
        if ns is None:
            cacheless.append([key])
        else:
            grouped[ns].append(key)
    # Largest groups first so workers pick up heavy tasks early — modest help on
    # makespan, costs nothing.
    grouped_tasks = sorted(grouped.values(), key=len, reverse=True)
    return grouped_tasks + cacheless


def load_checkpoints(checkpoint_dir: Path, spi_keys, M: int):
    """Return (done_results, remaining_keys).

    done_results: dict[key] -> (matrix, error_or_None, 0.0).
    A key is considered done if ``<key>.npy`` exists and has shape (M, M).
    """
    done: dict = {}
    remaining: list = []
    for key in spi_keys:
        npy = checkpoint_dir / f"{key}.npy"
        if not npy.exists():
            remaining.append(key)
            continue
        try:
            arr = np.load(npy)
        except Exception:
            remaining.append(key)
            continue
        if arr.shape != (M, M):
            remaining.append(key)
            continue
        err_path = checkpoint_dir / f"{key}.error"
        err = err_path.read_text() if err_path.exists() else None
        done[key] = (arr, err, 0.0)
    return done, remaining


def run_parallel(
    spis: dict,
    dataset,
    spi_keys: list[str],
    n_jobs: int,
    mp_context: str,
    checkpoint_dir: Optional[Path],
    progress: bool,
    configfile: str,
) -> dict:
    """Execute ``spi_keys`` across ``n_jobs`` workers; return dict[key] -> (S, err, elapsed)."""
    from tqdm import tqdm

    arr = np.ascontiguousarray(dataset._data)
    tasks = build_tasks(spi_keys, spis)
    cp_str = str(checkpoint_dir) if checkpoint_dir is not None else None

    shared = shm.SharedMemory(create=True, size=arr.nbytes)
    try:
        shared_view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shared.buf)
        shared_view[:] = arr

        ctx = mp.get_context(mp_context)
        results: dict = {}
        with cf.ProcessPoolExecutor(
            max_workers=n_jobs,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(
                shared.name, arr.shape, str(arr.dtype),
                list(dataset.procnames), getattr(dataset, "_name", None),
                configfile,
            ),
        ) as ex:
            futures = [ex.submit(_run_task, task, cp_str) for task in tasks]
            iterator = cf.as_completed(futures)
            if progress:
                iterator = tqdm(iterator, total=len(futures), desc="SPI buckets")
            for fut in iterator:
                for key, S, err, elapsed in fut.result():
                    results[key] = (S, err, elapsed)
        return results
    finally:
        shared.close()
        try:
            shared.unlink()
        except FileNotFoundError:
            pass
