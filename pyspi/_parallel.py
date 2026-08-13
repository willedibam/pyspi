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
- Progress is per-SPI, not per-bucket: workers post a lightweight event to a
  shared queue after each SPI so the tqdm bar advances one tick per SPI and
  a stuck SPI is visible (the bar stalls). Results themselves still travel
  back via the futures, which also surfaces a hard worker crash.
"""

from __future__ import annotations

import multiprocessing as mp
import multiprocessing.shared_memory as shm
import concurrent.futures as cf
import os
import queue as _queue
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from ._logging import get_logger

logger = get_logger("pyspi.parallel")


def default_mp_context() -> str:
    """Best start method for the current platform.

    fork on Linux: workers inherit the parent's already-imported modules and
    instantiated SPIs via copy-on-write, so worker startup is near-instant.
    Measured ~2x faster end-to-end than spawn for a 252-SPI run.

    spawn elsewhere: fork is unsafe on macOS (Accelerate/CoreFoundation after
    init) and absent on Windows; spawn re-imports per worker but is correct.
    """
    return "fork" if sys.platform.startswith("linux") else "spawn"

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
    data.zscore = False
    data.detrend = False
    data._data = arr
    data.data_type = arr.dtype.type
    data.n_processes = arr.shape[0]
    data.n_observations = arr.shape[1]
    data.n_replications = arr.shape[2]
    data._procnames = list(procnames)
    data._name = name or "N/A"
    return data, shared


_BLAS_ENV_VARS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
)


def _pin_blas_env() -> None:
    """Force BLAS/threading env vars to 1 in the parent, before workers spawn.

    threadpool_limits (see _pin_worker_thread_pools) pins OpenBLAS/MKL/OpenMP
    at runtime; these env vars additionally cover numba and — for spawn
    workers, which import numpy before _worker_init runs — make BLAS start
    single-threaded from process start. Caveat: macOS Accelerate only partly
    honours these (its vDSP/FFT path threads independently of any documented
    env var), so on macOS n_jobs>1 can still oversubscribe FFT-heavy SPIs;
    Linux (OpenBLAS/MKL) is fully covered.
    """
    for var in _BLAS_ENV_VARS:
        os.environ[var] = "1"


def available_cores() -> int:
    """Cores this process may actually use.

    ``sched_getaffinity`` honours cgroup/cpuset pinning, so under PBS or Slurm
    this returns the cores the scheduler actually granted -- not the machine's
    physical core count. That distinction is the whole point of the check in
    :func:`guard_oversubscription`.
    """
    try:
        return len(os.sched_getaffinity(0))  # Linux
    except AttributeError:
        return os.cpu_count() or 1


def _requested_threads() -> int:
    """Largest thread count any BLAS/OpenMP backend has been told to use."""
    counts = [1]
    for var in _BLAS_ENV_VARS:
        try:
            counts.append(int(os.environ.get(var, "1") or 1))
        except ValueError:
            pass
    return max(counts)


def guard_oversubscription(n_jobs: int) -> None:
    """Warn -- or intervene -- when threads x processes exceeds the cores we hold.

    The dangerous case is dataset-level parallelism on a cluster: many
    single-core pyspi processes, each inheriting a site-wide
    ``OMP_NUM_THREADS=8``, so a 48-core node runs 384 threads and thrashes.
    ``pyspi/__init__`` only *defaults* the variable to 1, so an inherited value
    survives by design -- a user who sets it deliberately should keep it.

    When the scheduler granted exactly one core, more than one thread is never
    right, so that case is pinned outright. Anything else only warns, since
    pyspi cannot see how many sibling processes the scheduler started.
    """
    threads = _requested_threads()
    cores = available_cores()
    requested = n_jobs * threads
    if requested <= cores:
        return

    if cores == 1 and threads > 1:
        _pin_blas_env()
        try:
            from threadpoolctl import threadpool_limits
            global _THREADPOOL_LIMITER
            _THREADPOOL_LIMITER = threadpool_limits(limits=1)
        except ImportError:
            pass
        logger.warning(
            "Only 1 core is available to this process but the BLAS thread count "
            "is %d; pinned it to 1. This is the usual symptom of a scheduler "
            "array job inheriting a site-wide OMP_NUM_THREADS -- set "
            "OMP_NUM_THREADS=1 in your job script to silence this.",
            threads,
        )
        return

    logger.warning(
        "Oversubscription: n_jobs=%d x %d BLAS thread(s) = %d workers for %d "
        "available core(s). If you are running one dataset per process, set "
        "OMP_NUM_THREADS=1; if you meant to parallelise within this dataset, "
        "lower n_jobs.",
        n_jobs, threads, requested, cores,
    )


_THREADPOOL_LIMITER = None  # module-global so the limiter is never GC'd


def _pin_worker_thread_pools():
    """Pin every nested thread/process pool to 1 so process workers don't oversubscribe.

    n_jobs workers each running a library that itself spawns cpu_count() threads
    = quadratic blow-up. Pinning BLAS alone is not enough. The pools:
      - BLAS + OpenMP (numpy/scipy/sklearn): threadpool_limits, all user APIs.
      - cdt (causal discovery toolbox, transitive dep): autosets SETTINGS.NJOBS
        to cpu_count() at import; drives ANM/CDS/RECI/IGCI.
      - torch (drives InterDependenceScore): intra- and inter-op thread counts.
    pyEDM (drives ConvergentCrossMapping) is process-based, not thread-based, so
    it can't be pinned here — instead this function exports PYSPI_PIN_BACKENDS=1
    and statistics/causal.py reads it to pass parallel=False to pyEDM.
    """
    global _THREADPOOL_LIMITER
    os.environ["PYSPI_PIN_BACKENDS"] = "1"
    try:
        from threadpoolctl import threadpool_limits
        _THREADPOOL_LIMITER = threadpool_limits(limits=1)  # blas + openmp
    except ImportError:
        pass
    try:
        import cdt
        cdt.SETTINGS.NJOBS = 1
    except Exception:
        pass
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def _worker_init(shm_name, shape, dtype_str, procnames, ds_name, configfile, progress_q):
    """ProcessPoolExecutor initializer. Runs once per worker.

    Re-instantiates SPIs from the configfile (some SPI classes use closures in
    ``__init__`` that aren't picklable, so we can't ship instances across the
    process boundary).
    """
    data, shared = _attach_data(shm_name, shape, dtype_str, procnames, ds_name)

    # Direct call to the shared loader — no throwaway Calculator instantiation,
    # no stdout suppression needed.
    from pyspi.calculator import load_spis_from_yaml
    spis = load_spis_from_yaml(configfile)

    # Pin nested thread pools AFTER SPI modules import (cdt autosets NJOBS to
    # cpu_count() on import; we override it back to 1 here).
    _pin_worker_thread_pools()

    _WORKER_STATE["data"] = data
    _WORKER_STATE["shm"] = shared
    _WORKER_STATE["spis"] = spis
    _WORKER_STATE["progress_q"] = progress_q


def _run_task(spi_keys, checkpoint_dir):
    """Compute a bucket of SPIs sequentially in this worker.

    Posts ``(key, failed)`` to the progress queue after each SPI, and returns
    the list of ``(key, matrix, error_str_or_None, elapsed)`` tuples.
    """
    import warnings

    data = _WORKER_STATE["data"]
    spis = _WORKER_STATE["spis"]
    progress_q = _WORKER_STATE["progress_q"]
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
        if progress_q is not None:
            progress_q.put((key, err is not None))
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

    # Pin BLAS env before any worker spawns — workers inherit single-threaded
    # BLAS from process start (the only lever for macOS Accelerate).
    _pin_blas_env()

    arr = np.ascontiguousarray(dataset._data)
    M = arr.shape[0]
    tasks = build_tasks(spi_keys, spis)
    cp_str = str(checkpoint_dir) if checkpoint_dir is not None else None

    # Create resources inside the try so a failure constructing either one
    # (e.g. mp.Manager() raising) still runs the cleanup in finally.
    shared = None
    manager = None
    try:
        shared = shm.SharedMemory(create=True, size=arr.nbytes)
        manager = mp.Manager()
        progress_q = manager.Queue()
        shared_view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shared.buf)
        shared_view[:] = arr

        ctx = mp.get_context(mp_context)
        results: dict = {}
        pbar = tqdm(total=len(spi_keys), desc="SPIs", disable=not progress)
        with cf.ProcessPoolExecutor(
            max_workers=n_jobs,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(
                shared.name, arr.shape, str(arr.dtype),
                list(dataset.procnames), getattr(dataset, "_name", None),
                configfile, progress_q,
            ),
        ) as ex:
            future_to_task = {ex.submit(_run_task, task, cp_str): task for task in tasks}
            pending = set(future_to_task)
            while pending:
                # Per-SPI progress ticks (cosmetic; bar stalls on a stuck SPI).
                while True:
                    try:
                        key, _failed = progress_q.get_nowait()
                        pbar.update(1)
                        pbar.set_postfix_str(key[:32])
                    except _queue.Empty:
                        break
                # Harvest finished futures (source of truth for results).
                done = {f for f in pending if f.done()}
                for fut in done:
                    task = future_to_task[fut]
                    try:
                        for key, S, err, elapsed in fut.result():
                            results[key] = (S, err, elapsed)
                    except Exception as exc:  # worker process died (segfault/OOM)
                        for key in task:
                            results.setdefault(
                                key,
                                (np.full((M, M), np.nan), f"worker died: {exc}", 0.0),
                            )
                pending -= done
                if pending:
                    time.sleep(0.05)
        # Drain any progress events that arrived after the last poll, then
        # hard-sync the bar (a dead worker can leave it a few ticks short).
        while True:
            try:
                progress_q.get_nowait()
                pbar.update(1)
            except _queue.Empty:
                break
        pbar.n = len(results)
        pbar.refresh()
        pbar.close()
        return results
    finally:
        if manager is not None:
            manager.shutdown()
        if shared is not None:
            shared.close()
            try:
                shared.unlink()
            except FileNotFoundError:
                pass
