# BLAS threading: default to single-threaded BLAS so that parallel workers
# (see Calculator.compute) don't oversubscribe (n_jobs workers x full BLAS).
# Workers pin BLAS to 1 thread in _parallel._worker_init; this line ensures
# the main process (and any serial run) does the same by default.
import os
import numpy as np

os.environ.setdefault('OMP_NUM_THREADS', '1')

# NumPy 2 removed np.NaN; some legacy code paths still reference it.
if not hasattr(np, "NaN"):
    np.NaN = np.nan
