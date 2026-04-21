# BLAS threading: default to single-threaded BLAS in the main process so that
# fork-based parallel workers (see calculator.compute) can opt back into
# multi-threading via threadpoolctl without nested oversubscription.
import os
import numpy as np

os.environ.setdefault('OMP_NUM_THREADS', '1')

# NumPy 2 removed np.NaN; some legacy code paths still reference it.
if not hasattr(np, "NaN"):
    np.NaN = np.nan
