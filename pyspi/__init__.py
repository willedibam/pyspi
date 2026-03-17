# JVM + OpenBLAS segfault prevention:
# OpenBLAS multi-threading conflicts with JPype's JVM. Set OMP_NUM_THREADS=1
# to force single-threaded BLAS in the main process. Forked worker processes
# that don't touch JVM can restore multi-threading via threadpoolctl.
import os, logging, sys
import numpy as np

os.environ['OMP_NUM_THREADS'] = '1'

# NumPy 2 removed np.NaN; some legacy code paths still reference it.
if not hasattr(np, "NaN"):
    np.NaN = np.nan
