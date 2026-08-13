# Science/maths/computing tools
import numpy as np
import pandas as pd
import copy, yaml, importlib, time, warnings, os
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from scipy import stats

# From this package
from .data import Data
from .utils import convert_mdf_to_ddf, inspect_calc_results
from . import _parallel
from ._logging import get_logger, configure as _configure_logging

logger = get_logger("pyspi.calculator")


# ---------------------------------------------------------------------------
# LaggedCorrelation config expansion: max_tau -> tau=1..max_tau
# ---------------------------------------------------------------------------

def _as_label_list(labels):
    if labels is None:
        return []
    if isinstance(labels, str):
        return [labels]
    return list(labels)


def _is_module_label(label):
    return (
        isinstance(label, str)
        and len(label) == 3
        and label[0] == "M"
        and (label[1:].isdigit() or label[1:] == "XX")
    )


def _merge_spi_labels(spi, family_labels=None, config_labels=None):
    labels = list(getattr(spi, "labels", []))
    family_labels = _as_label_list(family_labels)
    config_labels = _as_label_list(config_labels)

    if any(_is_module_label(label) for label in config_labels):
        labels = [label for label in labels if not _is_module_label(label)]
        family_labels = [
            label for label in family_labels if not _is_module_label(label)
        ]

    merged = []
    for label in labels + family_labels + config_labels:
        if label not in merged:
            merged.append(label)
    spi.labels = merged


def _split_config_params(params):
    params = dict(params or {})
    config_labels = params.pop("labels", None)
    return params, config_labels

CONFIG_DIR = Path(__file__).parent / "configs"


def bundled_configs():
    """Names of the bundled configs, i.e. the valid non-path values of ``config``."""
    return sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))


def resolve_config(config):
    """Resolve ``config`` to a config yaml path.

    Accepts either the name of a bundled config (``"full"``, ``"fast"``,
    ``"benchmarked_p90"``, ...) or a path to a user-written yaml. A value is
    treated as a path if it carries a directory component or a ``.yaml``/
    ``.yml`` suffix; otherwise it is looked up in :data:`CONFIG_DIR`.
    """
    text = str(config)
    looks_like_path = (
        os.sep in text
        or (os.altsep is not None and os.altsep in text)
        or text.endswith((".yaml", ".yml"))
    )
    if looks_like_path:
        path = Path(text).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        return str(path)

    path = CONFIG_DIR / f"{text}.yaml"
    if not path.is_file():
        raise ValueError(
            f"Unknown config '{text}'. Bundled configs are: "
            f"{', '.join(bundled_configs())}. "
            f"To use your own, pass a path to a .yaml file."
        )
    return str(path)


def load_spis_from_yaml(configfile):
    """Instantiate all SPIs from a configfile.

    Returns a dict mapping identifier to SPI instance.

    Shared between :class:`Calculator` and the parallel worker initializer
    (see :func:`pyspi._parallel._worker_init`) so workers don't need to
    instantiate a throwaway Calculator just to rebuild ``_spis``. Progress is
    emitted via the ``pyspi.calculator`` logger at INFO level.
    """
    spis = {}
    logger.info("Loading configuration file: %s", configfile)
    with open(configfile) as f:
        yf = yaml.load(f, Loader=yaml.FullLoader)
    for module_name, module_spis in yf.items():
        logger.info("Importing module %s", module_name)
        module = importlib.import_module(module_name, __package__)
        for fcn, entry in (module_spis or {}).items():
            family_labels = entry.get("labels")
            configs = entry.get("configs")
            if fcn == "LaggedCorrelation" and configs is not None:
                configs = _expand_lagged_correlation_configs(configs)
            if configs is None:
                spi = getattr(module, fcn)()
                _merge_spi_labels(spi, family_labels)
                spis[spi.identifier] = spi
                logger.info('[%d] %s.%s(x,y) -> "%s"', len(spis), module_name, fcn, spi.identifier)
                continue
            for params in configs:
                params, config_labels = _split_config_params(params)
                spi = getattr(module, fcn)(**params)
                _merge_spi_labels(spi, family_labels, config_labels)
                spis[spi.identifier] = spi
                logger.info('[%d] %s.%s(x,y,%s) -> "%s"', len(spis), module_name, fcn, params, spi.identifier)
    return spis


def _expand_lagged_correlation_configs(configs):
    expanded = []
    for params in configs or []:
        if "max_tau" in params:
            if "tau" in params:
                raise ValueError("LaggedCorrelation config cannot set both tau and max_tau.")
            max_tau = int(params["max_tau"])
            if max_tau < 1:
                raise ValueError("max_tau must be >= 1.")
            base = {key: value for key, value in params.items() if key != "max_tau"}
            for tau in range(1, max_tau + 1):
                entry = dict(base)
                entry["tau"] = tau
                expanded.append(entry)
        else:
            expanded.append(dict(params))
    return expanded


class Calculator:
    """Compute all pairwise interactions.

    The calculator takes in a multivariate time-series dataset (MTS), computes and stores all pairwise interactions for the dataset.
    It uses a YAML configuration file that can be modified in order to compute a reduced set of pairwise methods.

    Example:
        >>> import numpy as np
        >>> dataset = np.random.randn(5,500)    # create a random multivariate time series (MTS)
        >>> calc = Calculator(dataset=dataset)  # Instantiate the calculator
        >>> calc.compute()                      # Compute all pairwise interactions

    Args:
        dataset (:class:`~pyspi.data.Data`, array_like, optional):
            The multivariate time series of M processes and T observations, default=None.
        name (str, optional):
            The name of the calculator. Mainly used for printing the results but can be useful if you have multiple instances, default=None.
        labels (array_like, optional):
            Any set of strings by which you want to label the calculator. This can be useful later for classification purposes, default=None.
        config (str, optional):
            Which SPIs to compute. Either the name of a bundled config or a path
            to your own YAML file, default="full". Bundled configs are:

            - ``"full"`` -- every SPI (~328).
            - ``"fast"`` -- drops the slowest SPIs.
            - ``"sonnet"`` -- 14 representative SPIs, one per module (M01-M14).
            - ``"fabfour"`` -- 4 SPIs: covariance, Spearman, directed information,
              power-envelope correlation.
            - ``"benchmarked_p80"`` / ``"_p90"`` / ``"_p95"`` / ``"_p99"`` -- keep
              the fastest N% of SPIs by measured amortized compute cost, so
              ``benchmarked_p80`` is the cheapest and ``benchmarked_p99`` the most
              complete. See ``bench/README.md`` for how these were derived.
        detrend (bool, optional):
            If True, detrend each time series in the MTS dataset individually along the time axis, default=False.
        zscore (bool, optional):
            If True, z-score each time series in the MTS dataset individually along
            the time axis, default=True. Per-process (rather than whole-dataset)
            standardisation is deliberate: it removes each process's arbitrary
            gain/units without letting the choice of the other processes in the
            dataset influence any pairwise statistic.
    """
    def __init__(
        self, dataset=None, name=None, labels=None, config="full",
        detrend=False, zscore=True, verbose=True,
    ):
        self._spis = {}
        self._zscore = zscore
        self._detrend = detrend
        self._timings = {}
        self._verbose = verbose

        # verbose maps to a process-global pyspi logger level (INFO vs WARNING).
        _configure_logging(verbose)

        configfile = resolve_config(config)

        self._configfile = configfile  # stored so parallel workers can re-instantiate SPIs
        self._config = config
        self._spis = load_spis_from_yaml(configfile)

        duplicates = [
            n for n, count in Counter(self._spis.keys()).items() if count > 1
        ]
        if duplicates:
            raise ValueError(
                f"Duplicate SPI identifiers: {duplicates}.\n Check the config file for duplicates."
            )

        self._name = name
        self._labels = labels

        logger.info("%d SPI(s) were successfully initialised.", len(self.spis))

        if dataset is not None:
            self.load_dataset(dataset)

    @property
    def spis(self):
        """Dict of SPIs.

        Keys are the SPI identifier and values are their objects.
        """
        return self._spis

    @spis.setter
    def spis(self, s):
        raise Exception("Do not set this property externally.")

    @property
    def n_spis(self):
        """Number of SPIs in the calculator."""
        return len(self._spis)

    @property
    def dataset(self):
        """Dataset as a data object."""
        return self._dataset

    @dataset.setter
    def dataset(self, d):
        raise Exception(
            "Do not set this property externally. Use the load_dataset() method."
        )

    @property
    def name(self):
        """Name of the calculator."""
        return self._name

    @name.setter
    def name(self, n):
        self._name = n

    @property
    def labels(self):
        """List of calculator labels."""
        return self._labels

    @labels.setter
    def labels(self, ls):
        self._labels = ls

    @property
    def timings(self):
        """Per-SPI wall-clock times (seconds) from the last compute() call."""
        return dict(self._timings)

    @property
    def table(self):
        """Results table for all pairwise interactions."""
        return self._table

    @table.setter
    def table(self, a):
        raise Exception(
            "Do not set this property externally. Use the compute() method."
        )

    @property
    def group(self):
        """The numerical group assigned during :meth:`~pyspi.Calculator.calculator.set_group`."""
        try:
            return self._group
        except AttributeError as err:
            warnings.warn("Group undefined. Call set_group() method first.")
            raise AttributeError(err)

    @group.setter
    def group(self, g):
        raise Exception(
            "Do not set this property externally. Use the set_group() method."
        )

    @property
    def group_name(self):
        """The group name assigned during :meth:`~pyspi.Calculator.calculator.set_group`."""
        try:
            return self._group_name
        except AttributeError as err:
            warnings.warn(f"Group name undefined. Call set_group() method first.")
            return None

    @group_name.setter
    def group_name(self, g):
        raise Exception("Do not set this property externally. Use the group() method.")

    def load_dataset(self, dataset):
        """Load new dataset into existing instance.

        Args:
            dataset (:class:`~pyspi.data.Data`, array_list):
                New dataset to attach to calculator.
        """
        if not isinstance(dataset, Data):
            self._dataset = Data(
                Data.convert_to_numpy(dataset),
                zscore=self._zscore,
                detrend=self._detrend,
            )
        else:
            self._dataset = dataset

        columns = pd.MultiIndex.from_product(
            [self.spis.keys(), self._dataset.procnames], names=["spi", "process"]
        )
        self._table = pd.DataFrame(
            data=np.full(
                (self.dataset.n_processes, self.n_spis * self.dataset.n_processes),
                np.nan,
            ),
            columns=columns,
            index=self._dataset.procnames,
        )
        self._table.columns.name = "process"

    def compute(
        self,
        n_jobs=None,
        checkpoint_dir=None,
        resume=True,
        mp_context=None,
        progress=True,
    ):
        """Compute every SPI on the loaded dataset.

        Args:
            n_jobs (int, optional): Number of worker processes. ``None`` (default)
                falls back to the ``PYSPI_N_JOBS`` environment variable, or 1 if
                unset. ``1`` runs serially in this process.
            checkpoint_dir (str | Path, optional): If set, each finished SPI is
                written to ``<dir>/<identifier>.npy`` atomically. Enables resume.
            resume (bool): If True (default) and ``checkpoint_dir`` contains
                results from a prior run, those SPIs are loaded and skipped.
            mp_context (str, optional): Multiprocessing start method when
                ``n_jobs>1``. Default (``None``): ``"fork"`` on Linux (workers
                inherit imported state via copy-on-write — ~2x faster startup),
                ``"spawn"`` on macOS/Windows (fork is unsafe/absent there).
            progress (bool): Show a tqdm progress bar (default True).

        Backend threading: when ``n_jobs>1`` each worker pins its nested pools
        (OpenMP/OpenBLAS/MKL, cdt, torch, pyEDM) to one thread/process so the
        workers don't oversubscribe the cores. macOS is an exception — its
        Accelerate BLAS cannot be thread-pinned by threadpoolctl, so on macOS
        ``n_jobs>1`` can oversubscribe BLAS-heavy SPIs; prefer ``n_jobs=1``
        there for a single dataset. ``n_jobs=1`` always leaves the backends
        free to self-parallelise.
        """
        if not hasattr(self, "_dataset"):
            raise AttributeError(
                "Dataset not loaded yet. Please initialise with load_dataset."
            )

        if n_jobs is None:
            n_jobs = int(os.getenv("PYSPI_N_JOBS", "1"))
        _parallel.guard_oversubscription(n_jobs)

        spi_keys = list(self.spis.keys())
        M = self.dataset.n_processes
        cp_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        if cp_dir is not None:
            cp_dir.mkdir(parents=True, exist_ok=True)

        # Resume: skip SPIs whose checkpoint exists.
        if cp_dir is not None and resume:
            done, spi_keys = _parallel.load_checkpoints(cp_dir, spi_keys, M)
            for key, (S, err, _t) in done.items():
                self._table[key] = S
                self._timings[key] = 0.0
                if err is not None:
                    warnings.warn(f'Checkpoint contains prior error for "{key}": {err}')
            if done:
                logger.info("Resumed %d SPI(s) from %s", len(done), cp_dir)

        if not spi_keys:
            logger.info("All SPIs already cached; nothing to compute.")
            if self._verbose:
                inspect_calc_results(self)
            return

        t_start = time.perf_counter()

        if n_jobs <= 1:
            self._compute_serial(spi_keys, M, cp_dir, progress)
        else:
            n_workers = min(int(n_jobs), len(spi_keys))
            ctx = mp_context or _parallel.default_mp_context()
            logger.info("Parallel compute: %d SPI(s) via %d workers (mp=%s)",
                        len(spi_keys), n_workers, ctx)
            results = _parallel.run_parallel(
                self._spis, self._dataset, spi_keys,
                n_jobs=n_workers, mp_context=ctx,
                checkpoint_dir=cp_dir, progress=progress,
                configfile=self._configfile,
            )
            for key, (S, err, elapsed) in results.items():
                if err is not None:
                    warnings.warn(f'Caught error for SPI "{key}": {err}')
                self._table[key] = S
                self._timings[key] = elapsed

        elapsed = time.perf_counter() - t_start
        logger.info("Calculation complete. Time taken: %.4fs", elapsed)
        if self._verbose:
            inspect_calc_results(self)

    def _compute_serial(self, spi_keys, M, cp_dir, progress):
        iterable = tqdm(spi_keys) if progress else spi_keys
        for key in iterable:
            if progress:
                iterable.set_description(f"Processing [{self._name}: {key}]")
            t0 = time.perf_counter()
            err = None
            try:
                S = self._spis[key].multivariate(self.dataset)
                S = np.array(S, dtype=float, copy=True)
                np.fill_diagonal(S, np.nan)
            except Exception as e:
                warnings.warn(f'Caught {type(e).__name__} for SPI "{key}": {e}')
                S = np.full((M, M), np.nan)
                err = f"{type(e).__name__}: {e}"
            self._table[key] = S
            self._timings[key] = time.perf_counter() - t0
            if cp_dir is not None:
                _parallel._atomic_npy_write(cp_dir / f"{key}.npy", S)
                err_path = cp_dir / f"{key}.error"
                if err is not None:
                    err_path.write_text(err)
                elif err_path.exists():
                    err_path.unlink()

    def _rmmin(self):
        """Iterate through all spis and remove the minimum (fixes absolute value errors when correlating)"""
        for spi in self.spis:
            mpi = self.table[spi]
            if not self.spis[spi].issigned():
                self.table[spi] = mpi - np.nanmin(mpi)

    def set_group(self, classes):
        """Assigns a numeric value to this instance based on list of classes.

        Args:
            classes (list):
                If any of the labels in this instance matches one in the class list, then we assign the index
                value to this class.
        """
        self._group = None
        self._group_name = None

        # Ensure this is a list of lists
        for i, c in enumerate(classes):
            if not isinstance(c, list):
                classes[i] = [c]

        for i, i_cls in enumerate(classes):
            for j, j_cls in enumerate(classes):
                if i == j:
                    continue
                assert not set(i_cls).issubset(
                    set(j_cls)
                ), f"Class {i_cls} is a subset of class {j_cls}."

        labset = set(self.labels)
        matches = [set(cls).issubset(labset) for cls in classes]

        if np.count_nonzero(matches) > 1:
            warnings.warn(f"More than one match for classes {classes}")
        else:
            try:
                id = np.where(matches)[0][0]
                self._group = id
                self._group_name = ", ".join(classes[id])
            except (TypeError, IndexError):
                pass

    def _merge(self, other):
        """TODO: Merge two calculators (to include additional SPIs)"""
        raise NotImplementedError()
        if self.identifier is not other.name:
            raise TypeError(f"Calculator name does do not match. Aborting merge.")

        for attr in ["name", "n_processes", "n_observations"]:
            selfattr = getattr(self.dataset, attr)
            otherattr = getattr(other.dataset, attr)
            if selfattr is not otherattr:
                raise TypeError(
                    f"Attribute {attr} does not match between calculators ({selfattr} != {otherattr})"
                )

    def get_stat_labels(self):
        """Get the labels for each statistic.

        Returns:
            stat_labels (dict): dictionary of
        """
        return {k: v.labels for k, v in zip(self._spis.keys(), self._spis.values())}

    def _get_correlation_df(self, with_labels=False, rmmin=False):
        # Sorts out pesky numerical issues in the unsigned spis
        if rmmin:
            self._rmmin()

        # Flatten (get Edge-by-SPI matrix). future_stack=True is the pandas 3
        # behaviour; unlike the legacy default it keeps all-NaN rows, so the
        # explicit dropna reproduces the old semantics (self-pairs are all-NaN
        # because the SPI diagonals are NaN). Verified equivalent on pandas 2.3.
        edges = self.table.stack(future_stack=True).dropna(how="all")

        # Correlate the edge matrix (using pearson and/or spearman correlation)
        cf = pd.DataFrame(
            index=[c for c in edges.columns], columns=[c for c in edges.columns]
        )
        # Need to iterate through each pair to handle unsigned/signed statistics
        for i, s0 in enumerate(edges.columns):
            for j, s1 in enumerate(edges.columns[i + 1 :]):

                if self.spis[s0].issigned() and self.spis[s1].issigned():
                    # When they're both signed, just take the correlation
                    cf.iloc[[i, i + j + 1], [i, i + j + 1]] = edges[[s0, s1]].corr(
                        method="spearman"
                    )
                else:
                    # Otherwise, take the absolute value to make sure we compare like-for-like
                    cf.iloc[[i, i + j + 1], [i, i + j + 1]] = (
                        edges[[s0, s1]].abs().corr(method="spearman")
                    )

        cf.index.name = "SPI-1"
        cf.columns.name = "SPI-2"

        if with_labels:
            return cf, self.get_stat_labels()
        else:
            return cf


def forall(func):
    def do(self, *args, **kwargs):
        try:
            for i in self._calculators.index:
                calc_ser = self._calculators.loc[i]
                for calc in calc_ser:
                    func(calc, *args, **kwargs)
        except AttributeError:
            raise AttributeError(
                f"No calculators in frame yet. Initialise before calling {func}"
            )

    return do


class CalculatorFrame:
    """ CalculatorFrame
        Container for batch level commands, like computing/pruning/initialising multiple datasets at once
    """
    def __init__(
        self,
        calculators=None,
        name=None,
        datasets=None,
        names=None,
        labels=None,
        **kwargs,
    ):
        if calculators is not None:
            self.set_calculator(calculators)

        self.name = name

        if datasets is not None:
            if names is None:
                names = [None] * len(datasets)
            if labels is None:
                labels = [None] * len(datasets)
            self.init_from_list(datasets, names, labels, **kwargs)

    @property
    def name(self):
        if hasattr(self, "_name") and self._name is not None:
            return self._name
        else:
            return ""

    @name.setter
    def name(self, n):
        self._name = n

    @staticmethod
    def from_calculator(calculator):
        cf = CalculatorFrame()
        cf.add_calculator(calculator)
        return cf

    def set_calculator(self, calculators):
        if hasattr(self, "_calculators"):
            warnings.warn("Overwriting existing calculators without explicitly deleting.")
            del self._calculators

        if isinstance(calculators, Calculator):
            calculators = [calculators]

        if isinstance(calculators, CalculatorFrame):
            self.add_calculator(calculators)
        else:
            for calc in calculators:
                self.add_calculator(calc)

    def add_calculator(self, calc):

        if not hasattr(self, "_calculators"):
            self._calculators = pd.DataFrame()

        if isinstance(calc, CalculatorFrame):
            self._calculators = pd.concat(
                [self._calculators, calc._calculators], ignore_index=True
            )
        elif isinstance(calc, Calculator):
            self._calculators = pd.concat(
                [self._calculators, pd.Series(data=calc, name=calc.name)],
                ignore_index=True,
            )
        elif isinstance(calc, pd.DataFrame):
            if isinstance(calc.iloc[0], Calculator):
                self._calculators = calc
            else:
                raise TypeError("Received dataframe but it is not in known format.")
        else:
            raise TypeError(f"Unknown data type: {type(calc)}.")

        self.n_calculators = len(self.calculators.index)

    def init_from_list(self, datasets, names, labels, **kwargs):
        base_calc = Calculator(**kwargs)
        for i, dataset in enumerate(datasets):
            calc = copy.deepcopy(base_calc)
            calc.load_dataset(dataset)
            calc.name = names[i]
            calc.labels = labels[i]
            self.add_calculator(calc)

    def init_from_yaml(
        self, document, detrend=False, zscore=True, n_processes=None, n_observations=None, **kwargs
    ):
        datasets = []
        names = []
        labels = []
        with open(document) as f:
            yf = yaml.load(f, Loader=yaml.FullLoader)

            for config in yf:
                try:
                    file = config["file"]
                    dim_order = config["dim_order"]
                    names.append(config["name"])
                    labels.append(config["labels"])
                    datasets.append(
                        Data(
                            data=file,
                            dim_order=dim_order,
                            name=names[-1],
                            detrend=detrend,
                            zscore=zscore,
                            n_processes=n_processes,
                            n_observations=n_observations,
                        )
                    )
                except Exception as err:
                    warnings.warn(f"Loading dataset: {config} failed ({err}).")

        self.init_from_list(datasets, names, labels, **kwargs)

    @property
    def calculators(self):
        """Return data array."""
        try:
            return self._calculators
        except AttributeError:
            return None

    @calculators.setter
    def calculators(self, cs):
        if hasattr(self, "calculators"):
            raise AttributeError(
                "You can not assign a value to this attribute"
                " directly, use the set_data method instead."
            )
        else:
            self._calculators = cs

    @calculators.deleter
    def calculators(self):
        warnings.warn("Overwriting existing calculators.")
        del self._calculators

    def merge(self, other):
        try:
            self._calculators = pd.concat(
                [self._calculators, other._calculators], ignore_index=True
            )
        except AttributeError:
            self._calculators = other._calculators

    def compute(self, **kwargs):
        """Compute every calculator in the frame, one dataset after another.

        Keyword arguments are forwarded to :meth:`Calculator.compute`, so
        ``frame.compute(n_jobs=4)`` parallelises *within* each dataset.

        Note that for many datasets on many cores, running one dataset per
        process (e.g. a scheduler array job) beats ``n_jobs>1`` here: SPIs
        sharing a cache run serially inside a single worker, which floors the
        achievable speedup at roughly 2-4x irrespective of ``n_jobs``. See
        "Running at scale" in the README.
        """
        if not hasattr(self, "_calculators"):
            raise AttributeError("No calculators in frame yet. Initialise before computing.")
        for i in self._calculators.index:
            for calc in self._calculators.loc[i]:
                calc.compute(**kwargs)

    @property
    def groups(self):
        groups = []
        for i in self._calculators.index:
            calc_ser = self._calculators.loc[i]
            for calc in calc_ser:
                groups.append(calc.group)
        return groups

    @forall
    def set_group(calc, *args):
        calc.set_group(*args)

    @forall
    def _rmmin(calc):
        calc._rmmin()

    def get_correlation_df(self, with_labels=False, **kwargs):
        if with_labels:
            mlabels = {}
            dlabels = {}

        shapes = pd.DataFrame()
        mdf = pd.DataFrame()
        for calc in [c[0] for c in self.calculators.values]:
            out = calc._get_correlation_df(with_labels=with_labels, **kwargs)

            s = pd.Series(
                dict(
                    n_processes=calc.dataset.n_processes,
                    n_observations=calc.dataset.n_observations,
                )
            )
            if calc.name is not None:
                s.name = calc.name
                shapes = pd.concat([shapes, pd.DataFrame(s).T])
            else:
                s.name = "N/A"
                shapes = pd.concat([shapes, pd.DataFrame(s).T])
            if with_labels:
                df = pd.concat({calc.name: out[0]}, names=["Dataset"])
                try:
                    mlabels = mlabels | out[1]
                except TypeError:
                    mlabels.update(out[1])
                dlabels[calc.name] = calc.labels
            else:
                df = pd.concat({calc.name: out}, names=["Dataset"])

            # Adds another hierarchical level giving the dataset name
            mdf = pd.concat([mdf, df])
        shapes.index.name = "Dataset"

        if with_labels:
            return mdf, shapes, mlabels, dlabels
        else:
            return mdf, shapes


class CorrelationFrame:
    def __init__(self, cf=None, **kwargs):
        self._slabels = {}
        self._dlabels = {}
        self._mdf = pd.DataFrame()
        self._shapes = pd.DataFrame()

        if cf is not None:
            if isinstance(cf, Calculator):
                cf = CalculatorFrame(cf)

            if isinstance(cf, CalculatorFrame):
                # Store the statistic-focused dataframe, statistic labels, and dataset labels
                (
                    self._mdf,
                    self._shapes,
                    self._slabels,
                    self._dlabels,
                ) = cf.get_correlation_df(with_labels=True, **kwargs)
                self._name = cf.name
            else:
                self.merge(cf)

    @property
    def name(self):
        if not hasattr(self, "_name"):
            return ""
        else:
            return self._name

    @name.setter
    def name(self, n):
        self._name = n

    @property
    def shapes(self):
        return self._shapes

    @property
    def mdf(self):
        return self._mdf

    @property
    def ddf(self):
        if not hasattr(self, "_ddf") or self._ddf.size != self._mdf.size:
            self._ddf = convert_mdf_to_ddf(self.mdf)
        return self._ddf

    @property
    def n_datasets(self):
        return self.ddf.shape[1]

    @property
    def n_spis(self):
        return self.mdf.shape[1]

    @property
    def mlabels(self):
        return self._slabels

    @property
    def dlabels(self):
        return self._dlabels

    @mdf.setter
    def mdf(self):
        raise AttributeError("Do not directly set the mdf attribute.")

    @mlabels.setter
    def mlabels(self):
        raise AttributeError("Do not directly set the mlabels attribute.")

    @dlabels.setter
    def dlabels(self):
        raise AttributeError("Do not directly set the dlabels attribute.")

    def merge(self, other):
        if not all(isinstance(i[0], str) for i in self._mdf.index):
            raise TypeError(
                f"This operation only works with named calculators (set each calc.name property)."
            )

        try:
            self._ddf = self.ddf.join(other.ddf)
            self._mdf = pd.concat([self._mdf, other.mdf], verify_integrity=True)
            self._shapes = pd.concat(
                [self._shapes, other.shapes], verify_integrity=True
            )
        except KeyError:
            self._ddf = copy.deepcopy(other.ddf)
            self._mdf = copy.deepcopy(other.mdf)
            self._shapes = copy.deepcopy(other.shapes)

        try:
            self._slabels = self._slabels | other.mlabels
            self._dlabels = self._dlabels | other.dlabels
        except TypeError:
            self._slabels.update(other.mlabels)
            self._dlabels.update(other.dlabels)

    def get_pvalues(self):
        if not hasattr(self, "_pvalues"):
            n = self.shapes["n_observations"]
            nstats = self.mdf.shape[1]
            ns = np.repeat(n.values, nstats**2).reshape(
                self.mdf.shape[0], self.mdf.shape[1]
            )
            rsq = self.mdf.values**2
            fval = ns * rsq / (1 - rsq)
            self._pvalues = stats.f.sf(fval, 1, ns - 1)
        return pd.DataFrame(
            data=self._pvalues, index=self.mdf.index, columns=self.mdf.columns
        )

    def compute_significant_values(self):
        pvals = self.get_pvalues()
        nstats = self.mdf.shape[1]
        self._insig_ind = pvals > 0.05 / nstats / (nstats - 1) / 2

        if not hasattr(self, "_insig_group"):
            pvals = pvals.droplevel(["Dataset", "Type"])
            group_pvalue = pd.DataFrame(
                data=np.full([pvals.columns.size] * 2, np.nan),
                columns=pvals.columns,
                index=pvals.columns,
            )
            for f1 in pvals.columns:
                print(f"Computing significance for {f1}...")
                for f2 in [
                    f
                    for f in pvals.columns
                    if f is not f1 and np.isnan(group_pvalue[f1][f])
                ]:
                    cp = pvals[f1][f2]
                    group_pvalue[f1][f2] = stats.combine_pvalues(cp[~cp.isna()])[1]
                    group_pvalue[f2][f1] = group_pvalue[f1][f2]
            self._insig_group = group_pvalue > 0.05

    def get_average_correlation(
        self, thresh=0.2, absolute=True, summary="mean", remove_insig=False
    ):
        mdf = copy.deepcopy(self.mdf)

        if absolute:
            ss_adj = getattr(mdf.abs().groupby("SPI-1"), summary)()
        else:
            ss_adj = getattr(mdf.groupby("SPI-1"), summary)()
        ss_adj = (
            ss_adj.dropna(thresh=ss_adj.shape[0] * thresh, axis=0)
            .dropna(thresh=ss_adj.shape[1] * thresh, axis=1)
            .sort_index(axis=1)
        )
        if remove_insig:
            ss_adj[self._insig_group.sort_index()] = np.nan

        return ss_adj

    def get_feature_matrix(self, sthresh=0.8, dthresh=0.2, dropduplicates=True):

        fm = self.ddf
        if dropduplicates:
            fm = fm.drop_duplicates()

        # Drop datasets based on NaN threshold
        num_dnans = dthresh * fm.shape[0]
        fm = fm.dropna(axis=1, thresh=num_dnans)

        # Drop measures based on NaN threshold
        num_snans = sthresh * fm.shape[1]
        fm = fm.dropna(axis=0, thresh=num_snans)
        return fm

    @staticmethod
    def _verify_classes(classes):
        # Ensure this is a list of lists
        for i, cls in enumerate(classes):
            if not isinstance(cls, list):
                classes[i] = [cls]

        for i, i_cls in enumerate(classes):
            for j, j_cls in enumerate(classes):
                if i == j:
                    continue
                assert not set(i_cls).issubset(
                    set(j_cls)
                ), f"Class {i_cls} is a subset of class {j_cls}."

    @staticmethod
    def _get_group(labels, classes, instance, verbose=False):
        labset = set(labels)
        matches = [set(cls).issubset(labset) for cls in classes]

        # Iterate through all
        if np.count_nonzero(matches) > 1:
            if verbose:
                print(
                    f"More than one match in for {instance} whilst searching for {classes} within {labels}). Choosing first one."
                )

        try:
            myid = np.where(matches)[0][0]
            return myid
        except (TypeError, IndexError):
            if verbose:
                print(f"{instance} has no match in {classes}. Options are {labels}")
            return -1

    @staticmethod
    def _set_groups(classes, labels, group_names, group):
        CorrelationFrame._verify_classes(classes)
        for m in labels:
            group[m] = CorrelationFrame._get_group(labels[m], classes, m)

    def set_sgroups(self, classes):
        # Initialise the classes
        self._sgroup_names = {i: ", ".join(c) for i, c in enumerate(classes)}
        self._sgroup_names[-1] = "N/A"

        self._sgroup_ids = {m: -1 for m in self._slabels}
        CorrelationFrame._set_groups(
            classes, self._slabels, self._sgroup_names, self._sgroup_ids
        )

    def set_dgroups(self, classes):
        self._dgroup_names = {i: ", ".join(c) for i, c in enumerate(classes)}
        self._dgroup_names[-1] = "N/A"

        self._dgroup_ids = {d: -1 for d in self._dlabels}
        CorrelationFrame._set_groups(
            classes, self._dlabels, self._dgroup_names, self._dgroup_ids
        )

    def get_dgroup_ids(self, names=None):
        if names is None:
            names = self._ddf.columns

        return [self._dgroup_ids[n] for n in names]

    def get_dgroup_names(self, names=None):
        if names is None:
            names = self._ddf.columns

        return [self._dgroup_names[i] for i in self.get_dgroup_ids(names)]

    def get_sgroup_ids(self, names=None):
        if names is None:
            names = self._mdf.columns

        return [self._sgroup_ids[n] for n in names]

    def get_sgroup_names(self, names=None):
        if names is None:
            names = self._mdf.columns

        return [self._sgroup_names[i] for i in self.get_sgroup_ids(names)]

    def relabel_spis(self, names, labels):
        assert len(names) == len(labels), "Length of spis must equal length of labels."

        for n, l in zip(names, labels):
            try:
                self._slabels[n] = l
            except AttributeError:
                self._slabels = {n: l}

    def relabel_data(self, names, labels):
        assert len(names) == len(
            labels
        ), "Length of datasets must equal length of labels."

        for n, l in zip(names, labels):
            self._dlabels[n] = l
