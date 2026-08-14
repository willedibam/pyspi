"""Red tests: a checkpoint must identify the run that produced it.

Resume currently validates only the SPI identifier and an ``(M, M)`` shape, so
any other run of the same width silently inherits the previous run's numbers.
That is the most dangerous class of defect in the package: it produces
valid-looking results with no warning and no trace.

Also covered: identifier collisions. Identifiers are formatted with ``.3g``/
``.4g``, so distinct parameterisations can render to the same string, and
dictionary insertion overwrites the loser before any duplicate check runs.

See tests/test_state_integrity.py for the xfail(strict=True) rationale.
"""
import numpy as np
import pytest

from pyspi.calculator import Calculator
from pyspi.data import Data

CONFIG = "fabfour"


def _data(seed, m=3, t=80, name=None):
    rng = np.random.default_rng(seed)
    return Data(data=rng.standard_normal((m, t)), dim_order="ps",
                zscore=False, name=name)


def _run(dataset, cp_dir, config=CONFIG, **kw):
    calc = Calculator(dataset=dataset, config=config, verbose=False)
    calc.compute(checkpoint_dir=cp_dir, progress=False, **kw)
    return calc


def _first_spi_values(calc):
    key = sorted(calc.spis)[0]
    return key, np.asarray(calc.table[key].to_numpy(dtype=float))


# --------------------------------------------------------------------------
# Checkpoint identity
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="checkpoints are not bound to the input data")
def test_checkpoint_rejects_a_different_dataset(tmp_path):
    """A different dataset of the same width must not reuse checkpoints."""
    first = _run(_data(seed=1), tmp_path)
    key, v1 = _first_spi_values(first)

    second = _run(_data(seed=2), tmp_path)
    _, v2 = _first_spi_values(second)

    reference = _run(_data(seed=2), tmp_path / "clean")
    _, ref = _first_spi_values(reference)

    assert np.allclose(v2, ref, equal_nan=True), (
        f"'{key}' resumed the first dataset's result for a different dataset "
        f"(got {np.nanmean(v2):.6g}, correct value {np.nanmean(ref):.6g})."
    )
    assert not np.allclose(v1, v2, equal_nan=True)


@pytest.mark.xfail(strict=True, reason="checkpoints are not bound to the config")
def test_checkpoint_rejects_a_different_config(tmp_path):
    dataset = _data(seed=3)
    _run(dataset, tmp_path, config="fabfour")

    calc = Calculator(dataset=dataset, config="fast", verbose=False)
    calc.compute(checkpoint_dir=tmp_path, progress=False)

    shared = set(calc.spis) & {"cov_EmpiricalCovariance"}
    assert shared, "Precondition: the two configs must share at least one SPI."
    # A config change must be detected even when identifiers overlap.
    assert getattr(calc, "_resume_rejected", False), (
        "Resume accepted checkpoints written under a different config."
    )


@pytest.mark.xfail(strict=True, reason="checkpoints are not bound to process names/order")
def test_checkpoint_rejects_permuted_processes(tmp_path):
    rng = np.random.default_rng(7)
    arr = rng.standard_normal((3, 80))

    _run(Data(data=arr, dim_order="ps", zscore=False, procnames=["a", "b", "c"]),
         tmp_path)

    perm = [2, 0, 1]
    permuted = _run(
        Data(data=arr[perm], dim_order="ps", zscore=False,
             procnames=["c", "a", "b"]),
        tmp_path,
    )
    key, got = _first_spi_values(permuted)

    ref_calc = _run(
        Data(data=arr[perm], dim_order="ps", zscore=False,
             procnames=["c", "a", "b"]),
        tmp_path / "clean",
    )
    _, ref = _first_spi_values(ref_calc)

    assert np.allclose(got, ref, equal_nan=True), (
        f"'{key}' resumed a checkpoint computed under a different process order."
    )


@pytest.mark.xfail(strict=True, reason="failed .error checkpoints are treated as complete")
def test_failed_checkpoints_are_retried_by_default(tmp_path):
    """An SPI that failed previously must be recomputed, not resumed as NaN."""
    dataset = _data(seed=4)
    calc = Calculator(dataset=dataset, config=CONFIG, verbose=False)
    key = sorted(calc.spis)[0]

    # Simulate a prior run in which `key` failed.
    tmp_path.mkdir(parents=True, exist_ok=True)
    np.save(tmp_path / f"{key}.npy", np.full((3, 3), np.nan))
    (tmp_path / f"{key}.error").write_text("RuntimeError: simulated prior failure")

    calc.compute(checkpoint_dir=tmp_path, progress=False)
    got = np.asarray(calc.table[key].to_numpy(dtype=float))

    assert np.isfinite(got[~np.eye(3, dtype=bool)]).any(), (
        f"'{key}' was resumed from a failed checkpoint instead of being retried."
    )


# --------------------------------------------------------------------------
# Identifier collisions
# --------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="duplicate detection runs after dict insertion has merged keys")
def test_duplicate_identifiers_are_rejected_at_insertion():
    """Two SPIs with the same identifier must raise, not silently overwrite."""
    from pyspi.calculator import load_spis_from_yaml
    import tempfile, textwrap, os

    yaml_text = textwrap.dedent("""
        .statistics.basic:
          Covariance:
            labels: [undirected]
            configs:
              - estimator: EmpiricalCovariance
              - estimator: EmpiricalCovariance
        """)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write(yaml_text)
        path = fh.name
    try:
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            load_spis_from_yaml(path)
    finally:
        os.unlink(path)


@pytest.mark.xfail(strict=True, reason="identifiers round floats with .3g/.4g, so distinct params collide")
def test_identifier_does_not_collide_under_float_rounding():
    """Parameterisations that differ numerically must differ in identifier."""
    from pyspi.statistics.spectral import CoherenceMagnitude

    a = CoherenceMagnitude(fmin=0.123456, fmax=0.5)
    b = CoherenceMagnitude(fmin=0.123499, fmax=0.5)

    assert a.identifier != b.identifier, (
        f"Distinct fmin values collide after .3g rounding: {a.identifier!r}."
    )
