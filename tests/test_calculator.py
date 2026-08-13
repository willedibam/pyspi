from pyspi.calculator import (Calculator, Data, CalculatorFrame, CorrelationFrame,
                              load_spis_from_yaml, resolve_config, bundled_configs)
from pyspi.data import available_datasets, load_dataset
import numpy as np 
import os
import pytest

############################# Test Calculator Object ########################
def test_whether_calculator_instantiates():
    """Basic test to check whether or not the calculator will instantiate."""
    calc = Calculator()
    assert isinstance(calc, Calculator), "Calculator failed to instantiate."

def test_whether_calculator_computes():
    # check whether the calculator runs
    data = np.random.randn(3, 100)
    calc = Calculator(dataset=data)
    calc.compute()

@pytest.mark.parametrize("config", [
    'fabfour',
    'fast',
    'sonnet'
])
def test_whether_calculator_instantiates_with_bundled_configs(config):
    """Test whether the calculator instantiates with each of the bundled configs"""
    calc = Calculator(config=config)
    assert isinstance(calc, Calculator), "Calculator failed to instantiate"

def test_whether_invalid_config_throws_error():
    """Test whether the calculator fails to instantiate with an unknown config name."""
    with pytest.raises(ValueError) as excinfo:
        Calculator(config='nviutw')
    assert "Unknown config 'nviutw'" in str(excinfo.value), "Unknown-config error not displaying."

def test_whether_calculator_compute_fails_with_no_dataset():
    """Test whether the calculator fails to compute SPIs when no dataset is provided."""
    calc = Calculator()
    with pytest.raises(AttributeError) as excinfo:
        calc.compute()
    assert "Dataset not loaded yet" in str(excinfo.value), "Dataset not loaded yet error not displaying."

def test_calculator_name():
    """Test whether the calculator name is retrieved correctly."""
    calc = Calculator(name="test name")
    assert calc.name == "test name", "Calculator name property did not return the expected string 'test name'"

def test_calculator_labels():
    """Test whether the calculator labels are retreived correctly, when provided."""
    test_labels = ['label1', 'label2']
    calc = Calculator(labels = test_labels)
    assert calc.labels == ['label1', 'label2'], f"Calculator labels property did not return the expected list: {test_labels} "

def test_yaml_spi_labels_inherit_and_override(tmp_path):
    """YAML family labels apply to SPIs; config labels override module labels."""
    configfile = tmp_path / "labels_config.yaml"
    configfile.write_text(
        """
.statistics.basic:
  CrossCorrelation:
    labels:
      - family-label
      - M14
    dependencies:
    configs:
      - statistic: "max"
        labels:
          - override-label
          - M10
      - statistic: "mean"
  KendallTau:
    labels:
      - MXX
    dependencies:
    configs:
      - squared: False
        labels:
          - M14
""",
        encoding="utf-8",
    )

    spis = load_spis_from_yaml(str(configfile))

    inherited = spis["xcorr_mean_sig-True"].labels
    assert "family-label" in inherited
    assert "M14" in inherited
    assert "M10" not in inherited

    overridden = spis["xcorr_max_sig-True"].labels
    assert "family-label" in overridden
    assert "override-label" in overridden
    assert "M10" in overridden
    assert "M14" not in overridden

    mxx_overridden = spis["kendalltau"].labels
    assert "M14" in mxx_overridden
    assert "MXX" not in mxx_overridden

def test_pass_single_integer_as_dataset():
    """Test whether correct error is thrown when incorrect data type passed into calculator."""
    with pytest.raises(TypeError) as excinfo:
        calc = Calculator(dataset=42)
    assert "Unknown data type" in str(excinfo.value), "Incorrect data type error not displaying for integer dataset."

def test_pass_incorrect_shape_dataset_into_calculator():
    """Test whether an error is thrown when incorrect dataset shape is passed into calculator."""
    dataset_with_wrong_dim = np.random.randn(3, 5, 10)
    with pytest.raises(RuntimeError) as excinfo:
        calc = Calculator(dataset=dataset_with_wrong_dim)
    assert "Data array dimension (3)" in str(excinfo.value), "Incorrect dimension error message not displaying for incorrect shape dataset."

@pytest.mark.parametrize("nan_loc, expected_output", [
    ([1], "[1]"),
    ([1, 2], "[1 2]"),
    ([0, 2, 3], "[0 2 3]")
    ])
def test_pass_dataset_with_nan_into_calculator(nan_loc, expected_output):
    """Check whether ValueError is raised when a dataset containing a NaN is passed into the calculator object"""
    base_dataset = np.random.randn(5, 100)
    for loc in nan_loc:
        base_dataset[loc, 0] = np.nan
    with pytest.raises(ValueError) as excinfo:
        calc = Calculator(dataset=base_dataset)
    assert f"non-numerics (NaNs) in processes: {expected_output}" in str(excinfo), "NaNs not detected in dataset when loading into Calculator!"

def test_pass_dataset_with_inf_into_calculator():
    """Check whether ValueError is raised when a dataset containing an inf/-inf value is passed into the calculator object"""
    base_dataset = np.random.randn(5, 100)
    base_dataset[0, 1] = np.inf
    base_dataset[2, 2] = -np.inf
    with pytest.raises(ValueError) as excinfo:
        calc = Calculator(dataset=base_dataset)
    assert f"non-numerics (NaNs) in processes: [0 2]" in str(excinfo), "NaNs not detected in dataset when loading into Calculator!"

@pytest.mark.parametrize("shape, n_procs_expected, n_obs_expected", [
    ((2, 23), 2, 23),
    ((5, 4), 5, 4),
    ((100, 32), 100, 32)
])
def test_data_object_process_and_observations(shape, n_procs_expected, n_obs_expected):
    """Test whether the number of processes and observations for a given dataset is correct"""
    dat = np.random.randn(shape[0], shape[1])
    calc = Calculator(dataset=dat)
    assert calc.dataset.n_observations == n_obs_expected, f"Number of observations returned by Calculator ({calc.dataset.n_observations}) does not match exepected: {n_obs_expected}"
    assert calc.dataset.n_processes == n_procs_expected, f"Number of processes returned by Calculator ({calc.dataset.n_processes}) does not match exepected: {n_procs_expected}"

EXPECTED_CONFIGS = [
    "full", "fast", "sonnet", "fabfour",
    "benchmarked_p80", "benchmarked_p90", "benchmarked_p95", "benchmarked_p99",
]

@pytest.mark.parametrize("name", EXPECTED_CONFIGS)
def test_bundled_config_resolves(name):
    """Every advertised config name resolves to a file that exists."""
    assert os.path.isfile(resolve_config(name)), f"config '{name}' did not resolve to a file."

def test_bundled_configs_matches_shipped_set():
    """bundled_configs() is exactly the advertised set - catches a stray or missing yaml."""
    assert sorted(bundled_configs()) == sorted(EXPECTED_CONFIGS)

def test_unknown_config_name_raises():
    with pytest.raises(ValueError, match="Unknown config"):
        Calculator(config="does_not_exist")

def test_config_accepts_a_path(tmp_path):
    """A path is resolved as a path, not looked up as a bundled name."""
    cfg = tmp_path / "mine.yaml"
    cfg.write_text(".statistics.basic:\n  Covariance:\n    configs:\n      - squared: False\n")
    calc = Calculator(config=str(cfg))
    assert calc.n_spis == 1

def test_missing_config_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Calculator(config=str(tmp_path / "nope.yaml"))

@pytest.mark.parametrize("config, procs, obs", [
    ("full", 2, 100),
    ("full", 5, 100),
    ("fabfour", 8, 100),
    ("fast", 10, 100),
    ("sonnet", 3, 100)
])
def test_whether_table_shape_correct_before_compute(config, procs, obs):
    """Test whether the pre-configured table is the correct shape prior to computing SPIs."""
    dat = np.random.randn(procs, obs)
    calc = Calculator(dataset=dat, config=config)
    num_spis = calc.n_spis
    expected_table_shape = (procs, num_spis*procs)
    assert calc.table.shape == expected_table_shape, f"Calculator table ({subset}) shape: ({calc.table.shape}) does not match expected shape: {expected_table_shape}"

############################# Test Data Object ########################
def test_data_object_has_been_converted_to_numpyfloat64():
    """Test whether the data object converts passed dataset to numpy array by default."""
    dat = np.random.randn(5, 10)
    calc = Calculator(dataset=dat)
    assert calc.dataset.data_type == np.float64, "Dataset was not converted into a numpy array when loaded into Calculator."

def test_whether_data_instantiates():
    """Test whether the data object instantiates without issue."""
    data_obj = Data()
    assert isinstance(data_obj, Data), "Data object failed to instantiate!"

def test_whether_data_throws_error_when_retrieving_nonexistent_dataset():
    """Test whether the data object throws correct message when trying to access a non-existent dataset."""
    data_obj = Data()
    with pytest.raises(AttributeError) as excinfo:
        dataset = data_obj.data
    assert "'Data' object has no attribute 'data'" in str(excinfo), "Unexpected error message when trying to retrieve non-existent dataset!"

def test_whether_data_throws_error_when_incorrect_dataset_type():
    """Test if correct message is shown when passing invalid dataset data type into data object."""
    with pytest.raises(TypeError) as excinfo:
        d = Data(data=3)
    assert f"Unknown data type" in str(excinfo), "Incorrect error message thrown when invalid dataset loaded into data object."

@pytest.mark.parametrize("order, shape, n_procs_expected, n_obs_expected", [
    ("ps", (3, 100), 3, 100),
    ("sp", (100, 3), 3, 100)
])
def test_whether_dim_order_works(order, shape, n_procs_expected, n_obs_expected):
    """Check that ps and sp correctly specify order of process/obseravtions"""
    dataset = np.random.randn(shape[0], shape[1])
    d = Data(data=dataset, dim_order=order)
    assert d.n_processes == n_procs_expected, f"Number of processes does not match expected for specified dim order: {order}"
    assert d.n_observations == n_obs_expected, f"Number of observations does not match expected for specified dim order: {order}"

def test_whether_data_name_assigned_only_with_dataset():
    """If no dataset is provided, there is no name for the data object (N/A)"""
    d = Data(name='test')
    assert d.name == 'N/A', "Data object name is not N/A when no dataset provided."

def test_whether_data_object_has_name_with_dataset():
    """If dataset is provided, the name will be returned"""
    dataset = np.random.randn(4, 100)
    d = Data(data=dataset, name='test')
    assert d.name == "test", f"Data object name 'test' is not being returned. Instead, {d.name} is returned."

def test_whether_data_normalise_works():
    """Check whether the data is being normalised by default when loading into data object"""
    dataset = 4 * np.random.randn(10, 500)
    d = Data(data=dataset, zscore=True)
    returned_dataset = d.to_numpy(squeeze=True)
    assert returned_dataset.mean() == pytest.approx(0, 1e-8), f"Returned dataset mean is not close to zero: {returned_dataset.mean()}"
    assert returned_dataset.std() == pytest.approx(1, 0.01), f"Returned dataset std is not close to one: {returned_dataset.std()}"

def test_whether_set_data_works():
    """Check whether existing dataset is overwritten by new dataset"""
    old_dataset = np.random.randn(1, 100)
    d = Data(data=old_dataset) # start with empty data object
    new_dataset = np.random.randn(5, 100)
    d.set_data(data=new_dataset)
    # just check the shapes since new datast will be normalised and not equal to the dataset passed in
    assert d.to_numpy(squeeze=True).shape[0] == 5, "Unexpected dataset returned when overwriting existing dataset!"

def test_add_univariate_process_to_existing_data_object():
    # start with initial data object
    dataset = np.random.randn(5, 100)
    orig_data_object = Data(data=dataset)
    # now add additional proc to existing data object
    new_univariate_proc = np.random.randn(1, 100)
    orig_data_object.add_process(proc=new_univariate_proc)
    assert orig_data_object.n_processes == 6, "New dataset number of processes not equal to expected number of processes."

def test_add_multivariate_process_to_existing_data_object():
    """Should not work, can only add univariate process with add_process function"""
    dataset = np.random.randn(5, 100)
    orig_data_object = Data(data=dataset)
    # now add additional procs to existing data object
    new_multivariate_proc = np.random.randn(2, 100)
    with pytest.raises(TypeError) as excinfo:
        orig_data_object.add_process(proc=new_multivariate_proc)
    assert "Process must be a 1D numpy array" in str(excinfo.value), "Expected 1D array error NOT thrown."

@pytest.mark.parametrize("dataset_name", sorted(available_datasets()))
def test_load_valid_dataset(dataset_name):
    """Every dataset advertised by available_datasets() must actually load."""
    dataset = load_dataset(dataset_name)
    assert isinstance(dataset, Data), f"Could not load dataset: {dataset_name}"

def test_load_invalid_dataset():
    """Test whether the load_dataset function throws the correct error/message when trying to load non-existent dataset."""
    with pytest.raises(NameError) as excinfo:
        dataset = load_dataset(name="test")
    assert "Unknown dataset: test" in str(excinfo.value), "Did not get expected error when loading invalid dataset."

def test_calculator_frame_normal_operation():
    """Test whether the calculator frame instantiates as expected."""
    datasets = [np.random.randn(3, 100) for _ in range(3)]
    dataset_names = ['d1', 'd2', 'd3']
    dataset_labels = ['label1', 'label2', 'label3']

    # create calculator frame
    calc_frame = CalculatorFrame(name="MyCalcFrame", datasets=[Data(data=data, dim_order='ps') for data in datasets], 
                                 names=dataset_names, labels=dataset_labels, config='fabfour')
    assert(isinstance(calc_frame, CalculatorFrame)), "CalculatorFrame failed to instantiate."

    # check the properties of the frame
    # check expected number of calcs in frame - 3 for 3 datasets
    num_calcs_in_frame = calc_frame.n_calculators
    assert num_calcs_in_frame == 3, f"Unexpected number ({num_calcs_in_frame}) of calculators in the frame. Expected 3."
    
    # get the frame name
    assert calc_frame.name == "MyCalcFrame", "Calculator frame has unexpected name."

    # ensure dataset names, labels passed along to inidividual calculators
    for (index, calc) in enumerate(calc_frame.calculators[0]):
        assert calc.name == dataset_names[index], "Indiviudal calculator has unexpected name."
        assert calc.labels == dataset_labels[index], "Indiviudal calculator has unexpected label."
    
    # check that compute runs
    calc_frame.compute()

def test_correlation_frame_normal_operation():
    """Test whether the correlation frame instantiates as expected.""" 
    datasets = [np.random.randn(3, 100) for _ in range(3)]
    dataset_names = ['d1', 'd2', 'd3']
    dataset_labels = ['label1', 'label2', 'label3']
    calc_frame = CalculatorFrame(name="MyCalcFrame", datasets=[Data(data=data, dim_order='ps') for data in datasets], 
                                 names=dataset_names, labels=dataset_labels, config='fabfour')
    
    calc_frame.compute()
    cf = calc_frame.get_correlation_df()

    assert not(cf[0].empty), "Correlation frame is empty."


def test_correlation_frame_with_labels():
    """The with_labels=True path was silently broken by a stale method name.

    It called Calculator.getstatlabels(), which does not exist (the method is
    get_stat_labels). Nothing exercised it, so it never surfaced.
    """
    datasets = [Data(data=np.random.randn(3, 100)) for _ in range(3)]
    frame = CalculatorFrame(datasets=datasets, names=['d1', 'd2', 'd3'],
                            labels=['a', 'b', 'c'], config='fabfour')
    frame.compute()
    mdf, shapes, mlabels, dlabels = frame.get_correlation_df(with_labels=True)
    assert not mdf.empty
    assert mlabels and dlabels


def test_correlation_frame_constructs():
    """CorrelationFrame itself, which no test previously instantiated."""
    datasets = [Data(data=np.random.randn(3, 100)) for _ in range(3)]
    frame = CalculatorFrame(datasets=datasets, names=['d1', 'd2', 'd3'],
                            labels=['a', 'b', 'c'], config='fabfour')
    frame.compute()
    corr = CorrelationFrame(frame)
    assert corr.n_datasets == 3
    assert corr.n_spis == 4
    assert not corr.mdf.empty

def test_normalisation_flag():
    """Test whether the normalisation flag when instantiating
    the calculator works as expected."""
    data = np.random.randn(3, 100)
    calc = Calculator(dataset=data, zscore=False, detrend=False)
    calc_loaded_dataset = calc.dataset.to_numpy().squeeze()
    
    assert (calc_loaded_dataset == data).all(), f"Calculator zscore=False not producing the correct output." 
    


def test_save_load_npz_roundtrip(tmp_path):
    """.npz is the canonical on-disk format and must round-trip exactly."""
    import pyspi
    d = Data(np.random.randn(4, 120), procnames=['w', 'x', 'y', 'z'])
    calc = Calculator(dataset=d, config='fabfour')
    calc.compute()

    path = calc.save(tmp_path / "r.npz")
    back = pyspi.load_table(path)
    assert back.equals(calc.table)
    assert list(back.index) == ['w', 'x', 'y', 'z']


def test_save_csv_and_rejects_unknown_suffix(tmp_path):
    d = Data(np.random.randn(3, 100))
    calc = Calculator(dataset=d, config='fabfour')
    calc.compute()

    assert calc.save(tmp_path / "r.csv").exists()
    with pytest.raises(ValueError, match="Unsupported suffix"):
        calc.save(tmp_path / "r.pkl")


def test_load_table_rejects_non_npz(tmp_path):
    import pyspi
    p = tmp_path / "r.csv"
    p.write_text("not npz")
    with pytest.raises(ValueError, match="Can only load"):
        pyspi.load_table(p)
