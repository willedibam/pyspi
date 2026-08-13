from pyspi.utils import filter_spis
import pytest
import yaml
from unittest.mock import mock_open, patch

@pytest.fixture
def config_file(tmp_path, mock_yaml_content):
    """A real config yaml on disk, so filter_spis exercises real path resolution."""
    path = tmp_path / "mock_config.yaml"
    path.write_text(yaml.dump(mock_yaml_content))
    return path


@pytest.fixture
def mock_yaml_content():
    return {
        "module1": {
            "spi1": {"labels": ["keyword1", "keyword2"], "configs": [1, 2]},
            "spi2": {"labels": ["keyword1"], "configs": [3]},
        },
        "module2": {
            "spi3": {"labels": ["keyword3"], "configs": [1, 2, 3]},
        },
    }

def test_filter_spis_invalid_keywords():
    """Pass in a dataype other than a list for the keywords"""
    with pytest.raises(ValueError) as excinfo:
        filter_spis(keywords="linear", configfile="full")
    assert "Keywords must be provided as a list of strings" in str(excinfo.value)
    # check for passing in an empty list
    with pytest.raises(ValueError) as excinfo:
        filter_spis(keywords=[], configfile="full")
    assert "At least one keyword must be provided" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        filter_spis(keywords=[4], configfile="full")
    assert "All keywords must be strings" in str(excinfo.value)  

def test_filter_spis_with_invalid_config():
    """Pass in an invalid/missing config file"""
    with pytest.raises(FileNotFoundError):
        filter_spis(keywords=["test"], configfile="invalid_config.yaml")

def test_filter_spis_no_matches(config_file, tmp_path, monkeypatch):
    """Pass in keywords that return no spis and check for ValueError"""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError) as excinfo:
        filter_spis(keywords=["random_keyword"], output_name="mock_filtered_config",
                    configfile=str(config_file))
    assert "0 SPIs were found" in str(excinfo.value), "Incorrect error message returned when no keywords match found."

def test_filter_spis_normal_operation(config_file, tmp_path, monkeypatch):
    """Filter a config down to the SPIs carrying every keyword."""
    monkeypatch.chdir(tmp_path)
    filter_spis(keywords=["keyword1", "keyword2"], output_name="mock_filtered_config",
                configfile=str(config_file))

    written = yaml.safe_load((tmp_path / "mock_filtered_config.yaml").read_text())
    assert written == {"module1": {"spi1": {"labels": ["keyword1", "keyword2"], "configs": [1, 2]}}}, \
        "Expected filtered YAML does not match actual filtered YAML."


def test_filter_spis_io_error_on_read():
    # check to see whether io error is raised when trying to access the configfile
    with patch("builtins.open", mock_open(read_data="data")) as mocked_file:
        mocked_file.side_effect = IOError("error")
        with pytest.raises(IOError):
            filter_spis(["keyword"], "output", "config.yaml")
            
def test_filter_spis_saves_with_random_name_if_no_name_provided(mock_yaml_content):
    # mock os.urandom to return a predictable name
    random_bytes = bytes([1, 2, 3, 4])
    expected_random_part = "01020304"

    with patch("builtins.open", mock_open()) as mocked_file, patch("os.path.isfile", return_value=True), \
         patch("yaml.load", return_value=mock_yaml_content), patch("os.urandom", return_value=random_bytes):
        
        # run the filter function without providing an output name
        filter_spis(["keyword1"])

        # construct the expected output name
        expected_file_name_pattern = f"config_{expected_random_part}.yaml"

        # check the mocked open function to see if file with expected name is opened (for writing)
        call_args_list = mocked_file.call_args_list
        found_expected_call = any(
            expected_file_name_pattern in call_args.args[0] and
            ('w' in call_args.args[1] if len(call_args.args) > 1 else 'w' in call_args.kwargs.get('mode', ''))
            for call_args in call_args_list
        )

        assert found_expected_call, f"no file with the expected name {expected_file_name_pattern} was saved."

def test_loads_default_config_if_no_config_specified(tmp_path, monkeypatch):
    """With no configfile, filter_spis falls back to the bundled 'full' config."""
    monkeypatch.chdir(tmp_path)
    filter_spis(["nonlinear"], output_name="from_default")

    written = yaml.safe_load((tmp_path / "from_default.yaml").read_text())
    assert written, "Filtering the default config produced an empty result."
    assert all(
        "nonlinear" in entry["labels"]
        for module in written.values() for entry in module.values()
    ), "Default-config filtering returned SPIs missing the requested label."
