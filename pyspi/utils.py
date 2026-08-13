import numpy as np
from scipy.stats import zscore
import warnings
import pandas as pd
import os
import yaml

def acf(x, mode='positive'):
    """Return the autocorrelation function using FFT-based computation.

    O(N log N) via FFT, replacing the original O(N^2) np.correlate approach.
    """
    if x.ndim > 1:
        x = np.squeeze(x)

    x = x - x.mean()
    s = x.std()
    if s == 0:
        n = len(x)
        return np.zeros(n) if mode == 'positive' else np.zeros(2 * n - 1)
    x = x / s

    n = len(x)
    fft_size = 2 * n
    X = np.fft.rfft(x, n=fft_size)
    acf_full = np.fft.irfft(X * np.conj(X), n=fft_size)[:n]
    acf_full = acf_full / acf_full[0]  # normalize so acf[0] = 1

    if mode == 'positive':
        return acf_full
    # full symmetric ACF
    return np.concatenate([acf_full[::-1], acf_full[1:]])

def swap_chars(s, i_1, i_2):
    """Swap to characters in a string.

    Example:
        >>> print(swap_chars('heLlotHere', 2, 6))
        'heHlotLere'
    """
    if i_1 > i_2:
        i_1, i_2 = i_2, i_1
    return ''.join([s[0:i_1], s[i_2], s[i_1+1:i_2], s[i_1], s[i_2+1:]])

def convert_mdf_to_ddf(df):
    ddf = pd.pivot_table(data=df.stack(dropna=False).reset_index(),index='Dataset',columns=['SPI-1', 'SPI-2'],dropna=False).T.droplevel(0)
    return ddf

def filter_spis(keywords, output_name = None, configfile= None):
    """
    Filter a YAML using a list of keywords, and save the reduced set as a new
    YAML with a user-specified name (or a random one if not provided) in the
    current directory.

    Args:
        keywords (list): A list of keywords (as strings) to filter the YAML.
        output_name (str, optional): The desired name for the output file. Defaults to a random name.
        configfile (str, optional): The path to the input YAML file. Defaults to the `config.yaml' in the pyspi dir.

    Raises:
        ValueError: If `keywords` is not a list or if no SPIs match the keywords.
        FileNotFoundError: If the specified `configfile` or the default `config.yaml` is not found.
        IOError: If there's an error reading the YAML file.
    """
    # handle invalid keyword input
    if not keywords:
        raise ValueError("At least one keyword must be provided.")
    if not all(isinstance(keyword, str) for keyword in keywords):
        raise ValueError("All keywords must be strings.")
    if not isinstance(keywords, list):
        raise ValueError("Keywords must be provided as a list of strings.")

    # Default to the full bundled config; otherwise accept a bundled name or path.
    from pyspi.calculator import resolve_config
    if configfile is None:
        configfile = resolve_config("full")
        source_file_info = f"Default bundled config '{configfile}' was used as the source file."
    else:
        configfile = resolve_config(configfile)
        source_file_info = f"User-specified config file '{configfile}' was used as the source file."

    # load in user-specified yaml
    try:
        with open(configfile) as f:
            yf = yaml.load(f, Loader=yaml.FullLoader)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file '{configfile}' not found.")
    except Exception as e:
        # handle all other exceptions
        raise IOError(f"An error occurred while trying to read '{configfile}': {e}")

    # new dictionary to be converted to final YAML
    filtered_subset = {}
    spis_found = 0

    for module in yf:
        module_spis = {}
        for spi in yf[module]:
            spi_labels = yf[module][spi].get('labels') or []
            if all(keyword in spi_labels for keyword in keywords):
                module_spis[spi] = yf[module][spi]
                if yf[module][spi].get('configs'):
                    spis_found += len(yf[module][spi].get('configs'))
                else:
                    spis_found += 1

        if module_spis:
            filtered_subset[module] = module_spis

    # check that > 0 SPIs found
    if spis_found == 0:
        raise ValueError(f"0 SPIs were found with the specific keywords: {keywords}.")

    # construct output file path
    if output_name is None:
        # use a unique name
        output_name = "config_" + os.urandom(4).hex()

    output_file = os.path.join(os.getcwd(), f"{output_name}.yaml")

    # write to YAML
    with open(output_file, "w") as outfile:
        yaml.dump(filtered_subset, outfile, default_flow_style=False, sort_keys=False)

    # output relevant information
    print(f"""\nOperation Summary:
-----------------
- {source_file_info}
- Total SPIs Matched: {spis_found} SPI(s) were found with the specific keywords: {keywords}.
- New File Created: A YAML file named `{output_name}.yaml` has been saved in the current directory: `{output_file}'
- Next Steps: To utilise the filtered set of SPIs, please initialise a new Calculator instance with the following command:
`Calculator(config='{output_file}')`
""")

def inspect_calc_results(calc):
    """
    Display a summary of the computed SPI results, including counts of successful computations,
    outputs with NaNs, and partially computed results.
    """
    total_num_spis = calc.n_spis
    num_procs = calc.dataset.n_processes
    spi_results = dict({'Successful': list(), 'NaNs': list(), 'Partial NaNs': list()})
    for key in calc.spis.keys():
        if calc.table[key].isna().all().all():
            spi_results['NaNs'].append(key)
        elif calc.table[key].isnull().values.sum() > num_procs:
            # off-diagonal NaNs
            spi_results['Partial NaNs'].append(key)
        else:
            # returned numeric values (i.e., not NaN)
            spi_results['Successful'].append(key)

    # print summary
    double_line_60 = "="*60
    single_line_60 = "-"*60
    print("\nSPI Computation Results Summary")
    print(double_line_60)
    print(f"\nTotal number of SPIs attempted: {total_num_spis}")
    print(f"Number of SPIs successfully computed: {len(spi_results['Successful'])} ({len(spi_results['Successful']) / total_num_spis * 100:.2f}%)")
    print(single_line_60)
    print("Category       | Count | Percentage")
    print(single_line_60)
    for category, spis in spi_results.items():
        count = len(spis)
        percentage = (count / total_num_spis) * 100
        print(f"{category:14} | {count:5} | {percentage:6.2f}%")
    print(single_line_60)

    if spi_results['NaNs']:
        print(f"\n[{len(spi_results['NaNs'])}] SPI(s) produced NaN outputs:")
        print(single_line_60)
        for i, spi in enumerate(spi_results['NaNs']):
            print(f"{i+1}. {spi}")
        print(single_line_60 + "\n")
    if spi_results['Partial NaNs']:
        print(f"\n[{len(spi_results['Partial NaNs'])}] SPIs which produced partial NaN outputs:")
        print(single_line_60)
        for i, spi in enumerate(spi_results['Partial NaNs']):
            print(f"{i+1}. {spi}")
        print(single_line_60 + "\n")
