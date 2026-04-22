import pytest

@pytest.fixture(scope="session")
def spi_warning_logger(request):
    warnings_log = list()

    def add_warning(spi, module_name, max_abs, max_rel, num_exceed, num_interactions):
        warnings_log.append((spi, module_name, max_abs, max_rel, num_exceed, num_interactions))

    request.session.spi_warnings = warnings_log
    return add_warning

def pytest_sessionfinish(session, exitstatus):
    spi_warnings = getattr(session, 'spi_warnings', [])

    header_line = "=" * 90
    content_line = "-" * 90
    footer_line = "=" * 90
    header = " SPI DRIFT SUMMARY (abs/rel tolerance vs upstream baseline mean) "
    footer = f" Session completed with exit status: {exitstatus} "
    padded_header = f"{header:^90}"
    padded_footer = f"{footer:^90}"

    print("\n")
    print(header_line)
    print(padded_header)
    print(header_line)

    if spi_warnings:
        print(f"\nDetected {len(spi_warnings)} (dataset, SPI) pair(s) with drift exceeding "
              f"ATOL=1e-6 and RTOL=1e-2.\n")

        print(f"{'Dataset:SPI':<40}{'Cat':<10}{'Max |Δ|':>12}{'Max rel':>12}"
              f"{'# Exceed':>10}{'Unq Pairs':>12}")
        print(content_line)

        for est, module_name, max_abs, max_rel, num_exceed, num_interactions in spi_warnings:
            marker = " **" if max_rel > 0.1 or max_abs > 0.1 else ""
            rel_str = f"{max_rel:>12.4g}" if max_rel == max_rel else f"{'n/a':>12}"
            print(f"{est+marker:<40}{module_name:<10}{max_abs:>12.4g}{rel_str}"
                  f"{num_exceed:>10}{num_interactions:>12}")
    else:
        print("\n\nNo (dataset, SPI) pair exceeded the drift thresholds.\n")

    print(footer_line)
    print(padded_footer)
