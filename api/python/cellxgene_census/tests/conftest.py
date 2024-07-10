import multiprocessing

import pytest
import tiledbsoma as soma

TEST_MARKERS_SKIPPED_BY_DEFAULT = ["expensive", "experimental"]

# tiledb will complain if this isn't set and a process is spawned. May cause segfaults on the proxy test if this isn't set.
multiprocessing.set_start_method("spawn", force=True)


def pytest_addoption(parser: pytest.Parser) -> None:
    for test_option in TEST_MARKERS_SKIPPED_BY_DEFAULT:
        parser.addoption(
            f"--{test_option}",
            action="store_true",
            dest=test_option,
            default=False,
            help=f"enable '{test_option}' decorated tests",
        )

    # Add option to set the census_version (not set by default)
    parser.addoption("--census_version", action="store", default="stable")


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """This is called for every test"""

    # Configure census_version if used
    census_version = metafunc.config.option.census_version
    if "census_version" in metafunc.fixturenames:
        metafunc.parametrize("census_version", [census_version])


def pytest_configure(config: pytest.Config) -> None:
    """
    Exclude tests marked with any of the TEST_MARKERS_SKIPPED_BY_DEFAULT values, unless the corresponding explicit
    flag is specified by the user.
    """
    excluded_markexprs = []

    for test_option in TEST_MARKERS_SKIPPED_BY_DEFAULT:
        if not vars(config.option).get(test_option, False):
            excluded_markexprs.append(test_option)

    if config.option.markexpr and excluded_markexprs:
        config.option.markexpr += " and "
    config.option.markexpr += " and ".join([f"not {m}" for m in excluded_markexprs])


@pytest.fixture
def small_mem_context() -> soma.SOMATileDBContext:
    """used to keep memory usage smaller for GHA runners."""
    from cellxgene_census import get_default_soma_context

    return get_default_soma_context(tiledb_config={"soma.init_buffer_bytes": 32 * 1024**2})


@pytest.fixture(scope="session")
def census() -> soma.Collection:
    import cellxgene_census

    return cellxgene_census.open_soma(census_version="latest")


@pytest.fixture(scope="session")
def lts_census() -> soma.Collection:
    import cellxgene_census

    return cellxgene_census.open_soma(census_version="stable")


@pytest.fixture(scope="session")
def dec_lts_census() -> soma.Collection:
    """Fixture for the 2023-12-15 LTS Census."""
    import cellxgene_census

    return cellxgene_census.open_soma(census_version="2023-12-15")
