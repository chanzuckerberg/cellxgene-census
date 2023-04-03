import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--expensive", action="store_true", dest="expensive", default=False, help="enable 'expensive' decorated tests"
    )


def pytest_configure(config: pytest.Config) -> None:
    if not config.option.expensive:
        config.option.markexpr = "not expensive"
