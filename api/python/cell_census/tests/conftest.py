def pytest_addoption(parser):
    parser.addoption(
        "--expensive", action="store_true", dest="expensive", default=False, help="enable 'expensive' decorated tests"
    )


def pytest_configure(config):
    if not config.option.expensive:
        setattr(config.option, "markexpr", "not expensive")
