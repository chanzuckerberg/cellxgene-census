import sys


def test_ci() -> None:
    if sys.version_info[1] == 10:
        raise Exception("Python 3.10!")
