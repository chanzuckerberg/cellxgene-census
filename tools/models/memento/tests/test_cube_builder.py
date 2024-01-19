from os import path
from tempfile import TemporaryDirectory

import numpy as np
import tiledb

from tools.models.memento.src.estimators_cube_builder.cube_builder import build


def test_cube_builder_regression() -> None:
    """ This test is useful to ensure that the cube builder is producing the same cube as the original cube builder, after
    any refactoring. Any substantive changes to the schema of the cube or the computation will necessarily cause this
    test fail. However, if manual verification of the cube is done, a new static fixture should be generated as follows:

    cd tools/models/memento/

    python tests/fixtures/census_fixture.py \
    s3://cellxgene-data-public/cell-census/2023-10-23/soma/census_data/homo_sapiens \
    "is_primary_data == True and tissue_general in ['tongue']" \
    "feature_id in ['ENSG00000000419', 'ENSG00000002330']" \
    tests/fixtures/census-homo-sapiens-small

    python -m estimators_cube_builder \
    --experiment-uri tests/fixtures/human-tongue-2genes \
    --cube-uri tests/fixtures/estimators-cube-expected
    """

    pwd = path.dirname(__file__)
    with TemporaryDirectory() as cube_dir:
        build(cube_uri=cube_dir, experiment_uri=path.join(pwd, "fixtures", "census-homo-sapiens-small"))

        expected_cube_fixture_dir = path.join(pwd, "fixtures", "estimators-cube-expected")

        with tiledb.open(path.join(cube_dir, "obs_groups")) as actual_obs_groups:
            with tiledb.open(path.join(expected_cube_fixture_dir, "obs_groups")) as expected_obs_groups:
                assert actual_obs_groups.df[:].equals(expected_obs_groups.df[:])

        with tiledb.open(path.join(cube_dir, "estimators")) as actual_estimators:
            with tiledb.open(path.join(expected_cube_fixture_dir, "estimators")) as expected_estimators:
                actual_estimators = actual_estimators.df[:]
                expected_estimators = expected_estimators.df[:]
                for col in ["mean", "sem"]:
                    assert np.allclose(
                        actual_estimators[col], expected_estimators[col]
                    ), f"estimators mismatch for '{col}'"
