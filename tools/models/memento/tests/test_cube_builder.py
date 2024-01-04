from os import path
from tempfile import TemporaryDirectory

import tiledb

from tools.models.memento.src.estimators_cube_builder.cube_builder import build
from tools.models.memento.src.estimators_cube_builder.cube_schema import CUBE_LOGICAL_DIMS


# @pytest.mark.skip(reason="Not ready for prime time")
def test_cube_builder_regression() -> None:
    """ This test is useful to ensure that the cube builder is producing the same cube as the original cube builder, after
    any refactoring. Any substantive changes to the schema of the cube or the computation will necessarily cause this
    test fail. However, if manual verification of the cube is done, a new static fixture should be generated as follows:

    cd tools/models/memento/

    python tests/fixtures/census_fixture.py \
    s3://cellxgene-data-public/cell-census/2023-10-23/soma/census_data/homo_sapiens \
    "is_primary_data == True and tissue_general in ['tongue']" \
    "feature_id in ['ENSG00000000419', 'ENSG00000002330']" \
    tests/fixtures/census-human-tongue-2genes-2

    python -m estimators_cube_builder \
    --experiment-uri tests/fixtures/census-human-tongue-2genes \
    --cube-uri tests/fixtures/estimators-cube-human-tongue-2genes-<COMMIT>



    """
    pwd = path.dirname(__file__)
    with TemporaryDirectory() as cube_dir:
        build(cube_uri=cube_dir, experiment_uri=path.join(pwd, "fixtures", "census-human-tongue-2genes"))
        with tiledb.open(path.join(cube_dir, "obs_groups")) as actual_obs_groups:
            with tiledb.open(path.join(cube_dir, "estimators")) as actual_estimators:
                with tiledb.open(
                    path.join(pwd, "fixtures", "estimators-cube-human-tongue-2genes-ae091b2f")
                ) as expected_cube:
                    expected_cube_df = expected_cube.df[:].set_index(CUBE_LOGICAL_DIMS).sort_index()
                    actual_cube_df = (
                        actual_obs_groups.df[:]
                        .set_index("obs_group_joinid")
                        .join(actual_estimators.df[:].set_index("obs_group_joinid"))
                        .set_index(CUBE_LOGICAL_DIMS)
                        .sort_index()
                    )
                    assert all(actual_cube_df == expected_cube_df)
