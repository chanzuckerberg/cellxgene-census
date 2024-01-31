import pathlib
import sys
from typing import Callable, List, Optional, Sequence, Union
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pytest
import tiledbsoma as soma
from scipy import sparse
from scipy.sparse import coo_matrix, spmatrix
from somacore import AxisQuery
from tiledbsoma import Experiment, _factory
from tiledbsoma._collection import CollectionBase
from torch.utils.data._utils.worker import WorkerInfo

# conditionally import torch, as it will not be available in all test environments
try:
    from torch import Tensor, float32

    from cellxgene_census.experimental.ml.pytorch import (
        ExperimentDataPipe,
        experiment_dataloader,
    )
except ImportError:
    # this should only occur when not running `experimental`-marked tests
    pass


def pytorch_x_value_gen(obs_range: range, var_range: range) -> spmatrix:
    occupied_shape = (obs_range.stop - obs_range.start, var_range.stop - var_range.start)
    checkerboard_of_ones = coo_matrix(np.indices(occupied_shape).sum(axis=0) % 2)
    checkerboard_of_ones.row += obs_range.start
    checkerboard_of_ones.col += var_range.start
    return checkerboard_of_ones


def pytorch_seq_x_value_gen(obs_range: range, var_range: range) -> spmatrix:
    """A sparse matrix where the values of each col are the obs_range values. Useful for checking the
    X values are being returned in the correct order."""
    data = np.vstack([list(obs_range)] * len(var_range)).flatten()
    rows = np.vstack([list(obs_range)] * len(var_range)).flatten()
    cols = np.column_stack([list(var_range)] * len(obs_range)).flatten()
    return coo_matrix((data, (rows, cols)))


@pytest.fixture
def X_layer_names() -> List[str]:
    return ["raw"]


@pytest.fixture
def obsp_layer_names() -> Optional[List[str]]:
    return None


@pytest.fixture
def varp_layer_names() -> Optional[List[str]]:
    return None


def add_dataframe(coll: CollectionBase, key: str, value_range: range) -> None:
    df = coll.add_new_dataframe(
        key,
        schema=pa.schema(
            [
                ("soma_joinid", pa.int64()),
                ("label", pa.large_string()),
            ]
        ),
        index_column_names=["soma_joinid"],
    )
    df.write(
        pa.Table.from_pydict(
            {
                "soma_joinid": list(value_range),
                "label": [str(i) for i in value_range],
            }
        )
    )


def add_sparse_array(
    coll: CollectionBase,
    key: str,
    obs_range: range,
    var_range: range,
    value_gen: Callable[[range, range], spmatrix],
) -> None:
    a = coll.add_new_sparse_ndarray(key, type=pa.float32(), shape=(obs_range.stop, var_range.stop))
    tensor = pa.SparseCOOTensor.from_scipy(value_gen(obs_range, var_range))
    a.write(tensor)


@pytest.fixture(scope="function")
def soma_experiment(
    tmp_path: pathlib.Path,
    obs_range: Union[int, range],
    var_range: Union[int, range],
    X_value_gen: Callable[[range, range], sparse.spmatrix],
    obsp_layer_names: Sequence[str],
    varp_layer_names: Sequence[str],
) -> soma.Experiment:
    with soma.Experiment.create((tmp_path / "exp").as_posix()) as exp:
        if isinstance(obs_range, int):
            obs_range = range(obs_range)
        if isinstance(var_range, int):
            var_range = range(var_range)

        add_dataframe(exp, "obs", obs_range)
        ms = exp.add_new_collection("ms")
        rna = ms.add_new_collection("RNA", soma.Measurement)
        add_dataframe(rna, "var", var_range)
        rna_x = rna.add_new_collection("X", soma.Collection)
        add_sparse_array(rna_x, "raw", obs_range, var_range, X_value_gen)

        if obsp_layer_names:
            obsp = rna.add_new_collection("obsp")
            for obsp_layer_name in obsp_layer_names:
                add_sparse_array(obsp, obsp_layer_name, obs_range, var_range, X_value_gen)

        if varp_layer_names:
            varp = rna.add_new_collection("varp")
            for varp_layer_name in varp_layer_names:
                add_sparse_array(varp, varp_layer_name, obs_range, var_range, X_value_gen)
    return _factory.open((tmp_path / "exp").as_posix())


@pytest.mark.experimental
# noinspection PyTestParametrized
@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen,use_eager_fetch",
    [(6, 3, pytorch_x_value_gen, use_eager_fetch) for use_eager_fetch in (True, False)],
)
def test_non_batched(soma_experiment: Experiment, use_eager_fetch: bool) -> None:
    exp_data_pipe = ExperimentDataPipe(
        soma_experiment,
        measurement_name="RNA",
        X_name="raw",
        obs_column_names=["label"],
        use_eager_fetch=use_eager_fetch,
    )
    row_iter = iter(exp_data_pipe)

    row = next(row_iter)
    assert row[0].int().tolist() == [0, 1, 0]
    assert row[1].tolist() == [0, 0]


@pytest.mark.experimental
# noinspection PyTestParametrized,DuplicatedCode
@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen,use_eager_fetch",
    [(6, 3, pytorch_x_value_gen, use_eager_fetch) for use_eager_fetch in (True, False)],
)
def test_batching__all_batches_full_size(soma_experiment: Experiment, use_eager_fetch: bool) -> None:
    exp_data_pipe = ExperimentDataPipe(
        soma_experiment,
        measurement_name="RNA",
        X_name="raw",
        obs_column_names=["label"],
        batch_size=3,
        use_eager_fetch=use_eager_fetch,
    )
    batch_iter = iter(exp_data_pipe)

    batch = next(batch_iter)
    assert batch[0].int().tolist() == [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    assert batch[1].tolist() == [[0, 0], [1, 1], [2, 2]]

    batch = next(batch_iter)
    assert batch[0].int().tolist() == [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
    assert batch[1].tolist() == [[3, 3], [4, 4], [5, 5]]

    with pytest.raises(StopIteration):
        next(batch_iter)


@pytest.mark.experimental
# noinspection PyTestParametrized,DuplicatedCode
@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen,use_eager_fetch",
    [(range(100_000_000, 100_000_003), 3, pytorch_x_value_gen, use_eager_fetch) for use_eager_fetch in (True, False)],
)
def test_unique_soma_joinids(soma_experiment: Experiment, use_eager_fetch: bool) -> None:
    exp_data_pipe = ExperimentDataPipe(
        soma_experiment,
        measurement_name="RNA",
        X_name="raw",
        obs_column_names=["label"],
        batch_size=3,
        use_eager_fetch=use_eager_fetch,
    )

    soma_joinids = np.concatenate([batch[1][:, 0].numpy() for batch in exp_data_pipe])

    assert len(np.unique(soma_joinids)) == len(soma_joinids)


@pytest.mark.experimental
# noinspection PyTestParametrized
@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen,use_eager_fetch",
    [(5, 3, pytorch_x_value_gen, use_eager_fetch) for use_eager_fetch in (True, False)],
)
def test_batching__partial_final_batch_size(soma_experiment: Experiment, use_eager_fetch: bool) -> None:
    exp_data_pipe = ExperimentDataPipe(
        soma_experiment,
        measurement_name="RNA",
        X_name="raw",
        obs_column_names=["label"],
        batch_size=3,
        use_eager_fetch=use_eager_fetch,
    )
    batch_iter = iter(exp_data_pipe)

    next(batch_iter)
    batch = next(batch_iter)
    assert batch[0].int().tolist() == [[1, 0, 1], [0, 1, 0]]

    with pytest.raises(StopIteration):
        next(batch_iter)


@pytest.mark.experimental
# noinspection PyTestParametrized,DuplicatedCode
@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen,use_eager_fetch",
    [(3, 3, pytorch_x_value_gen, use_eager_fetch) for use_eager_fetch in (True, False)],
)
def test_batching__exactly_one_batch(soma_experiment: Experiment, use_eager_fetch: bool) -> None:
    exp_data_pipe = ExperimentDataPipe(
        soma_experiment,
        measurement_name="RNA",
        X_name="raw",
        obs_column_names=["label"],
        batch_size=3,
        use_eager_fetch=use_eager_fetch,
    )
    batch_iter = iter(exp_data_pipe)

    batch = next(batch_iter)
    assert batch[0].int().tolist() == [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    assert batch[1].tolist() == [[0, 0], [1, 1], [2, 2]]

    with pytest.raises(StopIteration):
        next(batch_iter)


@pytest.mark.experimental
# noinspection PyTestParametrized
@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen,use_eager_fetch",
    [(6, 3, pytorch_x_value_gen, use_eager_fetch) for use_eager_fetch in (True, False)],
)
def test_batching__empty_query_result(soma_experiment: Experiment, use_eager_fetch: bool) -> None:
    exp_data_pipe = ExperimentDataPipe(
        soma_experiment,
        measurement_name="RNA",
        X_name="raw",
        obs_query=AxisQuery(coords=([],)),
        obs_column_names=["label"],
        batch_size=3,
        use_eager_fetch=use_eager_fetch,
    )
    batch_iter = iter(exp_data_pipe)

    with pytest.raises(StopIteration):
        next(batch_iter)


@pytest.mark.experimental
# noinspection PyTestParametrized
@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen,use_eager_fetch",
    [(6, 3, pytorch_x_value_gen, use_eager_fetch) for use_eager_fetch in (True, False)],
)
def test_sparse_output__non_batched(soma_experiment: Experiment, use_eager_fetch: bool) -> None:
    exp_data_pipe = ExperimentDataPipe(
        soma_experiment,
        measurement_name="RNA",
        X_name="raw",
        obs_column_names=["label"],
        return_sparse_X=True,
        use_eager_fetch=use_eager_fetch,
    )
    batch_iter = iter(exp_data_pipe)

    batch = next(batch_iter)
    assert isinstance(batch[1], Tensor)
    assert batch[0].to_dense().tolist() == [0, 1, 0]


@pytest.mark.experimental
# noinspection PyTestParametrized
@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen,use_eager_fetch",
    [(6, 3, pytorch_x_value_gen, use_eager_fetch) for use_eager_fetch in (True, False)],
)
def test_sparse_output__batched(soma_experiment: Experiment, use_eager_fetch: bool) -> None:
    exp_data_pipe = ExperimentDataPipe(
        soma_experiment,
        measurement_name="RNA",
        X_name="raw",
        obs_column_names=["label"],
        batch_size=3,
        return_sparse_X=True,
        use_eager_fetch=use_eager_fetch,
    )
    batch_iter = iter(exp_data_pipe)

    batch = next(batch_iter)
    assert isinstance(batch[1], Tensor)
    assert batch[0].to_dense().tolist() == [[0, 1, 0], [1, 0, 1], [0, 1, 0]]


@pytest.mark.experimental
# noinspection PyTestParametrized,DuplicatedCode
@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen,use_eager_fetch",
    [(10, 1, pytorch_x_value_gen, use_eager_fetch) for use_eager_fetch in (True, False)],
)
def test_batching__partial_soma_batches_are_concatenated(soma_experiment: Experiment, use_eager_fetch: bool) -> None:
    exp_data_pipe = ExperimentDataPipe(
        soma_experiment,
        measurement_name="RNA",
        X_name="raw",
        obs_column_names=[],
        batch_size=3,
        # set SOMA batch read size such that PyTorch batches will span the tail and head of two SOMA batches
        soma_chunk_size=4,
        use_eager_fetch=use_eager_fetch,
    )

    full_result = list(exp_data_pipe)

    assert [len(batch[1]) for batch in full_result] == [3, 3, 3, 1]


@pytest.mark.experimental
# noinspection PyTestParametrized
@pytest.mark.parametrize("obs_range,var_range,X_value_gen", [(3, 3, pytorch_x_value_gen)])
def test_encoders(soma_experiment: Experiment) -> None:
    exp_data_pipe = ExperimentDataPipe(
        soma_experiment,
        measurement_name="RNA",
        X_name="raw",
        obs_column_names=["label"],
        batch_size=3,
    )
    batch_iter = iter(exp_data_pipe)

    batch = next(batch_iter)
    assert isinstance(batch[1], Tensor)

    labels_encoded = batch[1][:, 1]
    labels_decoded = exp_data_pipe.obs_encoders["label"].inverse_transform(labels_encoded)
    assert labels_decoded.tolist() == ["0", "1", "2"]


@pytest.mark.experimental
@pytest.mark.skipif(
    (sys.version_info.major, sys.version_info.minor) == (3, 9), reason="fails intermittently with OOM error for 3.9"
)
# noinspection PyTestParametrized
@pytest.mark.parametrize("obs_range,var_range,X_value_gen", [(6, 3, pytorch_x_value_gen)])
def test_multiprocessing__returns_full_result(soma_experiment: Experiment) -> None:
    """Tests the ExperimentDataPipe provides all data, as collected from multiple processes that are managed by a
    PyTorch DataLoader with multiple workers configured."""

    dp = ExperimentDataPipe(
        soma_experiment,
        measurement_name="RNA",
        X_name="raw",
        obs_column_names=["label"],
        soma_chunk_size=3,  # two chunks, one per worker
    )
    # Note we're testing the ExperimentDataPipe via a DataLoader, since this is what sets up the multiprocessing
    dl = experiment_dataloader(dp, num_workers=2)

    full_result = list(iter(dl))

    soma_joinids = [t[1][0].item() for t in full_result]
    assert sorted(soma_joinids) == list(range(6))


@pytest.mark.experimental
# noinspection PyTestParametrized
@pytest.mark.parametrize("obs_range,var_range,X_value_gen", [(6, 3, pytorch_x_value_gen)])
def test_distributed__returns_data_partition_for_rank(soma_experiment: Experiment) -> None:
    """Tests pytorch._partition_obs_joinids() behavior in a simulated PyTorch distributed processing mode,
    using mocks to avoid having to do real PyTorch distributed setup."""

    with patch("cellxgene_census.experimental.ml.pytorch.dist.is_initialized") as mock_dist_is_initialized, patch(
        "cellxgene_census.experimental.ml.pytorch.dist.get_rank"
    ) as mock_dist_get_rank, patch(
        "cellxgene_census.experimental.ml.pytorch.dist.get_world_size"
    ) as mock_dist_get_world_size:
        mock_dist_is_initialized.return_value = True
        mock_dist_get_rank.return_value = 1
        mock_dist_get_world_size.return_value = 3

        dp = ExperimentDataPipe(
            soma_experiment, measurement_name="RNA", X_name="raw", obs_column_names=["label"], soma_chunk_size=2
        )
        full_result = list(iter(dp))

        soma_joinids = [t[1][0].item() for t in full_result]

        # Of the 6 obs rows, the PyTorch process of rank 1 should get [2, 3]
        # (rank 0 gets [0, 1], rank 2 gets [4, 5])
        assert sorted(soma_joinids) == [2, 3]


@pytest.mark.experimental
# noinspection PyTestParametrized
@pytest.mark.parametrize("obs_range,var_range,X_value_gen", [(12, 3, pytorch_x_value_gen)])
def test_distributed_and_multiprocessing__returns_data_partition_for_rank(soma_experiment: Experiment) -> None:
    """Tests pytorch._partition_obs_joinids() behavior in a simulated PyTorch distributed processing mode and
    DataLoader multiprocessing mode, using mocks to avoid having to do distributed pytorch
    setup or real DataLoader multiprocessing."""

    with patch("torch.utils.data.get_worker_info") as mock_get_worker_info, patch(
        "cellxgene_census.experimental.ml.pytorch.dist.is_initialized"
    ) as mock_dist_is_initialized, patch(
        "cellxgene_census.experimental.ml.pytorch.dist.get_rank"
    ) as mock_dist_get_rank, patch(
        "cellxgene_census.experimental.ml.pytorch.dist.get_world_size"
    ) as mock_dist_get_world_size:
        mock_get_worker_info.return_value = WorkerInfo(id=1, num_workers=2, seed=1234)
        mock_dist_is_initialized.return_value = True
        mock_dist_get_rank.return_value = 1
        mock_dist_get_world_size.return_value = 3

        dp = ExperimentDataPipe(
            soma_experiment, measurement_name="RNA", X_name="raw", obs_column_names=["label"], soma_chunk_size=2
        )

        full_result = list(iter(dp))

        soma_joinids = [t[1][0].item() for t in full_result]

        # Of the 12 obs rows, the PyTorch process of rank 1 should get [4..7], and then within that partition,
        # the 2nd DataLoader process should get the second half of the rank's partition, which is just [6, 7]
        # (rank 0 gets [0..3], rank 2 gets [8..11])
        assert sorted(soma_joinids) == [6, 7]


@pytest.mark.experimental
# noinspection PyTestParametrized,DuplicatedCode
@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen,use_eager_fetch",
    [(3, 3, pytorch_x_value_gen, use_eager_fetch) for use_eager_fetch in (True, False)],
)
def test_experiment_dataloader__non_batched(soma_experiment: Experiment, use_eager_fetch: bool) -> None:
    dp = ExperimentDataPipe(
        soma_experiment,
        measurement_name="RNA",
        X_name="raw",
        obs_column_names=["label"],
        use_eager_fetch=use_eager_fetch,
    )
    dl = experiment_dataloader(dp)
    torch_data = [row for row in dl]

    row = torch_data[0]
    assert row[0].to_dense().tolist() == [0, 1, 0]
    assert row[1].tolist() == [0, 0]


@pytest.mark.experimental
# noinspection PyTestParametrized,DuplicatedCode
@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen,use_eager_fetch",
    [(6, 3, pytorch_x_value_gen, use_eager_fetch) for use_eager_fetch in (True, False)],
)
def test_experiment_dataloader__batched(soma_experiment: Experiment, use_eager_fetch: bool) -> None:
    dp = ExperimentDataPipe(
        soma_experiment,
        measurement_name="RNA",
        X_name="raw",
        obs_column_names=["label"],
        batch_size=3,
        use_eager_fetch=use_eager_fetch,
    )
    dl = experiment_dataloader(dp)
    torch_data = [row for row in dl]

    batch = torch_data[0]
    assert batch[0].to_dense().tolist() == [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    assert batch[1].tolist() == [[0, 0], [1, 1], [2, 2]]


@pytest.mark.experimental
# noinspection PyTestParametrized,DuplicatedCode
@pytest.mark.parametrize(
    "obs_range,var_range,X_value_gen,use_eager_fetch",
    [(6, 3, pytorch_x_value_gen, use_eager_fetch) for use_eager_fetch in (True, False)],
)
def test__X_tensor_dtype_matches_X_matrix(soma_experiment: Experiment, use_eager_fetch: bool) -> None:
    dp = ExperimentDataPipe(
        soma_experiment,
        measurement_name="RNA",
        X_name="raw",
        obs_column_names=["label"],
        batch_size=3,
        use_eager_fetch=use_eager_fetch,
    )
    torch_data = next(iter(dp))

    assert torch_data[0].dtype == float32


@pytest.mark.experimental
# noinspection PyTestParametrized,DuplicatedCode
@pytest.mark.parametrize("obs_range,var_range,X_value_gen", [(10, 1, pytorch_x_value_gen)])
def test__pytorch_splitting(soma_experiment: Experiment) -> None:
    dp = ExperimentDataPipe(soma_experiment, measurement_name="RNA", X_name="raw", obs_column_names=["label"])
    dp_train, dp_test = dp.random_split(weights={"train": 0.7, "test": 0.3}, seed=1234)
    dl = experiment_dataloader(dp_train)

    all_rows = list(iter(dl))
    assert len(all_rows) == 7


@pytest.mark.experimental
# noinspection PyTestParametrized,DuplicatedCode
@pytest.mark.parametrize("obs_range,var_range,X_value_gen", [(16, 1, pytorch_seq_x_value_gen)])
def test__shuffle(soma_experiment: Experiment) -> None:
    dp = ExperimentDataPipe(
        soma_experiment, measurement_name="RNA", X_name="raw", obs_column_names=["label"], shuffle=True
    )

    all_rows = list(iter(dp))

    soma_joinids = [row[1][0].item() for row in all_rows]
    X_values = [row[0][0].item() for row in all_rows]

    # same elements
    assert set(soma_joinids) == set(range(16))
    # not ordered! (...with a `1/16!` probability of being ordered)
    assert soma_joinids != list(range(16))
    # randomizes X in same order as obs
    # note: X values were explicitly set to match obs_joinids to allow for this simple assertion
    assert X_values == soma_joinids


@pytest.mark.experimental
@pytest.mark.skip(reason="Not implemented")
def test_experiment_dataloader__multiprocess_sparse_matrix__fails() -> None:
    pass


@pytest.mark.experimental
@pytest.mark.skip(reason="Not implemented")
def test_experiment_dataloader__multiprocess_dense_matrix__ok() -> None:
    pass


@pytest.mark.experimental
@patch("cellxgene_census.experimental.ml.pytorch.ExperimentDataPipe")
def test_experiment_dataloader__unsupported_params__fails(dummy_exp_data_pipe: ExperimentDataPipe) -> None:
    with pytest.raises(ValueError):
        experiment_dataloader(dummy_exp_data_pipe, shuffle=True)
    with pytest.raises(ValueError):
        experiment_dataloader(dummy_exp_data_pipe, batch_size=3)
    with pytest.raises(ValueError):
        experiment_dataloader(dummy_exp_data_pipe, batch_sampler=[])
    with pytest.raises(ValueError):
        experiment_dataloader(dummy_exp_data_pipe, sampler=[])
    with pytest.raises(ValueError):
        experiment_dataloader(dummy_exp_data_pipe, collate_fn=lambda x: x)
