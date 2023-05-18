import pathlib
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pytest
import tiledbsoma as soma
from scipy import sparse
from scipy.sparse import coo_matrix, spmatrix
from somacore import AxisQuery
from tiledbsoma import Experiment, _factory
from tiledbsoma._collection import CollectionBase
from torch import Tensor

from cellxgene_census.incubation.pytorch import ExperimentDataPipe


def pytorch_x_value_gen(shape: Tuple[int, int]) -> spmatrix:
    checkerboard_of_ones = coo_matrix(np.indices(shape).sum(axis=0) % 2)
    return checkerboard_of_ones


@pytest.fixture
def X_layer_names() -> List[str]:
    return ["raw"]


@pytest.fixture
def obsp_layer_names() -> Optional[List[str]]:
    return None


@pytest.fixture
def varp_layer_names() -> Optional[List[str]]:
    return None


@pytest.fixture
def X_value_gen() -> Callable[[Tuple[int, int]], sparse.spmatrix]:
    def _x_value_gen(shape: Tuple[int, int]) -> sparse.coo_matrix:
        return sparse.random(
            shape[0],
            shape[1],
            density=0.1,
            format="coo",
            dtype=np.float32,
            random_state=np.random.default_rng(),
        )

    return _x_value_gen


def add_dataframe(coll: CollectionBase, key: str, sz: int) -> None:
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
                "soma_joinid": [i for i in range(sz)],
                "label": [str(i) for i in range(sz)],
            }
        )
    )


def add_sparse_array(
    coll: CollectionBase, key: str, shape: Tuple[int, int], value_gen: Callable[[Tuple[int, int]], sparse.spmatrix]
) -> None:
    a = coll.add_new_sparse_ndarray(key, type=pa.float32(), shape=shape)
    tensor = pa.SparseCOOTensor.from_scipy(value_gen(shape))
    a.write(tensor)


@pytest.fixture(scope="function")
def soma_experiment(
    tmp_path: pathlib.Path,
    n_obs: int,
    n_vars: int,
    X_layer_names: Sequence[str],
    X_value_gen: Callable[[Tuple[int, int]], sparse.spmatrix],
    obsp_layer_names: Sequence[str],
    varp_layer_names: Sequence[str],
) -> soma.Experiment:
    with soma.Experiment.create((tmp_path / "exp").as_posix()) as exp:
        add_dataframe(exp, "obs", n_obs)
        ms = exp.add_new_collection("ms")
        rna = ms.add_new_collection("RNA", soma.Measurement)
        add_dataframe(rna, "var", n_vars)
        rna_x = rna.add_new_collection("X", soma.Collection)
        for X_layer_name in X_layer_names:
            add_sparse_array(rna_x, X_layer_name, (n_obs, n_vars), X_value_gen)

        if obsp_layer_names:
            obsp = rna.add_new_collection("obsp")
            for obsp_layer_name in obsp_layer_names:
                add_sparse_array(obsp, obsp_layer_name, (n_obs, n_obs), X_value_gen)

        if varp_layer_names:
            varp = rna.add_new_collection("varp")
            for varp_layer_name in varp_layer_names:
                add_sparse_array(varp, varp_layer_name, (n_vars, n_vars), X_value_gen)
    return _factory.open((tmp_path / "exp").as_posix())


# noinspection PyTestParametrized
@pytest.mark.parametrize("n_obs,n_vars,X_layer_names,X_value_gen", [(6, 3, ("raw",), pytorch_x_value_gen)])
def test_non_batched(soma_experiment: Experiment) -> None:
    exp_data_pipe = ExperimentDataPipe(
        exp_uri=soma_experiment.uri, ms_name="RNA", layer_name="raw", obs_column_names=["label"]
    )
    row_iter = iter(exp_data_pipe)

    row = next(row_iter)
    assert row[0].tolist() == [0, 0]
    assert row[1].to_dense().tolist() == [0, 1, 0]


# noinspection PyTestParametrized,DuplicatedCode
@pytest.mark.parametrize("n_obs,n_vars,X_layer_names,X_value_gen", [(6, 3, ("raw",), pytorch_x_value_gen)])
def test_batching__all_batches_full_size(soma_experiment: Experiment) -> None:
    exp_data_pipe = ExperimentDataPipe(
        exp_uri=soma_experiment.uri, ms_name="RNA", layer_name="raw", obs_column_names=["label"], batch_size=3
    )
    batch_iter = iter(exp_data_pipe)

    batch = next(batch_iter)
    assert batch[0].tolist() == [[0, 0], [1, 1], [2, 2]]
    assert batch[1].to_dense().tolist() == [[0, 1, 0], [1, 0, 1], [0, 1, 0]]

    batch = next(batch_iter)
    assert batch[0].tolist() == [[3, 3], [4, 4], [5, 5]]
    assert batch[1].to_dense().tolist() == [[1, 0, 1], [0, 1, 0], [1, 0, 1]]

    with pytest.raises(StopIteration):
        next(batch_iter)


# noinspection PyTestParametrized
@pytest.mark.parametrize("n_obs,n_vars,X_layer_names,X_value_gen", [(5, 3, ("raw",), pytorch_x_value_gen)])
def test_batching__partial_final_batch_size(soma_experiment: Experiment) -> None:
    exp_data_pipe = ExperimentDataPipe(
        exp_uri=soma_experiment.uri, ms_name="RNA", layer_name="raw", obs_column_names=["label"], batch_size=3
    )
    batch_iter = iter(exp_data_pipe)

    next(batch_iter)
    batch = next(batch_iter)
    assert batch[1].to_dense().tolist() == [[1, 0, 1], [0, 1, 0]]

    with pytest.raises(StopIteration):
        next(batch_iter)


# noinspection PyTestParametrized,DuplicatedCode
@pytest.mark.parametrize("n_obs,n_vars,X_layer_names,X_value_gen", [(3, 3, ("raw",), pytorch_x_value_gen)])
def test_batching__exactly_one_batch(soma_experiment: Experiment) -> None:
    exp_data_pipe = ExperimentDataPipe(
        exp_uri=soma_experiment.uri, ms_name="RNA", layer_name="raw", obs_column_names=["label"], batch_size=3
    )
    batch_iter = iter(exp_data_pipe)

    batch = next(batch_iter)
    assert batch[0].tolist() == [[0, 0], [1, 1], [2, 2]]
    assert batch[1].to_dense().tolist() == [[0, 1, 0], [1, 0, 1], [0, 1, 0]]

    with pytest.raises(StopIteration):
        next(batch_iter)


# noinspection PyTestParametrized
@pytest.mark.parametrize("n_obs,n_vars,X_layer_names,X_value_gen", [(6, 3, ("raw",), pytorch_x_value_gen)])
def test_batching__empty_query_result(soma_experiment: Experiment) -> None:
    exp_data_pipe = ExperimentDataPipe(
        exp_uri=soma_experiment.uri,
        ms_name="RNA",
        layer_name="raw",
        obs_query=AxisQuery(coords=([],)),
        obs_column_names=["label"],
        batch_size=3,
    )
    batch_iter = iter(exp_data_pipe)

    with pytest.raises(StopIteration):
        next(batch_iter)


# noinspection PyTestParametrized
@pytest.mark.parametrize("n_obs,n_vars,X_layer_names,X_value_gen", [(6, 3, ("raw",), pytorch_x_value_gen)])
def test_dense_output(soma_experiment: Experiment) -> None:
    exp_data_pipe = ExperimentDataPipe(
        exp_uri=soma_experiment.uri,
        ms_name="RNA",
        layer_name="raw",
        obs_column_names=["label"],
        batch_size=3,
        dense_X=True,
    )
    batch_iter = iter(exp_data_pipe)

    batch = next(batch_iter)
    assert isinstance(batch[1], Tensor)
    assert batch[1].tolist() == [[0, 1, 0], [1, 0, 1], [0, 1, 0]]


# noinspection PyTestParametrized
@pytest.mark.parametrize("n_obs,n_vars,X_layer_names,X_value_gen", [(3, 3, ("raw",), pytorch_x_value_gen)])
def test_encoders(soma_experiment: Experiment) -> None:
    exp_data_pipe = ExperimentDataPipe(
        exp_uri=soma_experiment.uri,
        ms_name="RNA",
        layer_name="raw",
        obs_column_names=["label"],
        batch_size=3,
    )
    batch_iter = iter(exp_data_pipe)

    batch = next(batch_iter)
    assert isinstance(batch[0], Tensor)

    labels_encoded = batch[0][:, 1]
    labels_decoded = exp_data_pipe.obs_encoders()["label"].inverse_transform(labels_encoded)
    assert labels_decoded.tolist() == ["0", "1", "2"]
