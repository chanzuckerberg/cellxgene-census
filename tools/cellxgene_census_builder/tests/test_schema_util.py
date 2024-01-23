import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from cellxgene_census_builder.build_soma.schema_util import FieldSpec, TableSpec


def test_create_spec() -> None:
    fields = [
        ("soma_joinid", pa.int64()),
        ("a", pa.string()),
        ("b", pa.large_string()),
        ("c", pa.float32()),
        FieldSpec(name="d", type=pa.int32()),
        ("e", pa.bool_()),
    ]
    ts = TableSpec.create(fields)

    assert ts is not None
    assert isinstance(ts, TableSpec)
    assert not ts.use_arrow_dictionaries  # default is False
    assert len(ts.fields) == len(fields)
    assert all(isinstance(f, FieldSpec) for f in ts.fields)
    assert all((a.type == (b[1] if isinstance(b, tuple) else b.type)) for a, b in zip(ts.fields, fields))
    assert list(ts.field_names()) == [f[0] if isinstance(f, tuple) else f.name for f in fields]
    assert ts.field("soma_joinid").name == "soma_joinid"
    assert ts.field("d").name == "d"
    with pytest.raises(ValueError):
        ts.field("no-such-key")

    ts = TableSpec.create(fields, use_arrow_dictionary=True)
    assert ts.use_arrow_dictionaries

    ts = TableSpec.create(fields, use_arrow_dictionary=False)
    assert not ts.use_arrow_dictionaries


def test_dict_feature_flag() -> None:
    fields = [
        FieldSpec(name="a", type=pa.string(), is_dictionary=True),
        FieldSpec(name="b", type=pa.large_string(), is_dictionary=True),
        FieldSpec(name="c", type=pa.float32(), is_dictionary=True),
        FieldSpec(name="d", type=pa.int32(), is_dictionary=True),
        FieldSpec(name="e", type=pa.bool_(), is_dictionary=True),
    ]
    schema = TableSpec.create(fields, use_arrow_dictionary=False).to_arrow_schema()
    assert all(not pa.types.is_dictionary(schema.field(fname).type) for fname in schema.names)

    schema = TableSpec.create(fields, use_arrow_dictionary=True).to_arrow_schema()
    assert all(pa.types.is_dictionary(schema.field(fname).type) for fname in schema.names)


def test_fieldspec_to_pandas_dtype() -> None:
    assert FieldSpec(name="test", type=pa.int32()).to_pandas_dtype() == np.int32
    assert FieldSpec(name="test", type=pa.string()).to_pandas_dtype() == np.object_
    with pytest.raises(TypeError):
        FieldSpec(name="test", type=pa.string(), is_dictionary=True).to_pandas_dtype()
    assert (
        FieldSpec(name="test", type=pa.bool_(), is_dictionary=True).to_pandas_dtype(ignore_dict_type=True) == np.bool_
    )


def test_fieldspec_is_type_equivalent() -> None:
    # primitives
    assert FieldSpec(name="test", type=pa.int32()).is_type_equivalent(pa.int32())
    assert not FieldSpec(name="test", type=pa.int8()).is_type_equivalent(pa.int32())

    # non-primitive synonyms
    assert FieldSpec(name="test", type=pa.string()).is_type_equivalent(pa.string())
    assert FieldSpec(name="test", type=pa.string()).is_type_equivalent(pa.large_string())
    assert FieldSpec(name="test", type=pa.large_string()).is_type_equivalent(pa.string())
    assert FieldSpec(name="test", type=pa.large_string()).is_type_equivalent(pa.large_string())
    assert FieldSpec(name="test", type=pa.binary()).is_type_equivalent(pa.binary())
    assert FieldSpec(name="test", type=pa.binary()).is_type_equivalent(pa.large_binary())
    assert FieldSpec(name="test", type=pa.large_binary()).is_type_equivalent(pa.binary())
    assert FieldSpec(name="test", type=pa.large_binary()).is_type_equivalent(pa.large_binary())

    # dictionary
    assert FieldSpec(name="test", type=pa.string(), is_dictionary=True).is_type_equivalent(
        pa.dictionary(value_type=pa.string(), index_type=pa.int8())
    )
    assert not FieldSpec(name="test", type=pa.string(), is_dictionary=True).is_type_equivalent(
        pa.dictionary(value_type=pa.int32(), index_type=pa.int8())
    )
    assert not FieldSpec(name="test", type=pa.string()).is_type_equivalent(
        pa.dictionary(value_type=pa.string(), index_type=pa.int8())
    )
    assert not FieldSpec(name="test", type=pa.string(), is_dictionary=True).is_type_equivalent(pa.string())

    # allow null/non-primitive equivalance
    assert FieldSpec(name="test", type=pa.string()).is_type_equivalent(pa.null(), null_non_primitive_equivalence=True)
    assert not FieldSpec(name="test", type=pa.string()).is_type_equivalent(
        pa.null(), null_non_primitive_equivalence=False
    )
    assert not FieldSpec(name="test", type=pa.int8()).is_type_equivalent(pa.null(), null_non_primitive_equivalence=True)
    assert not FieldSpec(name="test", type=pa.int8()).is_type_equivalent(
        pa.null(), null_non_primitive_equivalence=False
    )

    # internal API
    with pytest.raises(TypeError):
        FieldSpec(name="test", type=pa.int8())._check_type_compat(pa.int32(), False)
    with pytest.raises(TypeError):
        FieldSpec(name="test", type=pa.string())._check_type_compat(pa.null(), False)
    # doesn't raise
    FieldSpec(name="test", type=pa.string())._check_type_compat(pa.null(), True)


def test_tablespec_recategoricalize() -> None:
    df_nocat = pd.DataFrame({"a": [0, 1, 2], "b": ["a", "b", "c"], "c": [True, False, True]})
    df_cat = pd.DataFrame(
        {"a": df_nocat.a.astype("category"), "b": df_nocat.b.astype("category"), "c": df_nocat.c.astype("category")}
    )

    # no dicts
    ts = TableSpec.create(
        [
            FieldSpec(name="a", type=pa.int32(), is_dictionary=False),
            FieldSpec(name="b", type=pa.string(), is_dictionary=False),
            FieldSpec(name="c", type=pa.bool_(), is_dictionary=False),
        ],
        use_arrow_dictionary=False,
    )
    assert df_nocat.equals(ts.recategoricalize(df_nocat))
    assert df_nocat.equals(ts.recategoricalize(df_cat))

    # all dicts
    ts = TableSpec.create(
        [
            FieldSpec(name="a", type=pa.int32(), is_dictionary=True),
            FieldSpec(name="b", type=pa.string(), is_dictionary=True),
            FieldSpec(name="c", type=pa.bool_(), is_dictionary=True),
        ],
        use_arrow_dictionary=True,
    )
    assert df_cat.equals(ts.recategoricalize(df_nocat))
    assert df_cat.equals(ts.recategoricalize(df_cat))
