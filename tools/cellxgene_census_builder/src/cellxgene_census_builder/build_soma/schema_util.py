from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, TypeVar, Union, cast

import attrs
import numpy.typing as npt
import pandas as pd
import pyarrow as pa

"""
Specification of DataFrame/Table schema. Uses motivating this (over a simple pyarrow.Schema):

1. allow for late-binding to various types, e.g., Dictionary index_type (set to smallest width,
based upon number of labels).

2. allow explicit type choices where there are options (e.g., string vs large_string)

"""

OptDataFrame = TypeVar("OptDataFrame", pd.DataFrame, None, Optional[pd.DataFrame])


@attrs.define(frozen=True, kw_only=True, slots=True)
class FieldSpec:
    """Define the schema spec for a single field (aka column). Fields with is_dictionary==True are candidates
    for encoding as a Arrow dictionary type.
    """

    name: str  # field name
    type: pa.DataType  # field type, or the value_type if `is_dictionary` is True
    is_dictionary: bool = False  # dictionary or primitive field type

    def to_pandas_dtype(self, *, ignore_dict_type: bool = False) -> npt.DTypeLike:
        """
        Return the Pandas dtype for this field.

        This is only possible if the field is not a dictionary-type field.

        The caller may request the dictionary value type, effectively ignoring the is_dictionary
        attribute, by setting `ignore_dict_type` to True.
        """
        if not ignore_dict_type and self.is_dictionary and not pa.types.is_dictionary(self.type):
            raise TypeError("Unable to determine final Pandas type for dictionary.")
        return cast(npt.DTypeLike, self.type.to_pandas_dtype())

    def is_type_equivalent(self, other_type: pa.DataType, *, null_non_primitive_equivalence: bool = False) -> bool:
        """
        Return True if this FieldSpec is equivalent to the Arrow `other_type`.
        For convenience in comparing with types inferred from Pandas DataFrames,
        where strings and other Arrow non-primitives are stored as objects, allow a
        pa.null DataType to be equivalent to Arrow non-primitive.
        """
        if self.is_dictionary:
            if pa.types.is_dictionary(other_type):
                return FieldSpec(name=self.name, type=self.type, is_dictionary=False).is_type_equivalent(
                    other_type.value_type, null_non_primitive_equivalence=null_non_primitive_equivalence
                )
            else:
                return False

        if self.type == other_type:
            return True

        # Non-primitives (e.g., Pandas 'object' types) become pa.null in an empty dataframe
        # as the type can't be inferred from the data.
        if null_non_primitive_equivalence and not pa.types.is_primitive(self.type) and pa.types.is_null(other_type):
            return True

        # Treat large/small strings and binary/bytes as equivalent. TileDB promotes foo->large_foo
        def is_string(typ: pa.DataType) -> bool:
            return cast(bool, pa.types.is_large_string(typ)) or cast(bool, pa.types.is_string(typ))

        def is_binary(typ: pa.DataType) -> bool:
            return cast(bool, pa.types.is_large_binary(typ)) or cast(bool, pa.types.is_binary(typ))

        if is_string(self.type) and is_string(other_type):
            return True

        if is_binary(self.type) and is_binary(other_type):
            return True

        return False

    def _check_type_compat(self, other_type: pa.DataType, empty_dataframe: bool) -> None:
        """Internal - peforms the type check and raises on non-equivalent types (False)."""
        if not self.is_type_equivalent(other_type, null_non_primitive_equivalence=empty_dataframe):
            raise TypeError(
                f"Incompatible or unsupported type for field {self.name}: expected {repr(self.type)}, got {repr(other_type)}"
            )


@attrs.define(frozen=True, kw_only=True, slots=True)
class TableSpec:
    """
    List of FieldSpec defining an Arrow Table, with a table-wide feature flag enabling/disabling
    use of Dictionary types.

    Instantiate ONLY with the class method `create`.
    """

    fields: List[FieldSpec]
    use_arrow_dictionaries: bool  # Feature flag to enable/disable dictionary/enum support

    @classmethod
    def create(
        cls, fields: Sequence[Union[FieldSpec, Tuple[str, pa.DataType]]], *, use_arrow_dictionary: bool = False
    ) -> TableSpec:
        u = []
        for f in fields:
            if isinstance(f, FieldSpec):
                if not use_arrow_dictionary and f.is_dictionary:
                    f = attrs.evolve(f, is_dictionary=False)
                u.append(f)
            else:
                name, type = f
                u.append(FieldSpec(name=name, type=type, is_dictionary=False))

        # quick unique check
        if len(set(f.name for f in u)) != len(fields):
            raise ValueError("All field names must be unique.")

        return TableSpec(fields=u, use_arrow_dictionaries=use_arrow_dictionary)

    def to_arrow_schema(self, df: Optional[pd.DataFrame] = None) -> pa.Schema:
        """
        Returns Arrow schema for a Table.

        Use the specified types, but check for equivalence. Where the field spec is a
        dictionary, create the narrowest possible dictionary index_type sufficient for
        the dataframe.
        """
        pa_fields = []
        for field_spec in self.fields:
            if df is not None:
                # Verify data we have is data we expect! This is not part of schema building, but
                # helps catch accidental data casts/transformations
                assert field_spec.name in df
                pa_type = pa.Schema.from_pandas(df[[field_spec.name]], preserve_index=False).field(field_spec.name).type
                field_spec._check_type_compat(pa_type, df.empty)  # raises on mismatch

            if field_spec.is_dictionary:
                if df is not None:
                    # Assume that Pandas will pick smallest index type
                    index_type = pa.from_numpy_dtype(df[field_spec.name].cat.codes.dtype)
                else:
                    index_type = pa.int8()  # smallest, as a default
                pa_fields.append((field_spec.name, pa.dictionary(index_type, field_spec.type, ordered=0)))
            else:
                pa_fields.append((field_spec.name, field_spec.type))

        return pa.schema(pa_fields)

    def field_names(self) -> Sequence[str]:
        """Return field names for this TableSpec as a sequence of string"""
        return list(field.name for field in self.fields)

    def field(self, key: str) -> FieldSpec:
        """Return the named field, or raise ValueError if no such key."""
        r = [f for f in self.fields if f.name == key]
        if not r:
            raise ValueError("No such item")
        assert len(r) == 1
        return r[0]

    def recategoricalize(self, df: OptDataFrame) -> OptDataFrame:
        """Apply/unapply categorical typing to match table schema spec"""
        if df is None or df.empty:
            return df

        df = df.copy()
        for fld in self.fields:
            is_categorical = isinstance(df[fld.name].dtype, pd.CategoricalDtype)
            if fld.is_dictionary and not is_categorical:
                df[fld.name] = df[fld.name].astype("category")
            elif not fld.is_dictionary and is_categorical:
                df[fld.name] = df[fld.name].astype(df[fld.name].cat.categories.dtype)

        return df


def decategoricalize(df: pd.DataFrame, *, inplace: bool = False) -> pd.DataFrame:
    if not inplace:
        df = df.copy()
    for key in df:
        if isinstance(df[key].dtype, pd.CategoricalDtype):
            df[key] = df[key].astype(df[key].cat.categories.dtype)
    return df
