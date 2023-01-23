# General unit tests for cell_census_builder. Intention is to add more fine-grained tests for builder.
import os
from tempfile import mkstemp, TemporaryDirectory

import pandas as pd
import pyarrow as pa
import tiledbsoma as soma


def test_unicode_support() -> None:
    """
    Regression test that unicode is supported correctly in tiledbsoma.
    This test is not strictly necessary, but it validates the requirements that Cell Census
    support unicode in DataFrame columns.
    """
    with TemporaryDirectory() as d:
        pd_df = pd.DataFrame(data={'value': ["Ünicode", "S̈upport"]}, columns=['value'])
        pd_df['soma_joinid'] = pd_df.index
        s_df = soma.DataFrame(uri=os.path.join(d, "unicode_support")).\
            create(pa.Schema.from_pandas(pd_df, preserve_index=False), index_column_names=['soma_joinid'])
        s_df.write(pa.Table.from_pandas(pd_df, preserve_index=False))

        pd_df_in = soma.DataFrame(uri=os.path.join(d, "unicode_support")).read().concat().to_pandas()

        assert pd_df_in['value'].to_list() == ["Ünicode", "S̈upport"]


