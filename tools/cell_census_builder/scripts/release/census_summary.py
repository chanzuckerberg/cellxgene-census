import sys
import cell_census
import pandas as pd

from cell_census_builder.globals import CENSUS_DATA_NAME

if __name__ == '__main__':
    census_version = sys.argv[1] if len(sys.argv) > 1 else "latest"

    census = cell_census.open_soma(census_version=census_version)

    COLS_TO_QUERY = [
        ("soma_joinid", "cells"),
        ("dataset_id", "datasets"),
        ("cell_type_ontology_term_id", "cell types"),
        ("tissue_ontology_term_id", "tissues"),
        ("assay_ontology_term_id", "assays"),
    ]

    obs_df = {
        name: experiment.obs.read(column_names=[c[0] for c in COLS_TO_QUERY]).concat().to_pandas()
        for name, experiment in census[CENSUS_DATA_NAME].items()
    }

    # Use Pandas to summarize and display
    stats = [(organism, col[1], df[col[0]].nunique()) for organism, df in obs_df.items() for col in COLS_TO_QUERY]
    print(census['census_info']['summary'].read().concat().to_pandas()[['label', 'value']].to_string(index=False))
    stats_df = pd.DataFrame(stats, columns=['organism', 'attribute', 'unique count'])
    display_stats_df = pd.pivot(stats_df, index=['organism'], columns=['attribute'], values=['unique count'])
    print(display_stats_df)
