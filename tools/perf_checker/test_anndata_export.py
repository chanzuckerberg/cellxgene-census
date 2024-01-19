from sys import stderr

import cellxgene_census
import tiledbsoma as soma

print("Starting bm 1", file=stderr)
census_S3_latest = dict(census_version="2024-01-01")


def main() -> None:
    with cellxgene_census.open_soma(**census_S3_latest) as census:
        with census["census_data"]["homo_sapiens"].axis_query(
            measurement_name="RNA",
            obs_query=soma.AxisQuery(value_filter="""tissue_general == 'eye'"""),
        ) as query:
            query.to_anndata(X_name="raw")


main()
