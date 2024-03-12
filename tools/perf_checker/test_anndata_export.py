import cellxgene_census
import tiledbsoma as soma


def main() -> None:
    with cellxgene_census.open_soma(census_version="stable") as census:
        with census["census_data"]["homo_sapiens"].axis_query(
            measurement_name="RNA",
            obs_query=soma.AxisQuery(value_filter="""tissue_general == 'eye'"""),
        ) as query:
            query.to_anndata(X_name="raw")


main()
