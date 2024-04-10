import cellxgene_census
import tiledbsoma


def main() -> None:
    with cellxgene_census.open_soma(census_version="stable") as census:
        human = census["census_data"]["homo_sapiens"]

        query = human.axis_query(
            measurement_name="RNA",
            obs_query=tiledbsoma.AxisQuery(
                coords=[
                    slice(0, 300000),
                ]
            ),
        )

        ### Reading X

        X = query.X("raw")

        # Load a memory-fixed slice into Arrow based on SOMA's buffer size
        # Read the full table into Arrow
        full_table = X.tables().concat()

        # Add a blockwise iterator example

        ### Reading obs

        obs = query.obs()

        # Read the full table into Arrow
        full_table = obs.concat()

        ## Anndata
        query.to_anndata(X_name="raw")

        ## Indexer

        # Add indexer examples


main()
