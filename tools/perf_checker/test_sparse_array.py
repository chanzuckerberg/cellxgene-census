import cellxgene_census


def main() -> None:
    print("Got X")
    with cellxgene_census.open_soma(census_version="stable") as census:
        # create pointer
        X = (
            census["census_data"]["homo_sapiens"]
            .ms["RNA"]
            .X["raw"]
            .read(
                coords=[
                    slice(0, 500000),
                ]
            )
        )
        # Load a memory-fixed slice into Arrow based on SOMA's buffer size
        # next(X)

        # Read the full table into Arrow
        X.tables().concat()
        # full_table = X.concat()
        # X.to_pandas()


main()
