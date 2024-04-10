import cellxgene_census


def main() -> None:
    with cellxgene_census.open_soma(census_version="stable") as census:
        # create pointer
        obs = census["census_data"]["homo_sapiens"].obs.read(
            coords=[
                slice(0, 200000000),
            ]
        )

        # Load a memory-fixed slice into Arrow based on SOMA's buffer size
        next(obs)
        # Read the full table into Arrow
        full_table = obs.concat()

        # To pandas
        full_table.to_pandas()


main()
