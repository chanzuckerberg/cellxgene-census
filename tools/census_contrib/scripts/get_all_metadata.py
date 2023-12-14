import json
import sys

import tiledbsoma as soma


def main(fnames: list[str]) -> int:
    md = {}
    for fn in fnames:
        with soma.open(fn) as A:
            d = json.loads(A.metadata["CxG_embedding_info"])
            md[d["id"]] = d

    json.dump(md, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
