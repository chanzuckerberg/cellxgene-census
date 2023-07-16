#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

if [ -e vignettes ]; then
    >&2 echo "Error: 'vignettes' already exists in the current directory; clean it up before running this script (see README.md for explanation)"
    exit 1
fi

cleanup() {
    unlink vignettes
}
trap cleanup EXIT

ln -s vignettes_ vignettes

Rscript -e 'pkgdown::build_site()'

echo "** pkgdown::build_site() succeeded. Check in the updated docs/ to git: git rm -r --cached docs/ && git add docs/"
