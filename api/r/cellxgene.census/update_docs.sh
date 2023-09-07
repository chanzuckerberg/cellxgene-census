#!/bin/bash
# SEE vignettes_/README.md

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

# if succesfull rendering let's change the source link to 
# vignettes to the real location "vignettes_"
if [ $? -eq 0 ]; then
    for file in ./docs/articles/*html; do
        echo "Fixing source to '_vignettes' $file"
        sed -E "s/(.*Source:.*)\/vignettes(.*)/\1\/vignettes_\2/g" $file > temp.html && mv temp.html $file
    done
fi

echo "** pkgdown::build_site() succeeded. Check in the updated docs/ to git: git rm -r --cached docs/ && git add docs/"
