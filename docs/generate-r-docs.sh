#!/bin/bash

cd ../api/r/cellxgene.census
Rscript -e 'pkgdown::build_site()'

mkdir -p ../../../docs/r/
cp -r docs/* ../../../docs/.