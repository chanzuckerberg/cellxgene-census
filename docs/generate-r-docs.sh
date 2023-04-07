#!/bin/bash

cd ../api/r/cellxgene.census
# Rscript -e 'install.packages("pkgdown")'
# Rscript -e 'install.packages("usethis")'
# Rscript -e 'usethis::use_pkgdown()'
Rscript -e 'pkgdown::build_site()'

mkdir -p ../../../docs/r/
cp -r docs/* ../../../docs/.