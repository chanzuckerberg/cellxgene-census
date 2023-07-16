#!/bin/bash

set -euxo pipefail

cd "$(dirname $0)/.."

# install dependencies
Rscript -e 'install.packages("rmarkdown", repos="https://cloud.r-project.org")'
Rscript -e 'remotes::install_deps(".", dependencies=TRUE)'
R CMD install -d .

# build each vignette Rmd to an HTML file alongside
for Rmd in $(find vignettes_ -name '*.Rmd'); do
  Rscript -e "rmarkdown::render('${Rmd}', 'html_document')"
done

echo "** build_vignettes.sh SUCCESS"
