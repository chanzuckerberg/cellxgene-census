#!/usr/bin/bash -x

python -m census_contrib inject -v --cwd CxG-czi-1 --census-path 2023-10-23/soma/ --obsm-key geneformer
python -m census_contrib inject -v --cwd CxG-czi-2 --census-path 2023-10-23/soma/ --obsm-key scvi
python -m census_contrib inject -v --cwd CxG-czi-3 --census-path 2023-10-23/soma/ --obsm-key scvi
