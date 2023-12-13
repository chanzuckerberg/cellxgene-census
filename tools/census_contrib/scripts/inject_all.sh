#!/usr/bin/bash -x

python -m census_contrib inject -v --cwd CxG-czi-1 --census-write-path 2023-12-15/soma/ --census-uri 2023-12-15/soma/ --obsm-key geneformer
python -m census_contrib inject -v --cwd CxG-czi-2 --census-write-path 2023-12-15/soma/ --census-uri 2023-12-15/soma/ --obsm-key scvi
python -m census_contrib inject -v --cwd CxG-czi-3 --census-write-path 2023-12-15/soma/ --census-uri 2023-12-15/soma/ --obsm-key scvi
