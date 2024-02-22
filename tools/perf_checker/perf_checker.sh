.sh
#!/bin/sh
set -euox pipefail



dbpath="census-profiler-tests"

pip install psutil
pip install gitpython
pip install somacore
pip install tiledbsoma
pip install cellxgene_census

# Download the repo including the profiler
cd ../
git clone https://github.com/single-cell-data/TileDB-SOMA.git
# Downloading TileDB-SOMA (remove the next line once the branch is merged)
cd TileDB-SOMA/profiler
git checkout census_profiler
pip install .
cd ../../cellxgene-census/

# New benchmarks must be added to this list
declare -a benchmarks=("./tools/perf_checker/test_anndata_export.py")

# Running all benchmarks and checking performance changes
for benchmark in ${benchmarks}
do
  python -m profiler "python ${benchmark}" $dbpath
  python ./tools/perf_checker/perf_checker.py "python ${benchmark}" $dbpath
done