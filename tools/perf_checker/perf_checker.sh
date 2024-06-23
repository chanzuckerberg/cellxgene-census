.sh
#!/bin/sh
set -euox pipefail

dbpath="census-profiler-tests-trial"
sudo apt install time

python3.11 -m venv ~/venv
. ~/venv/bin/activate

pip install psutil; pip install gitpython; pip install somacore; pip install tiledbsoma; pip install cellxgene_census; pip install boto3

# Install the profiler
pip install 'git+https://github.com/single-cell-data/TileDB-SOMA.git#subdirectory=profiler'

# New benchmarks must be added to this list
declare -a benchmarks=("./tools/perf_checker/test_anndata_export.py")

# Running all benchmarks and checking performance changes
for benchmark in ${benchmarks}
do
  python -m profiler "python ${benchmark}" $dbpath
  python ./tools/perf_checker/perf_checker.py "python ${benchmark}" $dbpath
done