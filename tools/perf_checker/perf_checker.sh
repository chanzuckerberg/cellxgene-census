.sh
#!/bin/sh
set -euox pipefail

# Installing the requirements
python -m venv perf
source perf/bin/activate
pip install psutil
pip install gitpython
pip install somacore
pip install tiledbsoma
pip install cellxgene_census

# Installing mount-s3
sudo wget https://s3.amazonaws.com/mountpoint-s3-release/latest/x86_64/mount-s3.deb
sudo apt install -y ./mount-s3.deb

# Setting up mount-s3. We use S3 file system as it is necessary to persist the
# profiling run data that are performed below
mkdir ./census-profiler-tests
mkdir ./s3_cache
mount-s3 census-profiler-tests ./mount-s3 --cache ./s3_cache  --metadata-ttl 300
dbpath=`pwd`/census-profiler-tests

# New benchmarks must be added to this list
declare -a benchmarks=("./tools/perf_checker/benchmark1.py")

# Download the repo including the profiler
git clone https://github.com/single-cell-data/TileDB-SOMA.git
# Downloading TileDB-SOMA (remove the next line once the branch is merged)
cd TileDB-SOMA
git checkout census_profiler
cd ../

# Download gnu time tool
sudo apt-get update -y
sudo apt-get install -y time

# Running all benchmarks and checking performance changes
for benchmark in ${benchmarks}
do
  echo "Start profiling"
  python ./TileDB-SOMA/profiler/profiler.py "python ${benchmark}" $dbpath -t gtime
  echo "Done profiling"
  python ./TileDB-SOMA/profile_report.py "python ${benchmark}" $dbpath
done
