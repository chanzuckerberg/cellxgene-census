.sh
#!/bin/sh
set -euox pipefail

python -m venv perf
source perf/bin/activate
pip install psutil
pip install gitpython
pip install somacore
pip install tiledbsoma
pip install cellxgene_census


sudo wget https://s3.amazonaws.com/mountpoint-s3-release/latest/x86_64/mount-s3.deb
sudo apt install -y ./mount-s3.deb
mkdir ./mount-s3
mkdir ./s3_cache
mount-s3 census-profiler-tests ./mount-s3 --cache ./s3_cache  --metadata-ttl 300
dbpath=`pwd`/mount-s3
echo "s3 mount path = ${dbpath}"
# new benchmarks must be added to this list
declare -a benchmarks=("./tools/perf_checker/benchmark1.py")

git clone https://github.com/single-cell-data/TileDB-SOMA.git
cd TileDB-SOMA
git checkout census_profiler
cd ../


arraylength=${#benchmarks[@]}
for (( i=0; i<${arraylength}; i++ ))
do
  python  ./TileDB-SOMA/profiler "python ${benchmarks[$i]}" $dbpath -t time
  python ./TileDB-SOMA/profile_report.py "python ${benchmarks[$i]}" $dbpath
done
