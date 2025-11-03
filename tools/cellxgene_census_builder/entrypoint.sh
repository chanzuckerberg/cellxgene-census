#!/bin/bash

# These configure thread allocation for BLAS, OpenMP, etc. We don't use these
# packages, but they will allocate a large number of threads at startup (usually
# one thread per host CPU). Turn down the allocation which gets excessive in
# high-CPU hosts, such as those used to build the Census.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Optimize AWS CLI for large uploads
AWS_CONFIG_DIR="${AWS_CONFIG_DIR:-$HOME/.aws}"
AWS_CONFIG_FILE="$AWS_CONFIG_DIR/config"
if [ ! -f "$AWS_CONFIG_FILE" ]; then
	mkdir -p "$AWS_CONFIG_DIR"
	cat > "$AWS_CONFIG_FILE" <<'EOF'
[default]
s3 =
		max_concurrent_requests = 64
		max_queue_size = 10000
		multipart_threshold = 64MB
		multipart_chunksize = 64MB
EOF
	chmod 600 "$AWS_CONFIG_FILE" || true
fi

# Log tiledbsoma package versions
python3 -c 'import tiledbsoma; tiledbsoma.show_package_versions()'

# Log pip freeze 
echo "---- pip freeze ----"
pip freeze

# Log system config
echo "---- sysctl ----"
sysctl vm.max_map_count
sysctl kernel.pid_max
sysctl kernel.threads-max

echo "----"
python3 -m cellxgene_census_builder .
BUILDER_STATUS=$?
# On error, log dmesg tail to aid troubleshooting
# Note: requires docker --privileged option
if [[ $BUILDER_STATUS -ne 0 ]]; then dmesg | tail -n 128; fi
if [[ $BUILDER_STATUS -ne 0 ]]; then df; fi
exit $BUILDER_STATUS
