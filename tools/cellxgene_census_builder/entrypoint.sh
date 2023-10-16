#!/bin/bash

# Log tiledbsoma package versions
python3 -c 'import tiledbsoma; print(tiledbsoma.show_package_versions())'

# Log pip freeze 
pip freeze

python3 -m cellxgene_census_builder .
BUILDER_STATUS=$?
# On error, log dmesg tail to aid troubleshooting
# Note: requires docker --privileged option
if [[ $BUILDER_STATUS -ne 0 ]]; then dmesg | tail -n 128; fi
if [[ $BUILDER_STATUS -ne 0 ]]; then df; fi
exit $BUILDER_STATUS
