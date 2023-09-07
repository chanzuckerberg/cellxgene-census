#!/bin/bash

python3 -m cellxgene_census_builder .
BUILDER_STATUS=$?
# On error, log dmesg tail to aid troubleshooting
# Note: requires docker --privileged option
if [[ $BUILDER_STATUS -ne 0 ]]; then dmesg | tail -n 128; fi
exit $BUILDER_STATUS
