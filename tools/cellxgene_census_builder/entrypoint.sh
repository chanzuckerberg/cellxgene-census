#!/bin/bash

python3 -m cellxgene_census_builder .
BUILDER_STATUS=$?
dmesg
exit $BUILDER_STATUS
