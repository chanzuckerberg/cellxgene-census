#!/usr/bin/env bash

# This automates adding all instance (ephemeral) storage as swap
#
# https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instance-store-swap-volumes.html

# exit immediately when a command fails
set -e
# treat unset variables as an error and exit immediately
set -u
# echo each line of the script to stdout so we can see what is happening
# to turn off echo do 'set +o xtrace'
set -o xtrace

DEVICE_PREFIX="nvme"

# Must be run as privileged user
if [[ $(id -u) != 0 ]]; then
  echo "ERROR: not root. You must run using sudo. Exiting with no action taken."
  exit
fi

# Detect all block devices that are disks, and do not have
# partitions or other holder devices (eg, part of raid group, etc) 
function detect_devices {
  PY_CMD='
import sys, json
device_prefix = sys.argv[1]
bdevs = [
  dev for dev in json.load(sys.stdin)["blockdevices"] 
  if dev["type"] == "disk" and "children" not in dev and dev["name"].startswith(device_prefix)
]
for d in bdevs:
  name = d["name"]
  print(f"/dev/{name}")
'
  lsblk --json --output NAME,TYPE,MOUNTPOINT | python3 -c "${PY_CMD}" "$1"
}

for bdev in $(detect_devices ${DEVICE_PREFIX}); do
  echo "Adding ${bdev}"
  mkswap ${bdev}
  swapon -v ${bdev}
done

echo "Done, swapping on devices:"
swapon -s
