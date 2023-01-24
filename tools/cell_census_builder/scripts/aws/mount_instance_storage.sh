#!/usr/bin/env bash

# This automates mounting either all of the instance (ephemeral) storage 
# devices or the specified devices onto a file system.  If a single device
# is found, it creates an ext4 file system. If multiple devices are found or specified, 
# it creates a RAID0 group, and an ext4 file system on top of it.
# 
# Usage:
# mount_instance_storage.sh [DEVICE]...
# Example:
# mount_instance_storage.sh /dev/nvme{4,5}n1
#
# https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/add-instance-store-volumes.html
# https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/raid-config.html
#

DEVICES="$@"

# exit immediately when a command fails
set -e
# treat unset variables as an error and exit immediately
set -u
# echo each line of the script to stdout so we can see what is happening
# to turn off echo do 'set +o xtrace'
set -o xtrace


DEVICE_PREFIX="nvme"
MOUNTPOINT="/mnt/scratch"
RAID_VOLUME="/dev/md0"
VOLUME_LABEL="scratch_volume"


# Must be run as privileged user
if [[ $(id -u) != 0 ]]; then
  echo "ERROR: not root. You must run using sudo. Exiting with no action taken."
  exit
fi

# Test for a conflict on the mount point
if grep -qs ' ${MOUNTPOINT} ' /proc/mounts; then
  echo "ERROR: ${MOUNTPOINT} already in use. Exiting with no action taken."
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

function create_volume {
  devices_count=$(wc -w <<< $@)
  if [[ ${devices_count} == 0 ]]; then
    echo "No devices found, no volume created."
    exit 1
  elif [[ ${devices_count} == 1 ]]; then
    echo "Found single device, creating volume"
    mkfs.ext4 -L ${VOLUME_LABEL} $@
  else
    echo "Found ${devices_count} devices, creating RAID0 volume"
    mdadm --create --verbose ${RAID_VOLUME} --level=0 --name=${VOLUME_LABEL} --raid-devices=${devices_count} $@
    mkfs.ext4 -L ${VOLUME_LABEL} ${RAID_VOLUME}
  fi
}

function mount_volume {
  mkdir -p ${MOUNTPOINT}
  mount LABEL=${VOLUME_LABEL} ${MOUNTPOINT}
  chmod 777 ${MOUNTPOINT}
}

create_volume ${DEVICES:-$(detect_devices ${DEVICE_PREFIX})}
mount_volume
echo "Done. Mounted on ${MOUNTPOINT}."
