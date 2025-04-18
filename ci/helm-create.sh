#!/bin/sh
set -euox pipefail

kubectl config use-context dev-sc

export INSTALLATION_NAME="sc-dev-64g-runner"
export NAMESPACE_CONTROLLER="sc-dev-arc-systems"
export NAMESPACE_RUNNERS="sc-dev-arc-runners"
export YAML_FILE="values.yaml"

kubectl create namespace ${NAMESPACE_CONTROLLER} || true
kubectl create namespace ${NAMESPACE_RUNNERS} || true

helm install arc \
    --namespace "${NAMESPACE_CONTROLLER}" \
    --values "${YAML_FILE}" \
    oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set-controller

helm install "${INSTALLATION_NAME}" \
    --namespace "${NAMESPACE_RUNNERS}" \
    --values "${YAML_FILE}" \
    oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set 
