#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

NAMESPACE=tianhaowang-ucsd
JOB=fig4b-one-step-algorithm-specific-0721
CONFIGMAP=fig4b-algorithm-specific-code-0721
HELPER=pvc-copy-helper
REMOTE_DIR=/mnt/results/fig4_wang2509/7_21/nautilus
LOCAL_FIG_DIR=experiments/fig4_wang2509/figures/7_21/nautilus/stage1_one_step_eta
LOCAL_RESULT_DIR=experiments/fig4_wang2509/results/7_21/nautilus

mkdir -p "${LOCAL_FIG_DIR}" "${LOCAL_RESULT_DIR}"

kubectl -n "${NAMESPACE}" create configmap "${CONFIGMAP}" \
  --from-file=run_one_step.py=experiments/fig4_wang2509/run_one_step.py \
  --from-file=plot_fig4.py=experiments/fig4_wang2509/plot_fig4.py \
  --from-file=plot_fig4b_high_precision.py=experiments/fig4_wang2509/plot_fig4b_high_precision.py \
  --dry-run=client -o yaml | kubectl apply -f -

JOB_COMPLETE="$(kubectl -n "${NAMESPACE}" get job "${JOB}" -o jsonpath='{.status.succeeded}' 2>/dev/null || true)"
if [[ "${JOB_COMPLETE}" != "1" ]]; then
  kubectl -n "${NAMESPACE}" delete job "${JOB}" --ignore-not-found=true
  kubectl apply -f experiments/fig4_wang2509/k8s/fig4b-eta-1e-1-1e7-0721.yaml
fi
kubectl -n "${NAMESPACE}" wait --for=condition=complete "job/${JOB}" --timeout=45m

kubectl -n "${NAMESPACE}" delete pod "${HELPER}" --ignore-not-found=true
kubectl apply -f experiments/fig4_wang2509/k8s/pvc-copy-helper.yaml
kubectl -n "${NAMESPACE}" wait --for=condition=Ready "pod/${HELPER}" --timeout=3m

kubectl -n "${NAMESPACE}" cp \
  "${HELPER}:${REMOTE_DIR}/fig4b_one_step_algorithm_specific_eta_nautilus.csv" \
  "${LOCAL_RESULT_DIR}/fig4b_one_step_algorithm_specific_eta_nautilus.csv"

for figure in \
  fig4b_one_step_algorithm_specific_eta_normal_nautilus.png \
  fig4b_one_step_algorithm_specific_eta_precision1e-16_nautilus.png
do
  kubectl -n "${NAMESPACE}" cp \
    "${HELPER}:${REMOTE_DIR}/${figure}" \
    "${LOCAL_FIG_DIR}/${figure}"
done

kubectl -n "${NAMESPACE}" logs "job/${JOB}"
kubectl -n "${NAMESPACE}" delete pod "${HELPER}" --wait=false
