#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

NAMESPACE=tianhaowang-ucsd
JOB=multistep-eta-wide-k300-0721
CONFIGMAP=multistep-eta-wide-k300-code-0721
HELPER=pvc-copy-helper
REMOTE_DIR=/mnt/results/fig4_wang2509/7_21/nautilus/multistep_eta_sweep_K300_wide_300etas
LOCAL_FIG_DIR=experiments/fig4_wang2509/figures/7_21/nautilus/multistep_eta_sweep_K300_wide_300etas
LOCAL_RESULT_DIR=experiments/fig4_wang2509/results/7_21/nautilus/multistep_eta_sweep_K300_wide_300etas

mkdir -p "${LOCAL_FIG_DIR}" "${LOCAL_RESULT_DIR}"

kubectl -n "${NAMESPACE}" create configmap "${CONFIGMAP}" \
  --from-file=run_multistep_eta_sweep.py=experiments/fig4_wang2509/run_multistep_eta_sweep.py \
  --from-file=plot_multistep_eta_sweep.py=experiments/fig4_wang2509/plot_multistep_eta_sweep.py \
  --dry-run=client -o yaml | kubectl apply -f -

JOB_EXISTS="$(kubectl -n "${NAMESPACE}" get job "${JOB}" -o name 2>/dev/null || true)"
if [[ -z "${JOB_EXISTS}" ]]; then
  kubectl apply -f experiments/fig4_wang2509/k8s/multistep-eta-wide-k300-0721.yaml
fi
kubectl -n "${NAMESPACE}" wait --for=condition=complete "job/${JOB}" --timeout=6h

kubectl -n "${NAMESPACE}" delete pod "${HELPER}" --ignore-not-found=true
kubectl apply -f experiments/fig4_wang2509/k8s/pvc-copy-helper.yaml
kubectl -n "${NAMESPACE}" wait --for=condition=Ready "pod/${HELPER}" --timeout=3m

kubectl -n "${NAMESPACE}" cp \
  "${HELPER}:${REMOTE_DIR}/multistep_eta_sweep_K300_300etas_1e-8_1e8.csv" \
  "${LOCAL_RESULT_DIR}/multistep_eta_sweep_K300_300etas_1e-8_1e8.csv"

for optimizer in gd signgd muon; do
  for regime in decoupled coupled; do
    FIGURE="eta_loss_${optimizer}_${regime}.png"
    kubectl -n "${NAMESPACE}" cp \
      "${HELPER}:${REMOTE_DIR}/${FIGURE}" \
      "${LOCAL_FIG_DIR}/${FIGURE}"
  done
done

kubectl -n "${NAMESPACE}" logs "job/${JOB}"
kubectl -n "${NAMESPACE}" delete pod "${HELPER}" --wait=false
