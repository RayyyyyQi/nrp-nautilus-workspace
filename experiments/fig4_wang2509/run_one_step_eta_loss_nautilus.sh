#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

NAMESPACE=tianhaowang-ucsd
JOB=fig4-one-step-eta-loss-0721
CONFIGMAP=fig4-one-step-eta-loss-plotter-0721
HELPER=pvc-copy-helper
REMOTE_DIR=/mnt/results/fig4_wang2509/7_21/nautilus
LOCAL_FIG_DIR=experiments/fig4_wang2509/figures/7_21/nautilus
LOCAL_RESULT_DIR=experiments/fig4_wang2509/results/7_21/nautilus

mkdir -p "${LOCAL_FIG_DIR}" "${LOCAL_RESULT_DIR}"

kubectl -n "${NAMESPACE}" create configmap "${CONFIGMAP}" \
  --from-file=plot_one_step_eta_loss.py=experiments/fig4_wang2509/plot_one_step_eta_loss.py \
  --dry-run=client -o yaml | kubectl apply -f -

JOB_COMPLETE="$(kubectl -n "${NAMESPACE}" get job "${JOB}" -o jsonpath='{.status.succeeded}' 2>/dev/null || true)"
if [[ "${JOB_COMPLETE}" != "1" ]]; then
  kubectl -n "${NAMESPACE}" delete job "${JOB}" --ignore-not-found=true
  kubectl apply -f experiments/fig4_wang2509/k8s/fig4-one-step-eta-loss-0721.yaml
fi
kubectl -n "${NAMESPACE}" wait --for=condition=complete "job/${JOB}" --timeout=45m

kubectl -n "${NAMESPACE}" delete pod "${HELPER}" --ignore-not-found=true
kubectl apply -f experiments/fig4_wang2509/k8s/pvc-copy-helper.yaml
kubectl -n "${NAMESPACE}" wait --for=condition=Ready "pod/${HELPER}" --timeout=3m

kubectl -n "${NAMESPACE}" cp \
  "${HELPER}:${REMOTE_DIR}/one_step_eta_loss_wide_nautilus.csv" \
  "${LOCAL_RESULT_DIR}/one_step_eta_loss_wide_nautilus.csv"

for optimizer in gd signgd muon; do
  kubectl -n "${NAMESPACE}" cp \
    "${HELPER}:${REMOTE_DIR}/one_step_eta_loss_${optimizer}.png" \
    "${LOCAL_FIG_DIR}/one_step_eta_loss_${optimizer}.png"
done

kubectl -n "${NAMESPACE}" logs "job/${JOB}"
kubectl -n "${NAMESPACE}" delete pod "${HELPER}" --wait=false
