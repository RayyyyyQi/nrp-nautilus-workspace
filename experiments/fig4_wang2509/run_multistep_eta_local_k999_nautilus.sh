#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
NS=tianhaowang-ucsd
JOB=multistep-eta-local-k999-step200-0721
CM=multistep-eta-local-k999-code-0721
HELPER=pvc-copy-helper
REMOTE=/mnt/results/fig4_wang2509/7_21/nautilus/multistep_eta_local_K999_step200
LOCAL_FIG=experiments/fig4_wang2509/figures/7_21/nautilus/multistep_eta_local_K999_step200
LOCAL_RESULT=experiments/fig4_wang2509/results/7_21/nautilus/multistep_eta_local_K999_step200
mkdir -p "${LOCAL_FIG}" "${LOCAL_RESULT}"

kubectl -n "${NS}" create configmap "${CM}" \
  --from-file=run_multistep_eta_sweep.py=experiments/fig4_wang2509/run_multistep_eta_sweep.py \
  --from-file=plot_multistep_eta_sweep.py=experiments/fig4_wang2509/plot_multistep_eta_sweep.py \
  --dry-run=client -o yaml | kubectl apply -f -
if ! kubectl -n "${NS}" get job "${JOB}" >/dev/null 2>&1; then
  kubectl apply -f experiments/fig4_wang2509/k8s/multistep-eta-local-k999-step200-0721.yaml
fi
kubectl -n "${NS}" wait --for=condition=complete "job/${JOB}" --timeout=4h

kubectl -n "${NS}" delete pod "${HELPER}" --ignore-not-found=true
kubectl apply -f experiments/fig4_wang2509/k8s/pvc-copy-helper.yaml
kubectl -n "${NS}" wait --for=condition=Ready "pod/${HELPER}" --timeout=3m
kubectl -n "${NS}" cp "${HELPER}:${REMOTE}/multistep_eta_local_K999_step200.csv" "${LOCAL_RESULT}/multistep_eta_local_K999_step200.csv"
for optimizer in gd signgd muon; do
  for regime in decoupled coupled; do
    kubectl -n "${NS}" cp "${HELPER}:${REMOTE}/eta_loss_${optimizer}_${regime}.png" "${LOCAL_FIG}/eta_loss_${optimizer}_${regime}.png"
  done
done
kubectl -n "${NS}" delete pod "${HELPER}" --wait=false
