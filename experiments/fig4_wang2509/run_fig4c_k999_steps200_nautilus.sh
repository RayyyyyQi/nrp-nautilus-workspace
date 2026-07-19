#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

NS=tianhaowang-ucsd
JOB=fig4c-k999-steps200-tuned-eta-0721
CM=fig4c-k999-steps200-code-0721
HELPER=pvc-copy-helper
REMOTE=/mnt/results/fig4_wang2509/7_21/nautilus/fig4c_K999_steps200_tuned_eta
LOCAL_FIG=experiments/fig4_wang2509/figures/7_21/nautilus/stage3_figure4c
LOCAL_RESULT=experiments/fig4_wang2509/results/7_21/nautilus

kubectl -n "${NS}" create configmap "${CM}" \
  --from-file=run_multi_step.py=experiments/fig4_wang2509/run_multi_step.py \
  --from-file=plot_fig4.py=experiments/fig4_wang2509/plot_fig4.py \
  --from-file=plot_fig4c_high_precision.py=experiments/fig4_wang2509/plot_fig4c_high_precision.py \
  --dry-run=client -o yaml | kubectl apply -f -

if ! kubectl -n "${NS}" get job "${JOB}" >/dev/null 2>&1; then
  kubectl apply -f experiments/fig4_wang2509/k8s/fig4c-k999-steps200-tuned-eta-0721.yaml
fi
kubectl -n "${NS}" wait --for=condition=complete "job/${JOB}" --timeout=2h

kubectl -n "${NS}" delete pod "${HELPER}" --ignore-not-found=true
kubectl apply -f experiments/fig4_wang2509/k8s/pvc-copy-helper.yaml
kubectl -n "${NS}" wait --for=condition=Ready "pod/${HELPER}" --timeout=3m
kubectl -n "${NS}" cp "${HELPER}:${REMOTE}/fig4c_K999_steps200_tuned_eta_nautilus.csv" "${LOCAL_RESULT}/fig4c_K999_steps200_tuned_eta_nautilus.csv"
for figure in fig4c_K999_steps200_tuned_eta_normal_nautilus.png fig4c_K999_steps200_tuned_eta_precision1e-16_nautilus.png; do
  kubectl -n "${NS}" cp "${HELPER}:${REMOTE}/${figure}" "${LOCAL_FIG}/${figure}"
done
kubectl -n "${NS}" delete pod "${HELPER}" --wait=false
