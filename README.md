# NRP Nautilus Workspace

This repository contains my NRP Nautilus Kubernetes configuration files, experiment scripts, Docker setup, and notes.

## Namespace

Current namespace:

```bash
tianhaowang-ucsd
```

## Common Commands

Check pods:

```bash
kubectl get pods
```

Check jobs:

```bash
kubectl get jobs
```

Apply a YAML file:

```bash
kubectl apply -f jobs/example-job.yaml
```

Delete a YAML-created resource:

```bash
kubectl delete -f jobs/example-job.yaml
```

View logs:

```bash
kubectl logs <pod-name>
```

## Notes

Do not commit kubeconfig files, tokens, secrets, datasets, checkpoints, or large experiment outputs.
