"""Focused K=999 diagnostics for Muon representation/environment jumps."""

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import torch

from .config import DEFAULT_CONFIG, get_dtype, make_prob_vector
from .embeddings import make_embeddings
from .model import init_W, logits


def state(W, E, Et, p, cfg, tol):
    Z = logits(W, E, Et)
    logP = torch.log_softmax(Z, dim=0)
    P = torch.exp(logP)
    correct = torch.diag(P)
    I = torch.eye(cfg.K, device=W.device, dtype=W.dtype)
    grad = Et @ ((P - I) * p[None, :]) @ E.T
    G = -grad
    polar_method = "svd"
    try:
        U, S, Vh = torch.linalg.svd(G, full_matrices=False)
        mask = S > tol
        D = U[:, mask] @ Vh[mask, :] if mask.any() else torch.zeros_like(G)
    except torch._C._LinAlgError:
        polar_method = "eigh_fallback"
        A = 0.5 * (G.T @ G + (G.T @ G).T)
        evals, V = torch.linalg.eigh(A)
        evals = torch.clamp(evals, min=0.0)
        order = torch.argsort(evals, descending=True)
        evals, V = evals[order], V[:, order]
        S = torch.sqrt(evals)
        mask = S > tol
        if mask.any():
            Vk = V[:, mask]
            D = (G @ Vk) @ (torch.rsqrt(evals[mask])[:, None] * Vk.T)
        else:
            D = torch.zeros_like(G)
    return {
        "Z": Z,
        "correct": correct,
        "loss": float((-(p * torch.diag(logP)).sum()).item()),
        "delta": float((correct.max() - correct.min()).item()),
        "G": G,
        "D": D,
        "S": S,
        "polar_method": polar_method,
    }


def probe_values(A, probes):
    scale = torch.linalg.norm(A) + 1e-30
    return [float((A * q).sum().item() / scale.item()) for q in probes]


def run(args):
    cfg = replace(DEFAULT_CONFIG, K=args.K, d=args.K, L=args.L,
                  device=args.device, svd_tol=args.tol)
    dtype = get_dtype(cfg.dtype_name)
    p = make_prob_vector(cfg, device=cfg.device, dtype=dtype)
    embeddings = {r: make_embeddings(r, cfg) for r in ("decoupled", "coupled")}
    weights = {r: init_W(cfg) for r in embeddings}

    generator = torch.Generator(device=cfg.device).manual_seed(20260718)
    probes = [torch.randn((cfg.K, cfg.K), generator=generator,
                          device=cfg.device, dtype=dtype) for _ in range(4)]
    probes = [q / torch.linalg.norm(q) for q in probes]
    selected = set(range(0, 6)) | set(range(97, 107))
    rows = []
    checkpoints = {}
    prev_D = {r: None for r in embeddings}

    for step in range(args.steps + 1):
        states = {
            r: state(weights[r], *embeddings[r], p, cfg, args.tol)
            for r in embeddings
        }

        # Map the coupled update and logits back to the common fact/logit space.
        E_c, Et_c = embeddings["coupled"]
        Dc_z = Et_c.T @ states["coupled"]["D"] @ E_c
        Dd_z = states["decoupled"]["D"]
        update_diff = float(
            (torch.linalg.norm(Dc_z - Dd_z) /
             (torch.linalg.norm(Dd_z) + 1e-30)).item()
        )
        prediction_diff = float(torch.max(torch.abs(
            states["coupled"]["correct"] - states["decoupled"]["correct"]
        )).item())

        if step in selected:
            for regime, s in states.items():
                row = {
                    "regime": regime,
                    "step": step,
                    "eta": args.eta,
                    "tol": args.tol,
                    "loss": s["loss"],
                    "delta": s["delta"],
                    "prediction_diff_coupled_decoupled": prediction_diff,
                    "update_diff_coupled_decoupled": update_diff,
                    "rank": int((s["S"] > args.tol).sum().item()),
                    "rank_1e-14": int((s["S"] > 1e-14).sum().item()),
                    "rank_1e-12": int((s["S"] > 1e-12).sum().item()),
                    "rank_1e-10": int((s["S"] > 1e-10).sum().item()),
                    "direction_change": float("nan") if prev_D[regime] is None else float(
                        (torch.linalg.norm(s["D"] - prev_D[regime]) /
                         (torch.linalg.norm(prev_D[regime]) + 1e-30)).item()
                    ),
                    "polar_method": s["polar_method"],
                }
                for i in range(1, 11):
                    row[f"s_tail_{i}"] = float(s["S"][-i].item())
                for i, value in enumerate(probe_values(s["G"], probes), 1):
                    row[f"gradient_probe_{i}"] = value
                for i, value in enumerate(probe_values(s["D"], probes), 1):
                    row[f"direction_probe_{i}"] = value
                rows.append(row)

        if step in (5, 98, 99, 100, 101, 102, 103):
            for regime, s in states.items():
                checkpoints[f"{regime}_gradient_step{step}"] = s["G"].cpu()
                checkpoints[f"{regime}_direction_step{step}"] = s["D"].cpu()

        if step == args.steps:
            break
        for regime, s in states.items():
            prev_D[regime] = s["D"]
            weights[regime] = weights[regime] + args.eta * s["D"]

        if step % 10 == 0:
            print(f"completed step {step}/{args.steps}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    torch.save(checkpoints, out.with_suffix(".pt"))
    print(f"[done] wrote {out}", flush=True)
    print(f"[done] wrote {out.with_suffix('.pt')}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=999)
    parser.add_argument("--L", type=int, default=200)
    parser.add_argument("--steps", type=int, default=106)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
