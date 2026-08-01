#!/usr/bin/env python3
"""
Evaluate the Cauchy-Riemann consistency experiment.

The question is NOT whether the penalty reduces the residual it is trained on
-- that would be circular. It is whether reducing it buys anything the penalty
was never shown:

  * long-horizon iteration  -- interior IoU, escape-time MAE, set disagreement,
    all produced by iterating the learned map ~64 times
  * C_w symmetry defect     -- a finite-rotation property, not a local one
  * pointwise fit           -- does the constraint cost accuracy?

CR is local and infinitesimal. Iteration is global and compounding. There is
no mechanical reason the first must help the second, which is exactly why it
is worth measuring.

    python3 cr_evaluate.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import neural_dynamics as nd
import cr_consistency as cr
import symmetry_probe as sp
import differential_probe as dp


def evaluate(run_dir: Path, resolution=200, max_iter=64, seed=0):
    net = nd.load_map(run_dir / "map.npz")
    meta = json.loads(str(np.load(run_dir / "map.npz",
                                  allow_pickle=False)["meta"]))
    data = np.load(run_dir / "map.npz", allow_pickle=False)

    rng = np.random.default_rng(seed)
    cx, cy, cw = cr.collocation_batch(rng, 20000)

    # what it was trained on
    cr_val = cr.cr_penalty(net, cx, cy, cw)

    # normalized CR from the independent probe (different stencil, different
    # normalization, measured against its own analytic floor)
    probe = dp.summarize(net.power, [2.0, 3.0, 4.0, 5.0, 6.0], n=8000)
    floor = dp.summarize(nd.true_power, [2.0, 3.0, 4.0, 5.0, 6.0], n=8000)

    # what it was NOT trained on: long-horizon iteration
    stats = nd.render_comparison(net, run_dir / "_eval.png", w_value=2.0,
                                 resolution=resolution, max_iter=max_iter)

    # and a finite-rotation property
    defects = [sp.symmetry_defect(net, float(k), n_samples=60000)
               for k in (2, 3, 4, 5, 6)]

    return {
        "cr_weight": meta.get("cr_weight", 0.0),
        "seed": meta.get("seed"),
        "final_fit": float(data["losses"][-1]),
        "cr_trained": cr_val,
        "cr_probe": float(np.median(probe["cauchy_riemann"])),
        "cr_floor": float(np.median(floor["cauchy_riemann"])),
        "iou": float(stats["iou"]),
        "mae": float(stats["mae"]),
        "disagreement": float(stats["disagreement_fraction"]),
        "symmetry_mean": float(np.mean(defects)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="crx_w*_s*")
    parser.add_argument("--out", type=Path, default=Path("./differential"))
    args = parser.parse_args()

    runs = sorted(Path(".").glob(args.glob))
    runs = [r for r in runs if (r / "map.npz").exists()]
    if not runs:
        raise SystemExit(f"no runs matching {args.glob}")

    rows = []
    for r in runs:
        print(f"evaluating {r.name} ...", flush=True)
        rows.append({"run": r.name, **evaluate(r)})

    rows.sort(key=lambda d: (d["cr_weight"], d["seed"]))

    print("\nCAUCHY-RIEMANN CONSISTENCY EXPERIMENT")
    print("  trained-on ->            | not trained on ------------------------>")
    print(f"  {'weight':>8} {'seed':>4} {'fit':>10} {'cr(train)':>10} "
          f"{'cr(probe)':>10} {'IoU':>8} {'MAE':>8} {'disagree':>9} {'sym':>8}")
    for d in rows:
        print(f"  {d['cr_weight']:>8.0e} {d['seed']:>4} {d['final_fit']:>10.3e} "
              f"{d['cr_trained']:>10.3e} {d['cr_probe']:>10.3e} "
              f"{d['iou']:>8.4f} {d['mae']:>8.4f} "
              f"{d['disagreement'] * 100:>8.2f}% {d['symmetry_mean']:>8.4f}")

    # paired deltas against the matched-seed baseline
    base = {d["seed"]: d for d in rows if d["cr_weight"] == 0.0}
    print("\n  PAIRED vs matched-seed baseline (negative = better for "
          "fit/MAE/disagree/sym, positive = better for IoU)")
    weights = sorted({d["cr_weight"] for d in rows if d["cr_weight"] > 0})
    for wv in weights:
        group = [d for d in rows if d["cr_weight"] == wv and d["seed"] in base]
        if not group:
            continue
        def rel(key):
            vals = [(d[key] - base[d["seed"]][key]) / abs(base[d["seed"]][key])
                    for d in group if base[d["seed"]][key] != 0]
            return 100.0 * float(np.mean(vals)) if vals else float("nan")
        print(f"    weight {wv:.0e}: cr(train) {rel('cr_trained'):+7.1f}%  "
              f"cr(probe) {rel('cr_probe'):+7.1f}%  |  "
              f"IoU {rel('iou'):+6.2f}%  MAE {rel('mae'):+7.2f}%  "
              f"disagree {rel('disagreement'):+7.2f}%  sym {rel('symmetry_mean'):+7.2f}%")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cr_experiment.json").write_text(json.dumps(rows, indent=2),
                                                 encoding="utf-8")
    print(f"\nwrote {args.out}/cr_experiment.json")


if __name__ == "__main__":
    main()
