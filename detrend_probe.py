"""detrend_probe.py — is the deep-zoom correlation carrying structure, or a ramp?

    python detrend_probe.py --checkpoint run/model.npz --label anchored
    python detrend_probe.py --checkpoint run_uniform/model.npz --label uniform

`zoom_probe.py` reports corr(truth, net) over the whole window. At deep zoom a
window contains two very different things: a large-scale ramp (escape time
rising away from the set) and the fine self-similar structure. A band-limited
net cannot represent the second but can still get the first roughly right, and
plain correlation cannot tell those apart — a smooth net that only follows the
ramp scores positive, and a ringing net whose ripples land out of phase scores
negative, while neither has any of the structure.

So: fit a plane (least squares, 1/x/y) to truth and to net at each zoom,
subtract, and correlate the residuals. corr_raw is what zoom_probe reports;
corr_detrended is what survives when the ramp is gone. ramp_share is the
fraction of each field's variance the plane explains.

Ground truth and field construction are copied from zoom_probe.zoom_sweep so
the two agree by construction; only the statistics are new.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

import BranchCutCathedral as bcc
import cathedral_grad as cg
import zoom_probe as zp


def plane_fit(field):
    """Least-squares plane over the grid. Returns (fitted, residual, r2)."""
    n = field.shape[0]
    u = np.linspace(-1, 1, n, dtype=np.float64)
    X, Y = np.meshgrid(u, u)
    A = np.column_stack([np.ones(field.size), X.ravel(), Y.ravel()])
    coef, *_ = np.linalg.lstsq(A, field.ravel().astype(np.float64), rcond=None)
    fit = (A @ coef).reshape(field.shape)
    resid = field - fit
    var = float(field.var())
    r2 = float(1.0 - resid.var() / var) if var > 1e-12 else 0.0
    return fit, resid, r2


def corr(a, b):
    da, db = a - a.mean(), b - b.mean()
    d = da.std() * db.std()
    return float((da * db).mean() / d) if d > 1e-12 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--label", default="run")
    ap.add_argument("--resolution", type=int, default=192)
    ap.add_argument("--w", type=float, default=2.0)
    ap.add_argument("--out", type=Path, default=Path("./detrend"))
    a = ap.parse_args()

    from neural_dynamics import _resolve_checkpoint
    ckpt = _resolve_checkpoint(a.checkpoint)
    data = np.load(ckpt, allow_pickle=False)
    cfg = bcc.Config(**json.loads(str(data["config"])))
    net = bcc.BranchCutNet(cfg)
    bcc.load_checkpoint(net, bcc.Adam(net.params(), lr=cfg.lr), ckpt)
    cx, cy = -0.745428, 0.113009
    base_width = cfg.x_max - cfg.x_min
    rows = []

    print(f"{'zoom':>6} {'corr_raw':>9} {'corr_detr':>10} {'ramp_truth':>11} {'ramp_net':>9} {'net_std':>8}")
    for zoom in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000):
        half = 0.5 * base_width / zoom
        half_y = half * (cfg.y_max - cfg.y_min) / (cfg.x_max - cfg.x_min)
        xs = np.linspace(cx - half, cx + half, a.resolution, dtype=np.float32)
        ys = np.linspace(cy - half_y, cy + half_y, a.resolution, dtype=np.float32)
        X, Y = np.meshgrid(xs, ys)
        truth = bcc.power_escape_cpu(
            X.ravel(), Y.ravel(), np.full(X.size, a.w, np.float32), cfg
        ).reshape(a.resolution, a.resolution).astype(np.float64)
        pred, _, _ = cg.grid_field_and_grad(
            net, a.w, cfg, a.resolution,
            x_min=cx - half, x_max=cx + half,
            y_min=cy - half_y, y_max=cy + half_y,
        )
        pred = np.asarray(pred, dtype=np.float64)

        _, rt, r2t = plane_fit(truth)
        _, rp, r2p = plane_fit(pred)
        row = {
            "zoom": zoom,
            "corr_raw": corr(truth, pred),
            "corr_detrended": corr(rt, rp),
            "ramp_share_truth": r2t,
            "ramp_share_net": r2p,
            "truth_std": float(truth.std()),
            "net_std": float(pred.std()),
        }
        rows.append(row)
        print(f"{zoom:>5}x {row['corr_raw']:>+9.3f} {row['corr_detrended']:>+10.3f} "
              f"{r2t:>11.3f} {r2p:>9.3f} {row['net_std']:>8.3f}")

    a.out.mkdir(parents=True, exist_ok=True)
    p = a.out / f"detrend_{a.label}.json"
    json.dump({"label": a.label, "checkpoint": str(a.checkpoint), "w": a.w,
               "centre": [cx, cy], "resolution": a.resolution, "rows": rows},
              open(p, "w", encoding="utf-8", newline="\n"), indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
