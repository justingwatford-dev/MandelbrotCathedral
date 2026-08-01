#!/usr/bin/env python3
"""
DIFFERENTIAL IDENTITY PROBE -- does the learned map obey the algebra of z^w?

Pointwise MSE can be excellent while the analytic structure is broken. The
`none`-basis models are the proof: at harm_none_s2 w=5.0 the median relative
map error is 0.031 -- indistinguishable from w=4.5 (0.030) -- while the C_w
symmetry defect is 2.62 against a true value of exactly zero. Good values,
broken algebra, in a band narrow enough that no value-based metric noticed.

THE IDENTITIES

Off the origin and off the branch cut (the negative real axis), f(z,w) = z^w
satisfies three exact relations. With f = u + iv and z = x + iy:

  1. CAUCHY-RIEMANN       u_x = v_y   and   u_y = -v_x
     f is holomorphic in z. Nothing in the network's parameterization
     enforces this, which makes it the strongest of the three.

  2. EULER HOMOGENEITY    x*f_x + y*f_y = w*f
     f(t*z) = t^w f(z) for real t > 0; differentiate at t = 1. Positive
     scaling does not change arg z, so this is branch-safe.

  3. W-DERIVATIVE         f_w = f * Log z = f * (log|z| + i*arg z)
     d/dw exp(w Log z).

CALIBRATION IS THE POINT

Every residual here is finite-differenced in float32, so the measurement has
its own error floor. That floor is measured, not assumed: the same FD pipeline
runs on the ANALYTIC map, where the true residual is exactly zero, and
whatever comes out is the noise floor. A learned residual only means something
relative to it. `--calibrate` sweeps h and prints the floor.

    python3 differential_probe.py --calibrate
    python3 differential_probe.py --checkpoint dyn_w26/map.npz
    python3 differential_probe.py --compare harm_integer_s1 harm_none_s1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import neural_dynamics as nd

# Domain. Stay off the origin (log|z| blows up) and off the negative real axis
# (the principal branch cut, where holomorphy genuinely fails). Radii keep
# w*log|z| well inside the +/-60,30 clip in from_targets.
R_MIN, R_MAX = 0.5, 3.0
CUT_MARGIN = 0.35          # radians excluded either side of arg z = +/- pi
H_DEFAULT = 3e-3           # chosen by --calibrate, see the sweep


def sample_domain(n, w_value, seed=0):
    """Points on an annulus with the branch cut excised."""
    rng = np.random.default_rng(seed)
    radius = np.sqrt(rng.uniform(R_MIN ** 2, R_MAX ** 2, n))
    angle = rng.uniform(-np.pi + CUT_MARGIN, np.pi - CUT_MARGIN, n)
    zx = (radius * np.cos(angle)).astype(np.float32)
    zy = (radius * np.sin(angle)).astype(np.float32)
    w = np.full(n, w_value, np.float32)
    return zx, zy, w


def fd_derivatives(fn, zx, zy, w, h=H_DEFAULT):
    """Central differences of fn -> (u,v) with respect to x, y and w."""
    def call(a, b, c):
        re, im = fn(a, b, c)
        return np.asarray(re, np.float64), np.asarray(im, np.float64)

    hx = np.float32(h)
    u0, v0 = call(zx, zy, w)
    uxp, vxp = call(zx + hx, zy, w)
    uxm, vxm = call(zx - hx, zy, w)
    uyp, vyp = call(zx, zy + hx, w)
    uym, vym = call(zx, zy - hx, w)
    uwp, vwp = call(zx, zy, w + hx)
    uwm, vwm = call(zx, zy, w - hx)

    d = 2.0 * float(h)
    return {
        "u": u0, "v": v0,
        "u_x": (uxp - uxm) / d, "v_x": (vxp - vxm) / d,
        "u_y": (uyp - uym) / d, "v_y": (vyp - vym) / d,
        "u_w": (uwp - uwm) / d, "v_w": (vwp - vwm) / d,
    }


def identity_residuals(fn, zx, zy, w, h=H_DEFAULT):
    """
    Normalized residual of each identity, per sample.

    Each is |lhs - rhs| divided by the natural scale of the terms involved, so
    the numbers are comparable across the ~e^27 dynamic range of |f|.
    """
    g = fd_derivatives(fn, zx, zy, w, h)
    x = np.asarray(zx, np.float64)
    y = np.asarray(zy, np.float64)
    ww = np.asarray(w, np.float64)
    eps = 1e-30

    # 1. Cauchy-Riemann
    cr = np.abs(g["u_x"] - g["v_y"]) + np.abs(g["u_y"] + g["v_x"])
    cr_scale = (np.abs(g["u_x"]) + np.abs(g["v_y"])
                + np.abs(g["u_y"]) + np.abs(g["v_x"]))
    cr_res = cr / (cr_scale + eps)

    # 2. Euler homogeneity: x f_x + y f_y = w f  (real and imaginary parts)
    eu_re = x * g["u_x"] + y * g["u_y"] - ww * g["u"]
    eu_im = x * g["v_x"] + y * g["v_y"] - ww * g["v"]
    eu_scale = (np.abs(x * g["u_x"]) + np.abs(y * g["u_y"]) + np.abs(ww * g["u"])
                + np.abs(x * g["v_x"]) + np.abs(y * g["v_y"])
                + np.abs(ww * g["v"]))
    eu_res = (np.abs(eu_re) + np.abs(eu_im)) / (eu_scale + eps)

    # 3. f_w = f * Log z
    log_r = 0.5 * np.log(x * x + y * y + eps)
    arg_z = np.arctan2(y, x)
    rhs_re = g["u"] * log_r - g["v"] * arg_z
    rhs_im = g["v"] * log_r + g["u"] * arg_z
    dw = np.abs(g["u_w"] - rhs_re) + np.abs(g["v_w"] - rhs_im)
    dw_scale = (np.abs(g["u_w"]) + np.abs(rhs_re)
                + np.abs(g["v_w"]) + np.abs(rhs_im))
    dw_res = dw / (dw_scale + eps)

    return {"cauchy_riemann": cr_res, "euler": eu_res, "w_derivative": dw_res}


IDENTITIES = ("cauchy_riemann", "euler", "w_derivative")


def summarize(fn, w_values, n=20000, h=H_DEFAULT, seed=0):
    """Median normalized residual per identity, per w."""
    out = {k: [] for k in IDENTITIES}
    for wv in w_values:
        zx, zy, w = sample_domain(n, wv, seed=seed)
        res = identity_residuals(fn, zx, zy, w, h=h)
        for k in IDENTITIES:
            out[k].append(float(np.median(res[k])))
    return {k: np.asarray(v) for k, v in out.items()}


def calibrate(w_values=(2.0, 3.5, 5.0), n=8000):
    """
    Sweep h on the ANALYTIC map, where every residual is exactly zero. What
    comes out is the finite-difference noise floor -- the number every learned
    residual has to be read against.
    """
    print("FD NOISE FLOOR -- same pipeline on the analytic map, true residual = 0")
    print(f"  domain: {R_MIN} <= |z| <= {R_MAX}, |arg z| <= pi - {CUT_MARGIN}")
    print(f"\n  {'h':>9} " + " ".join(f"{k:>16}" for k in IDENTITIES))
    best, best_h = None, None
    for h in (3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4):
        s = summarize(nd.true_power, w_values, n=n, h=h)
        med = {k: float(np.median(s[k])) for k in IDENTITIES}
        worst = max(med.values())
        print(f"  {h:>9.0e} " + " ".join(f"{med[k]:>16.3e}" for k in IDENTITIES))
        if best is None or worst < best:
            best, best_h = worst, h
    print(f"\n  best h = {best_h:.0e}  (worst-identity floor {best:.3e})")
    print("  Below this h, float32 roundoff dominates; above it, truncation does.")
    return best_h, best


def report(net, label, w_values, n=20000, h=H_DEFAULT):
    learned = summarize(net.power, w_values, n=n, h=h)
    floor = summarize(nd.true_power, w_values, n=n, h=h)

    print(f"\n{label}")
    print(f"  {'w':>5} " + " ".join(f"{k:>26}" for k in IDENTITIES))
    print(f"  {'':>5} " + " ".join(f"{'learned / floor  ratio':>26}"
                                  for _ in IDENTITIES))
    for i, wv in enumerate(w_values):
        cells = []
        for k in IDENTITIES:
            l, f = learned[k][i], floor[k][i]
            cells.append(f"{l:>8.2e} /{f:>8.2e} {l / max(f, 1e-30):>6.0f}x")
        print(f"  {wv:>5.2f} " + " ".join(f"{c:>26}" for c in cells))

    print("  " + "-" * 84)
    agg = {}
    for k in IDENTITIES:
        l = float(np.median(learned[k]))
        f = float(np.median(floor[k]))
        agg[k] = {"learned": l, "floor": f, "ratio": l / max(f, 1e-30)}
        print(f"  {k:>16}: median learned {l:.3e}  floor {f:.3e}  "
              f"ratio {agg[k]['ratio']:.0f}x")
    return {"w": list(map(float, w_values)),
            "learned": {k: learned[k].tolist() for k in IDENTITIES},
            "floor": {k: floor[k].tolist() for k in IDENTITIES},
            "aggregate": agg}


def residual_map(net, out_path, w_value=5.0, resolution=220, h=H_DEFAULT):
    """Spatial map of each identity's residual, plus the analytic floor."""
    span = np.linspace(-R_MAX, R_MAX, resolution, dtype=np.float32)
    X, Y = np.meshgrid(span, span)
    zx, zy = X.ravel(), Y.ravel()
    w = np.full(zx.size, w_value, np.float32)

    radius = np.hypot(zx, zy)
    angle = np.arctan2(zy, zx)
    valid = ((radius >= R_MIN) & (radius <= R_MAX)
             & (np.abs(angle) <= np.pi - CUT_MARGIN))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), facecolor="#05050a")
    for row, (fn, name) in enumerate(((net.power, "learned"),
                                      (nd.true_power, "analytic (FD floor)"))):
        res = identity_residuals(fn, zx, zy, w, h=h)
        for col, key in enumerate(IDENTITIES):
            field = np.where(valid, res[key], np.nan).reshape(resolution, -1)
            ax = axes[row][col]
            ax.set_facecolor("#05050a")
            im = ax.imshow(np.log10(np.clip(field, 1e-12, None)),
                           extent=[-R_MAX, R_MAX, -R_MAX, R_MAX],
                           origin="lower", cmap="inferno", vmin=-9, vmax=0)
            ax.set_title(f"{key}  [{name}]", color="#eee", fontsize=10)
            ax.tick_params(colors="#888", labelsize=7)
            for s in ax.spines.values():
                s.set_color("#333")
            cb = fig.colorbar(im, ax=ax, fraction=0.046)
            cb.ax.tick_params(colors="#888", labelsize=7)
            cb.set_label("log10 residual", color="#aaa", fontsize=8)

    fig.suptitle(f"Differential identity residuals at w = {w_value}",
                 color="#eee", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, facecolor="#05050a", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--compare", nargs="*", default=None,
                        help="run directories to compare, e.g. harm_integer_s1")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--h", type=float, default=None)
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--map", action="store_true", help="write residual maps")
    parser.add_argument("--map-w", type=float, default=5.0)
    parser.add_argument("--out", type=Path, default=Path("./differential"))
    args = parser.parse_args()

    if args.calibrate:
        calibrate()
        return

    h = args.h if args.h is not None else H_DEFAULT
    args.out.mkdir(parents=True, exist_ok=True)
    w_values = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]

    targets = []
    if args.checkpoint:
        targets.append((args.checkpoint.parent.name or "model", args.checkpoint))
    for d in (args.compare or []):
        targets.append((d, Path(d) / "map.npz"))
    if not targets:
        targets = [("dyn_w26", Path("dyn_w26/map.npz"))]

    print(f"h = {h:.0e}   n = {args.n:,} per w   "
          f"domain {R_MIN}<=|z|<={R_MAX}, cut margin {CUT_MARGIN}")
    results = {}
    for label, path in targets:
        net = nd.load_map(path)
        results[label] = report(net, label, w_values, n=args.n, h=h)
        if args.map:
            residual_map(net, args.out / f"residuals_{label}.png",
                         w_value=args.map_w, h=h)

    (args.out / "differential_residuals.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}/differential_residuals.json")


if __name__ == "__main__":
    main()
