#!/usr/bin/env python3
"""
SYMMETRY PROBE — did the learned map discover the algebra, or just the pixels?

Motivated by the neural Julia panel at c = 0.285 + 0.01j losing its symmetry
while the other three held up.

THE EXACT FACT BEING TESTED
---------------------------
For integer w, the power map has an exact discrete rotational equivariance.
With rho = exp(2*pi*i / w):

    (rho * z) ** w  =  rho**w * z**w  =  z**w

so P(rho*z) = P(z) identically. For w = 2 this is the familiar
(-z)^2 = z^2, which is why every quadratic Julia set is invariant under
z -> -z.

Measured on the true map, the defect is machine zero:

    w = 2   ->  3.0e-16
    w = 3   ->  8.3e-16
    w = 4   ->  3.8e-16

and at non-integer w the principal branch cut destroys it
(`true_symmetry_defect`, n_samples=120000, seed=0):

    w = 2.5 ->  0.278522267
    w = 3.7 ->  0.116503367

So the defect curve of the TRUE map is a comb: exact zeros at the integers,
strictly positive between them under this metric, peaking around 0.1-0.3. The
defect falls smoothly to zero approaching each integer rather than dropping
off a cliff, so the contrast is a broad trough at every integer, not a spike
between them. Nothing about training points at this. A network fitting
pointwise MSE has no reason to respect it.

WHY THIS IS WORTH MEASURING
---------------------------
It converts "did the model learn the structure or memorize the surface?" into
one number per exponent, against an analytic ground-truth reference with no
fitted baseline. It is not parameter-free: the sampling disc, the
log-magnitude/direction feature representation, and the normalization are all
measurement choices, and the absolute scale depends on them. What does not
depend on them is the true-vs-learned comparison, since both sides go through
the same function.

This probe was originally paired with the FiLM embedding curve as a second,
independent route to the same question. THAT PAIRING NO LONGER HOLDS: the
embedding curve was shown to track its own training sampler, not the task
(see train_film.py). This measurement is not affected, and the reason is
worth stating precisely -- `neural_dynamics.orbit_dataset` draws w from
`rng.uniform(w_min, w_max)`, so there is no lattice anywhere in its inputs,
and the comparison here is against the analytic map rather than against a
chosen threshold.

That distinction is the whole lesson: this probe has a ground truth, so it can
be checked. Anything without one needs an intervention control before its
result means what it appears to mean.

Usage:
    python3 symmetry_probe.py --checkpoint dyn/map.npz --sweep
    python3 symmetry_probe.py --checkpoint dyn/map.npz --julia-test --w 2.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import neural_dynamics as nd


# ---------------------------------------------------------------------------
# THE MEASUREMENT
# ---------------------------------------------------------------------------

def rotate(zx, zy, angle):
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return zx * cos_a - zy * sin_a, zx * sin_a + zy * cos_a


def symmetry_defect(net, w_value, n_samples=120_000, bailout=4.0, seed=0,
                    relative=True):
    """
    Mean |P(rho*z) - P(z)| over the training disc, rho = exp(2*pi*i/w).

    Zero means the network reproduces the exact equivariance. Order-unity
    means it is fitting the surface pointwise with no algebraic structure
    underneath.
    """
    rng = np.random.default_rng(seed)
    radius = bailout * np.sqrt(rng.random(n_samples))
    theta = rng.uniform(-np.pi, np.pi, n_samples)
    zx = (radius * np.cos(theta)).astype(np.float32)
    zy = (radius * np.sin(theta)).astype(np.float32)
    w = np.full(n_samples, w_value, np.float32)

    base_re, base_im = net.power(zx, zy, w)

    rx, ry = rotate(zx, zy, 2.0 * np.pi / w_value)
    rot_re, rot_im = net.power(rx.astype(np.float32), ry.astype(np.float32), w)

    # Compare in log-magnitude / direction space, matching how the net was
    # trained: raw Cartesian differences are dominated by the few huge-|z^w|
    # samples and tell you almost nothing about the bulk.
    def as_features(re, im):
        # Promote before squaring. In float32, re ~ 1e-24 gives re*re ~ 1e-47,
        # which is below even the subnormal floor and flushes to exactly 0 --
        # so magnitude collapses onto the 1e-30 guard and re/magnitude blows
        # up to ~1e6 for a perfectly ordinary tiny output. That inflated the
        # measured defect of poorly-fit models by up to 134x while leaving
        # well-fit models untouched, which is the worst kind of measurement
        # bug: silent, and correlated with the thing being measured.
        re = np.asarray(re, dtype=np.float64)
        im = np.asarray(im, dtype=np.float64)
        magnitude = np.sqrt(re * re + im * im) + 1e-300
        return np.log(magnitude), re / magnitude, im / magnitude

    a = np.stack(as_features(base_re, base_im), axis=1)
    b = np.stack(as_features(rot_re, rot_im), axis=1)

    finite = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    a, b = a[finite], b[finite]

    defect = np.abs(a - b).mean()
    if relative:
        defect /= (np.abs(a).mean() + 1e-12)
    return float(defect)


def true_symmetry_defect(w_value, n_samples=120_000, bailout=4.0, seed=0):
    """Same measurement on the analytic map, as the reference comb."""
    rng = np.random.default_rng(seed)
    radius = bailout * np.sqrt(rng.random(n_samples))
    theta = rng.uniform(-np.pi, np.pi, n_samples)
    zx = radius * np.cos(theta)
    zy = radius * np.sin(theta)

    base_re, base_im = nd.true_power(zx, zy, np.full(n_samples, w_value))
    rx, ry = rotate(zx, zy, 2.0 * np.pi / w_value)
    rot_re, rot_im = nd.true_power(rx, ry, np.full(n_samples, w_value))

    def as_features(re, im):
        # Promote before squaring. In float32, re ~ 1e-24 gives re*re ~ 1e-47,
        # which is below even the subnormal floor and flushes to exactly 0 --
        # so magnitude collapses onto the 1e-30 guard and re/magnitude blows
        # up to ~1e6 for a perfectly ordinary tiny output. That inflated the
        # measured defect of poorly-fit models by up to 134x while leaving
        # well-fit models untouched, which is the worst kind of measurement
        # bug: silent, and correlated with the thing being measured.
        re = np.asarray(re, dtype=np.float64)
        im = np.asarray(im, dtype=np.float64)
        magnitude = np.sqrt(re * re + im * im) + 1e-300
        return np.log(magnitude), re / magnitude, im / magnitude

    a = np.stack(as_features(base_re, base_im), axis=1)
    b = np.stack(as_features(rot_re, rot_im), axis=1)
    finite = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    a, b = a[finite], b[finite]
    return float(np.abs(a - b).mean() / (np.abs(a).mean() + 1e-12))


# ---------------------------------------------------------------------------
# TEST-TIME ENFORCEMENT
# ---------------------------------------------------------------------------

class SymmetrizedMap:
    """
    Wraps a NeuralMap and averages its prediction over the C_w orbit, which
    makes the equivariance exact by construction for integer w.

    This is a free test-time intervention -- no retraining. If the Julia sets
    improve, the network's symmetry error was a real part of its error budget.
    If they do not, the asymmetry was cosmetic and the damage lies elsewhere.
    Either answer is worth having, and it costs w forward passes.
    """

    def __init__(self, net, w_value):
        if abs(w_value - round(w_value)) > 1e-6:
            raise ValueError("C_w orbit averaging is only defined for integer w")
        self.net = net
        self.order = int(round(w_value))
        self.bailout = net.bailout

    def power(self, zx, zy, w):
        zx = np.asarray(zx, np.float32).ravel()
        zy = np.asarray(zy, np.float32).ravel()

        acc_re = np.zeros(len(zx), np.float64)
        acc_im = np.zeros(len(zx), np.float64)

        for k in range(self.order):
            angle = 2.0 * np.pi * k / self.order
            rx, ry = rotate(zx, zy, angle)
            re, im = self.net.power(rx.astype(np.float32),
                                    ry.astype(np.float32), w)
            acc_re += re
            acc_im += im

        return ((acc_re / self.order).astype(np.float32),
                (acc_im / self.order).astype(np.float32))


# ---------------------------------------------------------------------------
# EXPERIMENTS
# ---------------------------------------------------------------------------

def sweep(net, out_path, w_min=2.0, w_max=6.0, n=161, n_samples=60_000):
    """
    Defect vs exponent. The reference curve is a comb with exact zeros at
    integers; the question is whether the network's curve has ANY feature
    at those points.
    """
    w_values = np.linspace(w_min, w_max, n)
    net_defect = np.array([symmetry_defect(net, float(w), n_samples=n_samples)
                           for w in w_values])
    true_defect = np.array([true_symmetry_defect(float(w), n_samples=20_000)
                            for w in w_values])

    integers = np.arange(int(np.ceil(w_min)), int(np.floor(w_max)) + 1)
    at_integers = np.array([net_defect[np.argmin(np.abs(w_values - k))]
                            for k in integers])
    baseline = np.median(net_defect)

    figure, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True,
                                facecolor="#05050a")

    axes[0].plot(w_values, true_defect, color="#4dd2ff", lw=1.4)
    axes[0].set_ylabel("true map", color="#aaa")
    axes[0].set_title("C_w symmetry defect — exact zeros at integer w",
                      color="white", fontsize=11)

    axes[1].plot(w_values, net_defect, color="#f5a623", lw=1.4)
    axes[1].axhline(baseline, color="#666", ls=":", lw=0.9)
    for k in integers:
        axes[1].axvline(k, color="#ff5c8a", ls="--", lw=0.8, alpha=0.6)
    axes[1].set_ylabel("learned map", color="#aaa")
    axes[1].set_xlabel("exponent w", color="#aaa")

    for ax in axes:
        ax.set_facecolor("#05050a")
        ax.tick_params(colors="#aaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    figure.suptitle("DID THE NETWORK FIND THE ALGEBRA?",
                    color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    figure.savefig(out_path, dpi=150, facecolor="#05050a", bbox_inches="tight")
    plt.close(figure)

    print(f"\n  median defect across w : {baseline:.4f}")
    for k, value in zip(integers, at_integers):
        ratio = value / baseline
        verdict = "DIP" if ratio < 0.75 else ("flat" if ratio < 1.25 else "peak")
        print(f"  w = {k}: defect {value:.4f}   {ratio:5.2f}x median   {verdict}")
    print("\n  Dips at the integers = the network reproduces the equivariance;")
    print("  a flat curve means it is fitting the surface pointwise.")
    print("  Whether 'found it unaided' follows depends on how the model's w")
    print("  was SAMPLED. neural_dynamics draws w continuously, so for those")
    print("  checkpoints it does. A model trained with an integer-anchored")
    print("  sampler was shown the lattice, and this plot cannot tell the")
    print("  difference -- compare against the true-map curve, not the median.")

    return w_values, net_defect, true_defect


def julia_test(net, out_path, w_value=2.0, resolution=200, max_iter=48):
    """Does enforcing the symmetry at test time actually improve anything?"""
    symmetrized = SymmetrizedMap(net, w_value)
    constants = [-0.4 + 0.6j, -0.8 + 0.156j, 0.285 + 0.01j, -0.70176 - 0.3842j]

    figure, axes = plt.subplots(3, len(constants),
                                figsize=(3.4 * len(constants), 10.2),
                                facecolor="#05050a")
    rows = []

    for column, c_value in enumerate(constants):
        axis = np.linspace(-1.8, 1.8, resolution, dtype=np.float32)
        ZX, ZY = np.meshgrid(axis, axis)

        def julia_with(mapper):
            zx = ZX.ravel().copy(); zy = ZY.ravel().copy()
            escape_iter = np.full(zx.size, max_iter, np.float32)
            alive = np.ones(zx.size, bool)
            for iteration in range(1, max_iter + 1):
                if not alive.any():
                    break
                index = np.flatnonzero(alive)
                px, py = mapper.power(zx[index], zy[index],
                                      np.full(len(index), w_value, np.float32))
                zx[index] = np.clip(px + c_value.real, -1e6, 1e6)
                zy[index] = np.clip(py + c_value.imag, -1e6, 1e6)
                magnitude = zx[index] ** 2 + zy[index] ** 2
                gone = (magnitude > 16.0) | ~np.isfinite(magnitude)
                escape_iter[index[gone]] = iteration
                alive[index[gone]] = False
            return (escape_iter / max_iter).reshape(resolution, resolution)

        class TrueMap:
            def power(self, zx, zy, w):
                return nd.true_power(zx, zy, w)

        truth = julia_with(TrueMap())
        plain = julia_with(net)
        fixed = julia_with(symmetrized)

        def iou(a, b):
            ai, bi = a >= 0.999, b >= 0.999
            return float((ai & bi).sum() / max((ai | bi).sum(), 1))

        rows.append({"c": str(c_value),
                     "iou_plain": iou(truth, plain),
                     "iou_symmetrized": iou(truth, fixed)})

        for row, (image, label) in enumerate([
            (truth, f"true  c={c_value:.3g}"),
            (plain, f"neural  IoU {rows[-1]['iou_plain']:.3f}"),
            (fixed, f"symmetrized  IoU {rows[-1]['iou_symmetrized']:.3f}"),
        ]):
            axes[row, column].imshow(image, origin="lower",
                                     cmap="twilight_shifted")
            axes[row, column].set_xticks([]); axes[row, column].set_yticks([])
            axes[row, column].set_title(label, color="white", fontsize=9)

    figure.suptitle("TEST-TIME SYMMETRY ENFORCEMENT — free, no retraining",
                    color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    figure.savefig(out_path, dpi=150, facecolor="#05050a", bbox_inches="tight")
    plt.close(figure)

    print()
    for row in rows:
        delta = row["iou_symmetrized"] - row["iou_plain"]
        print(f"  c={row['c']:>20s}  IoU {row['iou_plain']:.4f} -> "
              f"{row['iou_symmetrized']:.4f}   ({delta:+.4f})")
    mean_delta = np.mean([r['iou_symmetrized'] - r['iou_plain'] for r in rows])
    print(f"\n  mean change: {mean_delta:+.4f}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("./dyn/map.npz"))
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--julia-test", action="store_true")
    parser.add_argument("--w", type=float, default=2.0)
    parser.add_argument("--out", type=Path, default=Path("./symmetry"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    net = nd.load_map(args.checkpoint)

    defect = symmetry_defect(net, args.w)
    reference = true_symmetry_defect(args.w)
    print(f"\nC_w SYMMETRY DEFECT at w = {args.w}")
    print(f"  true map    : {reference:.3e}")
    print(f"  learned map : {defect:.4f}")

    if args.sweep:
        sweep(net, args.out / "symmetry_sweep.png")
    if args.julia_test:
        julia_test(net, args.out / "symmetry_julia.png", w_value=args.w)

    print(f"\nWrote to {args.out}/")


if __name__ == "__main__":
    main()
