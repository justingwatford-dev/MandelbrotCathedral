#!/usr/bin/env python3
"""
Train FiLMCathedralNet and read its w-embedding curve.

The missing half of extension #4. `film_exponent.py` can gradcheck the net and
extract the embedding curve, but has no training path -- so the curve could
only ever be reported on random weights. This trains it on the same escape-time
task as `train_small.py`, then asks the trained hypernetwork what it learned
about w.

    python3 train_film.py --epochs 40
    python3 train_film.py --curve-only --checkpoint run_film/model.npz

RESULT: NEGATIVE. The premise -- that a peak at integer w reports the network
found the branch-cut structure -- is false. The curve tracks where training w
was SAMPLED. `sample_w()` puts 45% of training w exactly on the integers, 25%
on half-integers, and `generate_dataset` re-snaps a further 30% of focused
samples to round(w).

    condition                       best offset        reading
    lattice on integers (default)   -0.005, +0.007     peak at integers
    lattice at k+0.25 (--w-anchor-phase 0.25)
                                    +0.254, +0.249     peak follows sampler
    no lattice (--w-anchor-mode uniform, 6 seeds)
                                    scatter +/-0.25    nothing

Paired movement for a +0.25 sampler shift is +0.2505. Under uniform w the mean
phase_exceedance is 0.526 against the 0.5 a true null predicts.

This survived a metric frozen in advance, five held-out seeds, two w-ranges,
and independent recomputation. None of those could see it, because all of them
sit downstream of the data generator. Only the intervention control could.

    python3 train_film.py --epochs 300 --n-data 300000 --w-anchor-mode uniform
    python3 train_film.py --curve-only --checkpoint run_film/model.npz
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import BranchCutCathedral as bcc
import film_exponent as fe


def build(cfg):
    return fe.FiLMCathedralNet(cfg)


def train(cfg, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(cfg.seed)
    bcc.xp.random.seed(cfg.seed)

    X, Y, W, labels = bcc.generate_dataset(cfg)
    # Capture how w was actually drawn so it travels with the checkpoint.
    w_audit = bcc.w_distribution_summary(
        bcc.to_numpy(W), bcc.to_numpy(labels), cfg)
    net = build(cfg)
    opt = bcc.Adam(net.params(), lr=cfg.lr)

    lab = bcc.to_numpy(labels)
    pos = float((lab >= .999).mean())
    pw = float(np.clip((1 - pos) / max(pos, 1e-6), 1, 20))
    print(f"params {net.parameter_count():,} | interior weight {pw:.2f}",
          flush=True)

    n = len(lab)
    t0 = time.time()
    losses = []
    for ep in range(1, cfg.epochs + 1):
        opt.lr = bcc.cosine_lr(ep, cfg.epochs, cfg)
        p = bcc.xp.asarray(np.random.permutation(n))
        X, Y, W, labels = X[p], Y[p], W[p], labels[p]

        tot = 0.0
        nb = 0
        for s in range(0, n, cfg.batch_size):
            e = min(s + cfg.batch_size, n)
            bs = e - s
            x, y, w, tg = X[s:e], Y[s:e], W[s:e], labels[s:e]

            pt, pi = net.fwd(x, y, w)
            it = (tg >= .999).astype(np.float32)
            be = 4.0 * np.exp(-((tg - .24) / .22) ** 2) * (1 - it)
            tw = 1.0 + 5.0 * np.sqrt(tg) + be + 3.0 * it
            d = pt - tg
            tl = (tw * d * d).mean()
            pr = pi.clip(1e-5, 1 - 1e-5)
            cw = 1.0 + it * (pw - 1.0)
            cl = -(cw * (it * np.log(pr) + (1 - it) * np.log(1 - pr))).mean()
            tot += float(tl + cfg.bce_weight * cl)
            nb += 1

            net.bwd(2 * tw * d / bs,
                    cfg.bce_weight * cw * (pr - it) / (pr * (1 - pr) + 1e-6) / bs)
            opt.step()

        losses.append(tot / nb)
        if ep % 5 == 0 or ep == 1:
            print(f"[{ep:3d}/{cfg.epochs}] loss={tot / nb:.6f} "
                  f"lr={opt.lr:.2e} {time.time() - t0:.0f}s", flush=True)

    bcc.save_checkpoint(net, opt, cfg.epochs, losses, cfg, out / "model.npz",
                        w_audit=w_audit)
    print(f"DONE {time.time() - t0:.0f}s -> {out}/model.npz", flush=True)
    return net


def load(cfg, path: Path):
    """Rebuild the net and restore saved parameters in params() order."""
    data = np.load(path, allow_pickle=True)
    net = build(cfg)
    net.B = bcc.xp.asarray(data["B"])
    for index, (parameter, _, _) in enumerate(net.params()):
        parameter[...] = bcc.xp.asarray(data[f"parameter_{index}"])
    return net


# ---------------------------------------------------------------------------
# FROZEN METRIC  (v1, fixed 2026-07-28 before any confirmatory seed was run)
#
# Every constant below is a measurement choice, not a discovered value. They
# are frozen here so that replication seeds are scored by a rule chosen in
# advance rather than one tuned until a seed passed.
#
# Fixes over the discovery run, all of which changed the numbers:
#   * The curve was sampled every 0.01 in w while offsets were scanned every
#     0.001 by NEAREST NEIGHBOUR. The 1001-point scan collapsed onto only 132
#     distinct values, so the reported best offset of +0.006 was below the
#     resolution of the data. Sample densely and interpolate instead.
#   * Point-sampling |d(embedding)/dw| at a single w is dominated by ripple.
#     Integrate arc length over a fixed window instead.
#   * Offsets are correlated positions along one deterministic curve, not
#     independent draws from a null. The exceedance fraction is NOT a p-value
#     and is no longer named like one.
# ---------------------------------------------------------------------------
CURVE_SPACING = 0.001      # w-resolution the embedding curve is sampled at
WINDOW_HALF   = 0.05       # arc length integrated over [w - h, w + h]
OFFSET_LIMIT  = 0.40       # lattice shift range; < 0.5 so every window stays
                           # inside the sampled band at both ends
OFFSET_STEP   = 0.001
PASS_EXCEEDANCE = 0.05     # phase_exceedance must be below this
PASS_OFFSET     = 0.02     # |best_offset| must be at or below this


def lattice_phase_test(w_values, speed, integers):
    """
    Score the integer lattice against the same lattice at every other phase.

    Returns (observed, null, phase_exceedance, best_offset). `phase_exceedance`
    is the fraction of phases scoring at least as high as the integers. It is a
    rank of one deterministic curve against shifted copies of itself, NOT a
    p-value over independent samples -- there is no sampling distribution here.
    """
    def windowed(v):
        """Mean arc-length speed over a fixed window centred on v."""
        lo, hi = v - WINDOW_HALF, v + WINDOW_HALF
        grid = np.arange(lo, hi + CURVE_SPACING * 0.5, CURVE_SPACING)
        return float(np.interp(grid, w_values, speed).mean())

    def lattice(offset):
        return float(np.mean([windowed(v + offset) for v in integers]))

    offsets = np.arange(-OFFSET_LIMIT, OFFSET_LIMIT + OFFSET_STEP * 0.5,
                        OFFSET_STEP)
    null = np.array([lattice(o) for o in offsets])
    observed = lattice(0.0)
    exceedance = float((null >= observed).mean())
    best_offset = float(offsets[int(np.argmax(null))])
    return observed, null, exceedance, best_offset


def report_curve(net, cfg, out: Path):
    n_points = int(round((cfg.w_max - cfg.w_min) / CURVE_SPACING)) + 1
    w_values = np.linspace(cfg.w_min, cfg.w_max, n_points)
    curve, speed = net.hyper.embedding_curve(w_values)

    median = float(np.median(speed))
    print("\nEMBEDDING CURVE (trained)")
    print(f"  curve shape       : {curve.shape}  "
          f"({cfg.blocks} blocks x 2 x {cfg.hidden} dims)")
    print(f"  sampled every     : {CURVE_SPACING} in w  ({n_points} points)")
    print(f"  arc-length speed  : mean {speed.mean():.4f}  "
          f"std {speed.std():.4f}  median {median:.4f}")

    # Scored integers must sit far enough inside the sampled range that every
    # shifted window stays in bounds. np.gradient is one-sided at the very
    # edges, so those points are not comparable to the rest either way.
    margin = OFFSET_LIMIT + WINDOW_HALF
    scored = [v for v in range(int(np.ceil(cfg.w_min)),
                               int(np.floor(cfg.w_max)) + 1)
              if cfg.w_min + margin <= v <= cfg.w_max - margin]
    band = (w_values > cfg.w_min + margin) & (w_values < cfg.w_max - margin)
    interior_speeds = speed[band]

    print(f"  {'w':>4}  {'windowed':>10}  {'x median':>9}  {'pctile':>7}")
    for v in range(int(np.ceil(cfg.w_min)), int(np.floor(cfg.w_max)) + 1):
        lo, hi = v - WINDOW_HALF, v + WINDOW_HALF
        grid = np.arange(lo, hi + CURVE_SPACING * 0.5, CURVE_SPACING)
        val = float(np.interp(grid, w_values, speed).mean())
        ratio = val / median if median else float("nan")
        pct = float((interior_speeds < val).mean()) * 100
        note = "" if v in scored else "  (too near edge - not scored)"
        print(f"  {v:>4}  {val:>10.4f}  {ratio:>8.2f}x  {pct:>6.1f}%{note}")

    observed, null, exceedance, best_offset = lattice_phase_test(
        w_values, speed, scored)

    print(f"\n  lattice phase test on {scored}   [frozen metric v1]")
    print(f"    window +/-{WINDOW_HALF}, offsets +/-{OFFSET_LIMIT} "
          f"step {OFFSET_STEP}, linear interpolation")
    print(f"    observed windowed speed : {observed:.4f}")
    print(f"    shifted-lattice null    : mean {null.mean():.4f}  "
          f"max {null.max():.4f}")
    print(f"    phase_exceedance        : {exceedance:.4f}   "
          f"(rank, not a p-value)")
    print(f"    best-fit offset         : {best_offset:+.3f}  "
          f"(0.000 = integers exactly optimal)")

    passed = exceedance < PASS_EXCEEDANCE and abs(best_offset) <= PASS_OFFSET
    print(f"\n    criterion: exceedance < {PASS_EXCEEDANCE} "
          f"and |offset| <= {PASS_OFFSET}  ->  {'PASS' if passed else 'FAIL'}")

    # THE CONFOUND. The sampler concentrates 45% of training w on a lattice,
    # plus a secondary bucket and a 30% re-snap. If that lattice sits on the
    # integers, "the embedding speeds up at integers" is equally well explained
    # by input density as by learned structure. Report where the sampler's
    # lattice was, so the two can be told apart.
    mode = getattr(cfg, "w_anchor_mode", "anchored")
    phase = float(getattr(cfg, "w_anchor_phase", 0.0))
    print(f"\n  sampler: w_anchor_mode={mode!r}  phase={phase:+.2f}")
    if mode == "uniform":
        print("    no lattice in the training distribution -- any alignment")
        print("    here cannot be an artifact of input density.")
    else:
        def lattice_at(o):
            grid_lo, grid_hi = -WINDOW_HALF, WINDOW_HALF
            g = np.arange(grid_lo, grid_hi + CURVE_SPACING * 0.5, CURVE_SPACING)
            return float(np.mean([
                float(np.interp(v + o + g, w_values, speed).mean())
                for v in scored]))
        print(f"    score at integers (offset 0.00) : {lattice_at(0.0):.4f}")
        print(f"    score at sampler  (offset {phase:+.2f}) : {lattice_at(phase):.4f}")
        if abs(phase) > 1e-9:
            follows = abs(best_offset - phase) < abs(best_offset)
            print(f"    best offset {best_offset:+.3f} is nearer "
                  f"{'the SAMPLER lattice' if follows else 'the INTEGERS'}")
            print("    -> learned the sampler" if follows
                  else "    -> learned the task structure")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4), facecolor="#05050a")
        ax.set_facecolor("#05050a")
        ax.plot(w_values, speed, color="#7fd4ff", lw=1.4)
        for v in range(int(cfg.w_min), int(cfg.w_max) + 1):
            ax.axvline(v, color="#ff9f45", ls="--", lw=0.8, alpha=0.6)
        ax.axhline(median, color="#888", ls=":", lw=0.8)
        ax.set_xlabel("w", color="#ddd")
        ax.set_ylabel("|d(embedding)/dw|", color="#ddd")
        ax.set_title("FiLM w-embedding arc-length speed", color="#eee")
        ax.tick_params(colors="#aaa")
        for spine in ax.spines.values():
            spine.set_color("#444")
        fig.savefig(out / "embedding_curve.png", dpi=150,
                    facecolor="#05050a", bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out}/embedding_curve.png")
    except Exception as exc:                       # plotting is not the point
        print(f"  (plot skipped: {type(exc).__name__}: {exc})")

    np.savez_compressed(out / "embedding_curve.npz",
                        w=w_values, curve=curve, speed=speed,
                        offsets_null=null, scored=np.asarray(scored, float))
    return {"median": median, "observed": observed,
            "phase_exceedance": exceedance, "best_offset": best_offset,
            "passed": bool(passed), "scored": scored}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--n-data", type=int, default=100_000)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("./run_film"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--curve-only", action="store_true")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed for data, init and shuffling (default: "
                             "Config's own)")
    # Widening the range makes every integer an interior point: with
    # w in [1.5, 6.5] all of 2..6 are scorable instead of just 3..5.
    parser.add_argument("--w-min", type=float, default=None)
    parser.add_argument("--w-max", type=float, default=None)
    # Sampler controls. The default training distribution puts 45% of w on an
    # integer lattice, so "anchored/0.0" cannot distinguish learned structure
    # from input density. "uniform" removes the lattice; phase 0.25 moves it
    # off the integers while leaving the true targets alone.
    parser.add_argument("--w-anchor-mode", choices=["anchored", "uniform"],
                        default=None)
    parser.add_argument("--w-anchor-phase", type=float, default=None)
    args = parser.parse_args()

    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.w_min is not None:
        overrides["w_min"] = args.w_min
    if args.w_max is not None:
        overrides["w_max"] = args.w_max
    if args.w_anchor_mode is not None:
        overrides["w_anchor_mode"] = args.w_anchor_mode
    if args.w_anchor_phase is not None:
        overrides["w_anchor_phase"] = args.w_anchor_phase

    cfg = bcc.Config(n_data=args.n_data, batch_size=4096, epochs=args.epochs,
                     hidden=args.hidden, blocks=args.blocks,
                     mixed_features=64, out_dir=str(args.out), **overrides)
    print(f"seed {cfg.seed} | w in [{cfg.w_min}, {cfg.w_max}] | "
          f"sampler {cfg.w_anchor_mode} phase {cfg.w_anchor_phase:+.2f}",
          flush=True)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.curve_only:
        path = args.checkpoint or (args.out / "model.npz")
        if not path.exists():
            raise SystemExit(f"No checkpoint at {path} -- train first.")
        net = load(cfg, path)
    else:
        net = train(cfg, args.out)

    report_curve(net, cfg, args.out)


if __name__ == "__main__":
    main()
