#!/usr/bin/env python3
"""
BRANCH-CUT CATHEDRAL

A companion/replacement experiment for MandelbrotLeary.py.

What changes:
  • Correct generalized smooth-escape estimate for varying exponent w
  • Conservative bailout radius
  • Mixed 3D Fourier features instead of axis-separated features
  • Dual network heads:
        1. smooth escape-time prediction
        2. non-escaping / interior probability
  • Boundary-weighted regression + balanced classification loss
  • Integer, half-integer, and continuous exponent sampling
  • Held-out slice validation with truth / prediction / error panels
  • Checkpointing, optimizer-state saving, and resume support
  • Psychedelic kaleidoscopic renderer driven by:
        - neural escape field
        - learned interior confidence
        - spatial gradients
        - branch-cut seam
        - model uncertainty

Place this file beside MandelbrotLeary.py.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

try:
    from PIL import Image
except ImportError:
    Image = None

import MandelbrotLeary as base


xp = base.xp
to_numpy = base.to_numpy
GPU = base.GPU

Linear = base.Linear
LayerNorm = base.LayerNorm
GELU = base.GELU
Sigmoid = base.Sigmoid
ResBlock = base.ResBlock
Adam = base.Adam


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

@dataclass
class Config:
    seed: int = 42

    n_data: int = 160_000
    batch_size: int = 4096
    epochs: int = 70

    hidden: int = 256
    blocks: int = 4
    mixed_features: int = 80

    max_iter: int = 256
    bailout: float = 4.0

    lr: float = 3e-4
    lr_min: float = 1e-6
    bce_weight: float = 0.20

    x_min: float = -2.5
    x_max: float = 1.0
    y_min: float = -1.35
    y_max: float = 1.35
    w_min: float = 2.0
    w_max: float = 6.0

    # How w is sampled. "anchored" concentrates mass on a lattice (the
    # original behaviour); "uniform" draws w continuously with no lattice at
    # all. w_anchor_phase shifts the lattice off the integers -- set it to
    # 0.25 to put the sampling spikes at k+0.25 while the true targets stay
    # where they are. Together these are the control conditions for any claim
    # that the network located the integers on its own: with "anchored" and
    # phase 0.0 the training distribution already identifies them.
    w_anchor_mode: str = "anchored"
    w_anchor_phase: float = 0.0
    w_snap_fraction: float = 0.30

    def __post_init__(self):
        # log(w) is the denominator of the smooth-escape estimate, so w=1
        # divides by zero. The infinity then survives np.clip as a finite
        # constant label, which means a downstream "are the labels finite?"
        # check passes while the data is silently degenerate. Refuse the
        # configuration instead -- this is the earliest point that can.
        if self.w_min <= 1.0:
            raise ValueError(
                f"w_min must be > 1 (got {self.w_min}): the smooth-escape "
                "estimate divides by log(w), and w=1 yields a constant-0 "
                "label block that no finiteness assertion will catch."
            )
        if self.w_max <= self.w_min:
            raise ValueError(f"w_max ({self.w_max}) must exceed w_min "
                             f"({self.w_min})")
        if self.w_anchor_mode not in ("anchored", "uniform"):
            raise ValueError(f"unknown w_anchor_mode {self.w_anchor_mode!r}")

    vis_chunk: int = 8192
    validation_res: int = 260
    art_res: int = 360
    art_frames: int = 72
    art_feedback: int = 2
    frame_ms: int = 55

    checkpoint_every: int = 10
    out_dir: str = "./branch_cut_outputs"


def preset(name: str) -> Config:
    if name == "smoke":
        return Config(
            n_data=20_000,
            batch_size=1024,
            epochs=4,
            hidden=128,
            blocks=2,
            mixed_features=36,
            validation_res=140,
            art_res=160,
            art_frames=12,
            checkpoint_every=2,
        )

    if name == "gpu":
        return Config(
            n_data=2_000_000,
            batch_size=65_536,
            epochs=300,
            hidden=512,
            blocks=6,
            mixed_features=144,
            validation_res=420,
            art_res=600,
            art_frames=150,
            checkpoint_every=20,
        )

    if name == "absurd":
        return Config(
            n_data=5_000_000,
            batch_size=65_536,
            epochs=450,
            hidden=768,
            blocks=8,
            mixed_features=192,
            validation_res=600,
            art_res=768,
            art_frames=240,
            checkpoint_every=25,
        )

    return Config()


# ---------------------------------------------------------------------------
# GENERALIZED QUADRATIC/POWER MAP
# ---------------------------------------------------------------------------

def power_escape_cpu(
    xs: np.ndarray,
    ys: np.ndarray,
    ws: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    """
    Escape field for:

        z_(n+1) = principal_branch(z_n ** w) + c
        z_0 = 0
        c = x + i y

    For non-integer w this is not a classical Multibrot polynomial.
    np.angle chooses the principal branch, producing a branch seam along
    the negative real axis. We retain it deliberately. It is now art.
    """
    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    ws = np.asarray(ws, dtype=np.float32)

    c = xs.astype(np.complex64) + 1j * ys.astype(np.complex64)
    z = np.zeros_like(c)

    # Non-escaping points remain exactly 1.
    out = np.ones(len(c), dtype=np.float32)
    alive = np.ones(len(c), dtype=bool)

    for iteration in range(1, cfg.max_iter + 1):
        if not alive.any():
            break

        za = z[alive]
        wa = ws[alive]

        radius = np.abs(za)
        angle = np.angle(za)

        # Principal-branch complex exponentiation.
        magnitude = np.power(radius, wa)
        powered = magnitude * (
            np.cos(wa * angle) + 1j * np.sin(wa * angle)
        )

        z[alive] = powered + c[alive]

        current_radius = np.abs(z)
        escaped = alive & (current_radius > cfg.bailout)

        if escaped.any():
            r = current_radius[escaped]
            w = ws[escaped]

            # Generalized continuous iteration count:
            # nu ≈ n + 1 - log(log|z|) / log(w)
            smooth = (
                iteration
                + 1.0
                - np.log(np.log(r)) / np.log(w)
            )

            out[escaped] = np.clip(
                smooth / cfg.max_iter,
                0.0,
                1.0,
            ).astype(np.float32)

            alive[escaped] = False

    return out


# ---------------------------------------------------------------------------
# SAMPLING
# ---------------------------------------------------------------------------

def anchor_values(cfg: Config) -> np.ndarray:
    """
    The lattice the sampler concentrates mass on, clipped to the configured
    range. Phase 0.0 gives the integers; phase 0.25 gives k+0.25, which is the
    control that separates "learned the task" from "learned the sampler".
    """
    phase = float(cfg.w_anchor_phase)
    first = np.ceil(cfg.w_min - phase)
    last = np.floor(cfg.w_max - phase)
    values = np.arange(first, last + 1, dtype=np.float64) + phase
    values = values[(values >= cfg.w_min) & (values <= cfg.w_max)]
    return values.astype(np.float32)


def sample_w(rng: np.random.Generator, n: int, cfg: Config) -> np.ndarray:
    """
    Mix exact integer anchors, half-integer branch-cut creatures, and
    fully continuous exponents.
    """
    if cfg.w_anchor_mode == "uniform":
        # Control condition: no lattice in the input distribution at all.
        return rng.uniform(cfg.w_min, cfg.w_max, n).astype(np.float32)

    if cfg.w_anchor_mode != "anchored":
        raise ValueError(f"unknown w_anchor_mode {cfg.w_anchor_mode!r}")

    anchors = anchor_values(cfg)
    secondary = anchors + 0.5
    secondary = secondary[(secondary >= cfg.w_min) & (secondary <= cfg.w_max)]
    if len(secondary) == 0:                      # degenerate narrow ranges
        secondary = anchors

    selector = rng.random(n)
    w = np.empty(n, dtype=np.float32)

    anchor_mask = selector < 0.45
    secondary_mask = (selector >= 0.45) & (selector < 0.70)
    continuous_mask = selector >= 0.70

    w[anchor_mask] = rng.choice(anchors, anchor_mask.sum())
    w[secondary_mask] = rng.choice(secondary, secondary_mask.sum())
    w[continuous_mask] = rng.uniform(
        cfg.w_min,
        cfg.w_max,
        continuous_mask.sum(),
    )

    # int(cfg.w_min) used to build this lattice, which put w=1 into a declared
    # range of [1.5, 6.5] -- 7.5% of samples outside the range, every one of
    # them landing on log(w)=0 in the smooth-escape formula and clipping to a
    # constant label. Never let a sampled w leave the configured interval.
    assert w.min() >= cfg.w_min - 1e-6 and w.max() <= cfg.w_max + 1e-6, (
        f"sample_w produced w outside [{cfg.w_min}, {cfg.w_max}]: "
        f"[{w.min()}, {w.max()}]"
    )
    return w


def w_distribution_summary(w: np.ndarray, labels: np.ndarray,
                           cfg: Config) -> dict:
    """
    Audit of how w was actually drawn, and what the labels look like as a
    function of it.

    This exists because a FiLM w-embedding result was replicated across five
    seeds under a frozen metric before anyone checked that 45% of training w
    sat exactly on the integers -- which turned out to be the whole effect.
    Metric discipline downstream cannot see a lattice planted upstream. Emit
    the data path alongside every checkpoint so it is evidence rather than an
    assumption underneath the evidence.
    """
    w = np.asarray(w, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    anchors = anchor_values(cfg).astype(np.float64)

    on_anchor = np.isclose(w[:, None], anchors[None, :], atol=1e-6).any(axis=1)
    on_integer = np.isclose(w, np.round(w), atol=1e-6)

    per_anchor = {}
    for a in anchors:
        m = np.isclose(w, a, atol=1e-6)
        if m.any():
            per_anchor[float(a)] = {
                "count": int(m.sum()),
                "fraction": float(m.mean()),
                "label_mean": float(labels[m].mean()),
                "label_unique": int(np.unique(labels[m]).size),
            }

    counts, edges = np.histogram(w, bins=20,
                                 range=(float(cfg.w_min), float(cfg.w_max)))
    return {
        "mode": cfg.w_anchor_mode,
        "phase": float(cfg.w_anchor_phase),
        "declared_range": [float(cfg.w_min), float(cfg.w_max)],
        "observed_range": [float(w.min()), float(w.max())],
        "out_of_range": int(((w < cfg.w_min - 1e-6) |
                             (w > cfg.w_max + 1e-6)).sum()),
        "anchor_lattice": anchors.tolist(),
        "anchor_mass": float(on_anchor.mean()),
        "integer_mass": float(on_integer.mean()),
        "labels_finite": bool(np.isfinite(labels).all()),
        "label_unique_total": int(np.unique(labels).size),
        "per_anchor": per_anchor,
        "histogram": {"counts": counts.tolist(),
                      "edges": edges.round(4).tolist()},
    }


def describe_w_distribution(w, labels, cfg: Config) -> dict:
    """Print the w-distribution audit and return it for checkpointing."""
    s = w_distribution_summary(w, labels, cfg)
    print(
        f"  w-audit: mode={s['mode']} phase={s['phase']:+.2f} | "
        f"range [{s['observed_range'][0]:.3f}, {s['observed_range'][1]:.3f}] "
        f"in [{s['declared_range'][0]}, {s['declared_range'][1]}] | "
        f"out-of-range {s['out_of_range']}"
    )
    print(
        f"  w-audit: anchor mass {s['anchor_mass'] * 100:.2f}% | "
        f"exact-integer mass {s['integer_mass'] * 100:.2f}% | "
        f"labels finite {s['labels_finite']}"
    )
    # Only judge degeneracy where there are enough samples for "few unique
    # labels" to mean anything -- a handful of incidental hits under uniform
    # sampling will always look degenerate.
    degenerate = [a for a, v in s["per_anchor"].items()
                  if v["count"] >= 256 and v["label_unique"] < v["count"] / 32]
    if degenerate:
        print(f"  w-audit: WARNING degenerate labels at w={degenerate} "
              f"(near-constant target -- check for log(w) blowup)")
    if s["anchor_mass"] > 0.05:
        print(f"  w-audit: NOTE the input distribution concentrates on "
              f"{len(s['anchor_lattice'])} lattice points. Any claim that the "
              f"model 'found' them needs a --w-anchor-mode uniform control.")
    return s


def generate_dataset(cfg: Config):
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_data

    print(f"\nGenerating {n:,} branch-cut samples...")
    start = time.time()

    n_focus = int(n * 0.70)
    n_global = n - n_focus

    # Global samples preserve the broad field.
    gx = rng.uniform(cfg.x_min, cfg.x_max, n_global).astype(np.float32)
    gy = rng.uniform(cfg.y_min, cfg.y_max, n_global).astype(np.float32)
    gw = sample_w(rng, n_global, cfg)

    # Scout a larger population for slow-escaping boundary candidates.
    scout_n = min(400_000, max(80_000, n // 2))

    sx = rng.uniform(cfg.x_min, cfg.x_max, scout_n).astype(np.float32)
    sy = rng.uniform(cfg.y_min, cfg.y_max, scout_n).astype(np.float32)
    sw = sample_w(rng, scout_n, cfg)

    scout_labels = power_escape_cpu(sx, sy, sw, cfg)

    # Slow-escaping but not fully interior.
    boundary_ids = np.flatnonzero(
        (scout_labels > 0.025) & (scout_labels < 0.999)
    )

    if len(boundary_ids) < 512:
        boundary_ids = np.flatnonzero(scout_labels > 0.01)

    if len(boundary_ids) == 0:
        boundary_ids = np.arange(scout_n)

    chosen = rng.choice(boundary_ids, n_focus, replace=True)

    cx = sx[chosen]
    cy = sy[chosen]
    cw = sw[chosen]

    # Log-uniform jitter creates fine and coarse boundary neighborhoods.
    log_radius = rng.uniform(
        np.log10(2e-5),
        np.log10(0.55),
        n_focus,
    )
    radius = np.power(10.0, log_radius).astype(np.float32)
    theta = rng.uniform(0.0, 2.0 * np.pi, n_focus).astype(np.float32)

    fx = np.clip(
        cx + radius * np.cos(theta),
        cfg.x_min,
        cfg.x_max,
    ).astype(np.float32)

    fy = np.clip(
        cy + radius * np.sin(theta),
        cfg.y_min,
        cfg.y_max,
    ).astype(np.float32)

    fw = np.clip(
        cw + rng.normal(0.0, 0.055, n_focus),
        cfg.w_min,
        cfg.w_max,
    ).astype(np.float32)

    # Re-snap some focused examples onto the anchor lattice. This is the third
    # mechanism concentrating training mass on the lattice, after the 45%
    # anchor and 25% secondary buckets in sample_w -- worth remembering before
    # concluding the network located that lattice unaided.
    if cfg.w_anchor_mode == "anchored" and cfg.w_snap_fraction > 0:
        phase = float(cfg.w_anchor_phase)
        snap = rng.random(n_focus) < cfg.w_snap_fraction
        fw[snap] = np.clip(
            np.round(fw[snap] - phase) + phase,
            cfg.w_min,
            cfg.w_max,
        ).astype(np.float32)

    x = np.concatenate([gx, fx])
    y = np.concatenate([gy, fy])
    w = np.concatenate([gw, fw])

    labels = power_escape_cpu(x, y, w, cfg)

    # log(w) sits in the denominator of the smooth-escape estimate, so w=1
    # silently produced a constant-0 label block rather than an error. Fail
    # loudly instead of training on it.
    assert np.isfinite(labels).all(), (
        f"{(~np.isfinite(labels)).sum()} non-finite labels"
    )
    assert w.min() >= cfg.w_min - 1e-6 and w.max() <= cfg.w_max + 1e-6, (
        f"w outside [{cfg.w_min}, {cfg.w_max}]: [{w.min()}, {w.max()}]"
    )

    order = rng.permutation(n)
    x = x[order]
    y = y[order]
    w = w[order]
    labels = labels[order]

    inside = labels >= 0.999

    print(
        f"Dataset complete in {time.time() - start:.1f}s | "
        f"inside={inside.mean() * 100:.2f}% | "
        f"slow boundary={((labels > 0.05) & (labels < 0.999)).mean() * 100:.2f}%"
    )
    describe_w_distribution(w, labels, cfg)

    return (
        xp.asarray(x),
        xp.asarray(y),
        xp.asarray(w),
        xp.asarray(labels),
    )


# ---------------------------------------------------------------------------
# MIXED FOURIER, DUAL-HEAD NETWORK
# ---------------------------------------------------------------------------

class BranchCutNet:
    def __init__(self, cfg: Config):
        self.cfg = cfg

        rng = np.random.default_rng(cfg.seed + 991)
        feature_count = cfg.mixed_features

        directions = rng.normal(size=(feature_count, 3))
        directions /= np.linalg.norm(
            directions,
            axis=1,
            keepdims=True,
        ).clip(1e-8)

        radii = np.power(
            2.0,
            np.linspace(0.0, 8.0, feature_count),
        )
        rng.shuffle(radii)

        # Mixed directions force x/y/w interactions into the encoding itself.
        matrix = directions * radii[:, None] * np.pi
        self.B = xp.asarray(matrix.astype(np.float32))

        # 3 normalized coordinates + 4 interaction terms + sin/cos features.
        encoded_dim = 7 + 2 * feature_count

        self.proj = Linear(encoded_dim, cfg.hidden)
        self.blocks = [
            ResBlock(cfg.hidden)
            for _ in range(cfg.blocks)
        ]
        self.norm = LayerNorm(cfg.hidden)

        self.time_head = Linear(cfg.hidden, 1)
        self.inside_head = Linear(cfg.hidden, 1)

        self.time_sigmoid = Sigmoid()
        self.inside_sigmoid = Sigmoid()

    def _normalize(self, x, y, w):
        xn = (
            2.0 * (x - self.cfg.x_min)
            / (self.cfg.x_max - self.cfg.x_min)
            - 1.0
        )
        yn = (
            2.0 * (y - self.cfg.y_min)
            / (self.cfg.y_max - self.cfg.y_min)
            - 1.0
        )
        wn = (
            2.0 * (w - self.cfg.w_min)
            / (self.cfg.w_max - self.cfg.w_min)
            - 1.0
        )
        return xn, yn, wn

    def _encode(self, x, y, w):
        xn, yn, wn = self._normalize(x, y, w)

        coords = xp.stack([xn, yn, wn], axis=1)
        phase = coords @ self.B.T

        interactions = xp.stack(
            [
                xn,
                yn,
                wn,
                xn * yn,
                xn * wn,
                yn * wn,
                xn * xn + yn * yn,
            ],
            axis=1,
        )

        return xp.concatenate(
            [
                interactions,
                xp.sin(phase),
                xp.cos(phase),
            ],
            axis=1,
        )

    def fwd(self, x, y, w):
        h = self._encode(x, y, w)
        h = self.proj.fwd(h)

        for block in self.blocks:
            h = block.fwd(h)

        h = self.norm.fwd(h)

        time_value = self.time_sigmoid.fwd(
            self.time_head.fwd(h)
        ).squeeze(-1)

        inside_probability = self.inside_sigmoid.fwd(
            self.inside_head.fwd(h)
        ).squeeze(-1)

        return time_value, inside_probability

    def bwd(self, time_gradient, inside_gradient):
        gt = self.time_sigmoid.bwd(time_gradient[:, None])
        gt = self.time_head.bwd(gt)

        gi = self.inside_sigmoid.bwd(inside_gradient[:, None])
        gi = self.inside_head.bwd(gi)

        gradient = self.norm.bwd(gt + gi)

        for block in reversed(self.blocks):
            gradient = block.bwd(gradient)

        self.proj.bwd(gradient)

    def params(self):
        parameters = list(self.proj.params())

        for block in self.blocks:
            parameters.extend(block.params())

        parameters.extend(self.norm.params())
        parameters.extend(self.time_head.params())
        parameters.extend(self.inside_head.params())

        return parameters

    def parameter_count(self):
        return sum(
            parameter.size
            for parameter, _, _ in self.params()
        )


# ---------------------------------------------------------------------------
# CHECKPOINTS
# ---------------------------------------------------------------------------

def save_checkpoint(
    net: BranchCutNet,
    optimizer: Adam,
    epoch: int,
    losses: list[float],
    cfg: Config,
    path: Path,
    w_audit: dict | None = None,
):
    payload = {
        "epoch": np.asarray(epoch, dtype=np.int32),
        "losses": np.asarray(losses, dtype=np.float32),
        "config": np.asarray(json.dumps(asdict(cfg))),
        "B": to_numpy(net.B),
        "optimizer_t": np.asarray(optimizer.t, dtype=np.int64),
    }

    # Ship the data-path audit with the weights. A checkpoint that cannot say
    # how its training w was drawn cannot support a claim about what the model
    # learned over w.
    if w_audit is not None:
        payload["w_audit"] = np.asarray(json.dumps(w_audit))

    for index, (parameter, _, _) in enumerate(net.params()):
        payload[f"parameter_{index}"] = to_numpy(parameter)

    for index, moment in enumerate(optimizer.ms):
        payload[f"moment_m_{index}"] = to_numpy(moment)

    for index, variance in enumerate(optimizer.vs):
        payload[f"moment_v_{index}"] = to_numpy(variance)

    np.savez_compressed(path, **payload)


def load_checkpoint(
    net: BranchCutNet,
    optimizer: Adam,
    path: Path,
):
    data = np.load(path, allow_pickle=False)

    net.B[...] = xp.asarray(data["B"])

    for index, (parameter, _, _) in enumerate(net.params()):
        parameter[...] = xp.asarray(data[f"parameter_{index}"])

    for index in range(len(optimizer.ms)):
        optimizer.ms[index][...] = xp.asarray(data[f"moment_m_{index}"])
        optimizer.vs[index][...] = xp.asarray(data[f"moment_v_{index}"])

    optimizer.t = int(data["optimizer_t"])
    epoch = int(data["epoch"])
    losses = data["losses"].astype(float).tolist()

    return epoch, losses


# ---------------------------------------------------------------------------
# INFERENCE
# ---------------------------------------------------------------------------

def predict_points(
    net: BranchCutNet,
    xs: np.ndarray,
    ys: np.ndarray,
    ws: np.ndarray,
    cfg: Config,
):
    xs = np.asarray(xs, dtype=np.float32).ravel()
    ys = np.asarray(ys, dtype=np.float32).ravel()
    ws = np.asarray(ws, dtype=np.float32).ravel()

    times = []
    inside = []

    for start in range(0, len(xs), cfg.vis_chunk):
        stop = start + cfg.vis_chunk

        bx = xp.asarray(xs[start:stop])
        by = xp.asarray(ys[start:stop])
        bw = xp.asarray(ws[start:stop])

        predicted_time, predicted_inside = net.fwd(bx, by, bw)

        times.append(to_numpy(predicted_time))
        inside.append(to_numpy(predicted_inside))

    return np.concatenate(times), np.concatenate(inside)


def predict_grid(
    net: BranchCutNet,
    w_value: float,
    cfg: Config,
    resolution: int,
):
    xs = np.linspace(
        cfg.x_min,
        cfg.x_max,
        resolution,
        dtype=np.float32,
    )
    ys = np.linspace(
        cfg.y_min,
        cfg.y_max,
        resolution,
        dtype=np.float32,
    )

    X, Y = np.meshgrid(xs, ys)
    W = np.full_like(X, w_value)

    time_field, inside_field = predict_points(
        net,
        X,
        Y,
        W,
        cfg,
    )

    time_field = time_field.reshape(resolution, resolution)
    inside_field = inside_field.reshape(resolution, resolution)

    # Classification head repairs saturated interior regions.
    combined = np.where(
        inside_field > 0.5,
        np.maximum(time_field, inside_field),
        time_field,
    )

    return combined, inside_field


# ---------------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------------

def save_validation(
    net: BranchCutNet,
    cfg: Config,
    output_dir: Path,
):
    test_w = [2.37, 3.14, 4.61, 5.73]
    resolution = cfg.validation_res

    figure, axes = plt.subplots(
        len(test_w),
        3,
        figsize=(13, 4.0 * len(test_w)),
        facecolor="#05050a",
    )

    metrics = []

    for row, w_value in enumerate(test_w):
        xs = np.linspace(
            cfg.x_min,
            cfg.x_max,
            resolution,
            dtype=np.float32,
        )
        ys = np.linspace(
            cfg.y_min,
            cfg.y_max,
            resolution,
            dtype=np.float32,
        )

        X, Y = np.meshgrid(xs, ys)
        W = np.full_like(X, w_value)

        truth = power_escape_cpu(
            X.ravel(),
            Y.ravel(),
            W.ravel(),
            cfg,
        ).reshape(resolution, resolution)

        prediction, inside_probability = predict_grid(
            net,
            w_value,
            cfg,
            resolution,
        )

        error = np.abs(truth - prediction)

        inside_truth = truth >= 0.999
        inside_prediction = inside_probability >= 0.5

        boundary = (truth > 0.025) & (truth < 0.999)

        mae = float(error.mean())
        boundary_mae = float(
            error[boundary].mean()
            if boundary.any()
            else np.nan
        )
        inside_accuracy = float(
            (inside_truth == inside_prediction).mean()
        )

        metrics.append(
            {
                "w": w_value,
                "mae": mae,
                "boundary_mae": boundary_mae,
                "inside_accuracy": inside_accuracy,
            }
        )

        panels = [
            (truth, "Ground truth", "twilight_shifted", 0.0, 1.0),
            (prediction, "Neural field", "twilight_shifted", 0.0, 1.0),
            (error, "Absolute error", "magma", 0.0, 0.35),
        ]

        for column, (image, title, cmap, low, high) in enumerate(panels):
            axis = axes[row, column]
            axis.imshow(
                image,
                origin="lower",
                cmap=cmap,
                vmin=low,
                vmax=high,
                aspect="equal",
            )
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_facecolor("#05050a")
            axis.set_title(
                f"{title} · w={w_value:.2f}",
                color="white",
                fontsize=10,
            )

        axes[row, 2].set_xlabel(
            f"MAE {mae:.4f} · boundary {boundary_mae:.4f} · "
            f"inside acc {inside_accuracy:.3f}",
            color="#bbbbbb",
            fontsize=9,
        )

    figure.suptitle(
        "BRANCH-CUT CATHEDRAL — HELD-OUT SLICE AUDIT",
        color="white",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout()

    panel_path = output_dir / "branch_cut_validation.png"
    figure.savefig(
        panel_path,
        dpi=160,
        bbox_inches="tight",
        facecolor="#05050a",
    )
    plt.close(figure)

    metrics_path = output_dir / "validation_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    return panel_path, metrics_path


# ---------------------------------------------------------------------------
# PSYCHEDELIC RENDERER
# ---------------------------------------------------------------------------

def soft_bloom(rgb: np.ndarray) -> np.ndarray:
    blur = (
        4.0 * rgb
        + np.roll(rgb, 1, axis=0)
        + np.roll(rgb, -1, axis=0)
        + np.roll(rgb, 1, axis=1)
        + np.roll(rgb, -1, axis=1)
        + 0.5 * np.roll(np.roll(rgb, 1, axis=0), 1, axis=1)
        + 0.5 * np.roll(np.roll(rgb, -1, axis=0), -1, axis=1)
    ) / 10.0

    highlights = np.clip(blur - 0.55, 0.0, 1.0)
    return np.clip(rgb + 0.65 * highlights, 0.0, 1.0)


def cathedral_frame(
    net: BranchCutNet,
    phase: float,
    cfg: Config,
    resolution: int,
):
    tau = 2.0 * np.pi

    # Seamless 2 → 6 → 2 exponent loop.
    w_value = cfg.w_min + (
        cfg.w_max - cfg.w_min
    ) * (0.5 - 0.5 * np.cos(tau * phase))

    xs = np.linspace(
        cfg.x_min,
        cfg.x_max,
        resolution,
        dtype=np.float32,
    )
    ys = np.linspace(
        cfg.y_min,
        cfg.y_max,
        resolution,
        dtype=np.float32,
    )

    X, Y = np.meshgrid(xs, ys)

    center_x = -0.62
    dx = X - center_x
    dy = Y

    radius = np.sqrt(dx * dx + dy * dy)
    angle = np.arctan2(dy, dx)

    # Continuous kaleidoscopic fold. Mathematicians may leave quietly.
    symmetry = max(2.0, w_value)
    sector = tau / symmetry

    folded = np.abs(
        np.mod(angle + 0.5 * sector, sector)
        - 0.5 * sector
    )

    spin = 0.18 * np.sin(tau * phase) + tau * phase / symmetry

    warped_x = center_x + radius * np.cos(folded + spin)
    warped_y = radius * np.sin(folded + spin)

    W = np.full_like(warped_x, w_value, dtype=np.float32)
    epsilon = 1e-7

    # Neural-gradient feedback warp.
    for feedback_index in range(cfg.art_feedback):
        field, confidence = predict_points(
            net,
            warped_x,
            warped_y,
            W,
            cfg,
        )

        field = field.reshape(resolution, resolution)
        confidence = confidence.reshape(resolution, resolution)

        combined = np.where(
            confidence > 0.5,
            np.maximum(field, confidence),
            field,
        )

        gradient_y, gradient_x = np.gradient(combined)
        magnitude = np.sqrt(
            gradient_x * gradient_x
            + gradient_y * gradient_y
            + epsilon
        )

        tangent_x = -gradient_y / magnitude
        tangent_y = gradient_x / magnitude

        pulse = (
            0.035
            / (feedback_index + 1)
            * np.tanh(28.0 * magnitude)
        )

        spiral = (
            tau * phase
            + 2.4 * angle
            + 6.0 * combined
            + feedback_index * 1.71
        )

        warped_x += pulse * (
            0.62 * tangent_x + 0.38 * np.cos(spiral)
        )
        warped_y += pulse * (
            0.62 * tangent_y + 0.38 * np.sin(spiral)
        )

        warped_x = np.clip(warped_x, cfg.x_min, cfg.x_max)
        warped_y = np.clip(warped_y, cfg.y_min, cfg.y_max)

    field, confidence = predict_points(
        net,
        warped_x,
        warped_y,
        W,
        cfg,
    )

    field = field.reshape(resolution, resolution)
    confidence = confidence.reshape(resolution, resolution)

    combined = np.where(
        confidence > 0.5,
        np.maximum(field, confidence),
        field,
    )

    gradient_y, gradient_x = np.gradient(combined)
    edge = np.tanh(
        38.0 * np.sqrt(
            gradient_x * gradient_x
            + gradient_y * gradient_y
            + epsilon
        )
    )

    # Maximum near classification uncertainty p=0.5.
    uncertainty = 4.0 * confidence * (1.0 - confidence)

    # Explicit glow along the principal-branch seam.
    branch_distance = np.abs(np.pi - np.abs(angle))
    seam = np.exp(
        -np.square(branch_distance / 0.055)
    )

    interference = (
        0.5
        + 0.5 * np.sin(
            58.0 * combined
            + symmetry * 3.0 * angle
            - 6.0 * radius
            + tau * phase
        )
    )

    hue = np.mod(
        0.61
        + 1.75 * combined
        + 0.18 * edge
        + 0.11 * uncertainty
        + 0.09 * seam
        + 0.14 * np.sin(3.0 * angle - tau * phase),
        1.0,
    )

    saturation = np.clip(
        0.58
        + 0.28 * interference
        + 0.22 * edge
        + 0.15 * uncertainty,
        0.0,
        1.0,
    )

    value = np.clip(
        0.015
        + 0.72 * np.power(combined, 0.38)
        + 0.42 * edge * interference
        + 0.22 * seam
        + 0.12 * uncertainty,
        0.0,
        1.0,
    )

    hsv = np.stack([hue, saturation, value], axis=-1)
    rgb = hsv_to_rgb(hsv)

    # Chromatic aberration, because restraint has left the building.
    red = np.roll(rgb[..., 0], 2, axis=1)
    green = rgb[..., 1]
    blue = np.roll(rgb[..., 2], -2, axis=1)

    rgb = np.stack([red, green, blue], axis=-1)
    rgb = soft_bloom(rgb)
    rgb = np.power(np.clip(rgb, 0.0, 1.0), 0.84)

    # Very subtle scan-wave modulation.
    scan = (
        0.965
        + 0.035
        * np.sin(
            np.arange(resolution)[:, None] * 0.31
            + tau * phase
        )
    )

    return np.clip(rgb * scan[..., None], 0.0, 1.0)


def save_cathedral_art(
    net: BranchCutNet,
    cfg: Config,
    output_dir: Path,
):
    poster = cathedral_frame(
        net,
        phase=0.271828,
        cfg=cfg,
        resolution=cfg.art_res,
    )

    poster_path = output_dir / "BRANCH_CUT_CATHEDRAL.png"
    plt.imsave(poster_path, poster)

    if Image is None:
        print("Pillow unavailable; animation skipped.")
        return poster_path, None

    frames = []

    print(f"\nRendering {cfg.art_frames} cathedral frames...")

    for frame_index in range(cfg.art_frames):
        phase = frame_index / cfg.art_frames

        rgb = cathedral_frame(
            net,
            phase,
            cfg,
            cfg.art_res,
        )

        frame = Image.fromarray(
            np.uint8(rgb * 255.0),
            mode="RGB",
        )
        frames.append(frame)

        if (
            frame_index == 0
            or (frame_index + 1) % 12 == 0
            or frame_index + 1 == cfg.art_frames
        ):
            print(
                f"  frame {frame_index + 1}/{cfg.art_frames}"
            )

    animation_path = output_dir / "BRANCH_CUT_CATHEDRAL.gif"

    frames[0].save(
        animation_path,
        save_all=True,
        append_images=frames[1:],
        duration=cfg.frame_ms,
        loop=0,
        disposal=2,
        optimize=False,
    )

    return poster_path, animation_path


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------

def cosine_lr(epoch: int, total: int, cfg: Config) -> float:
    return cfg.lr_min + 0.5 * (
        cfg.lr - cfg.lr_min
    ) * (
        1.0 + np.cos(np.pi * epoch / total)
    )


def train(
    cfg: Config,
    resume_path: Path | None = None,
):
    output_dir = Path(cfg.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    xp.random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    X, Y, W, labels = generate_dataset(cfg)
    # Capture the w-audit so every checkpoint this run writes carries the data
    # path with it, matching train_small.py and train_film.py.
    w_audit = w_distribution_summary(to_numpy(W), to_numpy(labels), cfg)

    net = BranchCutNet(cfg)
    optimizer = Adam(net.params(), lr=cfg.lr)

    start_epoch = 0
    losses: list[float] = []

    if resume_path is not None:
        start_epoch, losses = load_checkpoint(
            net,
            optimizer,
            resume_path,
        )
        print(f"Resumed from epoch {start_epoch}.")

    print(f"\nParameters: {net.parameter_count():,}")
    print(
        f"Architecture: mixed Fourier → {cfg.blocks} residual blocks "
        f"→ escape head + interior head"
    )

    label_numpy = to_numpy(labels)
    positive_fraction = float((label_numpy >= 0.999).mean())

    positive_weight = float(
        np.clip(
            (1.0 - positive_fraction)
            / max(positive_fraction, 1e-6),
            1.0,
            20.0,
        )
    )

    print(
        f"Interior class weight: {positive_weight:.2f}"
    )

    training_start = time.time()
    n = len(label_numpy)

    for epoch in range(start_epoch + 1, cfg.epochs + 1):
        optimizer.lr = cosine_lr(epoch, cfg.epochs, cfg)

        permutation = xp.random.permutation(n)
        X = X[permutation]
        Y = Y[permutation]
        W = W[permutation]
        labels = labels[permutation]

        epoch_loss = 0.0
        batch_count = 0

        for start in range(0, n, cfg.batch_size):
            stop = min(start + cfg.batch_size, n)
            batch_size = stop - start

            x = X[start:stop]
            y = Y[start:stop]
            w = W[start:stop]
            target = labels[start:stop]

            predicted_time, predicted_inside = net.fwd(x, y, w)

            inside_target = (
                target >= xp.float32(0.999)
            ).astype(xp.float32)

            # Emphasize late escapes, boundary points, and true interior.
            boundary_emphasis = (
                xp.float32(4.0)
                * xp.exp(
                    -xp.square(
                        (target - xp.float32(0.24))
                        / xp.float32(0.22)
                    )
                )
                * (xp.float32(1.0) - inside_target)
            )

            time_weight = (
                xp.float32(1.0)
                + xp.float32(5.0) * xp.sqrt(target)
                + boundary_emphasis
                + xp.float32(3.0) * inside_target
            )

            difference = predicted_time - target

            time_loss = (
                time_weight * difference * difference
            ).mean()

            probability = predicted_inside.clip(
                xp.float32(1e-5),
                xp.float32(1.0 - 1e-5),
            )

            class_weight = (
                xp.float32(1.0)
                + inside_target
                * xp.float32(positive_weight - 1.0)
            )

            classification_loss = -(
                class_weight
                * (
                    inside_target * xp.log(probability)
                    + (
                        xp.float32(1.0) - inside_target
                    )
                    * xp.log(
                        xp.float32(1.0) - probability
                    )
                )
            ).mean()

            total_loss = (
                time_loss
                + xp.float32(cfg.bce_weight)
                * classification_loss
            )

            time_gradient = (
                xp.float32(2.0)
                * time_weight
                * difference
                / xp.float32(batch_size)
            )

            inside_gradient = (
                xp.float32(cfg.bce_weight)
                * class_weight
                * (
                    probability - inside_target
                )
                / (
                    probability
                    * (
                        xp.float32(1.0) - probability
                    )
                    + xp.float32(1e-6)
                )
                / xp.float32(batch_size)
            )

            net.bwd(
                time_gradient,
                inside_gradient,
            )
            optimizer.step()

            epoch_loss += float(to_numpy(total_loss))
            batch_count += 1

        average_loss = epoch_loss / batch_count
        losses.append(average_loss)

        elapsed = time.time() - training_start
        eta = (
            elapsed
            / max(epoch - start_epoch, 1)
            * (cfg.epochs - epoch)
        )

        print(
            f"[{epoch:03d}/{cfg.epochs}] "
            f"loss={average_loss:.6f} "
            f"lr={optimizer.lr:.2e} "
            f"elapsed={elapsed:.0f}s "
            f"ETA={eta:.0f}s"
        )

        if (
            epoch % cfg.checkpoint_every == 0
            or epoch == cfg.epochs
        ):
            checkpoint_path = (
                output_dir / f"checkpoint_{epoch:04d}.npz"
            )

            save_checkpoint(
                net,
                optimizer,
                epoch,
                losses,
                cfg,
                checkpoint_path,
                w_audit=w_audit,
            )

            latest_path = output_dir / "checkpoint_latest.npz"
            save_checkpoint(
                net,
                optimizer,
                epoch,
                losses,
                cfg,
                latest_path,
                w_audit=w_audit,
            )

    return net, optimizer, losses, output_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train and render the Branch-Cut Cathedral."
    )

    parser.add_argument(
        "--preset",
        choices=["smoke", "cpu", "gpu", "absurd"],
        default="gpu" if GPU else "cpu",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--n-data", type=int)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--art-res", type=int)
    parser.add_argument("--out", type=str)
    parser.add_argument(
        "--skip-art",
        action="store_true",
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()
    cfg = preset(arguments.preset)

    if arguments.epochs is not None:
        cfg.epochs = arguments.epochs

    if arguments.n_data is not None:
        cfg.n_data = arguments.n_data

    if arguments.frames is not None:
        cfg.art_frames = arguments.frames

    if arguments.art_res is not None:
        cfg.art_res = arguments.art_res

    if arguments.out is not None:
        cfg.out_dir = arguments.out

    print("\nBRANCH-CUT CATHEDRAL")
    print(json.dumps(asdict(cfg), indent=2))

    net, optimizer, losses, output_dir = train(
        cfg,
        arguments.resume,
    )

    validation_path, metrics_path = save_validation(
        net,
        cfg,
        output_dir,
    )

    print(f"\nValidation panel: {validation_path}")
    print(f"Validation metrics: {metrics_path}")

    if not arguments.skip_art:
        poster_path, animation_path = save_cathedral_art(
            net,
            cfg,
            output_dir,
        )

        print(f"Cathedral poster: {poster_path}")

        if animation_path is not None:
            print(f"Cathedral animation: {animation_path}")

    print("\nThe branch cut has been promoted from bug to clergy.")


if __name__ == "__main__":
    main()