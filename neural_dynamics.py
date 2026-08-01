#!/usr/bin/env python3
"""
NEURAL DYNAMICS — learn the map, not the statistic.

Idea #3 of the roadmap, and the one worth chasing.

Every model so far predicts ESCAPE TIME: a summary statistic of the orbit.
This one learns the ORBIT STEP itself, and then iterates the learned
function in place of the analytic one.

Specifically it learns

    P(z, w)  ~=  z ** w          (principal branch)

and keeps the "+ c" exact. That split matters:

  * the additive c-dependence is trivial and free, so no capacity is
    wasted relearning addition;
  * every deviation from true dynamics is then attributable to the
    learned power map alone;
  * and the object you get at the end is a genuine dynamical system,

        z_{n+1} = P_theta(z_n, w) + c,

    which has its OWN Mandelbrot set, its OWN Julia sets, and its own
    bifurcation structure. None of these are the real ones.

The difference image between the true M-set and the neural M-set is not
an error plot. It is a map of what the network believes about dynamics --
the black box's misconceptions rendered as geography.

Representation notes:
  * |z^w| reaches 4**6 ~= 4096 over the training box, so targets live in
    asinh space (signed, invertible, log-like in the tail, linear near 0).
  * training pairs are drawn from REAL ORBITS, not uniform z. Uniform
    sampling puts almost all mass off the attractor and the iterated map
    walks straight off-manifold on step one.

Usage:
    python3 neural_dynamics.py --train --epochs 30
    python3 neural_dynamics.py --render --checkpoint dyn/map.npz
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import MandelbrotLeary as base

xp = base.xp
to_numpy = base.to_numpy
Linear = base.Linear
LayerNorm = base.LayerNorm
ResBlock = base.ResBlock
Adam = base.Adam


LOGMAG_SCALE = 8.0     # w*log|z| spans roughly [-27, 8] over the training box
N_HARMONICS = 10       # angular harmonics cos(k*theta), sin(k*theta)


def to_targets(zx, zy, w):
    """
    Decoupled target: (log|z^w|, cos(w*theta), sin(w*theta)).

    Predicting Re/Im directly couples a 4000x dynamic range to a 12*pi
    angular winding, and the regression collapses -- measured R^2 was
    0.25. Splitting magnitude from direction fixes the conditioning
    without handing the network the closed form: it still has to discover
    that log|z^w| = w*log|z| and that the direction winds at rate w.
    """
    radius = np.sqrt(zx * zx + zy * zy)
    angle = np.arctan2(zy, zx)
    log_magnitude = w * np.log(np.clip(radius, 1e-12, None))
    phi = w * angle
    return np.stack([
        log_magnitude / LOGMAG_SCALE,
        np.cos(phi),
        np.sin(phi),
    ], axis=1).astype(np.float32)


def from_targets(prediction):
    """Invert to Cartesian z^w."""
    log_magnitude = np.clip(prediction[:, 0] * LOGMAG_SCALE, -60.0, 30.0)
    magnitude = np.exp(log_magnitude)
    cos_phi = prediction[:, 1]
    sin_phi = prediction[:, 2]
    norm = np.sqrt(cos_phi ** 2 + sin_phi ** 2) + 1e-8
    return magnitude * cos_phi / norm, magnitude * sin_phi / norm


# ---------------------------------------------------------------------------
# THE LEARNED POWER MAP
# ---------------------------------------------------------------------------

class NeuralMap:
    """
    (zx, zy, w) -> (Re z^w, Im z^w) in asinh space.

    Linear output head, no sigmoid: this is a regression onto an unbounded
    quantity, not a normalized field.
    """

    def __init__(self, hidden=192, blocks=3, features=64, bailout=4.0, seed=17,
                 harmonic_mode="integer", harmonic_offset=0.25):
        self.hidden = hidden
        self.blocks_n = blocks
        self.features = features
        self.bailout = bailout
        self.harmonic_mode = harmonic_mode
        self.harmonic_offset = harmonic_offset

        rng = np.random.default_rng(seed)
        directions = rng.normal(size=(features, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True).clip(1e-8)
        radii = np.power(2.0, np.linspace(0.0, 5.0, features))
        rng.shuffle(radii)
        self.B = xp.asarray((directions * radii[:, None] * np.pi).astype(np.float32))

        # ARCHITECTURAL LATTICE WARNING. The target direction is
        # (cos(w*theta), sin(w*theta)). With integer harmonics k = 1..N, at
        # integer w = k that target is verbatim an input feature -- and the
        # k-th harmonic is EXACTLY invariant under the C_k rotation the
        # symmetry probe applies, since cos(k*(theta + 2pi/k)) = cos(k*theta).
        # So a near-zero symmetry defect at integer w can come from the basis
        # rather than from anything learned. `harmonic_mode` is the
        # intervention: shift the lattice off the integers, randomize it at
        # matched bandwidth, or remove it and keep only raw theta.
        harmonics = self._build_harmonics(rng)
        self.harmonics = xp.asarray(harmonics.astype(np.float32))
        n_harm = len(harmonics)

        # zx, zy, w, r, log r, theta/pi  + angular harmonics + Fourier
        encoded = 6 + 2 * n_harm + 2 * features

        self.proj = Linear(encoded, hidden)
        self.blocks = [ResBlock(hidden) for _ in range(blocks)]
        self.norm = LayerNorm(hidden)
        self.head = Linear(hidden, 3)

    def _build_harmonics(self, rng) -> np.ndarray:
        """The angular-frequency lattice. See the warning in __init__."""
        mode = self.harmonic_mode
        if mode == "integer":                       # original: 1, 2, ..., N
            return np.arange(1, N_HARMONICS + 1, dtype=np.float64)
        if mode == "shifted":                       # 1.25, 2.25, ..., N.25
            return np.arange(1, N_HARMONICS + 1, dtype=np.float64) \
                + float(self.harmonic_offset)
        if mode == "random":                        # matched bandwidth, no lattice
            return np.sort(rng.uniform(1.0, N_HARMONICS + 1.0, N_HARMONICS))
        if mode == "none":                          # raw theta only
            return np.zeros(0, dtype=np.float64)
        raise ValueError(f"unknown harmonic_mode {mode!r}")

    def _encode(self, zx, zy, w):
        zx = xp.asarray(zx, dtype=xp.float32)
        zy = xp.asarray(zy, dtype=xp.float32)
        w = xp.asarray(w, dtype=xp.float32)
        zn = zx / self.bailout
        yn = zy / self.bailout
        wn = (w - 4.0) / 2.0

        radius = xp.sqrt(zn * zn + yn * yn + xp.float32(1e-12))
        angle = xp.arctan2(yn, zn)
        log_radius = xp.log(radius + xp.float32(1e-12)) / xp.float32(3.0)

        coords = xp.stack([zn, yn, wn], axis=1)
        phase = coords @ self.B.T

        # Angular harmonics: the winding term cos(w*theta) is trivial in
        # theta and vicious in Cartesian Fourier features. Give it a basis
        # it can actually use -- the network still has to learn the rate.
        harmonic_phase = angle[:, None] * self.harmonics

        explicit = xp.stack([
            zn, yn, wn, radius, log_radius, angle / xp.float32(np.pi),
        ], axis=1)

        return xp.concatenate([
            explicit,
            xp.cos(harmonic_phase), xp.sin(harmonic_phase),
            xp.sin(phase), xp.cos(phase),
        ], axis=1)

    def fwd(self, zx, zy, w):
        h = self._encode(zx, zy, w)
        h = self.proj.fwd(h)
        for block in self.blocks:
            h = block.fwd(h)
        h = self.norm.fwd(h)
        return self.head.fwd(h)

    def bwd(self, gradient):
        g = self.head.bwd(gradient)
        g = self.norm.bwd(g)
        for block in reversed(self.blocks):
            g = block.bwd(g)
        self.proj.bwd(g)

    def params(self):
        p = list(self.proj.params())
        for block in self.blocks:
            p.extend(block.params())
        p.extend(self.norm.params())
        p.extend(self.head.params())
        return p

    def parameter_count(self):
        return sum(p.size for p, _, _ in self.params())

    # --- inference -------------------------------------------------------

    def power(self, zx, zy, w, chunk=65536):
        """Learned z**w, decompressed back to world scale."""
        zx = np.asarray(zx, np.float32).ravel()
        zy = np.asarray(zy, np.float32).ravel()
        w = np.asarray(w, np.float32).ravel()

        out_re, out_im = [], []
        for start in range(0, len(zx), chunk):
            stop = start + chunk
            prediction = to_numpy(self.fwd(
                xp.asarray(zx[start:stop]),
                xp.asarray(zy[start:stop]),
                xp.asarray(w[start:stop]),
            ))
            out_re.append(prediction)

        return from_targets(np.concatenate(out_re, axis=0))


# ---------------------------------------------------------------------------
# ORBIT-SAMPLED TRAINING DATA
# ---------------------------------------------------------------------------

def true_power(zx, zy, w):
    radius = np.sqrt(zx * zx + zy * zy)
    angle = np.arctan2(zy, zx)
    magnitude = np.power(np.clip(radius, 1e-12, None), w)
    return magnitude * np.cos(w * angle), magnitude * np.sin(w * angle)


def orbit_dataset(n_orbits=6000, max_steps=48, bailout=4.0,
                  w_min=2.0, w_max=6.0, seed=5, w_fixed=None):
    """
    Walk real orbits and record every (z_n, w) the dynamics actually
    visits. This is the distribution the iterated network will be asked
    about at inference, so it is the distribution it must be trained on.
    """
    rng = np.random.default_rng(seed)

    cx = rng.uniform(-2.5, 1.0, n_orbits).astype(np.float32)
    cy = rng.uniform(-1.35, 1.35, n_orbits).astype(np.float32)
    w = (np.full(n_orbits, w_fixed, np.float32) if w_fixed is not None
         else rng.uniform(w_min, w_max, n_orbits).astype(np.float32))

    zx = np.zeros(n_orbits, np.float32)
    zy = np.zeros(n_orbits, np.float32)
    alive = np.ones(n_orbits, bool)

    samples_zx, samples_zy, samples_w = [], [], []

    for _ in range(max_steps):
        if not alive.any():
            break
        samples_zx.append(zx[alive].copy())
        samples_zy.append(zy[alive].copy())
        samples_w.append(w[alive].copy())

        px, py = true_power(zx[alive], zy[alive], w[alive])
        zx[alive] = px + cx[alive]
        zy[alive] = py + cy[alive]

        alive[alive] = (zx[alive] ** 2 + zy[alive] ** 2) < bailout ** 2

    zx_all = np.concatenate(samples_zx)
    zy_all = np.concatenate(samples_zy)
    w_all = np.concatenate(samples_w)

    # Blend in a uniform disc so the map stays sane slightly off-orbit.
    n_extra = len(zx_all) // 3
    radius = bailout * np.sqrt(rng.random(n_extra)).astype(np.float32)
    theta = rng.uniform(0, 2 * np.pi, n_extra).astype(np.float32)
    zx_all = np.concatenate([zx_all, radius * np.cos(theta)])
    zy_all = np.concatenate([zy_all, radius * np.sin(theta)])
    extra_w = (np.full(n_extra, w_fixed, np.float32) if w_fixed is not None
               else rng.uniform(w_min, w_max, n_extra).astype(np.float32))
    w_all = np.concatenate([w_all, extra_w])

    order = rng.permutation(len(zx_all))
    zx_all, zy_all, w_all = zx_all[order], zy_all[order], w_all[order]

    targets = to_targets(zx_all, zy_all, w_all)

    return (zx_all.astype(np.float32), zy_all.astype(np.float32),
            w_all.astype(np.float32), targets.astype(np.float32))


# ---------------------------------------------------------------------------
# ITERATING THE NETWORK
# ---------------------------------------------------------------------------

def neural_escape(net: NeuralMap, cx, cy, w_value, max_iter=64,
                  bailout=4.0, shape=None):
    """
    The whole point. Replace z**w with the network and run the classic
    escape-time algorithm on the result.

    Escaped points are dropped each step, exactly as in the analytic
    version, so cost falls off fast.
    """
    cx = np.asarray(cx, np.float32).ravel()
    cy = np.asarray(cy, np.float32).ravel()
    n = len(cx)

    zx = np.zeros(n, np.float32)
    zy = np.zeros(n, np.float32)
    escape_iter = np.full(n, max_iter, np.float32)
    alive = np.ones(n, bool)

    for iteration in range(1, max_iter + 1):
        if not alive.any():
            break

        index = np.flatnonzero(alive)
        px, py = net.power(zx[index], zy[index],
                           np.full(len(index), w_value, np.float32))

        zx[index] = np.clip(px + cx[index], -1e6, 1e6)
        zy[index] = np.clip(py + cy[index], -1e6, 1e6)

        magnitude = zx[index] ** 2 + zy[index] ** 2
        escaped = magnitude > bailout ** 2
        escape_iter[index[escaped]] = iteration
        alive[index[escaped]] = False

        bad = ~np.isfinite(magnitude)
        escape_iter[index[bad]] = iteration
        alive[index[bad]] = False

    field = escape_iter / max_iter
    return field.reshape(shape) if shape else field


def true_escape(cx, cy, w_value, max_iter=64, bailout=4.0, shape=None):
    cx = np.asarray(cx, np.float32).ravel()
    cy = np.asarray(cy, np.float32).ravel()
    n = len(cx)
    zx = np.zeros(n, np.float32); zy = np.zeros(n, np.float32)
    escape_iter = np.full(n, max_iter, np.float32)
    alive = np.ones(n, bool)

    for iteration in range(1, max_iter + 1):
        if not alive.any():
            break
        index = np.flatnonzero(alive)
        px, py = true_power(zx[index], zy[index],
                            np.full(len(index), w_value, np.float32))
        zx[index] = px + cx[index]
        zy[index] = py + cy[index]
        magnitude = zx[index] ** 2 + zy[index] ** 2
        escaped = magnitude > bailout ** 2
        escape_iter[index[escaped]] = iteration
        alive[index[escaped]] = False

    field = escape_iter / max_iter
    return field.reshape(shape) if shape else field


def neural_julia(net: NeuralMap, c_value, w_value, resolution=192,
                 max_iter=48, bailout=4.0, extent=1.8):
    """Julia set of the LEARNED map for a fixed c."""
    axis = np.linspace(-extent, extent, resolution, dtype=np.float32)
    ZX, ZY = np.meshgrid(axis, axis)
    zx = ZX.ravel().copy(); zy = ZY.ravel().copy()

    escape_iter = np.full(zx.size, max_iter, np.float32)
    alive = np.ones(zx.size, bool)

    for iteration in range(1, max_iter + 1):
        if not alive.any():
            break
        index = np.flatnonzero(alive)
        px, py = net.power(zx[index], zy[index],
                           np.full(len(index), w_value, np.float32))
        zx[index] = np.clip(px + c_value.real, -1e6, 1e6)
        zy[index] = np.clip(py + c_value.imag, -1e6, 1e6)
        magnitude = zx[index] ** 2 + zy[index] ** 2
        gone = (magnitude > bailout ** 2) | ~np.isfinite(magnitude)
        escape_iter[index[gone]] = iteration
        alive[index[gone]] = False

    return (escape_iter / max_iter).reshape(resolution, resolution)


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------

def train(epochs=30, batch_size=4096, hidden=192, blocks=3, features=64,
          lr=6e-4, out_dir=Path("./dyn"), n_orbits=6000, w_fixed=None,
          seed=17, harmonic_mode="integer", harmonic_offset=0.25):
    out_dir.mkdir(parents=True, exist_ok=True)

    # One seed controls orbit sampling, weight init, and epoch shuffling, so
    # paired comparisons across architectures differ only in the architecture.
    np.random.seed(seed)
    shuffle_rng = np.random.default_rng(seed + 7919)

    print("Sampling real orbits...")
    zx, zy, w, targets = orbit_dataset(n_orbits=n_orbits, w_fixed=w_fixed,
                                       seed=seed)
    print(f"  {len(zx):,} orbit-visited states"
          + (f"  (w fixed at {w_fixed})" if w_fixed else ""))
    print(f"  target variance {targets.var():.4f}")

    net = NeuralMap(hidden=hidden, blocks=blocks, features=features, seed=seed,
                    harmonic_mode=harmonic_mode,
                    harmonic_offset=harmonic_offset)
    optimizer = Adam(net.params(), lr=lr)
    harm = to_numpy(net.harmonics)
    print(f"  {net.parameter_count():,} parameters")
    print(f"  seed {seed} | angular basis '{harmonic_mode}': "
          f"{'(none - raw theta only)' if harm.size == 0 else np.round(harm, 3).tolist()}")
    if harmonic_mode == "integer":
        print("  NOTE integer harmonics make the C_w-invariant feature "
              "available at integer w by construction;")
        print("       a symmetry-defect dip there needs a shifted/random/none "
              "control to interpret.\n")
    else:
        print()

    n = len(zx)
    start_time = time.time()
    losses = []

    for epoch in range(1, epochs + 1):
        optimizer.lr = 1e-6 + 0.5 * (lr - 1e-6) * (1 + np.cos(np.pi * epoch / epochs))
        order = shuffle_rng.permutation(n)
        zx, zy, w, targets = zx[order], zy[order], w[order], targets[order]

        total = 0.0
        count = 0

        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            size = stop - start

            prediction = net.fwd(
                xp.asarray(zx[start:stop]),
                xp.asarray(zy[start:stop]),
                xp.asarray(w[start:stop]),
            )
            target = xp.asarray(targets[start:stop])
            difference = prediction - target
            loss = (difference * difference).mean()

            net.bwd(xp.float32(2.0) * difference / xp.float32(size * 3))
            optimizer.step()

            total += float(to_numpy(loss))  # to_numpy: CuPy blocks implicit sync
            count += 1

        losses.append(total / count)
        if epoch % 5 == 0 or epoch == 1:
            r2 = 1.0 - (total / count) / float(targets.var())
            print(f"[{epoch:3d}/{epochs}] loss={total / count:.4e} "
                  f"R2={r2:.5f}  {time.time() - start_time:.0f}s", flush=True)

    payload = {
        "B": to_numpy(net.B),
        "meta": np.asarray(json.dumps({
            "hidden": hidden, "blocks": blocks,
            "features": features, "bailout": net.bailout,
            "harmonic_mode": net.harmonic_mode,
            "harmonic_offset": net.harmonic_offset,
            "seed": seed,
        })),
        # The angular basis is part of the data path for any claim about
        # integer structure -- ship it with the weights.
        "harmonics": to_numpy(net.harmonics),
        "losses": np.asarray(losses, np.float32),
    }
    for index, (parameter, _, _) in enumerate(net.params()):
        payload[f"parameter_{index}"] = to_numpy(parameter)

    np.savez_compressed(out_dir / "map.npz", **payload)
    print(f"\nSaved {out_dir}/map.npz")
    return net


def _resolve_checkpoint(path, pattern="*.npz"):
    """
    Fail with something actionable instead of a bare FileNotFoundError.
    Defaults point at the example output directory, which is almost never
    where the user actually trained.
    """
    from pathlib import Path as _Path
    path = _Path(path)
    if path.exists():
        return path

    noise = ("site-packages", ".cache", "node_modules", "envs",
             "miniconda", "anaconda", ".git", "OneDrive/Temp")
    found = []
    try:
        for candidate in sorted(_Path(".").rglob(pattern)):
            text = str(candidate).replace("\\\\", "/")
            if not any(bad in text for bad in noise):
                found.append(candidate)
    except (OSError, PermissionError):
        pass

    message = [f"No checkpoint at: {path}"]
    if found:
        message.append("")
        message.append("Checkpoints found nearby -- pass one with --checkpoint:")
        for candidate in found[:12]:
            size = candidate.stat().st_size / 1e6
            message.append(f"    {candidate}   ({size:.1f} MB)")
    else:
        message.append("")
        message.append("No .npz files found nearby. Train one first, or check")
        message.append("the --out directory you used (map.npz is written beside")
        message.append("the rendered PNGs).")
    raise FileNotFoundError("\n".join(message))


def load_map(path):
    path = _resolve_checkpoint(path)
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    net = NeuralMap(hidden=meta["hidden"], blocks=meta["blocks"],
                    features=meta["features"], bailout=meta["bailout"],
                    harmonic_mode=meta.get("harmonic_mode", "integer"),
                    harmonic_offset=meta.get("harmonic_offset", 0.25))
    if "harmonics" in data.files:
        net.harmonics = xp.asarray(data["harmonics"])
    net.B[...] = xp.asarray(data["B"])
    for index, (parameter, _, _) in enumerate(net.params()):
        parameter[...] = xp.asarray(data[f"parameter_{index}"])
    return net


# ---------------------------------------------------------------------------
# RENDERING THE NETWORK'S OWN MANDELBROT SET
# ---------------------------------------------------------------------------

def render_comparison(net, out_path, w_value=2.0, resolution=200, max_iter=64):
    xs = np.linspace(-2.5, 1.0, resolution, dtype=np.float32)
    ys = np.linspace(-1.35, 1.35, resolution, dtype=np.float32)
    CX, CY = np.meshgrid(xs, ys)
    shape = (resolution, resolution)

    print(f"  iterating the LEARNED map, w={w_value}...", flush=True)
    neural = neural_escape(net, CX, CY, w_value, max_iter=max_iter, shape=shape)
    truth = true_escape(CX, CY, w_value, max_iter=max_iter, shape=shape)

    neural_interior = neural >= 0.999
    truth_interior = truth >= 0.999
    disagreement = neural_interior ^ truth_interior

    iou = float((neural_interior & truth_interior).sum()
                / max((neural_interior | truth_interior).sum(), 1))

    figure, axes = plt.subplots(1, 4, figsize=(19, 4.6), facecolor="#05050a")

    axes[0].imshow(truth, origin="lower", cmap="twilight_shifted", vmin=0, vmax=1)
    axes[0].set_title(f"true dynamics  z^{w_value:g}+c", color="white", fontsize=10)

    axes[1].imshow(neural, origin="lower", cmap="twilight_shifted", vmin=0, vmax=1)
    axes[1].set_title("LEARNED dynamics  P(z)+c", color="white", fontsize=10)

    axes[2].imshow(np.abs(truth - neural), origin="lower", cmap="magma",
                   vmin=0, vmax=0.5)
    axes[2].set_title("escape-time difference", color="white", fontsize=10)

    axes[3].imshow(disagreement, origin="lower", cmap="inferno")
    axes[3].set_title(f"set disagreement · IoU {iou:.3f}", color="white", fontsize=10)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("#05050a")

    figure.suptitle(
        "THE MANDELBROT SET OF A NEURAL NETWORK — "
        "not an error plot, a different dynamical system",
        color="white", fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    figure.savefig(out_path, dpi=150, facecolor="#05050a", bbox_inches="tight")
    plt.close(figure)

    return {"iou": iou,
            "mae": float(np.abs(truth - neural).mean()),
            "disagreement_fraction": float(disagreement.mean())}


def render_julia_row(net, out_path, w_value=2.0, resolution=180):
    constants = [-0.4 + 0.6j, -0.8 + 0.156j, 0.285 + 0.01j, -0.70176 - 0.3842j]
    figure, axes = plt.subplots(2, len(constants),
                                figsize=(3.4 * len(constants), 7),
                                facecolor="#05050a")

    for column, c_value in enumerate(constants):
        neural = neural_julia(net, c_value, w_value, resolution=resolution)

        axis = np.linspace(-1.8, 1.8, resolution, dtype=np.float32)
        ZX, ZY = np.meshgrid(axis, axis)
        zx = ZX.ravel().copy(); zy = ZY.ravel().copy()
        escape_iter = np.full(zx.size, 48, np.float32)
        alive = np.ones(zx.size, bool)
        for iteration in range(1, 49):
            if not alive.any():
                break
            index = np.flatnonzero(alive)
            px, py = true_power(zx[index], zy[index],
                                np.full(len(index), w_value, np.float32))
            zx[index] = px + c_value.real
            zy[index] = py + c_value.imag
            magnitude = zx[index] ** 2 + zy[index] ** 2
            gone = magnitude > 16.0
            escape_iter[index[gone]] = iteration
            alive[index[gone]] = False
        truth = (escape_iter / 48).reshape(resolution, resolution)

        axes[0, column].imshow(truth, origin="lower", cmap="twilight_shifted")
        axes[0, column].set_title(f"true Julia  c={c_value:.3g}",
                                  color="white", fontsize=9)
        axes[1, column].imshow(neural, origin="lower", cmap="twilight_shifted")
        axes[1, column].set_title("neural Julia", color="white", fontsize=9)

        for row in (0, 1):
            axes[row, column].set_xticks([]); axes[row, column].set_yticks([])

    figure.suptitle("JULIA SETS OF THE LEARNED MAP", color="white",
                    fontsize=13, fontweight="bold")
    plt.tight_layout()
    figure.savefig(out_path, dpi=150, facecolor="#05050a", bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=200)
    parser.add_argument("--max-iter", type=int, default=64)
    parser.add_argument("--checkpoint", type=Path, default=Path("./dyn/map.npz"))
    parser.add_argument("--out", type=Path, default=Path("./dyn"))
    parser.add_argument("--w-fixed", type=float, default=None)
    parser.add_argument("--w-render", type=float, default=2.0)
    # Coverage matters far more for general w than for fixed w: the same
    # orbit budget has to span the whole [2,6] range instead of one slice.
    parser.add_argument("--n-orbits", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=17,
                        help="controls orbit sampling, init, and shuffling")
    # The architectural analogue of the sampler control in train_film.py.
    parser.add_argument("--harmonics", dest="harmonic_mode", default="integer",
                        choices=["integer", "shifted", "random", "none"],
                        help="angular basis: integer 1..10 (original), "
                             "shifted 1.25..10.25, random matched-bandwidth, "
                             "or none (raw theta only)")
    parser.add_argument("--harmonic-offset", type=float, default=0.25)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    if args.train:
        net = train(epochs=args.epochs, hidden=args.hidden,
                    blocks=args.blocks, out_dir=args.out,
                    w_fixed=args.w_fixed, n_orbits=args.n_orbits,
                    seed=args.seed, harmonic_mode=args.harmonic_mode,
                    harmonic_offset=args.harmonic_offset)
    else:
        net = load_map(args.checkpoint)

    if args.render or args.train:
        print("\nRendering the network's own Mandelbrot set...")
        stats = render_comparison(net, args.out / "neural_mandelbrot.png",
                                  w_value=args.w_render,
                                  resolution=args.resolution,
                                  max_iter=args.max_iter)
        print(f"  interior IoU            : {stats['iou']:.4f}")
        print(f"  escape-time MAE         : {stats['mae']:.4f}")
        print(f"  set disagreement        : {stats['disagreement_fraction'] * 100:.2f}% of plane")
        render_julia_row(net, args.out / "neural_julia.png",
                         w_value=args.w_render)
        print(f"  wrote {args.out}/neural_mandelbrot.png and neural_julia.png")


if __name__ == "__main__":
    main()
