#!/usr/bin/env python3
"""
FILM EXPONENT — give w its own pathway instead of a third coordinate.

Idea #4 of the roadmap.

Motivation, from the bandwidth measurement: w is the ROUGHEST axis of the
field, not the smoothest. Sweeping the exponent at fixed c drags a single
point in and out of the set repeatedly, so a w-line near the boundary is
a chaotic 1D signal -- median 153 cycles/domain against 12.5 for x and
5.0 for y. Feeding it through the same isotropic Fourier basis as
position is the wrong allocation.

FiLM (Perez et al., feature-wise linear modulation) splits the roles:

    "where am I"      -> (x, y) go through the Fourier encoder
    "which universe"  -> w goes through a small hypernetwork that emits
                         per-channel (gamma, beta) modulating each block

    h  <-  gamma(w) * h + beta(w)

Two payoffs, one practical and one interpretive:

  practical    the exponent gets a dedicated nonlinear pathway with its
               own capacity, instead of competing with position for the
               same basis functions;

  interpretive the learned w-embedding is a 1-D CURVE in modulation
               space. You can plot it. Its arc length, its curvature,
               its kinks are all directly readable, and `embedding_curve()`
               below extracts it in one call.

               It was built to ask whether the network discovers on its own
               that w = 2, 3, 4 are special. IT CANNOT ANSWER THAT. Controls
               in train_film.py show the curve tracks where training w was
               SAMPLED: shift the sampler's lattice to k+0.25 and the peak
               moves +0.2505 with it; remove the lattice and nothing remains
               across six seeds. Treat this as a sampler-sensitive diagnostic
               of the training distribution, not as interpretability evidence.

Backward pass is derived by hand to match the rest of the codebase:

    y = gamma * h + beta

    dL/dh      = g * gamma
    dL/dgamma  = g * h        (then into the hypernet)
    dL/dbeta   = g

Usage:
    python3 film_exponent.py --gradcheck
    python3 film_exponent.py --demo-curve
"""

from __future__ import annotations

import argparse

import numpy as np

import MandelbrotLeary as base
import BranchCutCathedral as bcc

xp = base.xp
to_numpy = base.to_numpy
Linear = base.Linear
LayerNorm = base.LayerNorm
GELU = base.GELU
Sigmoid = base.Sigmoid


# ---------------------------------------------------------------------------
# FiLM
# ---------------------------------------------------------------------------

class FiLM:
    """
    Applies y = gamma * h + beta with gamma, beta supplied per-sample.

    Kept as a bare layer so the ResBlock can own its own modulation and
    the hypernet can be shared across blocks.
    """

    def __init__(self):
        self._h = None
        self._gamma = None
        self.dgamma = None
        self.dbeta = None

    def fwd(self, h, gamma, beta):
        self._h = h
        self._gamma = gamma
        return gamma * h + beta

    def bwd(self, g):
        self.dgamma = g * self._h
        self.dbeta = g
        return g * self._gamma

    def params(self):
        return []


class ExponentHyperNet:
    """
    w -> Fourier lift -> Linear -> GELU -> Linear -> (gamma, beta) for
    every conditioned block at once.

    Emits gamma as 1 + raw so it starts as the identity: at init the
    conditioned network behaves exactly like the unconditioned one, which
    keeps early training stable.
    """

    def __init__(self, n_blocks, dim, n_freq=12, hidden=96,
                 w_min=2.0, w_max=6.0, seed=1234):
        self.n_blocks = n_blocks
        self.dim = dim
        self.w_min = w_min
        self.w_max = w_max

        rng = np.random.default_rng(seed)
        self.freqs = xp.asarray(
            (np.power(2.0, np.linspace(0.0, 4.5, n_freq)) * np.pi).astype(np.float32)
        )

        self.fc1 = Linear(1 + 2 * n_freq, hidden)
        self.act = GELU()
        self.fc2 = Linear(hidden, 2 * n_blocks * dim)

        # Start near identity: gamma ~ 1, beta ~ 0.
        self.fc2.W *= xp.float32(0.05)

        self._n_freq = n_freq
        self._cache = None

    def _encode(self, w):
        # CuPy will not accept a NumPy *array* as an operand against a device
        # array (scalars are fine, which is why this ran clean under NumPy).
        # Coerce at every public entry point.
        w = xp.asarray(w, dtype=xp.float32)
        wn = xp.float32(2.0) * (w - xp.float32(self.w_min)) / \
            xp.float32(self.w_max - self.w_min) - xp.float32(1.0)
        phase = wn[:, None] * self.freqs
        return xp.concatenate(
            [wn[:, None], xp.sin(phase), xp.cos(phase)], axis=1
        )

    def fwd(self, w):
        """Returns list of (gamma, beta), one pair per conditioned block."""
        encoded = self._encode(w)
        hidden = self.act.fwd(self.fc1.fwd(encoded))
        raw = self.fc2.fwd(hidden)

        n = w.shape[0]
        raw = raw.reshape(n, self.n_blocks, 2, self.dim)

        modulation = []
        for block in range(self.n_blocks):
            gamma = xp.float32(1.0) + raw[:, block, 0, :]
            beta = raw[:, block, 1, :]
            modulation.append((gamma, beta))

        self._cache = (n,)
        return modulation

    def bwd(self, gradients):
        """
        gradients: list of (dgamma, dbeta) per block, each (N, dim).
        d(gamma)/d(raw) = 1, so the +1 offset contributes nothing here.
        """
        (n,) = self._cache
        stacked = xp.zeros((n, self.n_blocks, 2, self.dim), dtype=xp.float32)
        for block, (dgamma, dbeta) in enumerate(gradients):
            stacked[:, block, 0, :] = dgamma
            stacked[:, block, 1, :] = dbeta

        g = self.fc2.bwd(stacked.reshape(n, 2 * self.n_blocks * self.dim))
        g = self.act.bwd(g)
        self.fc1.bwd(g)

    def params(self):
        return list(self.fc1.params()) + list(self.fc2.params())

    def embedding_curve(self, w_values):
        """
        Returns the (gamma, beta) vectors the network assigns to each
        exponent, stacked into a curve you can PCA, differentiate, or just
        look at.

        WHAT THIS MEASURES: the arc-length speed |d(embedding)/dw| peaks where
        training w was concentrated, not where the task has structure. Under
        the phase-shifted control in train_film.py the peak follows the
        sampler's lattice (+0.2505 for a +0.25 shift); under continuous
        uniform w it finds nothing across six seeds. So this is a
        sampler-sensitive diagnostic -- useful for seeing how training mass
        was distributed over w, and not usable as evidence that the network
        located anything on its own.

        Note np.gradient is called without a spacing argument, so speed is
        per-sample rather than per-unit-w: absolute values shift with the
        density of `w_values`. Rank and ratio comparisons are unaffected.
        """
        w = xp.asarray(np.asarray(w_values, np.float32))
        modulation = self.fwd(w)
        pieces = []
        for gamma, beta in modulation:
            pieces.append(to_numpy(gamma))
            pieces.append(to_numpy(beta))
        curve = np.concatenate(pieces, axis=1)

        speed = np.linalg.norm(np.gradient(curve, axis=0), axis=1)
        return curve, speed


class FiLMResBlock:
    """ResBlock with FiLM applied after the first normalization."""

    def __init__(self, dim):
        self.ln1 = LayerNorm(dim)
        self.film = FiLM()
        self.fc1 = Linear(dim, dim)
        self.act = GELU()
        self.ln2 = LayerNorm(dim)
        self.fc2 = Linear(dim, dim)

    def fwd(self, x, gamma, beta):
        h = self.ln1.fwd(x)
        h = self.film.fwd(h, gamma, beta)
        h = self.fc1.fwd(h)
        h = self.act.fwd(h)
        h = self.ln2.fwd(h)
        h = self.fc2.fwd(h)
        return h + x

    def bwd(self, g):
        g2 = self.fc2.bwd(g)
        g2 = self.ln2.bwd(g2)
        g2 = self.act.bwd(g2)
        g2 = self.fc1.bwd(g2)
        g2 = self.film.bwd(g2)
        g2 = self.ln1.bwd(g2)
        return g2 + g, (self.film.dgamma, self.film.dbeta)

    def params(self):
        p = []
        for layer in (self.ln1, self.fc1, self.ln2, self.fc2):
            p.extend(layer.params())
        return p


# ---------------------------------------------------------------------------
# THE CONDITIONED NETWORK
# ---------------------------------------------------------------------------

class FiLMCathedralNet:
    """
    (x, y) -> 2-D Fourier encoder -> FiLM-conditioned residual trunk
                                     ^
                                     |
                              w -> hypernetwork

    Drop-in for BranchCutNet: same fwd signature, same dual heads, so it
    works with the existing training loop and with cathedral_grad.
    """

    def __init__(self, cfg: bcc.Config, hyper_hidden=96, hyper_freq=12):
        self.cfg = cfg

        rng = np.random.default_rng(cfg.seed + 991)
        features = cfg.mixed_features

        directions = rng.normal(size=(features, 2))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True).clip(1e-8)
        radii = np.power(2.0, np.linspace(0.0, 8.0, features))
        rng.shuffle(radii)
        self.B = xp.asarray((directions * radii[:, None] * np.pi).astype(np.float32))

        # xn, yn, xn*yn, xn^2+yn^2 + sin/cos
        encoded_dim = 4 + 2 * features

        self.proj = Linear(encoded_dim, cfg.hidden)
        self.blocks = [FiLMResBlock(cfg.hidden) for _ in range(cfg.blocks)]
        self.hyper = ExponentHyperNet(
            cfg.blocks, cfg.hidden, n_freq=hyper_freq, hidden=hyper_hidden,
            w_min=cfg.w_min, w_max=cfg.w_max,
        )
        self.norm = LayerNorm(cfg.hidden)
        self.time_head = Linear(cfg.hidden, 1)
        self.inside_head = Linear(cfg.hidden, 1)
        self.time_sigmoid = Sigmoid()
        self.inside_sigmoid = Sigmoid()

    def _encode(self, x, y):
        cfg = self.cfg
        x = xp.asarray(x, dtype=xp.float32)
        y = xp.asarray(y, dtype=xp.float32)
        xn = xp.float32(2.0) * (x - cfg.x_min) / xp.float32(cfg.x_max - cfg.x_min) - xp.float32(1.0)
        yn = xp.float32(2.0) * (y - cfg.y_min) / xp.float32(cfg.y_max - cfg.y_min) - xp.float32(1.0)

        coords = xp.stack([xn, yn], axis=1)
        phase = coords @ self.B.T

        explicit = xp.stack([xn, yn, xn * yn, xn * xn + yn * yn], axis=1)
        return xp.concatenate([explicit, xp.sin(phase), xp.cos(phase)], axis=1)

    def fwd(self, x, y, w):
        modulation = self.hyper.fwd(w)

        h = self.proj.fwd(self._encode(x, y))
        for block, (gamma, beta) in zip(self.blocks, modulation):
            h = block.fwd(h, gamma, beta)
        h = self.norm.fwd(h)

        time_value = self.time_sigmoid.fwd(self.time_head.fwd(h)).squeeze(-1)
        inside_value = self.inside_sigmoid.fwd(self.inside_head.fwd(h)).squeeze(-1)
        return time_value, inside_value

    def bwd(self, time_gradient, inside_gradient):
        gt = self.time_head.bwd(self.time_sigmoid.bwd(time_gradient[:, None]))
        gi = self.inside_head.bwd(self.inside_sigmoid.bwd(inside_gradient[:, None]))

        g = self.norm.bwd(gt + gi)

        modulation_gradients = [None] * len(self.blocks)
        for index in range(len(self.blocks) - 1, -1, -1):
            g, mod_grad = self.blocks[index].bwd(g)
            modulation_gradients[index] = mod_grad

        self.hyper.bwd(modulation_gradients)
        self.proj.bwd(g)

    def params(self):
        p = list(self.proj.params())
        for block in self.blocks:
            p.extend(block.params())
        p.extend(self.hyper.params())
        p.extend(self.norm.params())
        p.extend(self.time_head.params())
        p.extend(self.inside_head.params())
        return p

    def parameter_count(self):
        return sum(p.size for p, _, _ in self.params())


# ---------------------------------------------------------------------------
# GRADIENT CHECK
# ---------------------------------------------------------------------------

def gradcheck(n_points=64, h=1e-3, seed=3, verbose=True):
    """
    Check every hand-derived parameter gradient, with special attention to
    the ones routed through FiLM and the hypernetwork -- those are the new
    code paths and the only place a sign error could hide.
    """
    cfg = bcc.Config(hidden=48, blocks=2, mixed_features=24)
    net = FiLMCathedralNet(cfg, hyper_hidden=32, hyper_freq=6)

    rng = np.random.default_rng(seed)
    x = xp.asarray(rng.uniform(cfg.x_min + .1, cfg.x_max - .1, n_points).astype(np.float32))
    y = xp.asarray(rng.uniform(cfg.y_min + .1, cfg.y_max - .1, n_points).astype(np.float32))
    w = xp.asarray(rng.uniform(cfg.w_min + .1, cfg.w_max - .1, n_points).astype(np.float32))
    target = xp.asarray(rng.uniform(0, 1, n_points).astype(np.float32))

    def loss_value():
        time_value, _ = net.fwd(x, y, w)
        difference = time_value - target
        return float(to_numpy((difference * difference).mean()))

    # Analytic
    time_value, _ = net.fwd(x, y, w)
    difference = time_value - target
    net.bwd(xp.float32(2.0) * difference / xp.float32(n_points),
            xp.zeros_like(difference))

    parameters = net.params()
    hyper_ids = {id(p) for p, _, _ in net.hyper.params()}

    results = []
    for parameter, attribute, owner in parameters:
        # NOT np.array(...): CuPy deliberately refuses implicit host transfer.
        analytic = to_numpy(getattr(owner, attribute)).copy()
        # ravel() returns a view for contiguous arrays on both backends;
        # reshape(-1) is not guaranteed to, and a copy would silently make the
        # in-place perturbation below a no-op.
        flat = parameter.ravel()
        flat_analytic = analytic.ravel()

        n_probe = min(12, flat.size)
        indices = rng.choice(flat.size, n_probe, replace=False)

        numeric = np.empty(n_probe, np.float32)
        for slot, index in enumerate(indices):
            original = float(flat[index])
            flat[index] = original + h
            plus = loss_value()
            flat[index] = original - h
            minus = loss_value()
            flat[index] = original
            numeric[slot] = (plus - minus) / (2 * h)

        a = flat_analytic[indices]
        denom = np.maximum(np.abs(a) + np.abs(numeric), 1e-7)
        results.append({
            "is_hyper": id(parameter) in hyper_ids,
            "rel": np.abs(a - numeric) / denom,
            "corr": float(np.corrcoef(a, numeric)[0, 1]) if n_probe > 2 else 1.0,
        })

    hyper_rel = np.concatenate([r["rel"] for r in results if r["is_hyper"]])
    trunk_rel = np.concatenate([r["rel"] for r in results if not r["is_hyper"]])
    all_corr = [r["corr"] for r in results if np.isfinite(r["corr"])]

    if verbose:
        print(f"\nFiLM GRADIENT CHECK  ({net.parameter_count():,} params, h={h:.0e})")
        print("-" * 62)
        print(f"  hypernetwork params : median rel err {np.median(hyper_rel):.2e}"
              f"   p95 {np.percentile(hyper_rel, 95):.2e}")
        print(f"  trunk params        : median rel err {np.median(trunk_rel):.2e}"
              f"   p95 {np.percentile(trunk_rel, 95):.2e}")
        print(f"  min per-tensor corr : {min(all_corr):.6f}")
        ok = (np.median(hyper_rel) < 5e-3 and np.median(trunk_rel) < 5e-3
              and min(all_corr) > 0.999)
        print("-" * 62)
        print("PASS" if ok else "FAIL")

    return {"hyper_median": float(np.median(hyper_rel)),
            "trunk_median": float(np.median(trunk_rel)),
            "min_corr": float(min(all_corr))}


def demo_curve():
    """Show the embedding-curve extraction working on an untrained net."""
    cfg = bcc.Config(hidden=64, blocks=3, mixed_features=32)
    net = FiLMCathedralNet(cfg)
    w_values = np.linspace(cfg.w_min, cfg.w_max, 401)
    curve, speed = net.hyper.embedding_curve(w_values)
    print(f"\nEMBEDDING CURVE (untrained -- expect no structure yet)")
    print(f"  curve shape       : {curve.shape}  "
          f"({cfg.blocks} blocks x 2 x {cfg.hidden} dims)")
    print(f"  arc-length speed  : mean {speed.mean():.4f}  std {speed.std():.4f}")
    print(f"  speed at integers : "
          + "  ".join(f"w={v:.0f}:{speed[np.argmin(np.abs(w_values - v))]:.4f}"
                      for v in range(2, 7)))
    print("\n  Peaks in this curve track where training w was SAMPLED, not")
    print("  where the task has structure -- see the controls in")
    print("  train_film.py. Read it as a diagnostic of the training")
    print("  distribution, not as evidence about what the network found.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gradcheck", action="store_true")
    parser.add_argument("--demo-curve", action="store_true")
    args = parser.parse_args()

    if args.gradcheck or not args.demo_curve:
        gradcheck()
    if args.demo_curve:
        demo_curve()


if __name__ == "__main__":
    main()
