#!/usr/bin/env python3
"""
CAUCHY-RIEMANN CONSISTENCY PENALTY for the learned power map.

`differential_probe.py` measured that a model at R^2 = 1.00000 violates
Cauchy-Riemann by ~650x the finite-difference noise floor. This trains against
that residual and asks whether it buys anything for long-horizon iteration, or
whether pointwise fit already captures what matters for dynamics.

WHY ONLY CAUCHY-RIEMANN
-----------------------
Of the three identities, CR is the only one that is a genuine inductive bias
rather than the answer in disguise:

  * Euler homogeneity  x*f_x + y*f_y = w*f, TOGETHER WITH holomorphy, implies
    f = C * z^w. Penalizing it hands over the closed form.
  * f_w = f * Log z is likewise nearly the closed form.
  * Cauchy-Riemann says only "f is holomorphic in z" -- a constraint satisfied
    by infinitely many functions. It constrains the shape of the solution
    without specifying it.

So CR is the honest thing to enforce. The others would make the experiment
circular.

THE LOG-POLAR FORM
------------------
Write z = exp(zeta) with zeta = u + i*v, u = log|z|, v = arg z. Since exp is
holomorphic, f(z) is holomorphic in zeta wherever it is in z, and so is
log f = g + i*phi (away from zeros). Cauchy-Riemann in zeta is then

    g_u = phi_v        and        g_v = -phi_u

which is ideal here, because the network *already* predicts g directly (its
log-magnitude head, times LOGMAG_SCALE) and predicts phi as a (cos, sin) pair.
Derivatives of phi come out branch-free without ever unwrapping the angle:

    phi_x = (c * s_x - s * c_x) / (c^2 + s^2)

and the two coordinate perturbations are multiplicative on z -- z*exp(+/-h)
for u and z*exp(+/-i*h) for v -- so nothing degenerates near the origin.

For the true map g = w*u and phi = w*v, so g_u = phi_v = w and
g_v = phi_u = 0. The residual is naturally O(w), needing no normalization.

    python3 cr_consistency.py --gradcheck
    python3 cr_consistency.py --train --cr-weight 0.1 --out ./cr_w01
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import neural_dynamics as nd
from neural_dynamics import xp, to_numpy, LOGMAG_SCALE

H_DEFAULT = 3e-3          # matches the calibrated floor in differential_probe
EPS = np.float32(1e-6)


# ---------------------------------------------------------------------------
# GRADIENT BOOKKEEPING
#
# Linear.bwd assigns (self.dW = ...) rather than accumulating, and caches its
# input on every fwd. A five-point stencil therefore cannot just call fwd/bwd
# five times -- each pass clobbers the last. Accumulate into our own buffers.
# ---------------------------------------------------------------------------

def zero_buffers(params):
    return [xp.zeros_like(p) for p, _, _ in params]


def add_current_grads(buffers, params, scale=1.0):
    for i, (_, attr, obj) in enumerate(params):
        g = getattr(obj, attr)
        if g is not None:
            buffers[i] += xp.float32(scale) * g


def write_back(buffers, params):
    for i, (_, attr, obj) in enumerate(params):
        setattr(obj, attr, buffers[i])


# ---------------------------------------------------------------------------
# THE RESIDUAL
# ---------------------------------------------------------------------------

def _stencil_points(zx, zy, h):
    """z*exp(+/-h) for the log-radius axis, z*exp(+/-i h) for the angle axis."""
    eh, emh = np.float32(np.exp(h)), np.float32(np.exp(-h))
    cs, sn = np.float32(np.cos(h)), np.float32(np.sin(h))
    return {
        "up": (zx * eh, zy * eh),
        "um": (zx * emh, zy * emh),
        "vp": (zx * cs - zy * sn, zx * sn + zy * cs),
        "vm": (zx * cs + zy * sn, -zx * sn + zy * cs),
    }


STENCIL_KEYS = ("up", "um", "vp", "vm", "centre")


def _stencil_batch(zx, zy, w, h):
    """
    All five stencil points as ONE concatenated batch.

    Five separate fwd/bwd calls on 1024 points each are launch-latency bound
    and cost 7.4x baseline. One call on 5*1024 costs ~1.25x. It also makes the
    gradient bookkeeping free: `dW = x.T @ g` already sums over the batch axis,
    so a single bwd on the concatenated gradient accumulates the stencil
    contributions exactly, with no manual buffering.
    """
    pts = _stencil_points(zx, zy, h)
    pts["centre"] = (zx, zy)
    bx = xp.concatenate([pts[k][0] for k in STENCIL_KEYS])
    by = xp.concatenate([pts[k][1] for k in STENCIL_KEYS])
    bw = xp.concatenate([w] * len(STENCIL_KEYS))
    return pts, bx, by, bw


def cr_residual(net, zx, zy, w, h=H_DEFAULT):
    """
    Returns (R1, R2, cache). R1 = g_u - phi_v, R2 = g_v + phi_u.

    cache holds everything the backward pass needs, so the forward work is
    done exactly once.
    """
    n = zx.shape[0]
    pts, bx, by, bw = _stencil_batch(zx, zy, w, h)
    packed = net.fwd(bx, by, bw)
    out = {k: packed[i * n:(i + 1) * n] for i, k in enumerate(STENCIL_KEYS)}
    centre = out["centre"]

    inv2h = xp.float32(1.0 / (2.0 * h))
    scale = xp.float32(LOGMAG_SCALE)

    g_u = (out["up"][:, 0] - out["um"][:, 0]) * scale * inv2h
    g_v = (out["vp"][:, 0] - out["vm"][:, 0]) * scale * inv2h

    c0, s0 = centre[:, 1], centre[:, 2]
    denom = c0 * c0 + s0 * s0 + EPS

    c_u = (out["up"][:, 1] - out["um"][:, 1]) * inv2h
    s_u = (out["up"][:, 2] - out["um"][:, 2]) * inv2h
    c_v = (out["vp"][:, 1] - out["vm"][:, 1]) * inv2h
    s_v = (out["vp"][:, 2] - out["vm"][:, 2]) * inv2h

    phi_u = (c0 * s_u - s0 * c_u) / denom
    phi_v = (c0 * s_v - s0 * c_v) / denom

    R1 = g_u - phi_v
    R2 = g_v + phi_u
    cache = dict(pts=pts, out=out, centre=centre, c0=c0, s0=s0, denom=denom,
                 c_u=c_u, s_u=s_u, c_v=c_v, s_v=s_v,
                 phi_u=phi_u, phi_v=phi_v, inv2h=inv2h, scale=scale, w=w)
    return R1, R2, cache


def cr_penalty(net, zx, zy, w, h=H_DEFAULT):
    R1, R2, _ = cr_residual(net, zx, zy, w, h)
    return float(to_numpy((R1 * R1 + R2 * R2).mean()))


def cr_backward(net, zx, zy, w, buffers, params, weight=1.0, h=H_DEFAULT):
    """
    Accumulate d(weight * mean(R1^2 + R2^2)) / d(params) into `buffers`.

    The penalty is a closed-form function of the network output at five
    stencil points, so the chain rule to each output is exact -- the only
    approximation is the finite difference itself.
    """
    R1, R2, ca = cr_residual(net, zx, zy, w, h)
    n = R1.shape[0]
    k = xp.float32(2.0 * weight / n)
    dR1, dR2 = k * R1, k * R2

    c0, s0, denom = ca["c0"], ca["s0"], ca["denom"]
    inv2h, scale = ca["inv2h"], ca["scale"]

    # d/d(shifted log-magnitude outputs): R1 = g_u - ..., R2 = g_v + ...
    d_up_L = dR1 * scale * inv2h
    d_um_L = -d_up_L
    d_vp_L = dR2 * scale * inv2h
    d_vm_L = -d_vp_L

    # phi_u enters R2 with +1, phi_v enters R1 with -1.
    a_u = dR2 / denom          # coefficient on (c0*s_u - s0*c_u)
    a_v = -dR1 / denom
    d_up_c = -a_u * s0 * inv2h
    d_um_c = -d_up_c
    d_up_s = a_u * c0 * inv2h
    d_um_s = -d_up_s
    d_vp_c = -a_v * s0 * inv2h
    d_vm_c = -d_vp_c
    d_vp_s = a_v * c0 * inv2h
    d_vm_s = -d_vp_s

    # Centre point enters through c0, s0 in both numerator and denominator.
    dphi_u_dc0 = (ca["s_u"] - ca["phi_u"] * xp.float32(2.0) * c0) / denom
    dphi_u_ds0 = (-ca["c_u"] - ca["phi_u"] * xp.float32(2.0) * s0) / denom
    dphi_v_dc0 = (ca["s_v"] - ca["phi_v"] * xp.float32(2.0) * c0) / denom
    dphi_v_ds0 = (-ca["c_v"] - ca["phi_v"] * xp.float32(2.0) * s0) / denom
    d_c0 = dR2 * dphi_u_dc0 - dR1 * dphi_v_dc0
    d_s0 = dR2 * dphi_u_ds0 - dR1 * dphi_v_ds0

    zero = xp.zeros_like(d_c0)
    grads = {
        "up": (d_up_L, d_up_c, d_up_s),
        "um": (d_um_L, d_um_c, d_um_s),
        "vp": (d_vp_L, d_vp_c, d_vp_s),
        "vm": (d_vm_L, d_vm_c, d_vm_s),
        "centre": (zero, d_c0, d_s0),
    }

    # Single bwd over the concatenated stencil. The forward cache from
    # cr_residual is still live, so no recomputation is needed.
    packed_grad = xp.concatenate(
        [xp.stack(grads[k], axis=1) for k in STENCIL_KEYS], axis=0)
    net.bwd(packed_grad)
    add_current_grads(buffers, params)

    return float(to_numpy((R1 * R1 + R2 * R2).mean()))


# ---------------------------------------------------------------------------
# VERIFICATION -- hand-derived gradients get checked before they get used
# ---------------------------------------------------------------------------

def gradcheck(n=256, n_params=40, h_fd=1e-3, seed=0):
    """
    Perturb actual network parameters and compare the analytic accumulated
    gradient of the penalty against central differences of the penalty. This
    exercises the whole chain: stencil -> output grads -> net.bwd.
    """
    rng = np.random.default_rng(seed)
    net = nd.NeuralMap(hidden=48, blocks=2, features=16, seed=3)
    params = net.params()

    radius = np.sqrt(rng.uniform(0.25, 9.0, n)).astype(np.float32)
    angle = rng.uniform(-np.pi + 0.4, np.pi - 0.4, n).astype(np.float32)
    zx = xp.asarray(radius * np.cos(angle))
    zy = xp.asarray(radius * np.sin(angle))
    w = xp.asarray(rng.uniform(2.0, 6.0, n).astype(np.float32))

    buffers = zero_buffers(params)
    cr_backward(net, zx, zy, w, buffers, params, weight=1.0)

    flat_idx = []
    for pi, (p, _, _) in enumerate(params):
        if p.size < 2:
            continue
        for _ in range(max(1, n_params // len(params))):
            flat_idx.append((pi, int(rng.integers(0, p.size))))

    analytic, numeric = [], []
    for pi, off in flat_idx:
        p = params[pi][0]
        flat = p.reshape(-1)
        original = float(to_numpy(flat[off]))

        flat[off] = xp.float32(original + h_fd)
        plus = cr_penalty(net, zx, zy, w)
        flat[off] = xp.float32(original - h_fd)
        minus = cr_penalty(net, zx, zy, w)
        flat[off] = xp.float32(original)

        numeric.append((plus - minus) / (2 * h_fd))
        analytic.append(float(to_numpy(buffers[pi].reshape(-1)[off])))

    analytic = np.asarray(analytic)
    numeric = np.asarray(numeric)
    scale = np.maximum(np.abs(analytic), np.abs(numeric))
    keep = scale > 1e-4
    rel = np.abs(analytic - numeric)[keep] / scale[keep]
    corr = float(np.corrcoef(analytic[keep], numeric[keep])[0, 1])

    print("\nCAUCHY-RIEMANN PENALTY GRADCHECK")
    print(f"  parameters probed : {len(analytic)}  ({keep.sum()} above 1e-4)")
    print(f"  median rel err    : {np.median(rel):.3e}")
    print(f"  p95 rel err       : {np.percentile(rel, 95):.3e}")
    print(f"  correlation       : {corr:.6f}")
    ok = np.median(rel) < 5e-2 and corr > 0.999
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def sanity_true_map(n=20000, h=H_DEFAULT, seed=0):
    """The analytic map must give ~0 residual. Calibrates the FD floor."""
    rng = np.random.default_rng(seed)
    radius = np.sqrt(rng.uniform(0.25, 9.0, n))
    angle = rng.uniform(-np.pi + 0.4, np.pi - 0.4, n)
    zx = (radius * np.cos(angle)).astype(np.float32)
    zy = (radius * np.sin(angle)).astype(np.float32)
    w = rng.uniform(2.0, 6.0, n).astype(np.float32)

    pts = _stencil_points(zx, zy, h)
    def g_phi(a, b):
        r = np.hypot(a, b)
        return w * np.log(r), w * np.arctan2(b, a)
    gu = (g_phi(*pts["up"])[0] - g_phi(*pts["um"])[0]) / (2 * h)
    gv = (g_phi(*pts["vp"])[0] - g_phi(*pts["vm"])[0]) / (2 * h)
    # phi via cos/sin to match how the network is measured (branch-free)
    def cs(a, b):
        p = w * np.arctan2(b, a)
        return np.cos(p), np.sin(p)
    c0, s0 = cs(zx, zy)
    cu, su = cs(*pts["up"]); cm, sm = cs(*pts["um"])
    cv, sv = cs(*pts["vp"]); cw, sw = cs(*pts["vm"])
    c_u, s_u = (cu - cm) / (2 * h), (su - sm) / (2 * h)
    c_v, s_v = (cv - cw) / (2 * h), (sv - sw) / (2 * h)
    d = c0 * c0 + s0 * s0 + 1e-12
    phi_u = (c0 * s_u - s0 * c_u) / d
    phi_v = (c0 * s_v - s0 * c_v) / d
    R1, R2 = gu - phi_v, gv + phi_u
    print("\nANALYTIC MAP THROUGH THE SAME STENCIL (residual should be ~0)")
    print(f"  mean R1^2 + R2^2 : {float((R1**2 + R2**2).mean()):.3e}")
    print(f"  median |R1|      : {float(np.median(np.abs(R1))):.3e}"
          f"   median |R2| : {float(np.median(np.abs(R2))):.3e}")
    print(f"  (typical |g_u| = w ~ 4, so this is the relative floor)")


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------

def collocation_batch(rng, n, w_min=2.0, w_max=6.0):
    """Fresh interior points for the penalty, independent of the data batch."""
    radius = np.sqrt(rng.uniform(0.04, 16.0, n))
    angle = rng.uniform(-np.pi + 0.35, np.pi - 0.35, n)
    return (xp.asarray((radius * np.cos(angle)).astype(np.float32)),
            xp.asarray((radius * np.sin(angle)).astype(np.float32)),
            xp.asarray(rng.uniform(w_min, w_max, n).astype(np.float32)))


def cr_schedule(epoch, epochs, weight, warmup_frac, ramp_frac):
    """
    Penalty weight for this epoch: off during warm-up, then ramped in.

    Applying the penalty from step 0 does not work. An untrained net with
    Fourier radii up to ~100 has input-derivatives of order 100, so the
    residual starts near 3e4 -- four orders above its converged value of ~1 --
    and its gradient swamps the data term, pulling the map toward a constant.
    Measured: cr_weight=1e-5 from scratch ends at fit 1.79 against a baseline
    of 2.5e-3, i.e. 700x worse.

    Warm-up also matches the question being asked. "Does enforcing an identity
    the model violates by ~1% improve long-horizon iteration?" is a statement
    about an already-fitted model, not about optimisation from scratch.
    """
    if weight <= 0.0:
        return 0.0
    start = warmup_frac * epochs
    if epoch <= start:
        return 0.0
    if ramp_frac <= 0.0:
        return weight
    t = (epoch - start) / max(ramp_frac * epochs, 1e-9)
    return weight * float(min(1.0, t))


def train(epochs=300, batch_size=4096, hidden=192, blocks=3, features=64,
          lr=6e-4, out_dir=Path("./cr_run"), n_orbits=24000, seed=1,
          cr_weight=0.0, cr_points=1024, h=H_DEFAULT,
          harmonic_mode="integer", cr_warmup=0.5, cr_ramp=0.1):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)
    shuffle_rng = np.random.default_rng(seed + 7919)
    colloc_rng = np.random.default_rng(seed + 104729)

    zx, zy, w, targets = nd.orbit_dataset(n_orbits=n_orbits, seed=seed)
    net = nd.NeuralMap(hidden=hidden, blocks=blocks, features=features,
                       seed=seed, harmonic_mode=harmonic_mode)
    params = net.params()
    optimizer = nd.Adam(params, lr=lr)
    print(f"  {len(zx):,} states | {net.parameter_count():,} params | "
          f"seed {seed} | cr_weight {cr_weight} | cr_points {cr_points}",
          flush=True)

    n = len(zx)
    start = time.time()
    history = []
    for epoch in range(1, epochs + 1):
        optimizer.lr = 1e-6 + 0.5 * (lr - 1e-6) * (1 + np.cos(np.pi * epoch / epochs))
        w_epoch = cr_schedule(epoch, epochs, cr_weight, cr_warmup, cr_ramp)
        order = shuffle_rng.permutation(n)
        zx, zy, w, targets = zx[order], zy[order], w[order], targets[order]

        tot_fit, tot_cr, count = 0.0, 0.0, 0
        for s0 in range(0, n, batch_size):
            s1 = min(s0 + batch_size, n)
            size = s1 - s0
            buffers = zero_buffers(params)

            # data term
            prediction = net.fwd(xp.asarray(zx[s0:s1]), xp.asarray(zy[s0:s1]),
                                 xp.asarray(w[s0:s1]))
            target = xp.asarray(targets[s0:s1])
            diff = prediction - target
            net.bwd(xp.float32(2.0) * diff / xp.float32(size * 3))
            add_current_grads(buffers, params)
            fit = float(to_numpy((diff * diff).mean()))

            # consistency term on fresh collocation points
            cr_val = 0.0
            if w_epoch > 0.0 and cr_points > 0:
                cx, cy, cw = collocation_batch(colloc_rng, cr_points)
                cr_val = cr_backward(net, cx, cy, cw, buffers, params,
                                     weight=w_epoch, h=h)

            write_back(buffers, params)
            optimizer.step()
            tot_fit += fit
            tot_cr += cr_val
            count += 1

        history.append({"epoch": epoch, "fit": tot_fit / count,
                        "cr": tot_cr / count, "lam": w_epoch})
        if epoch % 25 == 0 or epoch == 1:
            r2 = 1.0 - (tot_fit / count) / float(targets.var())
            print(f"[{epoch:3d}/{epochs}] fit={tot_fit / count:.4e} "
                  f"R2={r2:.5f} cr={tot_cr / count:.4e} "
                  f"lam={w_epoch:.1e} {time.time() - start:.0f}s", flush=True)

    payload = {
        "B": to_numpy(net.B),
        "meta": np.asarray(json.dumps({
            "hidden": hidden, "blocks": blocks, "features": features,
            "bailout": net.bailout, "harmonic_mode": net.harmonic_mode,
            "harmonic_offset": net.harmonic_offset, "seed": seed,
            "cr_weight": cr_weight, "cr_points": cr_points, "h": h,
            "cr_warmup": cr_warmup, "cr_ramp": cr_ramp,
        })),
        "harmonics": to_numpy(net.harmonics),
        "losses": np.asarray([r["fit"] for r in history], np.float32),
        "cr_history": np.asarray([r["cr"] for r in history], np.float32),
    }
    for i, (p, _, _) in enumerate(net.params()):
        payload[f"parameter_{i}"] = to_numpy(p)
    np.savez_compressed(out_dir / "map.npz", **payload)
    print(f"\nSaved {out_dir}/map.npz  ({time.time() - start:.0f}s)")
    return net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gradcheck", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--n-orbits", type=int, default=24000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cr-weight", type=float, default=0.0)
    parser.add_argument("--cr-points", type=int, default=1024)
    parser.add_argument("--h", type=float, default=H_DEFAULT)
    parser.add_argument("--cr-warmup", type=float, default=0.5,
                        help="fraction of training with the penalty OFF")
    parser.add_argument("--cr-ramp", type=float, default=0.1,
                        help="fraction of training over which it ramps in")
    parser.add_argument("--out", type=Path, default=Path("./cr_run"))
    args = parser.parse_args()

    if args.gradcheck:
        sanity_true_map()
        ok = gradcheck()
        raise SystemExit(0 if ok else 1)

    if args.train:
        train(epochs=args.epochs, n_orbits=args.n_orbits, seed=args.seed,
              cr_weight=args.cr_weight, cr_points=args.cr_points,
              h=args.h, out_dir=args.out, cr_warmup=args.cr_warmup,
              cr_ramp=args.cr_ramp)


if __name__ == "__main__":
    main()
