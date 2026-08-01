#!/usr/bin/env python3
"""
CATHEDRAL GRAD — analytic input gradients for the Branch-Cut network.

Idea #1 of the roadmap.

The insight: every `bwd(g)` in MandelbrotLeary already RETURNS dL/dx. The
training loop just throws that return value away at `proj.bwd(gradient)`.
Catch it, push it one more step back through the Fourier encoder, and you
have exact per-sample d(output)/d(x, y, w) at the cost of one extra
backward pass. No finite differences, no resolution dependence.

What that unlocks:

  1. Correct flow fields / normals for the renderer. `np.gradient` on a
     kaleidoscope-warped image measures d/d(pixel index), not d/d(domain).
     This measures the real thing.

  2. A NEURAL DISTANCE ESTIMATOR. Derivation, using the smooth iteration
     count nu = max_iter * y_hat and the Douady-Hubbard potential:

         nu  = n + 1 - log(log r) / log(w)          (the training target)
         G   = log(r) / w**n                        (potential)
         =>  G = w ** (1 - nu)
         =>  |grad G| = G * ln(w) * max_iter * |grad y_hat|
         =>  DE = G / |grad G| = 1 / (ln(w) * max_iter * |grad y_hat|)

     The potential cancels completely. The distance to the set is a pure
     function of the gradient magnitude of the learned field.

  3. THE RESOLUTION CEILING, made quantitative. The true field has
     unbounded gradient at the boundary. The network's is bounded by its
     Lipschitz constant, which is bounded by the largest row norm of the
     Fourier matrix B. So:

         min DE  =  1 / (ln(w) * max_iter * max|grad y_hat|)

     is the finest feature the model can physically represent, and it is
     readable off the weights before you render anything.

Usage:
    python3 cathedral_grad.py --gradcheck
    python3 cathedral_grad.py --bandwidth
"""

from __future__ import annotations

import argparse

import numpy as np

import BranchCutCathedral as bcc
import MandelbrotLeary as base

xp = base.xp
to_numpy = base.to_numpy


# ---------------------------------------------------------------------------
# ENCODER BACKWARD
# ---------------------------------------------------------------------------

def encode_bwd(net: bcc.BranchCutNet, x, y, w, g_encoded):
    """
    Backprop through BranchCutNet._encode.

    Forward was:
        xn, yn, wn      = affine rescale of x, y, w to [-1, 1]
        coords          = [xn, yn, wn]                       (N, 3)
        phase           = coords @ B.T                       (N, F)
        interactions    = [xn, yn, wn,
                           xn*yn, xn*wn, yn*wn,
                           xn^2 + yn^2]                      (N, 7)
        encoded         = [interactions, sin(phase), cos(phase)]

    Returns (dL/dx, dL/dy, dL/dw), one row per sample.
    """
    cfg = net.cfg
    F = cfg.mixed_features

    x = xp.asarray(x, dtype=xp.float32)
    y = xp.asarray(y, dtype=xp.float32)
    w = xp.asarray(w, dtype=xp.float32)
    xn, yn, wn = net._normalize(x, y, w)
    coords = xp.stack([xn, yn, wn], axis=1)
    phase = coords @ net.B.T

    g_int = g_encoded[:, :7]
    g_sin = g_encoded[:, 7:7 + F]
    g_cos = g_encoded[:, 7 + F:7 + 2 * F]

    # d sin(p)/dp = cos(p);  d cos(p)/dp = -sin(p)
    g_phase = g_sin * xp.cos(phase) - g_cos * xp.sin(phase)

    # phase = coords @ B.T  =>  dL/dcoords = g_phase @ B
    g_coords = g_phase @ net.B

    g_xn = g_coords[:, 0]
    g_yn = g_coords[:, 1]
    g_wn = g_coords[:, 2]

    # Polynomial interaction terms.
    g_xn = g_xn + g_int[:, 0] + g_int[:, 3] * yn + g_int[:, 4] * wn \
        + xp.float32(2.0) * g_int[:, 6] * xn
    g_yn = g_yn + g_int[:, 1] + g_int[:, 3] * xn + g_int[:, 5] * wn \
        + xp.float32(2.0) * g_int[:, 6] * yn
    g_wn = g_wn + g_int[:, 2] + g_int[:, 4] * xn + g_int[:, 5] * yn

    # Chain through the affine normalization.
    sx = xp.float32(2.0 / (cfg.x_max - cfg.x_min))
    sy = xp.float32(2.0 / (cfg.y_max - cfg.y_min))
    sw = xp.float32(2.0 / (cfg.w_max - cfg.w_min))

    return g_xn * sx, g_yn * sy, g_wn * sw


def trunk_bwd(net: bcc.BranchCutNet, time_gradient, inside_gradient):
    """
    Identical to BranchCutNet.bwd, except it RETURNS the encoder gradient
    instead of discarding it. Requires a matching fwd() call first.
    """
    gt = net.time_sigmoid.bwd(time_gradient[:, None])
    gt = net.time_head.bwd(gt)

    gi = net.inside_sigmoid.bwd(inside_gradient[:, None])
    gi = net.inside_head.bwd(gi)

    gradient = net.norm.bwd(gt + gi)

    for block in reversed(net.blocks):
        gradient = block.bwd(gradient)

    return net.proj.bwd(gradient)


def field_and_grad(net: bcc.BranchCutNet, x, y, w, want="time"):
    """
    Exact per-sample value and input gradient.

    want: "time"   -> d(escape field)/d(x, y, w)
          "inside" -> d(interior probability)/d(x, y, w)
          "both"   -> a dict with each

    Because every sample is processed independently, seeding the backward
    pass with a vector of ones yields d(out_i)/d(in_i) per row rather than
    the gradient of a summed scalar loss. Parameter gradients (.dW etc.)
    are clobbered as a side effect -- never call this mid-training-step.
    """
    x = xp.asarray(x, dtype=xp.float32).ravel()
    y = xp.asarray(y, dtype=xp.float32).ravel()
    w = xp.asarray(w, dtype=xp.float32).ravel()

    time_value, inside_value = net.fwd(x, y, w)

    ones = xp.ones_like(time_value)
    zeros = xp.zeros_like(time_value)

    out = {"time": time_value, "inside": inside_value}

    if want in ("time", "both"):
        g_enc = trunk_bwd(net, ones, zeros)
        out["time_grad"] = encode_bwd(net, x, y, w, g_enc)

    if want in ("inside", "both"):
        # Caches from the single fwd() above are still valid.
        g_enc = trunk_bwd(net, zeros, ones)
        out["inside_grad"] = encode_bwd(net, x, y, w, g_enc)

    return out


def grid_field_and_grad(net, w_value, cfg, resolution,
                        x_min=None, x_max=None, y_min=None, y_max=None,
                        chunk=None):
    """Dense grid evaluation. Returns (field, dfdx, dfdy) as (res, res)."""
    x_min = cfg.x_min if x_min is None else x_min
    x_max = cfg.x_max if x_max is None else x_max
    y_min = cfg.y_min if y_min is None else y_min
    y_max = cfg.y_max if y_max is None else y_max
    chunk = cfg.vis_chunk if chunk is None else chunk

    xs = np.linspace(x_min, x_max, resolution, dtype=np.float32)
    ys = np.linspace(y_min, y_max, resolution, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)

    fx = X.ravel()
    fy = Y.ravel()
    fw = np.full_like(fx, w_value)

    field, dx, dy = [], [], []

    for start in range(0, len(fx), chunk):
        stop = start + chunk
        result = field_and_grad(
            net,
            fx[start:stop],
            fy[start:stop],
            fw[start:stop],
            want="time",
        )
        gx, gy, _ = result["time_grad"]
        field.append(to_numpy(result["time"]))
        dx.append(to_numpy(gx))
        dy.append(to_numpy(gy))

    shape = (resolution, resolution)
    return (
        np.concatenate(field).reshape(shape),
        np.concatenate(dx).reshape(shape),
        np.concatenate(dy).reshape(shape),
    )


# ---------------------------------------------------------------------------
# DISTANCE ESTIMATOR
# ---------------------------------------------------------------------------

def neural_distance(field, dfdx, dfdy, w_value, max_iter,
                    koebe=2.0, eps=1e-12):
    """
    DE ~= koebe / (ln(w) * max_iter * |grad y_hat|)

    The Koebe 1/4 theorem only pins the true exterior distance to within a
    factor of 4, so `koebe` is a taste knob, not a constant of nature. The
    SHAPE of the field is what matters and that part is exact.

    Returns distance in world units. Meaningless inside the set (grad -> 0
    there, so DE -> inf, which is the correct behaviour for an exterior
    distance estimate).
    """
    magnitude = np.sqrt(dfdx * dfdx + dfdy * dfdy) + eps
    scale = np.log(max(w_value, 1.0000001)) * max_iter
    return koebe / (scale * magnitude)


def de_shade(field, dfdx, dfdy, w_value, max_iter, pixel_size,
             interior_value=0.999):
    """
    Distance-estimator shading: near-boundary pixels go bright, and the
    falloff is measured in PIXELS, so the boundary stays one crisp line
    wide at any zoom level. Free anti-aliasing.
    """
    distance = neural_distance(field, dfdx, dfdy, w_value, max_iter)
    normalized = np.clip(distance / (pixel_size * 3.0), 0.0, 1.0)
    shade = 1.0 - np.power(normalized, 0.35)
    return np.where(field >= interior_value, 0.0, shade)


def analytic_normals(dfdx, dfdy, relief=1.0):
    """True surface normals from the analytic gradient, for relief lighting."""
    nx = -relief * dfdx
    ny = -relief * dfdy
    nz = np.ones_like(dfdx)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    return nx / norm, ny / norm, nz / norm


def lambert(dfdx, dfdy, light=(0.6, 0.5, 0.62), relief=1.0):
    nx, ny, nz = analytic_normals(dfdx, dfdy, relief)
    lx, ly, lz = light
    ln = np.sqrt(lx * lx + ly * ly + lz * lz)
    return np.clip((nx * lx + ny * ly + nz * lz) / ln, 0.0, 1.0)


# ---------------------------------------------------------------------------
# BANDWIDTH / RESOLUTION CEILING
# ---------------------------------------------------------------------------

def encoder_bandwidth(net: bcc.BranchCutNet):
    """
    Read the model's spatial resolution ceiling straight off the Fourier
    matrix. Nothing needs to be trained or rendered for this to be true --
    it is a property of the encoding, and no amount of training can beat it.
    """
    cfg = net.cfg
    B = to_numpy(net.B)

    # Angular frequency per unit of NORMALIZED coordinate, per axis.
    per_axis_normalized = np.abs(B).max(axis=0)

    # Convert to world units via the normalization Jacobian.
    jacobian = np.array([
        2.0 / (cfg.x_max - cfg.x_min),
        2.0 / (cfg.y_max - cfg.y_min),
        2.0 / (cfg.w_max - cfg.w_min),
    ])
    per_axis_world = per_axis_normalized * jacobian

    spans = np.array([
        cfg.x_max - cfg.x_min,
        cfg.y_max - cfg.y_min,
        cfg.w_max - cfg.w_min,
    ])

    return {
        "row_norm_max": float(np.linalg.norm(B, axis=1).max()),
        "omega_world": per_axis_world,
        "min_wavelength_world": 2.0 * np.pi / per_axis_world,
        "cycles_across_domain": per_axis_world * spans / (2.0 * np.pi),
    }


def empirical_lipschitz(net, cfg, w_value=3.0, resolution=384):
    """Largest |grad y_hat| the trained model actually produces on a slice."""
    _, dfdx, dfdy = grid_field_and_grad(net, w_value, cfg, resolution)
    magnitude = np.sqrt(dfdx * dfdx + dfdy * dfdy)
    return {
        "max": float(magnitude.max()),
        "p999": float(np.percentile(magnitude, 99.9)),
        "median": float(np.median(magnitude)),
        "min_resolvable_de": float(
            2.0 / (np.log(w_value) * cfg.max_iter * magnitude.max())
        ),
    }


def true_field_gradient(cfg, w_value=3.0, resolution=384):
    """Same measurement on the ground truth, for comparison."""
    xs = np.linspace(cfg.x_min, cfg.x_max, resolution, dtype=np.float32)
    ys = np.linspace(cfg.y_min, cfg.y_max, resolution, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    truth = bcc.power_escape_cpu(
        X.ravel(), Y.ravel(), np.full(X.size, w_value, np.float32), cfg
    ).reshape(resolution, resolution)

    dy_pix, dx_pix = np.gradient(truth)
    dx = dx_pix / ((cfg.x_max - cfg.x_min) / (resolution - 1))
    dy = dy_pix / ((cfg.y_max - cfg.y_min) / (resolution - 1))
    magnitude = np.sqrt(dx * dx + dy * dy)
    return {"max": float(magnitude.max()),
            "p999": float(np.percentile(magnitude, 99.9))}


# ---------------------------------------------------------------------------
# GRADIENT CHECK
# ---------------------------------------------------------------------------

def gradcheck(n_points=256, h=1e-4, seed=7, cfg=None, net=None, verbose=True):
    """Analytic input gradients vs central finite differences."""
    cfg = cfg or bcc.Config(hidden=96, blocks=3, mixed_features=48)
    net = net or bcc.BranchCutNet(cfg)

    rng = np.random.default_rng(seed)
    pad = 0.05
    x = rng.uniform(cfg.x_min + pad, cfg.x_max - pad, n_points).astype(np.float32)
    y = rng.uniform(cfg.y_min + pad, cfg.y_max - pad, n_points).astype(np.float32)
    w = rng.uniform(cfg.w_min + pad, cfg.w_max - pad, n_points).astype(np.float32)

    report = {}

    for head, index in (("time", 0), ("inside", 1)):
        result = field_and_grad(net, x, y, w, want=head)
        analytic = [to_numpy(g) for g in result[f"{head}_grad"]]

        numeric = []
        for axis, coord in enumerate((x, y, w)):
            plus = coord.copy()
            minus = coord.copy()
            plus[:] = coord + h
            minus[:] = coord - h

            args_plus = [x, y, w]
            args_minus = [x, y, w]
            args_plus[axis] = plus
            args_minus[axis] = minus

            f_plus = to_numpy(net.fwd(*[xp.asarray(a) for a in args_plus])[index])
            f_minus = to_numpy(net.fwd(*[xp.asarray(a) for a in args_minus])[index])
            numeric.append((f_plus - f_minus) / (2.0 * h))

        for axis, name in enumerate("xyw"):
            a = analytic[axis]
            n = numeric[axis]
            denom = np.maximum(np.abs(a) + np.abs(n), 1e-6)
            rel = np.abs(a - n) / denom
            corr = float(np.corrcoef(a, n)[0, 1])
            report[f"{head}/d{name}"] = {
                "median_rel_err": float(np.median(rel)),
                "p95_rel_err": float(np.percentile(rel, 95)),
                "correlation": corr,
                "scale": float(np.abs(a).mean()),
            }

    if verbose:
        print("\nGRADIENT CHECK  (analytic vs central differences, h=%.0e)" % h)
        print("-" * 68)
        print(f"{'quantity':<16}{'median rel':>13}{'p95 rel':>11}"
              f"{'corr':>11}{'|grad|':>13}")
        for key, value in report.items():
            print(f"{key:<16}{value['median_rel_err']:>13.2e}"
                  f"{value['p95_rel_err']:>11.2e}"
                  f"{value['correlation']:>11.6f}"
                  f"{value['scale']:>13.3e}")
        worst = max(v["median_rel_err"] for v in report.values())
        best_corr = min(v["correlation"] for v in report.values())
        print("-" * 68)
        verdict = "PASS" if (worst < 5e-3 and best_corr > 0.9999) else "FAIL"
        print("  (h must sit near the float32 FD sweet spot; the analytic side\n"
              "   is exact, the finite differences are the noisy one.)")
        print(f"{verdict}   worst median rel err {worst:.2e}, "
              f"min correlation {best_corr:.6f}")

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gradcheck", action="store_true")
    parser.add_argument("--bandwidth", action="store_true")
    parser.add_argument("--preset", default="cpu")
    args = parser.parse_args()

    if args.gradcheck or not (args.gradcheck or args.bandwidth):
        gradcheck()

    if args.bandwidth:
        cfg = bcc.preset(args.preset)
        net = bcc.BranchCutNet(cfg)
        info = encoder_bandwidth(net)
        print("\nENCODER BANDWIDTH CEILING  (weights only, no training)")
        print("-" * 68)
        print(f"  max Fourier row norm      : {info['row_norm_max']:.1f} rad")
        for axis, name in enumerate(("x", "y", "w")):
            print(f"  {name}: omega_max {info['omega_world'][axis]:9.1f} rad/unit"
                  f" | min wavelength {info['min_wavelength_world'][axis]:.6f}"
                  f" | {info['cycles_across_domain'][axis]:7.1f} cycles across domain")
        print("\n  Nothing finer than the min wavelength can be represented,")
        print("  no matter how long you train.")


if __name__ == "__main__":
    main()
