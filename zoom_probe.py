#!/usr/bin/env python3
"""
ZOOM PROBE — the mirror of the outward-extrapolation experiment.

Idea #2 of the roadmap.

You already know what the network does when queried OUTSIDE its training
domain: it decays into a periodic lattice, because a Fourier-feature MLP
is a sum of sinusoids and sinusoids are periodic. This probe asks the
opposite question: what happens BELOW its training scale?

The setup is deliberately unfair to the "it just needs more data"
explanation. The training sampler already draws boundary neighbourhoods
with log-uniform radii down to 2e-5, so fine-scale data is present. Any
failure here is representational, not a coverage gap.

The falsifiable prediction, read straight off the weights before
rendering anything (see cathedral_grad.encoder_bandwidth):

    omega_max is FIXED in world units. Zoom by Z and the window shrinks
    by Z, so the structure the network can express inside that window
    falls as 1/Z:

        cycles_across_window(Z) = cycles_across_domain / Z

    The true Mandelbrot is self-similar, so its cycle count is roughly
    FLAT in Z. The two curves diverge immediately and the network hits
    "less than one wavelength per window" -- total structural death -- at

        Z_death ~= cycles_across_domain

For the gpu preset that is Z ~= 210. Not "somewhere around a hundred",
not "eventually". 210.

Usage:
    python3 zoom_probe.py --checkpoint run/model.npz
    python3 zoom_probe.py --checkpoint run/model.npz --render
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import BranchCutCathedral as bcc
import cathedral_grad as cg


# ---------------------------------------------------------------------------
# SPECTRAL STRUCTURE MEASURE
# ---------------------------------------------------------------------------

def cycles_across_window(image, energy_fraction=0.90):
    """
    Radial spatial frequency containing `energy_fraction` of the field's
    non-DC energy, in cycles per window width.

    This is the "how much structure is in here" number. A self-similar
    fractal holds it roughly constant under zoom. A band-limited function
    cannot.
    """
    field = image - image.mean()
    if not np.any(field):
        return 0.0

    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(field))) ** 2
    n = field.shape[0]
    centre = n // 2

    yy, xx = np.mgrid[0:n, 0:n]
    radius = np.sqrt((xx - centre) ** 2 + (yy - centre) ** 2)

    bins = np.arange(0, centre + 1)
    profile = np.array([
        spectrum[(radius >= r) & (radius < r + 1)].sum()
        for r in bins
    ])
    profile[0] = 0.0

    total = profile.sum()
    if total <= 0:
        return 0.0

    cumulative = np.cumsum(profile) / total
    index = int(np.searchsorted(cumulative, energy_fraction))
    return float(min(index, centre))


def lattice_score(image, net):
    """
    How much of the field's energy sits at frequencies the encoder can
    actually produce? Near 1.0 means the picture is made of Mandelbrot.
    Near 0.0 means you are looking at the basis itself.
    """
    field = image - image.mean()
    if not np.any(field):
        return 0.0
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(field))) ** 2
    return float(spectrum.sum())


# ---------------------------------------------------------------------------
# THE SWEEP
# ---------------------------------------------------------------------------

def zoom_sweep(net, cfg, centre=(-0.745428, 0.113009), w_value=2.0,
               zooms=(1, 2, 5, 10, 20, 50, 100, 200, 500, 1000),
               resolution=192, verbose=True):
    """
    Render truth and network at a ladder of zoom levels around one point
    and measure how much structure survives in each.

    Ground truth is computed at the SAME max_iter the model was trained
    on, so this is not "can it beat its teacher" -- it is "can it
    reproduce its own training target at fine scale".
    """
    cx, cy = centre
    base_width = cfg.x_max - cfg.x_min

    info = cg.encoder_bandwidth(net)
    ceiling = float(info["cycles_across_domain"][0])

    rows = []

    for zoom in zooms:
        half = 0.5 * base_width / zoom
        half_y = half * (cfg.y_max - cfg.y_min) / (cfg.x_max - cfg.x_min)

        xs = np.linspace(cx - half, cx + half, resolution, dtype=np.float32)
        ys = np.linspace(cy - half_y, cy + half_y, resolution, dtype=np.float32)
        X, Y = np.meshgrid(xs, ys)

        truth = bcc.power_escape_cpu(
            X.ravel(), Y.ravel(),
            np.full(X.size, w_value, np.float32), cfg,
        ).reshape(resolution, resolution)

        predicted, _, _ = cg.grid_field_and_grad(
            net, w_value, cfg, resolution,
            x_min=cx - half, x_max=cx + half,
            y_min=cy - half_y, y_max=cy + half_y,
        )

        mae = float(np.abs(truth - predicted).mean())
        denom = truth.std() * predicted.std()
        correlation = (
            float(((truth - truth.mean()) * (predicted - predicted.mean())).mean()
                  / denom) if denom > 1e-9 else 0.0
        )

        rows.append({
            "zoom": zoom,
            "window_width": 2 * half,
            "mae": mae,
            "correlation": correlation,
            "cycles_truth": cycles_across_window(truth),
            "cycles_net": cycles_across_window(predicted),
            "predicted_ceiling": ceiling / zoom,
            "truth_std": float(truth.std()),
            "net_std": float(predicted.std()),
        })

        if verbose:
            r = rows[-1]
            print(f"  zoom {zoom:>5d}x  width {r['window_width']:.2e}  "
                  f"MAE {mae:.4f}  corr {correlation:+.3f}  "
                  f"cycles truth {r['cycles_truth']:>5.1f} / "
                  f"net {r['cycles_net']:>5.1f}  "
                  f"(predicted {r['predicted_ceiling']:>6.1f})")

    return rows, ceiling


def outward_probe(net, cfg, w_value=2.0, factor=6.0, resolution=192):
    """
    The other side of the mirror: query far OUTSIDE the training box and
    measure the periodicity of what comes back. If the inward-collapse
    and the outward-decay are the same phenomenon, the lattice pitch
    should match the encoder's dominant wavelength in both.
    """
    span_x = (cfg.x_max - cfg.x_min) * factor
    cx = 0.5 * (cfg.x_min + cfg.x_max)
    cy = 0.5 * (cfg.y_min + cfg.y_max)

    field, _, _ = cg.grid_field_and_grad(
        net, w_value, cfg, resolution,
        x_min=cx - span_x / 2, x_max=cx + span_x / 2,
        y_min=cy - span_x / 2, y_max=cy + span_x / 2,
    )

    line = field[resolution // 2] - field[resolution // 2].mean()
    spectrum = np.abs(np.fft.rfft(line))
    # Skip the first few harmonics: a global ramp always dominates there and
    # is not the lattice. We want the repeating structure, not the trend.
    floor = 4
    peak = int(np.argmax(spectrum[floor:]) + floor)
    wavelength = span_x / peak if peak > 0 else np.inf

    return {"span": span_x, "peak_cycles": peak,
            "wavelength_world": float(wavelength), "field": field}


# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------

def plot_sweep(rows, ceiling, out_path):
    zooms = np.array([r["zoom"] for r in rows], float)

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.6), facecolor="#05050a")

    ax = axes[0]
    ax.loglog(zooms, [r["cycles_truth"] for r in rows], "o-",
              color="#4dd2ff", label="ground truth")
    ax.loglog(zooms, [r["cycles_net"] for r in rows], "o-",
              color="#f5a623", label="neural field")
    ax.loglog(zooms, [r["predicted_ceiling"] for r in rows], "--",
              color="#ff5c8a", label="predicted ceiling / Z")
    ax.axhline(1.0, color="#888", lw=0.8, ls=":")
    ax.set_xlabel("zoom factor"); ax.set_ylabel("cycles across window")
    ax.set_title("Structure survival", color="white")
    ax.legend(fontsize=8, facecolor="#111", labelcolor="white")

    ax = axes[1]
    ax.semilogx(zooms, [r["mae"] for r in rows], "o-", color="#f5a623")
    ax.set_xlabel("zoom factor"); ax.set_ylabel("MAE vs truth")
    ax.set_title("Error", color="white")

    ax = axes[2]
    ax.semilogx(zooms, [r["correlation"] for r in rows], "o-", color="#8fd14f")
    ax.axhline(0, color="#888", lw=0.8, ls=":")
    ax.set_xlabel("zoom factor"); ax.set_ylabel("correlation with truth")
    ax.set_title("Agreement", color="white")

    for ax in axes:
        ax.set_facecolor("#05050a")
        ax.tick_params(colors="#aaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        ax.xaxis.label.set_color("#aaa")
        ax.yaxis.label.set_color("#aaa")

    figure.suptitle(
        f"ZOOM PROBE — encoder ceiling {ceiling:.0f} cycles across domain, "
        f"death predicted at Z≈{ceiling:.0f}",
        color="white", fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    figure.savefig(out_path, dpi=150, facecolor="#05050a", bbox_inches="tight")
    plt.close(figure)


def render_ladder(net, cfg, out_path, centre=(-0.745428, 0.113009),
                  w_value=2.0, zooms=(1, 10, 50, 200, 1000), resolution=256):
    """Side-by-side truth / neural / distance-estimate at each zoom."""
    cx, cy = centre
    base_width = cfg.x_max - cfg.x_min

    figure, axes = plt.subplots(3, len(zooms),
                                figsize=(3.1 * len(zooms), 9.4),
                                facecolor="#05050a")

    for column, zoom in enumerate(zooms):
        half = 0.5 * base_width / zoom
        half_y = half * (cfg.y_max - cfg.y_min) / (cfg.x_max - cfg.x_min)

        xs = np.linspace(cx - half, cx + half, resolution, dtype=np.float32)
        ys = np.linspace(cy - half_y, cy + half_y, resolution, dtype=np.float32)
        X, Y = np.meshgrid(xs, ys)

        truth = bcc.power_escape_cpu(
            X.ravel(), Y.ravel(),
            np.full(X.size, w_value, np.float32), cfg,
        ).reshape(resolution, resolution)

        field, dfdx, dfdy = cg.grid_field_and_grad(
            net, w_value, cfg, resolution,
            x_min=cx - half, x_max=cx + half,
            y_min=cy - half_y, y_max=cy + half_y,
        )

        pixel = (2 * half) / resolution
        shade = cg.de_shade(field, dfdx, dfdy, w_value, cfg.max_iter, pixel)

        panels = [
            (truth, "twilight_shifted", f"truth  {zoom}x"),
            (field, "twilight_shifted", f"neural  {zoom}x"),
            (shade, "magma", f"neural DE  {zoom}x"),
        ]

        for row, (image, cmap, title) in enumerate(panels):
            ax = axes[row, column]
            ax.imshow(image, origin="lower", cmap=cmap, aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(title, color="white", fontsize=9)

    figure.suptitle("ZOOM LADDER — the fractal dissolving into its own basis",
                    color="white", fontsize=14, fontweight="bold")
    plt.tight_layout()
    figure.savefig(out_path, dpi=150, facecolor="#05050a", bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--preset", default="cpu")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--resolution", type=int, default=192)
    parser.add_argument("--out", type=Path, default=Path("./zoom_out"))
    args = parser.parse_args()

    from neural_dynamics import _resolve_checkpoint
    args.checkpoint = _resolve_checkpoint(args.checkpoint)
    data = np.load(args.checkpoint, allow_pickle=False)
    cfg = bcc.Config(**json.loads(str(data["config"])))

    net = bcc.BranchCutNet(cfg)
    optimizer = bcc.Adam(net.params(), lr=cfg.lr)
    bcc.load_checkpoint(net, optimizer, args.checkpoint)

    args.out.mkdir(parents=True, exist_ok=True)

    info = cg.encoder_bandwidth(net)
    print("\nENCODER CEILING")
    print(f"  min wavelength (x)     : {info['min_wavelength_world'][0]:.6f} world units")
    print(f"  cycles across domain   : {info['cycles_across_domain'][0]:.1f}")
    print(f"  => predicted structural death at zoom "
          f"Z ~= {info['cycles_across_domain'][0]:.0f}\n")

    print("ZOOM SWEEP")
    rows, ceiling = zoom_sweep(net, cfg, resolution=args.resolution)

    lipschitz = cg.empirical_lipschitz(net, cfg, w_value=2.0, resolution=256)
    truth_grad = cg.true_field_gradient(cfg, w_value=2.0, resolution=256)
    print(f"\nGRADIENT CEILING")
    print(f"  max |grad| network : {lipschitz['max']:.1f}")
    print(f"  max |grad| truth   : {truth_grad['max']:.1f}  "
          f"(unbounded in the limit)")
    print(f"  finest feature the net can resolve (min DE): "
          f"{lipschitz['min_resolvable_de']:.2e} world units")

    outward = outward_probe(net, cfg)
    print(f"\nOUTWARD PROBE (queried {outward['span']:.1f} units wide, "
          f"{outward['span'] / (cfg.x_max - cfg.x_min):.0f}x the training box)")
    print(f"  dominant lattice wavelength : {outward['wavelength_world']:.6f}")
    print(f"  encoder min wavelength      : {info['min_wavelength_world'][0]:.6f}")

    plot_sweep(rows, ceiling, args.out / "zoom_sweep.png")
    (args.out / "zoom_metrics.json").write_text(json.dumps(rows, indent=2))

    if args.render:
        render_ladder(net, cfg, args.out / "zoom_ladder.png")

    print(f"\nWrote {args.out}/zoom_sweep.png and zoom_metrics.json")


if __name__ == "__main__":
    main()
