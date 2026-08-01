#!/usr/bin/env python3
"""
Social-media renders of the learned dynamics.

The headline artifact is a w-sweep of the network's OWN Mandelbrot set: at
every frame the learned map P(z, w) is iterated in place of z**w + c, and w
slides continuously through [2, 6]. Non-integer w has a branch cut, so these
are Multibrots that do not appear in the usual family -- and they are being
produced by a neural network's approximation of the power map, not by the
closed form.

    python3 make_showcase.py --morph            # the main GIF
    python3 make_showcase.py --compare          # true vs learned, side by side
    python3 make_showcase.py --morph --frames 90 --resolution 560

GIF sizing: X caps GIFs at 15 MB, so --max-mb defaults to 14 and the script
re-encodes at a reduced palette / resolution until it fits, reporting what it
had to give up.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm

import neural_dynamics as nd

X_MIN, X_MAX = -2.5, 1.0
Y_MIN, Y_MAX = -1.35, 1.35


def escape_field(net, w_value, resolution, max_iter, learned=True):
    aspect = (X_MAX - X_MIN) / (Y_MAX - Y_MIN)
    width = int(resolution * aspect)
    xs = np.linspace(X_MIN, X_MAX, width, dtype=np.float32)
    ys = np.linspace(Y_MIN, Y_MAX, resolution, dtype=np.float32)
    CX, CY = np.meshgrid(xs, ys)
    shape = (resolution, width)
    if learned:
        return nd.neural_escape(net, CX, CY, w_value, max_iter=max_iter,
                                shape=shape)
    return nd.true_escape(CX, CY, w_value, max_iter=max_iter, shape=shape)


def colorize(field, cmap="magma", gamma=0.55, interior=(6, 4, 14)):
    """
    Escape time -> RGB, with the interior held flat and dark.

    Two things matter for this to read well. The raw field is heavily skewed
    (median 0.047, so nearly all of the exterior escapes immediately) and a
    linear ramp collapses it into one flat tone -- hence the gamma. And the
    interior sits at 1.0, the BRIGHT end of a sequential colormap, which
    inverts the usual convention and makes the set glow instead of the
    boundary. Painting the interior separately fixes both: the exterior keeps
    the full colour range, brightening toward the boundary where escape is
    slowest, and the set itself stays dark.
    """
    v = np.clip(field, 0.0, 1.0)
    inside = v >= 0.999
    shaded = np.power(np.where(inside, 0.0, v), gamma)
    rgb = (matplotlib.colormaps[cmap](shaded)[..., :3] * 255).astype(np.uint8)
    rgb[inside] = np.asarray(interior, np.uint8)
    return rgb


def stamp(img_array, text, scale=2):
    """Tiny 5x7 bitmap label -- avoids a font dependency."""
    glyphs = {
        "0": ["111", "101", "101", "101", "111"], "1": ["010", "110", "010", "010", "111"],
        "2": ["111", "001", "111", "100", "111"], "3": ["111", "001", "111", "001", "111"],
        "4": ["101", "101", "111", "001", "001"], "5": ["111", "100", "111", "001", "111"],
        "6": ["111", "100", "111", "101", "111"], "7": ["111", "001", "010", "010", "010"],
        "8": ["111", "101", "111", "101", "111"], "9": ["111", "101", "111", "001", "111"],
        ".": ["000", "000", "000", "000", "010"], "=": ["000", "111", "000", "111", "000"],
        "w": ["000", "101", "101", "111", "101"], " ": ["000", "000", "000", "000", "000"],
    }
    img = img_array
    x0, y0 = 8, img.shape[0] - 8 - 5 * scale
    for ch in text:
        pattern = glyphs.get(ch, glyphs[" "])
        for r, row in enumerate(pattern):
            for c, bit in enumerate(row):
                if bit == "1":
                    img[y0 + r * scale:y0 + (r + 1) * scale,
                        x0 + c * scale:x0 + (c + 1) * scale] = 255
        x0 += (len(pattern[0]) + 1) * scale
    return img


def write_gif(frames, path, duration_ms, max_mb, colors=200):
    """Encode, and step the palette/size down until it fits the cap."""
    attempt = 0
    while True:
        pil = [Image.fromarray(f).convert(
            "P", palette=Image.ADAPTIVE, colors=colors) for f in frames]
        pil[0].save(path, save_all=True, append_images=pil[1:],
                    duration=duration_ms, loop=0, optimize=True, disposal=2)
        size_mb = path.stat().st_size / 2**20
        if size_mb <= max_mb or attempt >= 4:
            return size_mb, colors, frames[0].shape
        attempt += 1
        colors = max(32, colors // 2)
        if attempt >= 2:                       # palette alone was not enough
            frames = [np.asarray(Image.fromarray(f).resize(
                (int(f.shape[1] * 0.8), int(f.shape[0] * 0.8)), Image.LANCZOS))
                for f in frames]


def morph(net, out_path, frames=72, resolution=440, max_iter=64,
          w_lo=2.0, w_hi=6.0, cmap="magma", max_mb=14.0,
          duration_ms=70, label=True):
    """w sweeping through [w_lo, w_hi], iterating the LEARNED map."""
    ws = np.linspace(w_lo, w_hi, frames)
    imgs = []
    for i, wv in enumerate(ws):
        field = escape_field(net, float(wv), resolution, max_iter, learned=True)
        img = colorize(field, cmap)
        if label:
            img = stamp(img.copy(), f"w={wv:.2f}")
        imgs.append(img)
        print(f"\r  frame {i + 1}/{frames}  w={wv:.3f}", end="", flush=True)
    print()
    # ping-pong so the loop is seamless without re-rendering
    loop = imgs + imgs[-2:0:-1]
    size_mb, colors, shape = write_gif(loop, out_path, duration_ms, max_mb)
    print(f"  wrote {out_path}  {size_mb:.1f} MB  "
          f"{len(loop)} frames  {shape[1]}x{shape[0]}  {colors} colors")


def compare(net, out_path, frames=72, resolution=340, max_iter=64,
            w_lo=2.0, w_hi=6.0, cmap="magma", max_mb=14.0,
            duration_ms=70):
    """True dynamics beside learned dynamics, same w, same sweep."""
    ws = np.linspace(w_lo, w_hi, frames)
    imgs = []
    for i, wv in enumerate(ws):
        left = colorize(escape_field(net, float(wv), resolution, max_iter,
                                     learned=False), cmap)
        right = colorize(escape_field(net, float(wv), resolution, max_iter,
                                      learned=True), cmap)
        gap = np.zeros((left.shape[0], 4, 3), np.uint8)
        gap[:] = 255
        img = np.concatenate([left, gap, right], axis=1)
        imgs.append(stamp(img, f"w={wv:.2f}"))
        print(f"\r  frame {i + 1}/{frames}  w={wv:.3f}", end="", flush=True)
    print()
    loop = imgs + imgs[-2:0:-1]
    size_mb, colors, shape = write_gif(loop, out_path, duration_ms, max_mb)
    print(f"  wrote {out_path}  {size_mb:.1f} MB  "
          f"{len(loop)} frames  {shape[1]}x{shape[0]}  {colors} colors")
    print("  (left: true z^w + c   right: the network's learned map)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("dyn_w26/map.npz"))
    parser.add_argument("--morph", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument("--resolution", type=int, default=440)
    parser.add_argument("--max-iter", type=int, default=96)
    parser.add_argument("--cmap", default="magma")
    parser.add_argument("--max-mb", type=float, default=14.0)
    parser.add_argument("--duration-ms", type=int, default=70)
    parser.add_argument("--out", type=Path, default=Path("./figures"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    net = nd.load_map(args.checkpoint)

    if args.morph or not (args.morph or args.compare):
        print("Rendering w-sweep of the LEARNED map...")
        morph(net, args.out / "showcase_w_morph.gif", frames=args.frames,
              resolution=args.resolution, max_iter=args.max_iter,
              cmap=args.cmap, max_mb=args.max_mb,
              duration_ms=args.duration_ms)
    if args.compare:
        print("Rendering true vs learned...")
        compare(net, args.out / "showcase_true_vs_learned.gif",
                frames=args.frames, resolution=int(args.resolution * 0.78),
                max_iter=args.max_iter, cmap=args.cmap, max_mb=args.max_mb,
                duration_ms=args.duration_ms)


if __name__ == "__main__":
    main()
