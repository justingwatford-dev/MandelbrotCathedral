#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║   MULTIBROT GENESIS — Neural Net from Scratch            ║
║   Swap `import numpy as xp` → `import cupy as xp`       ║
║   for full GPU acceleration on your machine              ║
╚══════════════════════════════════════════════════════════╝

Instead of just learning the standard Mandelbrot set (z^2 + c),
this network learns the continuous Multibrot family: z^w + c.
Inputs are (x, y, w). The animation sweeps through w, creating
a morphing, genetically engineered fractal continuum.
"""

# ── Console: Windows defaults stdout to cp1252, which cannot encode the box
#    drawing and emoji below and raises UnicodeEncodeError mid-print ─────────
import os
import sys
import warnings
from pathlib import Path as _Path
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


# ── CUDA discovery: the nvidia-* pip wheels install their DLLs under
#    site-packages/nvidia/, which is not on the Windows DLL search path and
#    is not somewhere CuPy looks. Without this, nvrtc is silently missing
#    even though every required library is installed ───────────────────────
def _register_cuda_dll_dirs():
    """Put pip-installed CUDA DLLs on the search path. Returns dirs added."""
    if not hasattr(os, "add_dll_directory"):        # POSIX: uses RPATH instead
        return []
    try:
        import nvidia
    except ImportError:
        return []
    added = []
    for root in map(_Path, nvidia.__path__):
        # CUDA 13 wheels use a flat cu13/bin/x86_64; older ones use <lib>/bin
        for cand in sorted(root.glob("cu*/bin/*")) + sorted(root.glob("*/bin")):
            if cand.is_dir() and any(cand.glob("*.dll")):
                try:
                    os.add_dll_directory(str(cand))
                    added.append(cand)
                except OSError:
                    pass
    return added


_CUDA_DLL_DIRS = _register_cuda_dll_dirs()

# ── Backend: CuPy on GPU, fall back to NumPy on CPU ──────────────────────────
try:
    if os.environ.get("MANDELBROT_FORCE_CPU"):
        raise RuntimeError("MANDELBROT_FORCE_CPU set")
    if _CUDA_DLL_DIRS:
        # We resolved the libraries above, so CuPy's CUDA_PATH probe warning
        # is a false alarm. Suppress that one message, nothing else.
        warnings.filterwarnings(
            "ignore", message=".*CUDA path could not be detected.*")
    import cupy as xp
    # `import cupy` succeeds even when the CUDA runtime DLLs are missing — the
    # failure only surfaces on first device touch. Probe before trusting it.
    _probe = xp.arange(4, dtype=xp.float32)
    float((_probe * 2).sum())
    del _probe
    to_numpy = xp.asnumpy
    GPU = True
    _dev = xp.cuda.Device(0)
    _name = xp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    _free, _total = _dev.mem_info
    print(f"🚀  CuPy / GPU mode — {_name}, sm_{_dev.compute_capability}, "
          f"{_free / 2**30:.1f}/{_total / 2**30:.1f} GiB free")
except Exception as _backend_err:
    import numpy as xp
    to_numpy = lambda x: np.asarray(x)
    GPU = False
    print("💻  NumPy / CPU mode — swap to CuPy for full GPU speed")
    print(f"    (CuPy unavailable: {type(_backend_err).__name__}: {_backend_err})")

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import time, sys
from matplotlib.colors import hsv_to_rgb

try:
    from PIL import Image
except ImportError:
    Image = None

# ──────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────────────────────────────────────
class Cfg:
    seed       = 42
    # ── scale these up on GPU ────────────────────────────────────────────────
    n_data     = 2_000_000 if GPU else 100_000
    batch_size = 65_536    if GPU else 2_048
    epochs     = 300       if GPU else 30
    hidden     = 512       if GPU else 192
    n_blocks   = 5         if GPU else 3
    # ── architecture ─────────────────────────────────────────────────────────
    n_freq     = 16        # Fourier frequencies per axis
    max_iter   = 256       # Mandelbrot escape iterations
    # ── training ─────────────────────────────────────────────────────────────
    lr         = 3e-4
    lr_min     = 1e-6
    # ── output ───────────────────────────────────────────────────────────────
    vis_res    = 400       # use 512+ on GPU
    vis_chunk  = 8192      # rows processed at a time during inference
    save_every = 10
    out_dir    = Path("./outputs")
    
    # ── multibrot continuum ──────────────────────────────────────────────────
    w_min      = 2.0       # Standard Mandelbrot
    w_max      = 6.0       # 6-armed star fractal

    # ── genesis sweep renderer ───────────────────────────────────────────────
    trip_res         = 512 if GPU else 320
    trip_frames      = 120 if GPU else 60
    trip_feedback    = 3
    trip_warp        = 0.05
    trip_duration_ms = 50

C = Cfg()
xp.random.seed(C.seed)
np.random.seed(C.seed)

# ──────────────────────────────────────────────────────────────────────────────
#  MULTIBROT GROUND TRUTH  (CPU only — called once)
# ──────────────────────────────────────────────────────────────────────────────
def multibrot_cpu(xs: np.ndarray, ys: np.ndarray, ws: np.ndarray) -> np.ndarray:
    """Smooth escape-time coloring for z^w + c, vectorised NumPy."""
    c     = xs.astype(np.complex64) + 1j * ys.astype(np.complex64)
    w     = ws.astype(np.float32)
    z     = np.zeros_like(c)
    out   = np.full(len(c), float(C.max_iter), dtype=np.float32)
    alive = np.ones(len(c), dtype=bool)
    
    for i in range(1, C.max_iter + 1):
        z_alive = z[alive]
        w_alive = ws[alive]
        
        # Compute z^w using polar form: z^w = r^w * (cos(w*theta) + i*sin(w*theta))
        r = np.abs(z_alive)
        theta = np.angle(z_alive)
        r_safe = np.clip(r, 1e-10, None) # avoid 0^w issues
        
        z_pow_w = (r_safe ** w_alive) * (np.cos(w_alive * theta) + 1j * np.sin(w_alive * theta))
        z[alive] = z_pow_w + c[alive]
        
        esc = alive & (np.abs(z) > 2.0)
        out[esc] = i - np.log2(np.log2(np.abs(z[esc]).clip(1.0001)))
        alive[esc] = False
        
    return (out / C.max_iter).astype(np.float32)

def gen_dataset(n: int):
    """
    Generate a 3D hybrid dataset: (x, y, w).
    """
    print(f"  Computing {n:,} Multibrot samples …", flush=True)
    t0 = time.time()

    n_focus  = int(n * 0.67)
    n_uniform = n - n_focus

    # ── Ordinary global samples ──────────────────────────────────────────────
    ux = np.random.uniform(-2.5,  1.0,  n_uniform).astype(np.float32)
    uy = np.random.uniform(-1.25, 1.25, n_uniform).astype(np.float32)
    uw = np.random.uniform(C.w_min, C.w_max, n_uniform).astype(np.float32)
    ul = multibrot_cpu(ux, uy, uw)

    # ── Scout for suspiciously slow-escaping locations ──────────────────────
    scout_n = min(500_000, max(100_000, n // 4))
    sx = np.random.uniform(-2.5,  1.0,  scout_n).astype(np.float32)
    sy = np.random.uniform(-1.25, 1.25, scout_n).astype(np.float32)
    sw = np.random.uniform(C.w_min, C.w_max, scout_n).astype(np.float32)
    sl = multibrot_cpu(sx, sy, sw)

    candidate_ids = np.flatnonzero((sl > 0.025) & (sl < 0.999))
    if len(candidate_ids) < 256:
        candidate_ids = np.flatnonzero(sl < 0.999)
    if len(candidate_ids) == 0:
        candidate_ids = np.arange(scout_n)

    center_ids = np.random.choice(candidate_ids, size=n_focus, replace=True)
    cx = sx[center_ids]
    cy = sy[center_ids]
    cw = sw[center_ids]

    # Log-uniform radii create a multi-scale boundary cloud.
    log_r = np.random.uniform(np.log10(1e-4), np.log10(0.75), n_focus)
    radius = np.power(10.0, log_r).astype(np.float32)
    theta  = np.random.uniform(0.0, 2.0 * np.pi, n_focus).astype(np.float32)

    fx = np.clip(cx + radius * np.cos(theta), -2.5, 1.0).astype(np.float32)
    fy = np.clip(cy + radius * np.sin(theta), -1.25, 1.25).astype(np.float32)
    # Add a little noise to w so the network learns the continuous morph
    fw = np.clip(cw + np.random.normal(0, 0.1, n_focus), C.w_min, C.w_max).astype(np.float32)
    fl = multibrot_cpu(fx, fy, fw)

    xs = np.concatenate([ux, fx])
    ys = np.concatenate([uy, fy])
    ws = np.concatenate([uw, fw])
    ls = np.concatenate([ul, fl])

    order = np.random.permutation(len(xs))
    xs, ys, ws, ls = xs[order], ys[order], ws[order], ls[order]

    print(f"  Done in {time.time() - t0:.1f}s  (in-set: {(ls == 1).mean() * 100:.1f}%)")
    return xp.array(xs), xp.array(ys), xp.array(ws), xp.array(ls)

def true_image(res=C.vis_res, w_val=3.0) -> np.ndarray:
    xs = np.linspace(-2.5,  1.0,  res, dtype=np.float32)
    ys = np.linspace(-1.25, 1.25, res, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    gw = np.full_like(gx, w_val)
    return multibrot_cpu(gx.ravel(), gy.ravel(), gw.ravel()).reshape(res, res)

# ──────────────────────────────────────────────────────────────────────────────
#  LAYERS (From Scratch)
# ──────────────────────────────────────────────────────────────────────────────
class Linear:
    def __init__(self, d_in: int, d_out: int):
        std      = float(xp.sqrt(xp.array(2.0 / d_in)))
        self.W   = (xp.random.randn(d_in, d_out) * std).astype(xp.float32)
        self.b   = xp.zeros(d_out, dtype=xp.float32)
        self.dW  = self.db = None
        self._x  = None

    def fwd(self, x):
        self._x  = x
        return x @ self.W + self.b

    def bwd(self, g):
        self.dW  = self._x.T @ g
        self.db  = g.sum(0)
        return g @ self.W.T

    def params(self):
        return [(self.W, "dW", self), (self.b, "db", self)]

class LayerNorm:
    _EPS = xp.float32(1e-5)
    def __init__(self, dim: int):
        self.g  = xp.ones(dim,  dtype=xp.float32)
        self.b  = xp.zeros(dim, dtype=xp.float32)
        self.dg = self.db = None
        self._hat = self._rs = None

    def fwd(self, x):
        mu      = x.mean(-1, keepdims=True)
        var     = x.var(-1,  keepdims=True)
        rs      = xp.float32(1.0) / xp.sqrt(var + self._EPS)
        hat     = (x - mu) * rs
        self._hat, self._rs = hat, rs
        return self.g * hat + self.b

    def bwd(self, g):
        hat, rs = self._hat, self._rs
        D       = xp.float32(g.shape[-1])
        self.dg = (g * hat).sum(0)
        self.db =  g.sum(0)
        dxh     = g * self.g
        return rs / D * (
            D * dxh - dxh.sum(-1, keepdims=True) - hat * (dxh * hat).sum(-1, keepdims=True)
        )

    def params(self):
        return [(self.g, "dg", self), (self.b, "db", self)]

_K1 = xp.float32(0.7978845608)
_K2 = xp.float32(0.044715)

class GELU:
    _x = _t = None
    def fwd(self, x):
        u        = _K1 * (x + _K2 * x ** 3)
        t        = xp.tanh(u)
        self._x, self._t = x, t
        return xp.float32(0.5) * x * (xp.float32(1.) + t)

    def bwd(self, g):
        x, t  = self._x, self._t
        sech2 = xp.float32(1.) - t ** 2
        du_dx = _K1 * (xp.float32(1.) + xp.float32(3.) * _K2 * x ** 2)
        dy_dx = (xp.float32(0.5) * (xp.float32(1.) + t) + xp.float32(0.5) * x * sech2 * du_dx)
        return g * dy_dx

class Sigmoid:
    _out = None
    def fwd(self, x):
        self._out = xp.float32(1.) / (xp.float32(1.) + xp.exp(-x.clip(-60, 60)))
        return self._out
    def bwd(self, g):
        s = self._out
        return g * s * (xp.float32(1.) - s)

class ResBlock:
    def __init__(self, dim: int):
        self.ln1 = LayerNorm(dim)
        self.fc1 = Linear(dim, dim)
        self.act = GELU()
        self.ln2 = LayerNorm(dim)
        self.fc2 = Linear(dim, dim)

    def fwd(self, x):
        h = self.ln1.fwd(x)
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
        g2 = self.ln1.bwd(g2)
        return g2 + g

    def params(self):
        p = []
        for layer in (self.ln1, self.fc1, self.ln2, self.fc2):
            p.extend(layer.params())
        return p

# ──────────────────────────────────────────────────────────────────────────────
#  FULL NETWORK
# ──────────────────────────────────────────────────────────────────────────────
class Net:
    """
    (x, y, w) → Fourier encode → proj → ResBlocks × N → LN → FC → Sigmoid → ŷ
    """
    def __init__(self):
        F        = C.n_freq
        in_dim   = 3 + 6 * F          # raw (x,y,w) + sin/cos for x, y, and w
        self.proj   = Linear(in_dim, C.hidden)
        self.blocks = [ResBlock(C.hidden) for _ in range(C.n_blocks)]
        self.ln_out = LayerNorm(C.hidden)
        self.fc_out = Linear(C.hidden, 1)
        self.sig    = Sigmoid()
        self.freqs  = xp.array(2.0 ** np.linspace(0, 8, F), dtype=xp.float32)

    def _encode(self, x, y, w):
        fx = x[:, None] * self.freqs
        fy = y[:, None] * self.freqs
        fw = w[:, None] * self.freqs
        return xp.concatenate([
            x[:, None], y[:, None], w[:, None],
            xp.sin(fx), xp.cos(fx),
            xp.sin(fy), xp.cos(fy),
            xp.sin(fw), xp.cos(fw),
        ], axis=1)

    def fwd(self, x, y, w):
        h = self._encode(x, y, w)
        h = self.proj.fwd(h)
        for blk in self.blocks:
            h = blk.fwd(h)
        h = self.ln_out.fwd(h)
        h = self.fc_out.fwd(h)
        return self.sig.fwd(h).squeeze(-1)

    def bwd(self, g):
        g = self.sig.bwd(g[:, None])
        g = self.fc_out.bwd(g)
        g = self.ln_out.bwd(g)
        for blk in reversed(self.blocks):
            g = blk.bwd(g)
        self.proj.bwd(g)

    def params(self):
        p = list(self.proj.params())
        for blk in self.blocks:
            p.extend(blk.params())
        p.extend(self.ln_out.params())
        p.extend(self.fc_out.params())
        return p

    def predict_grid(self, res=C.vis_res, w_val=3.0) -> np.ndarray:
        xs = np.linspace(-2.5,  1.0,  res, dtype=np.float32)
        ys = np.linspace(-1.25, 1.25, res, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)
        flat_x, flat_y = gx.ravel(), gy.ravel()
        chunks, chunk  = [], C.vis_chunk
        for i in range(0, len(flat_x), chunk):
            cx = xp.array(flat_x[i:i+chunk])
            cy = xp.array(flat_y[i:i+chunk])
            cw = xp.full_like(cx, w_val)
            chunks.append(to_numpy(self.fwd(cx, cy, cw)))
        return np.concatenate(chunks).reshape(res, res)

    def param_count(self) -> int:
        return sum(p.size for p, _, _ in self.params())

# ──────────────────────────────────────────────────────────────────────────────
#  ADAM OPTIMIZER
# ──────────────────────────────────────────────────────────────────────────────
class Adam:
    def __init__(self, params, lr=C.lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2 = lr, xp.float32(b1), xp.float32(b2)
        self.eps    = xp.float32(eps)
        self.params = params
        self.ms     = [xp.zeros_like(p) for p, _, _ in params]
        self.vs     = [xp.zeros_like(p) for p, _, _ in params]
        self.t      = 0

    def step(self):
        self.t += 1
        lr_t = xp.float32(self.lr) * (
            xp.sqrt(xp.float32(1. - self.b2 ** self.t))
            / xp.float32(1. - self.b1 ** self.t)
        )
        for i, (p, attr, obj) in enumerate(self.params):
            g = getattr(obj, attr)
            if g is None:
                continue
            self.ms[i] = self.b1 * self.ms[i] + (xp.float32(1.) - self.b1) * g
            self.vs[i] = self.b2 * self.vs[i] + (xp.float32(1.) - self.b2) * g * g
            p -= lr_t * self.ms[i] / (xp.sqrt(self.vs[i]) + self.eps)

# ──────────────────────────────────────────────────────────────────────────────
#  VISUALISATION
# ──────────────────────────────────────────────────────────────────────────────
_CM   = "twilight_shifted"
_BG   = "#05050a"

def _style_ax(ax):
    ax.tick_params(colors="#555")
    ax.set_facecolor(_BG)
    for sp in ax.spines.values():
        sp.set_edgecolor("#222")

def save_comparison(net: Net, epoch: int, loss: float, out_dir: Path) -> Path:
    print("  Rendering ground truth comparisons …", flush=True)
    # Show w=2 (Mandelbrot) and w=4 side by side
    nn2 = net.predict_grid(w_val=2.0)
    nn4 = net.predict_grid(w_val=4.0)
    tr2 = true_image(w_val=2.0)
    tr4 = true_image(w_val=4.0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 11), facecolor=_BG)
    kw = dict(cmap=_CM, origin="lower", aspect="equal",
              extent=[-2.5, 1.0, -1.25, 1.25], vmin=0, vmax=1)
    
    axes[0,0].imshow(tr2, **kw); axes[0,0].set_title("Ground Truth (w=2.0)", color="white", fontsize=11)
    axes[0,1].imshow(nn2, **kw); axes[0,1].set_title(f"Neural Net (w=2.0) · epoch {epoch}", color="white", fontsize=11)
    axes[1,0].imshow(tr4, **kw); axes[1,0].set_title("Ground Truth (w=4.0)", color="white", fontsize=11)
    axes[1,1].imshow(nn4, **kw); axes[1,1].set_title(f"Neural Net (w=4.0) · epoch {epoch}", color="white", fontsize=11)

    for ax in axes.ravel():
        _style_ax(ax)

    fig.suptitle(
        f"MULTIBROT GENESIS  ·  epoch {epoch:04d}  ·  loss {loss:.5f}",
        color="white", fontsize=14, fontweight="bold", y=0.98
    )
    plt.tight_layout()
    path = out_dir / f"epoch_{epoch:04d}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    return path

def save_final(net: Net, losses: list, out_dir: Path) -> Path:
    nn = net.predict_grid(w_val=3.0)
    tr = true_image(w_val=3.0)
    
    fig = plt.figure(figsize=(19, 6.5), facecolor=_BG)
    gs  = fig.add_gridspec(1, 3, wspace=0.06)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    kw = dict(cmap=_CM, origin="lower", aspect="equal",
              extent=[-2.5, 1.0, -1.25, 1.25], vmin=0, vmax=1)
    ax1.imshow(tr, **kw); ax1.set_title("Ground Truth (w=3.0)", color="white", fontsize=11)
    ax2.imshow(nn, **kw); ax2.set_title(f"Neural Net (w=3.0)", color="white", fontsize=11)
    for ax in (ax1, ax2): _style_ax(ax)

    ax3.plot(losses, color="#f5a623", lw=1.5)
    ax3.set_yscale("log")
    ax3.set_xlabel("Epoch", color="#888"); ax3.set_ylabel("MSE loss", color="#888")
    ax3.set_title("Training loss", color="white", fontsize=11)
    ax3.set_facecolor(_BG); ax3.tick_params(colors="#888")
    for sp in ax3.spines.values(): sp.set_edgecolor("#333")

    fig.suptitle("MULTIBROT GENESIS — FINAL", color="white", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "mandelbrot_FINAL.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    return path

# ──────────────────────────────────────────────────────────────────────────────
#  GENESIS SWEEP ENGINE
# ──────────────────────────────────────────────────────────────────────────────

def predict_points(net: Net, xs: np.ndarray, ys: np.ndarray, ws: np.ndarray) -> np.ndarray:
    xs = np.asarray(xs, dtype=np.float32).ravel()
    ys = np.asarray(ys, dtype=np.float32).ravel()
    ws = np.asarray(ws, dtype=np.float32).ravel()
    outputs = []
    for i in range(0, len(xs), C.vis_chunk):
        bx = xp.array(xs[i:i + C.vis_chunk])
        by = xp.array(ys[i:i + C.vis_chunk])
        bw = xp.array(ws[i:i + C.vis_chunk])
        outputs.append(to_numpy(net.fwd(bx, by, bw)))
    return np.concatenate(outputs)

def genesis_frame(net: Net, phase: float, res: int = C.trip_res) -> np.ndarray:
    """
    Sweeps through the power 'w' and applies flow-field warping.
    """
    # Phase 0.0 -> 1.0 maps to w 2.0 -> 6.0 -> 2.0 (looping sweep)
    w_val = C.w_min + (C.w_max - C.w_min) * (0.5 - 0.5 * np.cos(phase * 2.0 * np.pi))

    xs = np.linspace(-2.5, 1.0, res, dtype=np.float32)
    ys = np.linspace(-1.25, 1.25, res, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    W = np.full_like(X, w_val)

    original_x = X.copy()
    original_y = Y.copy()

    tau = np.float32(2.0 * np.pi)
    eps = np.float32(1e-7)

    # ── Recursive domain warping based on network gradients ──────────────────
    for k in range(C.trip_feedback):
        pred = predict_points(net, X, Y, W).reshape(res, res)
        grad_y, grad_x = np.gradient(pred)
        magnitude = np.sqrt(grad_x * grad_x + grad_y * grad_y + eps)

        polar_angle = np.arctan2(Y, X + 0.65)

        # Twist parameters shift with w_val to emphasize the morphing symmetry
        twist = (
            tau * (phase + 0.137 * k)
            + (w_val - 1.0) * pred
            + 2.8 * polar_angle
            + 0.9 * np.sin((w_val * 2.0) * polar_angle - tau * phase)
        )

        strength = (C.trip_warp / np.float32(k + 1)) * np.tanh(22.0 * magnitude)
        drift_x = np.cos(twist) - 0.42 * grad_y / magnitude
        drift_y = np.sin(twist) + 0.42 * grad_x / magnitude

        X += strength * drift_x
        Y += strength * drift_y
        
        # Global breathing tied to the sweeping w
        breath = 0.015 * np.sin(tau * phase + 2.0 * polar_angle)
        X += breath * (original_x + 0.65)
        Y += breath * original_y

        X = np.clip(X, -2.5, 1.0)
        Y = np.clip(Y, -1.25, 1.25)

    field = predict_points(net, X, Y, W).reshape(res, res)

    gy, gx = np.gradient(field)
    gradient = np.sqrt(gx * gx + gy * gy + eps)

    angle = np.arctan2(Y, X + 0.65)
    radius = np.sqrt((X + 0.65) ** 2 + Y ** 2)
    edge = np.tanh(35.0 * gradient)

    # Color shifts dramatically based on w_val, creating a temporal chromatic shift
    interference = 0.5 + 0.5 * np.sin(52.0 * field + (w_val * 3.0) * angle - 5.0 * radius + tau * phase)

    hue = np.mod(
        0.75 + 0.2 * (w_val / 4.0) + 1.5 * field + 0.17 * edge + 0.13 * np.sin(3.0 * angle - tau * phase),
        1.0,
    )
    saturation = np.clip(0.6 + 0.3 * interference + 0.15 * edge, 0.0, 1.0)
    value = np.clip(0.02 + 0.75 * np.power(np.clip(field, 0.0, 1.0), 0.4) + 0.35 * edge * interference, 0.0, 1.0)

    hsv = np.stack([hue, saturation, value], axis=-1)
    rgb = hsv_to_rgb(hsv)
    rgb = np.power(np.clip(rgb, 0.0, 1.0), 0.85) # Gamma adjust

    return rgb

def save_genesis_sweep(net: Net, out_dir: Path):
    """Save one poster and one looping neural-fractal morph animation."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n── Genesis Sweep Ritual ──────────────────────")

    poster = genesis_frame(net, phase=0.217)
    poster_path = out_dir / "multibrot_GENESIS_POSTER.png"
    plt.imsave(poster_path, poster)
    print(f"  Poster → {poster_path.name}")

    if Image is None:
        print("  Pillow unavailable; GIF skipped.")
        return poster_path, None

    frames = []
    for i in range(C.trip_frames):
        phase = i / C.trip_frames
        rgb = genesis_frame(net, phase=phase, res=C.trip_res)
        frame = Image.fromarray(np.uint8(np.clip(rgb, 0.0, 1.0) * 255.0), mode="RGB")
        frames.append(frame)

        if i == 0 or (i + 1) % 15 == 0:
            print(f"  Rendered frame {i + 1}/{C.trip_frames}")

    gif_path = out_dir / "multibrot_GENESIS_SWEEP.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=C.trip_duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )
    print(f"  Loop   → {gif_path.name}")
    return poster_path, gif_path

# ──────────────────────────────────────────────────────────────────────────────
#  TRAINING LOOP
# ──────────────────────────────────────────────────────────────────────────────
def cosine_lr(ep, total):
    return C.lr_min + 0.5 * (C.lr - C.lr_min) * (1 + np.cos(np.pi * ep / total))

def train():
    C.out_dir.mkdir(parents=True, exist_ok=True)

    print("\n── Dataset ──────────────────────────────────")
    X, Y, W, L = gen_dataset(C.n_data)
    N = C.n_data

    print("\n── Network ──────────────────────────────────")
    net = Net()
    n_params = net.param_count()
    n_batches = N // C.batch_size
    print(f"  Params : {n_params:,}")
    print(f"  Arch   : proj + {C.n_blocks} ResBlocks + output  (hidden={C.hidden}, freqs={C.n_freq})")
    print(f"  Batches: {n_batches} × {C.batch_size}")
    print(f"  Epochs : {C.epochs}")

    opt    = Adam(net.params())
    losses = []
    best   = float("inf")

    print("\n── Training ─────────────────────────────────")
    t0 = time.time()

    for ep in range(1, C.epochs + 1):
        opt.lr = cosine_lr(ep, C.epochs)

        idx  = xp.random.permutation(N)
        X, Y, W, L = X[idx], Y[idx], W[idx], L[idx]

        ep_loss = xp.float32(0.)

        for b in range(n_batches):
            sl  = slice(b * C.batch_size, (b + 1) * C.batch_size)
            x, y, w, t = X[sl], Y[sl], W[sl], L[sl]

            pred = net.fwd(x, y, w)
            diff = pred - t
            loss = (diff * diff).mean()
            ep_loss += loss

            g = xp.float32(2.) * diff / xp.float32(C.batch_size)
            net.bwd(g)
            opt.step()

        avg  = float(to_numpy(ep_loss)) / n_batches
        losses.append(avg)
        flag = " ★" if avg < best else ""
        if avg < best: best = avg

        elapsed = time.time() - t0
        eta     = elapsed / ep * (C.epochs - ep)
        print(f"  [{ep:3d}/{C.epochs}]  loss={avg:.5f}{flag}  lr={opt.lr:.1e}  {elapsed:.0f}s elapsed  ETA {eta:.0f}s")

        if ep % C.save_every == 0 or ep == 1 or ep == C.epochs:
            p = save_comparison(net, ep, avg, C.out_dir)
            print(f"    → {p.name}")

    final_path = save_final(net, losses, C.out_dir)
    poster_path, gif_path = save_genesis_sweep(net, C.out_dir)

    total = time.time() - t0
    print(f"\n✓  Done in {total:.0f}s  |  best loss {best:.5f}")
    print(f"   Final panel → {final_path}")
    print(f"   Genesis poster → {poster_path}")
    if gif_path is not None:
        print(f"   Genesis sweep → {gif_path}")

if __name__ == "__main__":
    train()