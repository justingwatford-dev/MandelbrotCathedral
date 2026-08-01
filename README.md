# Branch-Cut Cathedral — seven extensions

Seven modules built on top of `MandelbrotLeary.py` / `BranchCutCathedral.py`.
Everything here was gradient-checked or measured, not just written. The
measured results below came from a **single CPU core**, so they are
proof-of-correctness numbers — the shapes should hold, the constants should
not be trusted until you rerun them at scale.

Drop these beside your existing two files. `import cupy as xp` still works
throughout; nothing added here assumes NumPy.

---

## 0. Environment

Verified on Windows 11, Python 3.13, RTX 5070 (12 GB, sm_120), driver 610.62.

**Console must be UTF-8.** `MandelbrotLeary.py` prints box-drawing characters
and emoji. Windows defaults stdout to cp1252, which cannot encode them, and
the resulting `UnicodeEncodeError` fires *at import time* — so every module
here dies before reaching `main()`. The file now reconfigures `sys.stdout` /
`sys.stderr` to UTF-8 on import, with `errors="replace"` so a legacy console
degrades to `?` instead of crashing.

**GPU is optional and self-detecting.** The backend probe actually runs a
small array operation before trusting CuPy:

```
🚀  CuPy / GPU mode — NVIDIA GeForce RTX 5070, sm_120, 10.8/11.9 GiB free
💻  NumPy / CPU mode — swap to CuPy for full GPU speed
    (CuPy unavailable: RuntimeError: ...)
```

This matters because `import cupy` **succeeds even when the CUDA runtime
DLLs are missing** — the failure only surfaces on first device touch, as a
`RuntimeError`, not an `ImportError`. A plain `except ImportError` guard
sails straight past it and leaves you in a fake GPU mode with `n_data`
scaled to 2,000,000. Set `MANDELBROT_FORCE_CPU=1` to pin CPU regardless.

To check the backend without running anything:

```bash
python -c "import MandelbrotLeary as m; print('GPU:', m.GPU)"
```

**CUDA setup (Windows).** You do *not* need the full CUDA Toolkit. The pip
wheels are enough — note that for CUDA 13 NVIDIA dropped the `-cu13` suffix,
so the older `nvidia-*-cu13` names are deprecated shims that fail to build:

```bash
pip install cupy-cuda13x nvidia-cuda-nvrtc nvidia-cuda-runtime nvidia-cublas nvidia-cufft nvidia-curand nvidia-cusolver nvidia-cusparse nvidia-nvjitlink
```

Those wheels land in `site-packages/nvidia/cu13/bin/x86_64/`, which is **not**
on the Windows DLL search path and **not** somewhere CuPy looks — so `nvrtc`
stays missing even with everything correctly installed. `MandelbrotLeary.py`
registers that directory via `os.add_dll_directory()` at import. On Linux the
wheels carry RPATHs and the shim no-ops.

**Throughput**, 251k-param model, 100k samples, batch 4096:

| backend | per epoch | 2M-sample epoch | peak GPU mem |
|---------|----------:|----------------:|-------------:|
| CPU (1 core) | 2.06 s | ~41 s | — |
| RTX 5070 | 0.35 s | 5.78 s | 1.41 GiB |

The 2M-sample GPU default uses only 1.4 GiB of 12 GB, so there is room to
push `n_data` a lot further. At 100k the GPU is launch-latency bound and only
~6x the CPU; the gap widens with size.

---

## 1. `cathedral_grad.py` — analytic input gradients

Your `bwd(g)` methods already return dL/dx. The training loop discards it at
`proj.bwd(gradient)`. This catches it and pushes one more step back through
the Fourier encoder, giving exact per-sample d(output)/d(x, y, w) for one
extra backward pass.

```bash
python3 cathedral_grad.py --gradcheck
python3 cathedral_grad.py --bandwidth --preset gpu
```

**Verified:** all six derivatives at correlation 1.000000, worst median
relative error 6.7e-4.

> Note on the gradcheck: it first appeared to fail at 9e-2 relative error.
> An h-sweep showed textbook h^2 convergence down to the float32 roundoff
> floor at h=1e-4, then divergence as roundoff took over. The finite
> differences were the inaccurate side — the field has a ~0.02-unit
> wavelength and the default step was 1/7 of that. If you re-check this,
> keep h near 1e-4.

### The distance estimator

Using nu = max_iter * y_hat and the Douady-Hubbard potential, the potential
cancels completely:

```
G = w ** (1 - nu)
|grad G| = G * ln(w) * max_iter * |grad y_hat|
DE = G / |grad G| = 1 / (ln(w) * max_iter * |grad y_hat|)
```

Distance to the set is a pure function of the learned field's gradient
magnitude. `de_shade()` measures falloff in pixels, so the boundary stays one
crisp line wide at any zoom, with free anti-aliasing.

Also provides `analytic_normals()` / `lambert()` for correct relief lighting,
`encoder_bandwidth()` for the resolution ceiling, and `empirical_lipschitz()`.

---

## 2. `zoom_probe.py` — what happens below the training scale

The mirror of your outward-extrapolation experiment. The sampler already
draws boundary neighbourhoods down to radius 2e-5, so any failure here is
representational, not a coverage gap.

`run/model.npz` comes from `train_small.py` — the 251k-param, 100k-sample,
40-epoch escape model every number in this README is measured against. Train
it first or the probe has nothing to load (it will list any checkpoints it
finds nearby and exit):

```bash
python3 train_small.py
python3 zoom_probe.py --checkpoint run/model.npz --render
```

**Ceiling read off the weights before rendering anything:** min wavelength
0.0185 world units, 189 cycles across domain. A fresh `train_small.py` run
reproduces this to 4 significant figures (0.018478, 189.4), so the ceiling is
a property of the encoder, not of any particular training run.

**Measured:**

| zoom | MAE | corr | cycles truth | cycles net |
|-----:|----:|-----:|-------------:|-----------:|
| 1x | 0.032 | +0.959 | 7 | 6 |
| 10x | 0.082 | +0.778 | 9 | 17 |
| 50x | 0.280 | +0.416 | 7 | 5 |
| 100x | 0.453 | **-0.030** | 7 | 6 |
| 200x | 0.496 | -0.416 | 14 | 6 |
| 1000x | 0.642 | +0.222 | 51 | 7 |

Correlation crosses zero at **100x**, against a predicted 189x — same order,
2x early. Truth's structure *grows* under zoom, as a self-similar object must
(7 -> 14 -> 37 -> 51 cycles). The network's stays pinned at 6-8 the whole way.
That divergence is the result.

### Two things that did not work

**The 1/Z prediction was wrong in form.** Network cycles were predicted to
decay as ceiling/Z. They don't — the 90%-energy radial measure tracks where
the bulk of energy sits, not the maximum frequency present, so it never shows
the decay. A max-frequency or spectral-edge measure would be the right tool.

**Max |grad| is the opposite of what was claimed.** Network 598.7, truth 56.6.
The truth figure is an artifact of finite-differencing a 256-res grid, which
cannot see the real unbounded gradient. The network figure is real, and it is
Gibbs ringing — a band-limited function's only way to fake a discontinuity.
So the DE-derived "finest resolvable feature" of 1.9e-5 is measuring ringing
amplitude, not resolution, and overstates the model by ~1000x. **The encoder
wavelength, 0.0185, is the honest number.**

**The outward probe found nothing real.** Lattice pitch scaled with the query
window instead of staying fixed. Best guess: the sigmoid saturates far from
the training box, so the lattice lives in the *logits*, not the output.
Unresolved — try probing pre-sigmoid.

---

## 3. `neural_dynamics.py` — learn the map, not the statistic

Learns `P(z, w) ~= z**w` and keeps `+ c` exact, so every deviation from true
dynamics is attributable to the learned power map alone. Then iterates the
learned function in place of the analytic one.

```bash
python3 neural_dynamics.py --train --epochs 22 --w-fixed 2.0 --out ./dyn_w2
python3 neural_dynamics.py --render --checkpoint dyn_w2/map.npz
```

**Parameterization matters enormously.** First attempt predicted Re/Im in
asinh space: R^2 = 0.25, unusable. That couples a 4000x dynamic range to a
12*pi angular winding. Splitting magnitude from direction —
`(log|z^w|, cos w*theta, sin w*theta)` plus angular harmonics in the encoder —
fixes the conditioning without handing over the closed form. The network still
has to discover that log|z^w| = w*log|z| and that direction winds at rate w.

**Measured at w=2 fixed, 22 epochs:**

```
map fit R^2                : 0.99684
interior IoU vs true M-set : 0.7779
escape-time MAE            : 0.0561
set disagreement           : 3.87% of the plane
```

Recognizably the Mandelbrot set, and wrong on 3.87% of the plane. The
disagreement panel is not an error plot — it is a picture of what the network
believes about dynamics. `neural_julia()` renders Julia sets of the learned
map for fixed c.

**The current `dyn_w2/map.npz` is a much longer run — 1000 epochs**, final
training loss 1.05e-6 against 6.13e-1 at epoch 1. Re-rendering it:

| metric | 22 epochs | 1000 epochs |
|--------|----------:|------------:|
| interior IoU vs true M-set | 0.7779 | **0.9646** |
| escape-time MAE | 0.0561 | **0.0051** |
| set disagreement | 3.87% | **0.59%** |

So the 3.87% above is an undertrained number, not a capacity ceiling — nearly
an order of magnitude of it was just epochs. What survives 1000 epochs is the
more interesting residual, and it is still 0.59% of the plane.

### The general w in [2,6] case — trained

No longer the untrained hard version. 48,000 orbits (~800k orbit-visited
states), 1000 epochs, ~87 min on the 5070:

```bash
python3 neural_dynamics.py --train --epochs 1000 --n-orbits 48000 --out ./dyn_w26
```

(`--n-orbits` is new — it was a `train()` parameter with no CLI flag. Coverage
matters far more here than at fixed w, since the same orbit budget has to span
the whole range instead of one slice.)

**Generality cost nothing.** Evaluated at w=2, against the w=2 *specialist*:

| metric | w=2 specialist | w in [2,6] general |
|--------|---------------:|-------------------:|
| map fit R^2 | — | 1.00000 (loss 2.4e-7) |
| interior IoU | 0.9646 | **0.9651** |
| escape-time MAE | 0.0051 | **0.0054** |
| set disagreement | 0.59% | **0.58%** |

The general model matches the specialist at the specialist's own exponent
while also covering four units of w. There was no capacity tradeoff to pay.

---

## 4. `film_exponent.py` — give w its own pathway

Motivated by the bandwidth measurement: w is the *roughest* axis, not the
smoothest (median 153 cycles/domain near the boundary, against 12.5 for x and
5.0 for y). Feeding it through the same isotropic Fourier basis as position is
the wrong allocation.

```bash
python3 film_exponent.py --gradcheck
python3 film_exponent.py --demo-curve
```

`FiLMCathedralNet` is a drop-in for `BranchCutNet` — same `fwd` signature,
same dual heads, works with the existing training loop and with
`cathedral_grad`. Position goes through a 2-D Fourier encoder; w goes through
a hypernetwork emitting per-channel (gamma, beta) for each block. Gamma is
emitted as `1 + raw` so it starts at identity and early training stays stable.

**Verified:** hypernet median relative error 3.2e-3, trunk 8.2e-4, min
per-tensor correlation 0.999930.

### The interpretability payoff

`hyper.embedding_curve(w_values)` returns the learned w-embedding as a
plottable 1-D curve plus its arc-length speed. The premise was that peaks or
kinks at integer w would be the network reporting it had found the branch-cut
structure on its own.

**That premise is false, and the controls below establish why.** The curve
tracks where training w was *sampled*, not where the task has structure. Read
it as a diagnostic of training-mass distribution — which is a genuinely useful
thing to be able to see — and not as interpretability evidence.

`film_exponent.py` had no training path — only `--gradcheck` and
`--demo-curve` — so this curve could only ever be read off random weights.
`train_film.py` adds it (`FiLMCathedralNet` really is a drop-in; `Config`
already spans w in [2,6]):

```bash
python3 train_film.py --epochs 300 --n-data 300000
python3 train_film.py --curve-only --checkpoint run_film/model.npz
```

> ## ⚠ Status: the alignment replicates. The interpretation does not.
>
> Two separate claims were being run together here, and only one survives:
>
> * **"Embedding speed aligns with the integers across seeds."** Replicated,
>   3/3 held-out narrow-range seeds under a frozen metric. This stands.
> * **"The network discovered the integers without being told."** **Retracted,
>   and the mechanism is now identified.** `sample_w()` puts **45% of training
>   w exactly on the integer lattice**, a further 25% on half-integers, and
>   `generate_dataset` re-snaps another 30% of focused samples to `round(w)`.
>   Moving that lattice to k+0.25 moves the embedding peak to **+0.254 /
>   +0.249**; removing it leaves nothing (6 seeds, mean exceedance 0.53).
>   The curve measures sampling density. See *Result: the curve tracks the
>   sampler* below.
>
> This also invalidates the null below. "Best offset is uniform on ±0.40 absent
> integer structure" assumed nothing else in the pipeline identified the
> integers. Something did. The 3.6e-08 figure is not meaningful and is kept
> only to be struck through.
>
> Two further corrections, both from the same review:
>
> * **The five held-out runs were not five independent draws.** Seeds 1 and 2
>   were reused across the narrow and wide ranges, sharing initialization and
>   random streams — those are *paired*. Three narrow seeds are mutually
>   independent; the two wide seeds are independent of each other.
> * **The wide-range sampler was broken.** `int(cfg.w_min)` with `w_min=1.5`
>   built its lattice from `int(1.5)=1`, injecting **w=1 — outside the declared
>   range — into 7.5% of samples.** `log(w)=0` sits in the denominator of the
>   smooth-escape estimate, so every one of those samples got `±inf`, which
>   `np.clip` silently flattened to a constant label 0.0 (8,959 samples, one
>   unique value). Separately, `np.arange(1.5+0.5, 6.5, 1.0)` made the
>   "half-integer" bucket `[2,3,4,5,6]` — also integers — so **69.94% of
>   wide-range training w sat exactly on integers.** That, not a cleaner
>   signal, is the likely reason the wide runs looked better.
>
> The sampler is now fixed (range-clipped anchors, assertions on both w range
> and label finiteness) and the control experiment is below. The wide-range
> numbers in this section predate the fix and should not be cited.

**Replicated: the alignment, 3/3 under a frozen metric.**

Seed 42 is the discovery run — the one the hypothesis came from, and the one
the metric was designed against. Seeds 1, 2, 3 were run afterwards, scored by
a rule frozen before any of them was looked at:

| seed | phase_exceedance | best offset | verdict |
|------|-----------------:|------------:|---------|
| 42 *(discovery)* | 0.0375 | −0.018 | PASS |
| 1 | 0.0062 | −0.002 | PASS |
| 2 | 0.0275 | +0.013 | PASS |
| 3 | 0.0125 | −0.005 | PASS |

The confirmatory seeds are *stronger* than the discovery run, not weaker,
which is the opposite of what regression-to-the-mean on a fluke would give.

### Widening the range: all five integers interior

With w in [2,6] the endpoints are unscorable, leaving three integers. Training
over w in [1.5, 6.5] makes 2..6 all interior:

```bash
python3 train_film.py --epochs 300 --n-data 300000 --seed 1 --w-min 1.5 --w-max 6.5
```

| seed | scored integers | exceedance | best offset |
|------|-----------------|-----------:|------------:|
| 1 | 2,3,4,5,6 | 0.0125 | **+0.005** |
| 2 | 2,3,4,5,6 | 0.0137 | **+0.005** |

Both wide runs land on the *same* offset, +0.005 — five integers each, and all
five elevated (1.28–1.53x median, 89th–96th percentile) rather than the ragged
71st–98th spread the three-point version gave. Widening the range did not just
add points, it made the effect cleaner.

**The replication supplies a statistic a single run could not — but not a
sufficient one.** Within one curve, the phase offsets are correlated positions
along one deterministic object, so `phase_exceedance` is a rank and no p-value
is available. Across separately trained networks the best-fit offsets *are*
independent draws, which fixes that particular problem. It does not fix the
one that matters:

~~If there were no integer structure, each run's best offset would fall
anywhere in ±0.40. All five held-out runs land within ±0.013:~~

```
~~P(|offset| <= 0.013) per run  = 0.033~~
~~P(all five held-out runs)     = 3.6e-08~~
~~wide runs alone (offset 0.005) = 1.6e-04~~
```

Independence across seeds buys a null over *training runs*. It says nothing
about whether the integers were privileged before training started — and they
were, by the sampler. Independent runs reproducing a property of a shared
input distribution is exactly what they should do. A control null is required,
not more seeds.

### The control: was it the task, or the sampler?

Three conditions, identical in every other respect, on the corrected sampler
over w in [1.5, 6.5]:

| condition | training w | what it separates |
|-----------|-----------|-------------------|
| `--w-anchor-mode uniform` | continuous, no lattice | alignment here cannot be input density |
| `--w-anchor-phase 0.25` | 45% on k+0.25, targets unchanged | follows the sampler, or the task? |
| default (anchored, phase 0) | 45% on integers | the confounded original |

```bash
python3 train_film.py --epochs 300 --n-data 300000 --w-min 1.5 --w-max 6.5 --w-anchor-mode uniform
python3 train_film.py --epochs 300 --n-data 300000 --w-min 1.5 --w-max 6.5 --w-anchor-phase 0.25
```

The phase-shifted condition is the sharp one, because it puts the two
hypotheses in direct opposition: the sampling spikes move to k+0.25 while the
branch-cut structure of the true target stays at k. If the embedding peak
follows to +0.25, the curve was tracking input density all along. If it stays
near 0.00, it is tracking the task. `train_film.py` now reports the score at
both lattices and which one the best offset is nearer.

### Result: the curve tracks the sampler

**Move the sampler, the peak moves with it.**

| seed | sampler lattice | embedding peak | error |
|------|----------------:|---------------:|------:|
| 4 | +0.250 | **+0.254** | 0.004 |
| 5 | +0.250 | **+0.249** | 0.001 |

Mean error against the sampler lattice **0.0025**. The cleaner statistic is
the *paired* one — each shifted run against its own phase-0 control at the
same seed, which removes the per-seed baseline:

```
seed 4:  -0.005 -> +0.254   movement +0.2590
seed 5:  +0.007 -> +0.249   movement +0.2420
                mean movement +0.2505   (nominal +0.2500, error 0.0005)
```

(The unpaired figure — mean peak *location* +0.2515 — is the weaker version
of the same thing.) At phase 0.25 the windowed score at the sampler's lattice
is 1.33–1.48x the score at the integers. The peak moved by what the sampler
moved, to within 0.0005.

**Remove the lattice and nothing remains.** Six seeds, continuous uniform w:

| seed | 4 | 5 | 6 | 7 | 8 | 9 |
|------|--:|--:|--:|--:|--:|--:|
| exceedance | 0.578 | 0.044 | 0.613 | 0.830 | 0.986 | 0.102 |
| offset | −0.196 | +0.018 | −0.250 | −0.132 | +0.221 | −0.113 |

Mean exceedance **0.526**, median 0.596 — indistinguishable from the uniform
[0,1] a true null predicts. Offsets scatter across the whole ±0.25 range
(mean −0.075, sd 0.171) with no concentration at zero. The single PASS at seed
5 is exactly what chance delivers: P(at least one of six below 0.05) = 0.265
under the null, and it sits on both thresholds (0.0437 vs 0.05, +0.018 vs
0.020).

The three conditions differ *only* in how w was drawn:

```
lattice on integers  ->  peak at integers   (-0.005, +0.007)
lattice at k+0.25    ->  peak at k+0.25     (+0.254, +0.249)
no lattice           ->  peak nowhere       (scatter across +/-0.25)
```

**`embedding_curve` is a sampler-sensitive diagnostic, not an
interpretability probe.** Stated precisely: these controls establish that it
is sensitive to *this* lattice, in *this* training distribution — the peak
tracks the anchor lattice wherever it is put, and vanishes when it is removed.
That is enough to disqualify the interpretability reading. It is not enough to
claim the curve reconstructs training density in general, which would need
distributions other than a lattice-plus-uniform mixture. The "peaks at integer
w are the network telling you it found the branch cut" premise is wrong; the
weaker "this curve is telling you something about your sampler" is what
survives.

Everything that made the original result look strong — five seeds, a metric
frozen in advance, offsets agreeing to ±0.013, two ranges concurring at
+0.005 — was the pipeline faithfully reproducing a lattice that had been put
into the training data by hand. None of those safeguards could detect the
confound, and their combined effect was to make a confounded result *more*
convincing. The only check that would have caught it was reading `sample_w`.

> **Two methodological corrections, both of which moved the numbers.**
>
> *The obvious test does not work.* A "is this point higher than its
> neighbours" peak detector reports nothing — the curve carries 41 prominent
> ripples, so integers appear as broad elevated regions, not spikes. The first
> version printed "no integer structure" while its own table showed all five
> integers above the median.
>
> *The first fix overstated its own precision.* The curve was sampled every
> 0.01 in w while offsets were scanned every 0.001 by nearest neighbour — so
> the 1001-point scan collapsed onto **132 distinct values** and the reported
> best offset of +0.006 was finer than the data could resolve. Under
> interpolation it is +0.010, one curve sample. Under the frozen metric
> (dense sampling, window-integrated arc length) it is -0.018, and the
> exceedance moved 0.0200 -> 0.0120 -> 0.0375 across those three choices.
> The effect is real but **metric-sensitive**, which is exactly why the rule
> is now frozen in `train_film.py` before any confirmatory seed was scored.

`phase_exceedance` is deliberately not called a p-value. The offsets are
correlated positions along one deterministic curve, not independent draws from
a null distribution; it is the rank of the integer lattice against shifted
copies of itself. Three scored points is also thin.

With w in [2,6], w=2 and w=6 are endpoints where `np.gradient` goes one-sided,
and the frozen metric additionally requires a margin of `OFFSET_LIMIT +
WINDOW_HALF` so every shifted window stays in bounds — leaving three scorable
integers. Widening the training range fixes this; see below.

---

---

## 5. `symmetry_probe.py` — did it learn the algebra or the pixels?

> **⚠ Open: an architectural lattice, found after #4 was resolved.**
>
> #5's *sampler* is clean — `orbit_dataset` draws w continuously. Its *basis*
> is not. `NeuralMap._encode` supplies angular harmonics `cos(k·theta)`,
> `sin(k·theta)` for **integer** k = 1..10, while the regression target is
> `(log|z^w|, cos(w·theta), sin(w·theta))`. At integer w = k the target
> direction is verbatim an input feature. Worse for this probe specifically:
>
> ```
> the C_w rotation is theta -> theta + 2pi/w, and
> cos(k*(theta + 2pi/k)) = cos(k*theta + 2pi) = cos(k*theta)
> ```
>
> so at integer w the matching harmonic is **exactly invariant under the very
> rotation the probe applies** (w=2: k=2,4,6,8,10; w=3: k=3,6,9; w=4: k=4,8).
> A network reading direction off that feature satisfies C_w equivariance by
> construction. The integer dips below are therefore in the same position #4's
> integer peaks were in before the sampler control.
>
> This does **not** touch the off-integer result — no harmonic matches at
> non-integer w, so tracking the analytic curve to 1.6% there remains a real
> function fit. It is specifically the *integer dips*, the headline, that need
> an intervention.
>
> `--harmonics {integer,shifted,random,none}` is that intervention, exactly
> analogous to `--w-anchor-phase`.
>
> **RESOLVED — the basis was not the source. #5 survives.** See *The
> architectural control* below.



For integer w the power map has an exact discrete rotational equivariance:
with rho = exp(2*pi*i/w), `(rho*z)**w = rho**w * z**w = z**w`, so
`P(rho*z) = P(z)` identically. At non-integer w the principal branch cut
destroys it. The true map's defect curve is therefore a comb — machine zero
at the integers, strictly positive between them under this metric (peaking
around 0.1–0.3, falling smoothly to zero as each integer is approached) — and
**nothing in the pointwise MSE objective points at this.** It converts
"structure or surface?" into one number per exponent against an analytic
ground-truth reference with no fitted baseline. It is not parameter-free: the
sampling disc, the log-magnitude/direction feature representation and the
normalization are all measurement choices, and the absolute scale depends on
them. The true-vs-learned comparison does not, since both sides go through the
same function.

```bash
python3 symmetry_probe.py --checkpoint dyn_w2/map.npz --sweep
python3 symmetry_probe.py --checkpoint dyn_w2/map.npz --julia-test --w 2.0
```

**Measured on the w=2 checkpoint:**

```
true map at w=2   : 1.448e-16
learned map at w=2: 0.0061        <- 0.00x median, DIP
median across w   : 1.2397
w = 3,4,5,6       : 1.09, 1.24, 1.33, 1.28   (all flat)
```

The dip at w=2 is real: the network recovered C_2 equivariance to within
0.6% from pointwise regression alone. Unlike #4, this is not confounded by the
sampler: `neural_dynamics.py` draws w continuously from `rng.uniform(w_min,
w_max)` with no lattice anywhere in the training distribution.

> **Read the flat part carefully.** This checkpoint was trained at *fixed
> w=2*. It has never seen w=3..6, so flatness there is not evidence that it
> "fits the surface pointwise" — the tool's own printed interpretation
> assumes a model trained across the w range. One dip at the one exponent it
> was trained on is exactly the expected shape. This is a positive control,
> not the experiment.

### The experiment: the same sweep on the w in [2,6] model

```bash
python3 symmetry_probe.py --checkpoint dyn_w26/map.npz --sweep
```

```
median defect across w : 0.0745
w = 2: defect 0.0055    0.07x median   DIP
w = 3: defect 0.0016    0.02x median   DIP
w = 4: defect 0.0014    0.02x median   DIP
w = 5: defect 0.0012    0.02x median   DIP
w = 6: defect 0.0030    0.04x median   DIP
```

Every integer dips, 25–50x below the median. **But a low defect everywhere
would produce the same table for the wrong reason**, so check it against the
analytic comb rather than against the median — the true map is *supposed* to
be strongly asymmetric off the integers, and a model that had merely gone
globally smooth would fail there:

| w | true | learned | ratio |
|--:|-----:|--------:|------:|
| 2.25 | 0.2336 | 0.2354 | 1.008 |
| 2.50 | 0.2788 | 0.2805 | 1.006 |
| 2.75 | 0.1681 | 0.1696 | 1.009 |
| 3.50 | 0.1579 | 0.1593 | 1.009 |
| 4.50 | 0.1006 | 0.1018 | 1.012 |
| 5.50 | 0.0699 | 0.0710 | 1.016 |
| integers | 0.0000 | 0.0012–0.0055 | — |

It tracks the true asymmetry to within **1.6% at every non-integer w**, then
collapses by ~66x at the integers. That is the comb, learned — not smoothness
mistaken for it. Nothing in the pointwise MSE objective points at this.

For contrast, the fixed-w=2 model scored 1.09–1.33 at w=3..6, where the truth
is exactly zero: maximally wrong at precisely the points that matter.

`--julia-test` on the general model: mean IoU change +0.0298, and unlike the
fixed-w model **every** c improves (+0.008 to +0.056) rather than two-thirds
of them. Enforcing a symmetry the network already respects is a smaller, and
now uniformly positive, correction.

### The architectural control

The integer angular harmonics hand the network a free path to C_w
equivariance. Four angular bases, two paired seeds each, everything else
identical (300 epochs, 24k orbits, one seed driving orbit sampling, weight
init and shuffling):

```bash
python3 neural_dynamics.py --train --harmonics shifted --seed 1 --out ./harm_shifted_s1
python3 symmetry_probe.py --checkpoint harm_shifted_s1/map.npz --sweep
```

The intervention is real — `shifted` provably eliminates the free path:

| w | 2 | 3 | 4 | 5 | 6 |
|---|--:|--:|--:|--:|--:|
| features exactly C_w-invariant, `integer` basis | 10/20 | 6/20 | 4/20 | 4/20 | 2/20 |
| features exactly C_w-invariant, `shifted` basis | **0/20** | **0/20** | **0/20** | **0/20** | **0/20** |

**And the dips do not move.** Mean symmetry defect over w = 2..6:

| basis | seed 1 | seed 2 | vs `integer` |
|-------|-------:|-------:|-------------:|
| `integer` (original) | 0.0135 | 0.0129 | — |
| `shifted` (1.25…10.25) | 0.0135 | 0.0124 | **−1.9%** |
| `random` (matched bandwidth) | 0.0130 | 0.0131 | **−1.1%** |
| `none` (raw theta only) | 0.0219 | 0.0202 | +59.1% |

Removing every C_w-invariant feature changes the integer symmetry defect by
**1.9%** — i.e. nothing, and in the paired comparison seed 1 is flat
while seed 2 moves down, which is noise. Off-integer tracking is likewise
untouched: max deviation from the analytic curve is 9.34% (`integer`), 9.27%
(`shifted`), 9.40% (`random`) at this training budget. (The 1.6% quoted
earlier is `dyn_w26` at 1000 epochs / 48k orbits — the gap is budget, not
basis.)

**So the equivariance is learned, not handed over.** This is the outcome that
strengthens #5 rather than explaining it away: the result now survives an
intervention on its input basis in addition to having a clean sampler and an
analytic ground truth.

`none` is still the weakest condition, but only mildly and for ordinary
reasons: dropping the harmonics drops the input dimension (251,523 vs 255,363
parameters) and degrades the overall fit (R² 0.99999 vs 1.00000, IoU 0.904 vs
0.935). It dips at all five integers like everything else. `shifted` and
`random` are the clean matched controls, and they are decisive.

> **Correction — the `none` "blowups" were a bug in the probe, not the model.**
> This section first reported localized catastrophic failures (`none` s1 w=5.5
> defect 0.9912 = 14x truth; `none` s2 w=5.0 defect 2.6236, a *peak* where the
> truth is exactly zero) and read them as "good pointwise MSE, broken algebra."
> That was wrong. `symmetry_probe.as_features` computed
> `sqrt(re*re + im*im)` **in float32**. For a legitimately tiny output —
> `re ≈ 8.3e-24` at `|z| ≈ 0.02` — `re*re ≈ 7e-47` falls below even the
> float32 subnormal floor and flushes to exactly zero, so `magnitude`
> collapses onto the `1e-30` guard and `re/magnitude` explodes to ~1e6.
>
> | | float32 (as shipped) | float64 (fixed) |
> |---|---:|---:|
> | `none` s1, w=5.5 | 11.2284 | **0.0836** |
> | `none` s2, w=5.0 | 0.8620 | **0.0118** |
> | `integer` s1, w=5.0 | 0.0072 | 0.0072 |
> | `dyn_w26`, w=5.0 | 0.0012 | 0.0012 |
>
> Up to **134x** inflation — and note which rows moved. Well-fit models never
> emit outputs small enough to underflow, so the bug was invisible on exactly
> the models being showcased and fired only on the poorly-fit ones. A
> measurement error silently correlated with the quantity being measured.
> `as_features` now promotes to float64 before squaring. The headline #5
> numbers and the whole harmonics conclusion are unchanged (verified above:
> `shifted` −1.9%, `random` −1.1%); only the `none` anomalies were affected,
> and they were never real.

`--julia-test` symmetrizes the learned map and re-renders: mean IoU change
+0.043 across four c values, but not uniformly — c=0.285+0.01j gains +0.174
while c=-0.8+0.156j *loses* 0.033. Symmetrization is not a free win.

---

## 6. `differential_probe.py` — does it obey the algebra, not just the values?

Off the origin and off the branch cut, `z^w` satisfies three exact relations
that no basis function hands over for free:

```
1. CAUCHY-RIEMANN     u_x = v_y  and  u_y = -v_x      (f is holomorphic in z)
2. EULER HOMOGENEITY  x*f_x + y*f_y = w*f             (f(t*z) = t^w f(z))
3. W-DERIVATIVE       f_w = f * Log z
```

Cauchy–Riemann is the strongest: nothing in the network's parameterization
enforces holomorphy, and unlike the C_w symmetry it is a *local* statement, so
no single harmonic can satisfy it.

```bash
python3 differential_probe.py --calibrate
python3 differential_probe.py --compare dyn_w26 harm_integer_s1
```

### The floor is measured, not assumed

Every residual is finite-differenced in float32, so the measurement has its own
error floor. `--calibrate` runs the identical pipeline on the **analytic** map,
where the true residual is exactly zero, and reports what comes out:

| h | 3e-2 | 1e-2 | **3e-3** | 1e-3 | 3e-4 | 1e-4 |
|---|--:|--:|--:|--:|--:|--:|
| Cauchy–Riemann | 1.3e-4 | 1.7e-5 | **1.6e-5** | 5.3e-5 | 1.7e-4 | 5.2e-4 |
| Euler | 5.5e-5 | 8.1e-6 | **9.5e-6** | 3.4e-5 | 1.1e-4 | 3.4e-4 |
| w-derivative | 1.8e-4 | 2.1e-5 | **7.7e-6** | 3.9e-5 | 1.3e-4 | 5.3e-4 |

Textbook h² convergence down to h=3e-3, then float32 roundoff takes over —
floor ≈ **1.6e-5**, giving ~5 decades of headroom. A learned residual only
means anything read against this.

### Result: the values are right, the algebra is approximate

Median normalized residual over w = 2…6, against that floor:

| model | Cauchy–Riemann | Euler | w-derivative |
|-------|---------------:|------:|-------------:|
| `dyn_w26` (1000ep/48k) | 1.1e-2 = **657x** | 4.5e-3 = **468x** | 1.1e-2 = **598x** |
| `harm_integer_s1` (300ep/24k) | 7.1e-2 = **4339x** | 3.4e-2 = **3528x** | 7.3e-2 = **3976x** |

**A model at R² = 1.00000 with 0.58% set disagreement is not holomorphic to
about 1%.** It reproduces the values and the C_w symmetry while violating
Cauchy–Riemann by two to three orders of magnitude above the noise floor. That
is a genuinely different axis of "did it learn the structure" than anything
measured so far, and the network does worse on it than on everything else.

It is learnable, though, not a hard barrier: 3.3x more training takes the
residuals down ~6.5x across all three identities. And the `w_derivative`
residual is worst at the range endpoints (8025x at w=2, 1900x at w=6, versus
~300x in the interior) — exactly where a w-derivative has to extrapolate.

### What this probe did NOT find

It was built to explain the `none`-basis symmetry blowups, and it found no
spike at those w values at all — they were among the *lowest* residuals in
their rows. That disagreement is what exposed the float32 underflow bug in
`symmetry_probe` documented in #5: the blowups were never real, so there was
nothing for this probe to localize. The probe was right and the thing it was
built to explain was the artifact.

---

## 7. `cr_consistency.py` — enforcing the identity, and what it buys

Section 6 measured a ~650x Cauchy–Riemann violation. This trains against it.

**Only Cauchy–Riemann is penalized, deliberately.** Euler homogeneity
*together with* holomorphy implies `f = C·z^w`, so penalizing it hands over
the closed form and makes the experiment circular; `f_w = f·Log z` likewise.
CR alone says "holomorphic", which infinitely many functions satisfy — a
constraint on the shape of the solution, not the solution.

In log-polar coordinates (`z = e^zeta`, `zeta = u + iv`), holomorphy makes
`log f = g + i·phi` holomorphic in zeta, so CR is just

```
g_u = phi_v        g_v = -phi_u
```

which suits this network: it predicts `g` directly, phi-derivatives come
branch-free from `(c·s_x − s·c_x)/(c²+s²)` with no unwrapping, and both
perturbations are multiplicative on z (`z·e^{±h}`, `z·e^{±ih}`) so nothing
degenerates near the origin. Gradients are hand-derived, so they were checked
before use: **264 parameters, correlation 0.999996, median relative error
6.3e-3** against finite differences through the full chain.

```bash
python3 cr_consistency.py --gradcheck
python3 cr_consistency.py --train --cr-weight 1e-4 --out ./crx
python3 cr_evaluate.py
```

> **Warm-up is required, and it is not a fudge.** An untrained net with
> Fourier radii ~100 has input-derivatives of order 100, so the residual
> starts near 3e4 — four orders above its converged ~1 — and its gradient
> swamps the data term, collapsing the map toward a constant. Measured:
> `cr_weight=1e-5` applied from step 0 ends at fit 1.79 against a baseline of
> 2.5e-3, **700x worse**. The penalty is off for the first half of training
> then ramped in. That also matches the question being asked, which is about
> an already-fitted model, not optimisation from scratch.

### Result: no, it does not help iteration — but it fixes something else

Two seeds per weight, paired against the matched-seed baseline. `cr(probe)` is
the *independent* measurement from #6 (different stencil, different
normalization, its own analytic floor), included so the trained-on residual
cannot be gamed unnoticed.

| metric | seed spread | w=1e-4 | w=1e-3 | w=1e-2 |
|--------|------------:|-------:|-------:|-------:|
| cr (trained on) | 19.3% | −93.2% ** | −98.2% ** | −99.4% ** |
| cr (independent probe) | 18.8% | −61.9% ** | −79.6% ** | −87.5% ** |
| **C_w symmetry defect** | 4.3% | **−36.6% \*\*** | **−41.9% \*\*** | **−31.0% \*\*** |
| interior IoU | 1.6% | +0.0% | −0.1% | −1.5% |
| escape-time MAE | 15.3% | −5.9% | −3.1% | +15.6% |
| set disagreement | 21.4% | +0.3% | +3.6% | +23.9% |

`**` = effect exceeds 3x the baseline seed-to-seed spread.

**Long-horizon iteration does not improve.** IoU and disagreement move less
than seed noise, and the two seeds *disagree in sign* at w=1e-4 — the
signature of nothing happening. At w=1e-2 the constraint starts costing real
accuracy (IoU −1.5%, disagreement +23.9%, fit 2x worse). Pointwise fit already
captures what matters for dynamics; a model at R² = 1.00000 iterates about as
well as its holomorphy-corrected twin.

**But the C_w symmetry defect drops 37–42%**, on both seeds, at ~9x the seed
spread, for a property the penalty never saw — CR is local and infinitesimal,
C_w symmetry is a finite rotation.

### Why the transfer happens (it is not mysterious)

CR couples `g_u` to `phi_v`. Measuring the two separately against their true
value of `w`:

| run | \|g_u − w\| | \|phi_v − w\| |
|-----|-----------:|-------------:|
| baseline | **0.358** | 0.073 |
| w=1e-4 | 0.096 | 0.055 |
| w=1e-3 | 0.047 | 0.039 |
| w=1e-2 | **0.029** | 0.027 |

The network's *phase winding was already accurate*; its **log-magnitude slope
was the broken one**, off by 0.36 against w≈4. CR forces the two to agree, so
the accurate quantity drags the inaccurate one into line — a 12x improvement
in `g_u` against 2.7x in `phi_v`. And the symmetry defect is dominated by its
log-magnitude component, which is exactly what got fixed:

| run | symmetry: log-magnitude part | direction part |
|-----|-----------------------------:|---------------:|
| baseline | 0.0407 | 0.0098 |
| w=1e-4 | 0.0218 | 0.0089 |
| w=1e-3 | 0.0186 | 0.0088 |
| w=1e-2 | 0.0178 | 0.0115 |

So the story is complete rather than merely correlational: **CR acts as a
channel that transfers accuracy from the well-determined phase to the
poorly-determined magnitude slope, and the symmetry probe was measuring
magnitude consistency all along.** It also explains the null result on
iteration — a wrong *derivative* of log-magnitude barely perturbs pointwise
values, which is what iteration compounds.

**Practical upshot: w=1e-4 is free.** Fit is marginally *better* than baseline
(7.09e-6 vs 8.26e-6), CR falls 93%, symmetry falls 37%, iteration is
unchanged. Above 1e-3 the constraint starts being paid for.

---

## Caveats

Sections 1, 2 and the 22-epoch numbers in 3: one CPU core, 251k-param escape
model on 100k samples for 40 epochs, learned map 22 epochs at fixed w=2
(superseded — see #3). The 100x death point in particular will move with
model capacity.

The trained results in 3, 4 and 5 are GPU runs on the 5070: `dyn_w26` 1000
epochs / 48k orbits (~87 min), `run_film` 300 epochs / 300k samples (~12 min).

GPU is now working (see #0), so these are no longer expensive to rerun:
a 40-epoch 100k-sample training run is 13 s on the 5070, and the 2M-sample
configuration is ~4 min. Every gradcheck and probe in this README was
re-verified on GPU and agrees with the CPU numbers.

## The two probes do not agree — and only one was ever asking the question

The whole point of building #4 and #5 was that they ask the same question —
*did it learn the algebra or the pixels?* — through mechanisms that share
nothing. Both have now answered on models that see the full w range:

| probe | lattice in the *sampler*? | lattice in the *basis*? | verdict |
|-------|--------------------------|-------------------------|---------|
| #5 symmetry comb | no — continuous uniform w | yes, but **controlled**: shifting it changes the result by 1.6% | **stands, and stronger** |
| #4 embedding curve | **yes — 45% on integers** | n/a | **retracted** — tracks the sampler |

They looked like two independent routes to one conclusion. The difference
between them is entirely in the training distribution.

**#5 now has three independent things going for it.** Its sampler is clean —
`orbit_dataset` draws `rng.uniform(w_min, w_max)`, no lattice in the data. Its
basis *did* contain an integer lattice, and shifting it off the integers (which
provably removes every C_w-invariant feature) changes the integer symmetry
defect by 1.6%: the free path existed and the network was not using it. And
it is measured against analytic ground truth rather than a chosen threshold.

Clean sampler, controlled basis, ground truth. That is a considerably stronger
position than it was in when it was merely "the one that didn't fail."

**#4 measured its own sampler.** Under the phase-shifted control the embedding
peak moved to +0.254 and +0.249 when the sampling lattice moved to +0.25, and
under continuous-uniform w it found nothing across six seeds. The alignment was
real and replicable; it was never evidence about the branch cut.

**One probe found the shape of its own training data. The other found the
structure.** #4 tracked its sampler and is retracted. #5 was suspected of
tracking its input basis and — after an intervention that provably removed
that path — did not.

Both suspicions were worth chasing, and it is worth noting that only one of
them was right. The point of an intervention is that it can exonerate as well
as convict; running it because the confound is *plausible* is what makes the
surviving result mean something.

The generalization that matters: a lattice can enter through **any** path by
which information reaches the model — the sampler, the feature basis, the loss
weighting, the target parameterization. "Audit the data generator" was too
narrow a lesson. Audit every channel that could hand the model the answer, and
for each one ask what an intervention on it would look like.

The general lesson is not about this project: **a result that survives a frozen
metric, held-out seeds, and independent recomputation can still be an artifact
of the input distribution, and none of those safeguards can see it.** Those
safeguards were not defective — they correctly ruled out metric tuning,
discovery-seed overfitting, and arithmetic error. What was missing was
upstream *construct validity*: whether the measurement could distinguish task
structure from the data generator at all. Only an intervention that changes
the suspected cause tests that, and it belongs before the replication.

### Made executable

`generate_dataset` prints a w-audit on every run, and **every training entry
point** forwards it into **every** checkpoint it writes — `train_small.py`,
`train_film.py`, and `BranchCutCathedral.train()` for both
`checkpoint_NNNN.npz` and `checkpoint_latest.npz`:

```
w-audit: mode=anchored phase=+0.00 | range [1.500, 6.500] in [1.5, 6.5] | out-of-range 0
w-audit: anchor mass 34.17% | exact-integer mass 34.17% | labels finite True
w-audit: NOTE the input distribution concentrates on 5 lattice points. Any claim
         that the model 'found' them needs a --w-anchor-mode uniform control.
```

It reports observed vs declared range, out-of-range count, anchor and exact-
integer mass, per-anchor label statistics, a 20-bin histogram, and warns on
anchors whose labels are near-constant (the signature of the `log(w)` blowup).
`Config.__post_init__` now rejects `w_min <= 1` outright — the finiteness
assertion could never catch that case, because `np.clip` turns the infinity
into a valid-looking constant label.

A checkpoint that cannot say how its training w was drawn cannot support a
claim about what the model learned over w. Now they all can — with one honest
exception: the checkpoints in `branch_cut_outputs/` predate this and have no
`w_audit` key. Their stored config does not settle it either, since
`w_anchor_mode` and `w_anchor_phase` did not exist yet and deserialize to
`None`. That they were trained with a phase-0 integer lattice at ~40% mass is
inferred from the code as it stood, not read off the file. Which is precisely
the situation the audit exists to prevent.

## Suggested next run

Sweep the exponent while iterating the *learned* map, now that `dyn_w26`
covers w in [2,6]. That gives a morph through a family of dynamical systems
that do not exist — and unlike before, the model is accurate across the whole
range it would be swept through.

**Differential identities.** Away from the origin and the branch cut, `z^w`
satisfies exact relations that no single basis function hands over for free:
Cauchy–Riemann, the Euler homogeneity `x·f_x + y·f_y = w·f`, and
`f_w = f·Log z`. `cathedral_grad.py` already has the analytic-gradient
machinery; extending it to `NeuralMap` and mapping normalized residuals over
(z, w) would test whether the network learned the *analytic* structure rather
than a good pointwise approximation of it. The `none`-basis blowups above are
the argument for doing this: excellent MSE, broken algebra, in a band narrow
enough that no value-based metric noticed. A small differential-consistency
penalty then becomes a natural intervention — does enforcing the identities
improve long-horizon Julia/Mandelbrot accuracy?

**A Cartesian-only branch-cut probe.** Drop raw theta as well as the
harmonics, so the discontinuity cannot arrive as an input feature and must be
manufactured. On circles around the origin, measure the winding number of the
predicted direction, the location and width of the phase slip, and the minimum
direction-vector norm, as w moves between integers. A continuous model's
winding is integer-valued and can only change through a singular event — which
should produce a staircase of learned topological transitions.

**Withheld integer neighbourhoods.** Train with `|w - k| < delta` removed
around every integer, against matched models withholding bands around k+0.5,
and ask whether the symmetry zeros at unseen integers are reconstructed from
surrounding exponents. Sweeping delta turns that into a generalization radius
rather than a yes/no.

#4 is closed as a negative result. The open question it leaves is whether the
FiLM hypernetwork encodes branch-cut structure *at all* — the embedding curve
cannot answer that, because it answers a different question. #5 is the
template for what a real test looks like: measure something with a known
ground truth, then intervene on every channel that could supply the answer.

Two smaller loose ends:

- The frozen metric has three constants chosen by hand (`WINDOW_HALF`,
  `OFFSET_LIMIT`, `CURVE_SPACING`). They served their purpose — the anchored
  and phase-shifted conditions are cleanly separated under them — but the
  exceedance is window-sensitive while the offsets are not, which is why the
  offsets carry the argument.
- `embedding_curve` calls `np.gradient` with no spacing argument, so speed is
  per-sample rather than per-unit-w and absolute values shift with sampling
  density. Every test here is rank- or ratio-based so nothing is affected, but
  raw speeds are not comparable across configurations.

The sampler bugs are fixed (`anchor_values` clips the lattice to the configured
range; assertions guard both the w range and label finiteness), but note that
`w_anchor_mode="anchored"` with phase 0 remains the **default**, because
sections 1–3 and every escape-model number in this README were produced with
it. Anything making a claim about *learned* structure over w should train with
`--w-anchor-mode uniform`.
