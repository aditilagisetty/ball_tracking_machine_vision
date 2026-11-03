# Kalman Filter Notes

Study notes from the [rlabbe book](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python),
chapter 6 (Multivariate Kalman Filters) and chapter 8 (Designing Kalman Filters),
written up for our ball tracker. Chapter 8 has a "tracking a ball" example that is
worth running in the notebook alongside these notes.

---

## 1. The big idea

The filter keeps two things at every moment:

- **x** — its best guess of the state (where the ball is and how fast it's moving)
- **P** — how unsure it is about that guess (a covariance matrix)

Each frame it does two steps:

1. **Predict**: use physics to move the guess forward in time. Uncertainty **grows**,
   because the physics model isn't perfect.
2. **Update**: compare the prediction to the camera measurement and blend them.
   Uncertainty **shrinks**, because we learned something.

How much it trusts the measurement versus the prediction is decided automatically
by comparing their uncertainties. That blend weight is the **Kalman gain K**.

The camera only measures position, but the filter still estimates velocity: it
learns velocity from how position changes between frames, through the correlation
between position and velocity in P. This "hidden variable" idea is the key point of
chapter 6.

---

## 2. Our state and model

State (6 numbers), world frame, z up, meters and seconds:

```
x = [x, y, z, vx, vy, vz]
```

### Motion: F and gravity

Over a time step `dt`, with gravity `g = 9.81` pulling down in z:

```
position_new = position + velocity * dt - ½ g dt² (z only)
velocity_new = velocity                 -   g dt  (z only)
```

As matrices, `x_new = F x + B u`:

```
    [1 0 0 dt 0  0 ]          [   0     ]
    [0 1 0 0  dt 0 ]          [   0     ]
F = [0 0 1 0  0  dt]    Bu =  [-½ g dt² ]
    [0 0 0 1  0  0 ]          [   0     ]
    [0 0 0 0  1  0 ]          [   0     ]
    [0 0 0 0  0  1 ]          [ -g dt   ]
```

`F` is `transition_matrix(dt)` and `Bu` is `gravity_input(dt)`. Gravity is
known exactly, so it goes in as a control input rather than as part of the state.

### Measurement: H

The camera gives `z_meas = [x, y, z]`, so H just picks out the position:

```
H = [I₃  0₃]      (3×6)
```

---

## 3. The equations

**Predict**

```
x = F x + B u
P = F P Fᵀ + Q
```

**Update**

```
y = z_meas - H x          # residual: how far off was the prediction
S = H P Hᵀ + R            # how uncertain the residual is
K = P Hᵀ S⁻¹              # Kalman gain: blend weight
x = x + K y
P = (I - K H) P
```

That's the whole algorithm. Everything else is choosing F, Q, R and the
starting x and P well.

---

## 4. Choosing the noise (the actual design work, chapter 8)

### R: measurement noise

R is how noisy the camera's 3D points are. Start with a diagonal matrix:

```
R = diag(σx², σy², σz²)
```

Stereo is much worse in **depth** (distance from the camera) than sideways. Depth
error grows with distance squared:

```
σ_depth ≈ Z² · σ_disparity / (f · b)
```

- `Z` is the distance to the ball
- `f` is the focal length in pixels (from the calibration file)
- `b` is the baseline (12 cm on the ZED 2i)
- `σ_disparity` is the detection error in pixels (about 0.5–1 px)

Example: at Z = 2 m, f ≈ 700 px, σ_disparity = 0.5 px, this gives σ_depth ≈ 2.4 cm,
but only a few millimeters sideways.

The depth axis is the camera's axis, not world z. If the camera is tilted,
rotate the noise into the world frame: `R_world = Rot · R_cam · Rotᵀ`. For a
first version, a diagonal R with the largest value on the axis pointing roughly
away from the camera is fine.

**Better:** measure it. Hold the ball still, record a few seconds, and take the
variance of the triangulated positions.

### Q: process noise

Q is how much we distrust our physics model each step: drag we're ignoring,
spin, timing jitter. Too small and the filter ignores measurements and lags
behind; too large and it follows the noise.

The standard choice (chapter 7/8, `Q_discrete_white_noise` in filterpy) treats
unmodeled effects as a small random acceleration with standard deviation `σa`.
For each axis, the (position, velocity) block is:

```
σa² · [ dt⁴/4   dt³/2 ]
      [ dt³/2   dt²   ]
```

Because our state is ordered `[x, y, z, vx, vy, vz]` rather than
`[x, vx, y, vy, ...]`, those four numbers go into Q at the positions
(i, i), (i, i+3), (i+3, i), (i+3, i+3) for axis i = 0, 1, 2.

Starting value: `σa ≈ 1–3 m/s²`. Air drag on a tennis ball at throwing speeds is
roughly that size, so this also covers our gravity-only model leaving out drag.

### Starting x and P

- Position: the first measurement. Its variance is R.
- Velocity: unknown, so give it a big variance (e.g. 10² (m/s)²).
- Better: use the first two measurements, and set velocity to
  `(p₂ - p₁) / dt`.

A filter started this way trusts the first few measurements heavily and then
settles down.

---

## 5. Practical details for real data

- **Use real timestamps.** Compute `dt` from consecutive frame times, not 1/fps.
  Build F and Q fresh every step.
- **Missed detections.** Call `predict` and skip `update`. P grows, which is
  exactly right.
- **Reject bad detections (gating).** Before updating, compute
  `d² = yᵀ S⁻¹ y`. If `d² > 11.3` (the 99% cutoff for 3 measured numbers),
  the detection is probably wrong (a yellow shirt, a blur streak), so skip it.
- **Numerical safety.** `P = (I - K H) P` can drift and become non-symmetric.
  Using `P = (I - K H) P (I - K H)ᵀ + K R Kᵀ` (the "Joseph form") is safer,
  or at least re-symmetrize with `P = (P + Pᵀ) / 2`.

---

## 6. Checking the filter is honest (chapter 8)

These are the tests in `test/test_kalman_filter.py`.

### On simulated data (truth known): NEES

Error at each step: `e = x_true - x_est`. Then

```
NEES = eᵀ P⁻¹ e
```

Averaged over many steps and throws, NEES should be about **6** (the size of
the state).

- Much bigger than 6 means the filter is **overconfident**: Q or R is too small.
- Much smaller than 6 means it is **too cautious**: Q or R is too big.

A simpler check that says the same thing: about 95% of position errors should
be inside 2σ (σ from the diagonal of P).

### On real data (no truth): NIS

Use the residual instead: `NIS = yᵀ S⁻¹ y`, which should average about **3**
(the size of the measurement). This lets us tune on real throws without
knowing the true path.

---

## 7. How this connects to landing prediction

To predict the landing, run **predict only** (no updates) forward from the current
state until z hits the floor. P keeps growing with every step, so the further
ahead we look, the wider the landing uncertainty. Early in the flight there's a
long way to extrapolate, so the uncertainty ellipse is big. It shrinks as more of
the flight is seen. That shrinking is exactly what our main result plot shows.

---

## 8. Things to try in the notebook

- [ ] Run the chapter 8 ball-tracking example and change Q and R to see the filter
      lag (Q too small) or jitter (Q too big).
- [ ] Watch how P's velocity entries shrink over the first few updates.
- [ ] Simulate a throw *with* drag and filter it with our gravity-only model.
      How large does σa need to be for NEES to stay near 6?
