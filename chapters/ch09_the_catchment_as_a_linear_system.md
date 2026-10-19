---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# 9. The catchment as a linear system: unit hydrographs

```{admonition} Core ideas
:class: tip
1. Treat the catchment as a **system** that turns excess rainfall into direct runoff. If the system is **linear** and **time-invariant**, its response to any storm can be built from its response to one standard input.
2. The simplest such system is the **linear reservoir**, $S = kQ$. Its responses to an instantaneous input, a steady input and a pulse are an exponential decay, an exponential rise, and the difference of two rises.
3. The **unit hydrograph** is the direct runoff from 1 mm of excess rainfall falling uniformly over the catchment in a duration $D$. The response to any excess hyetograph is the **convolution** of the hyetograph with the unit hydrograph.
4. A unit hydrograph is derived from observed events by dividing by the excess depth (one burst) or by **deconvolution** (several bursts). Its volume must always equal **catchment area × 1 mm**.
5. The **S-curve** converts a unit hydrograph from one duration to another.
6. Where there are no data, **synthetic unit hydrographs** (Snyder, NRCS, Clark, Nash) estimate the response from catchment properties. The NRCS peak in SI form is $Q_p = 0.208\,A/T_p$ per mm of excess.
```

```{code-cell} python
:tags: [remove-cell]
# Plot style shared by all figures in this chapter.
import numpy as np
import matplotlib.pyplot as plt
from myst_nb import glue
%config InlineBackend.figure_format = "retina"

c1, c2, c3 = "#2a78d6", "#eb6834", "#1baf7a"   # categorical colors
ink, ink2, bg = "#0b0b0b", "#52514e", "#fcfcfb" # text, secondary text, background
plt.rcParams.update({
    "figure.facecolor": bg, "axes.facecolor": bg, "savefig.facecolor": bg,
    "text.color": ink, "axes.labelcolor": ink2, "axes.edgecolor": ink2,
    "xtick.color": ink2, "ytick.color": ink2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e4e3df", "axes.axisbelow": True,
    "font.size": 11, "axes.titlesize": 11, "axes.titlelocation": "left",
    "legend.frameon": False, "figure.constrained_layout.use": True,
})
```

## 9.1 The catchment as a system

Chapter 7 turned a storm into an excess rainfall hyetograph. This chapter turns the excess rainfall into a hydrograph of direct runoff. The catchment is treated as a **system**: a box with an input $i(t)$, the excess rainfall rate, and an output $Q(t)$, the direct runoff. We do not try to describe what happens inside the box in detail; we describe the relation between input and output.

Two properties make this relation simple.

- **Linearity.** If input $i_1$ produces output $Q_1$ and input $i_2$ produces $Q_2$, then $a\,i_1 + b\,i_2$ produces $a\,Q_1 + b\,Q_2$. Doubling the excess rainfall doubles the runoff at every moment (**proportionality**), and the responses to separate bursts of rain add up (**superposition**).
- **Time invariance.** The response to a given input does not depend on when it occurs. A storm on Monday produces the same hydrograph shape as the same storm on Friday, shifted in time.

A real catchment is neither exactly linear nor exactly time-invariant. Flow velocities increase with depth, so large floods travel faster than small ones; and the response changes with season, vegetation and land use. But the nonlinearity in the conversion of rain into runoff is mostly in the **losses** (Chapter 7), which are handled separately. Once the excess rainfall is known, the routing of that excess to the outlet is close enough to linear for many engineering purposes, as LeRoy Sherman recognized in 1932 when he introduced the unit hydrograph (see [*A Brief History*](https://daehakim-waterlab.github.io/water-history/history-of-hydrology/#floods-from-formulas-to-the-unit-hydrograph)).

## 9.2 The linear reservoir

The simplest linear system is a reservoir whose outflow is proportional to its storage:

$$
S = k\,Q ,
$$ (eq-linear-reservoir)

where $k$ is the **storage constant** (h). Combined with the water balance of Chapter 1, $dS/dt = i - Q$, this gives a first-order differential equation:

$$
k\,\frac{dQ}{dt} + Q = i(t).
$$ (eq-lr-ode)

Its responses to three standard inputs are the building blocks of this chapter. All inputs are expressed as a depth rate over the catchment, so $Q$ here is also a depth rate (mm h⁻¹); multiplying by $A/3.6$ converts it to m³ s⁻¹ (Chapter 1).

**Impulse response.** An input of 1 mm delivered instantaneously at $t = 0$ fills the reservoir at once, $S(0) = 1$ mm, and it then drains with $dQ/dt = -Q/k$:

$$
u(t) = \frac{1}{k}\,e^{-t/k}.
$$ (eq-iuh-lr)

This response to a unit impulse is the **instantaneous unit hydrograph** (IUH) of the reservoir. Its area is 1, as it must be: all of the 1 mm eventually flows out.

**Step response.** A steady input of 1 mm h⁻¹ starting at $t = 0$ gives

$$
g(t) = 1 - e^{-t/k},
$$ (eq-step-lr)

which rises towards the equilibrium $Q = i$. The step response is the integral of the impulse response, $g(t) = \int_0^t u(\tau)\,d\tau$.

**Pulse response.** An input of 1 mm spread evenly over a duration $D$ (a rate of $1/D$ mm h⁻¹) is a step that starts at $0$ minus a step that starts at $D$, each scaled by $1/D$:

$$
h_D(t) = \frac{1}{D}\bigl[g(t) - g(t - D)\bigr], \qquad g(t) = 0 \text{ for } t < 0 .
$$ (eq-pulse)

This is the **D-hour unit hydrograph** of the linear reservoir. These three ideas, the impulse, the step and the pulse, apply to any linear system, not just the reservoir.

```{code-cell} python
:tags: [hide-input]
k, D = 5.0, 2.0                                 # h
t = np.linspace(0, 30, 601)
g = lambda t: np.where(t > 0, 1 - np.exp(-np.clip(t, 0, None) / k), 0.0)
u = np.exp(-t / k) / k
hD = (g(t) - g(t - D)) / D
fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 5.6), sharex=True)
a1.plot(t, u, color=c1, lw=2.2, label="Impulse response $u(t)$ (1 mm at $t$ = 0)")
a1.plot(t, hD, color=c2, lw=2.2, label=f"Pulse response (1 mm over D = {D:.0f} h)")
a1.set_ylabel("Q (mm h⁻¹)"); a1.legend(loc="upper right", fontsize=9.5)
a1.set_title(f"(a) Responses to 1 mm of input, k = {k:.0f} h")
a2.plot(t, g(t), color=c3, lw=2.2, label="Step response $g(t)$ (1 mm h⁻¹ from $t$ = 0)")
a2.plot(t, g(t - D), color=c3, lw=1.4, ls="--", label="Same step, delayed by D")
a2.axhline(1, color=ink2, lw=0.8, ls=":")
a2.set_xlabel("Time (h)"); a2.set_ylabel("Q (mm h⁻¹)"); a2.set_xlim(0, 30); a2.set_ylim(0, 1.15)
a2.set_title("(b) Step responses; their difference ÷ D is the pulse response")
a2.legend(loc="lower right", fontsize=9.5)
plt.show()
```

*Figure 9.1. A linear reservoir with $k$ = 5 h. The impulse response decays exponentially from $1/k$; the step response rises towards the input rate; the pulse response is the difference of two step responses divided by $D$. All curves in (a) have an area of exactly 1 mm.*

A single linear reservoir peaks the instant the input stops, which is unrealistic for a catchment, where the rising limb takes time. Section 9.6 shows how adding a delay (Clark) or more reservoirs in series (Nash) produces a realistic shape.

## 9.3 The unit hydrograph and convolution

### Definition

The **unit hydrograph** (UH) of duration $D$ is the hydrograph of direct runoff at the outlet produced by **1 mm** of excess rainfall falling at a uniform rate over the whole catchment during $D$ hours. In SI its ordinates $U(t)$ are in m³ s⁻¹ per mm. Sherman used one inch; the definition and all the algebra are the same with any unit depth, but the ordinates differ by the factor 25.4.

Because its input is a known volume, a unit hydrograph has a fixed volume:

$$
\int_0^\infty U(t)\,dt = A \times 1\ \text{mm}
\quad\Longleftrightarrow\quad
\sum_n U_n\,\Delta t \times 3600 = 1000\,A \quad (\text{m}^3,\ A \text{ in km}^2,\ \Delta t \text{ in h}).
$$ (eq-uh-volume)

This is the most important check on any unit hydrograph.

### Convolution

Suppose the excess hyetograph consists of $M$ consecutive bursts of duration $D = \Delta t$, with depths $P_1, P_2, \ldots, P_M$ (mm). By proportionality, burst $m$ produces $P_m$ times the unit hydrograph. By time invariance, that response is shifted to start at the beginning of burst $m$. By superposition, the total direct runoff is the sum:

$$
Q_n = \sum_{m=1}^{\min(n, M)} P_m\, U_{n-m+1}, \qquad n = 1, 2, \ldots, N ,
$$ (eq-discrete-convolution)

where $Q_n$ and $U_j$ are ordinates at the ends of successive intervals. This is the **discrete convolution** of the hyetograph with the unit hydrograph. Its continuous form, with the instantaneous unit hydrograph $u$ and the excess rate $i$, is the **convolution integral**

$$
Q(t) = \int_0^t i(\tau)\, u(t - \tau)\, d\tau .
$$ (eq-convolution-integral)

Figure 9.2 shows the discrete form at work: each burst launches its own scaled copy of the unit hydrograph, and the copies add up.

```{code-cell} python
:tags: [hide-input]
from scipy.stats import gamma

def uh_from_iuh(cdf, A, D, dt, n):
    """D-hour unit hydrograph (m³/s per mm) at steps dt from the cumulative IUH (an S-curve in 1/1)."""
    t = np.arange(1, n + 1) * dt
    S = lambda x: np.where(x > 0, cdf(np.clip(x, 0, None)), 0.0)
    return t, (S(t) - S(t - D)) / D * A / 3.6

A, dt = 60.0, 1.0
iuh_cdf = lambda x: gamma.cdf(x, a=3.0, scale=2.0)            # a smooth, realistic shape (mean 6 h)
tU, U1 = uh_from_iuh(iuh_cdf, A, 1.0, dt, 30)
vol = U1.sum() * dt * 3600
assert abs(vol - 1000 * A) / (1000 * A) < 1e-3                 # volume = A × 1 mm

Pex = np.array([8.0, 20.0, 6.0])                               # mm in three successive hours
Qd = np.convolve(Pex, U1)
fig, ax = plt.subplots(figsize=(6.4, 4.0))
cols = [c3, c2, c1]
base = np.zeros(len(Qd) + 1)
tt = np.arange(0, len(Qd) + 1) * dt
for m, P in enumerate(Pex):
    part = np.zeros(len(Qd) + 1); part[m + 1: m + 1 + len(U1)] = P * U1
    ax.fill_between(tt, base, base + part, color=cols[m], alpha=0.35, step=None,
                    label=f"{P:.0f} mm in hour {m + 1} → {P:.0f} × UH, starting at {m} h")
    base = base + part
ax.plot(tt, base, color=ink, lw=2, label="Total direct runoff")
ax.set_xlim(0, 30); ax.set_xlabel("Time (h)"); ax.set_ylabel("Q (m³ s⁻¹)")
ax.legend(loc="upper right", fontsize=9)
plt.show()
glue("conv_peak", float(Qd.max()), display=False)
glue("conv_vol", float(Qd.sum() * 3600 / (A * 1e6) * 1000), display=False)
```

*Figure 9.2. Convolution. A 60 km² catchment receives 8, 20 and 6 mm of excess rainfall in three successive hours. Each burst produces a copy of the 1-hour unit hydrograph scaled by its depth and shifted by its start time (stacked areas); their sum is the direct runoff hydrograph, with a peak of {glue:text}`conv_peak:.0f` m³ s⁻¹ and a depth of {glue:text}`conv_vol:.1f` mm, equal to the 34 mm of excess rain.*

Two practical points follow. First, the time step of the unit hydrograph must equal the duration of the bursts in the hyetograph; a 1-hour UH cannot be convolved with 3-hour rainfall blocks without first converting it (Section 9.5). Second, the direct runoff from convolution must be added to baseflow to give the total streamflow.

## 9.4 Deriving a unit hydrograph from data

### A single burst

The easiest case is an isolated storm whose excess fell in a single burst of duration $D$. Separate the baseflow (Chapter 6), compute the direct runoff depth $P_e$ from its volume, and divide the direct runoff hydrograph by $P_e$:

$$
U(t) = \frac{Q_d(t)}{P_e}.
$$

Since $P_e$ was computed from the same hydrograph, the volume check is satisfied automatically. In practice, unit hydrographs from several such events are derived and averaged (aligning their peaks), to reduce the effect of rainfall that was not uniform in space or time.

### Several bursts: deconvolution

When the excess falls in several bursts, Equation {eq}`eq-discrete-convolution` must be solved for the unknown $U$ with known $P$ and $Q$. This is **deconvolution**. Written out, the equations form a matrix system:

$$
\begin{bmatrix} P_1 & & & \\ P_2 & P_1 & & \\ P_3 & P_2 & P_1 & \\ & P_3 & P_2 & \ddots \\ & & P_3 & \ddots \end{bmatrix}
\begin{bmatrix} U_1 \\ U_2 \\ U_3 \\ \vdots \end{bmatrix}
=
\begin{bmatrix} Q_1 \\ Q_2 \\ Q_3 \\ \vdots \end{bmatrix},
\qquad \mathbf{P}\,\mathbf{U} = \mathbf{Q}.
$$

There are $N$ equations (one per hydrograph ordinate) but only $N - M + 1$ unknowns. In exact arithmetic the first equations could be solved one by one (**forward substitution**: $U_1 = Q_1/P_1$, then $U_2 = (Q_2 - P_2 U_1)/P_1$, and so on). With real data, which are never exactly linear and always contain errors, the errors are amplified at each step and the unit hydrograph oscillates, often with negative ordinates. The robust approach is to use **all** the equations and find the $\mathbf{U}$ that fits them best in the least-squares sense, with the constraint $U_n \ge 0$:

$$
\min_{\mathbf{U} \ge 0}\ \|\mathbf{P}\,\mathbf{U} - \mathbf{Q}\|^2 .
$$

This problem is solved by standard non-negative least squares. Figure 9.3 compares the two approaches on a synthetic event whose true unit hydrograph is known.

```{code-cell} python
:tags: [hide-input]
from scipy.optimize import nnls
from scipy.linalg import toeplitz

rng = np.random.default_rng(9)
P_obs = np.array([5.0, 14.0, 9.0, 3.0])                       # excess, mm per hour
U_true = U1[:24]
Q_obs = np.convolve(P_obs, U_true) * (1 + 0.04 * rng.standard_normal(len(U_true) + len(P_obs) - 1))
N, M = len(Q_obs), len(P_obs)
nU = N - M + 1
Pmat = toeplitz(np.r_[P_obs, np.zeros(N - M)], np.r_[P_obs[0], np.zeros(nU - 1)])   # N × nU

U_fwd = np.zeros(nU)                                           # forward substitution
for n in range(nU):
    U_fwd[n] = (Q_obs[n] - sum(P_obs[m] * U_fwd[n - m] for m in range(1, min(n, M - 1) + 1))) / P_obs[0]
U_ls, _ = nnls(Pmat, Q_obs)                                    # non-negative least squares

fig, ax = plt.subplots(figsize=(6.4, 3.9))
tu = np.arange(1, nU + 1)
ax.plot(tu, U_true, color=ink2, lw=1.4, ls=":", label="True 1-hour UH")
off = int(np.argmax(np.abs(U_fwd) > 3.4)) if (np.abs(U_fwd) > 3.4).any() else nU
ax.plot(tu[: off + 1], U_fwd[: off + 1], color=c2, lw=1.8, marker="o", ms=3, label="Forward substitution")
ax.plot(tu, U_ls, color=c1, lw=2.2, label="Non-negative least squares")
ax.axhline(0, color=ink2, lw=0.8)
ax.set_xlabel("Time (h)"); ax.set_ylabel("U (m³ s⁻¹ per mm)"); ax.set_xlim(0, nU)
ax.set_ylim(-1.5, 3.5)
ax.legend(loc="upper right", fontsize=9.5)
plt.show()
glue("fwd_off", int(tu[min(off, nU - 1)]), display=False)
glue("ls_vol", float(U_ls.sum() * 3600 / (A * 1e6) * 1000), display=False)
glue("fwd_min", float(U_fwd.min()), display=False)
```

*Figure 9.3. Deriving a 1-hour unit hydrograph from a four-hour burst of excess rainfall on the 60 km² catchment, when the "observed" runoff contains 4% random error. Forward substitution amplifies the errors at every step; its oscillations grow until they leave the plot at hour {glue:text}`fwd_off`; least squares with a non-negativity constraint stays close to the true shape, and its volume corresponds to {glue:text}`ls_vol:.3f` mm.*

## 9.5 The S-curve: changing the duration

A unit hydrograph is tied to its duration $D$. Design storms (Chapter 11) may be given at other time steps, so the UH must often be converted. The tool is the **S-curve** (or S-hydrograph): the response to continuous excess rainfall at a rate of 1 mm per $D$ hours. By superposition, it is the sum of an endless series of $D$-hour unit hydrographs, each lagged by $D$:

$$
S(t) = \sum_{j=0}^{\infty} U_D(t - jD).
$$

The S-curve rises to an equilibrium discharge equal to the input rate times the area, $A/(3.6\,D)$ m³ s⁻¹, reached after a time equal to the time base of the unit hydrograph. Then, exactly as for the linear reservoir in Section 9.2, a $D'$-hour unit hydrograph is the difference between the S-curve and the same curve lagged by $D'$, rescaled for the new rate:

$$
U_{D'}(t) = \frac{D}{D'}\,\bigl[S(t) - S(t - D')\bigr].
$$ (eq-s-curve)

```{code-cell} python
:tags: [hide-input]
def s_curve(U, n_out):
    """S-curve from a unit hydrograph given at steps equal to its duration."""
    S = np.zeros(n_out)
    for j in range(n_out):
        S[j:] += U[: n_out - j] if len(U) >= n_out - j else np.r_[U, np.zeros(n_out - j - len(U))]
    return S

nS = 40
S1 = s_curve(U1, nS)                                           # from the 1-hour UH, rate 1 mm/h
tS = np.arange(1, nS + 1)
U3 = (1.0 / 3.0) * (S1 - np.r_[np.zeros(3), S1[:-3]])          # 3-hour UH
for UU, name in [(U1, "1-h"), (U3, "3-h")]:
    assert abs(UU.sum() * 3600 - 1000 * A) / (1000 * A) < 2e-3, name

fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.plot(tS, S1, color=ink, lw=2, label="S-curve (1 mm h⁻¹)")
ax.plot(tS, np.r_[np.zeros(3), S1[:-3]], color=ink2, lw=1.4, ls="--", label="S-curve lagged 3 h")
ax.plot(tU, U1, color=c1, lw=2, label="1-hour UH")
ax.plot(tS, U3, color=c2, lw=2.2, label="3-hour UH = (1/3) × difference")
ax.axhline(A / 3.6, color=ink2, lw=0.8, ls=":"); ax.text(33, A / 3.6 + 0.6, "A / 3.6", color=ink2, fontsize=9.5)
ax.set_xlim(0, nS); ax.set_xlabel("Time (h)"); ax.set_ylabel("m³ s⁻¹ (per mm for the UHs)")
ax.legend(loc="center right", fontsize=9.5)
plt.show()
glue("u1_pk", float(U1.max()), display=False); glue("u3_pk", float(U3.max()), display=False)
```

*Figure 9.4. Converting the 1-hour unit hydrograph of the 60 km² catchment into a 3-hour unit hydrograph with the S-curve. The S-curve levels off at $A/3.6$ = 16.7 m³ s⁻¹, the discharge from 1 mm h⁻¹ over 60 km². The 3-hour UH has the same volume but a lower, later peak ({glue:text}`u3_pk:.2f` against {glue:text}`u1_pk:.2f` m³ s⁻¹ per mm), because the same 1 mm is spread over three hours.*

When the S-curve is built from a UH derived from data, small errors make it oscillate around the equilibrium value; the oscillations are smoothed by hand before computing the new UH. Converting to a duration that is not a multiple of the original requires interpolating the S-curve.

## 9.6 Synthetic unit hydrographs

Most catchments where engineers must build something have no streamflow record. **Synthetic unit hydrographs** estimate the shape from catchment properties, using relations calibrated on gauged catchments. Each method gives a time parameter (lag or time to peak) and a peak discharge or a shape; the volume must always be checked against Equation {eq}`eq-uh-volume`.

### Snyder (1938)

Franklin Snyder related the unit hydrograph of catchments in the Appalachian Mountains to their size and shape. The **basin lag** $t_p$ (h), from the centroid of excess rainfall to the peak, is

$$
t_p = 0.75\,C_t\,(L\,L_c)^{0.3} \qquad (L, L_c \text{ in km}),
$$ (eq-snyder-lag)

where $L$ is the main-stream length from the outlet to the divide, $L_c$ the length along the main stream from the outlet to the point nearest the catchment centroid, and $C_t$ a regional coefficient (Snyder found values of about 1.8–2.2 in the Appalachians; steeper regions have smaller values). The original form is $t_p = C_t (L L_c)^{0.3}$ with lengths in miles; the factor $0.75 = 1.609^{-0.6}$ converts the lengths, so the same $C_t$ is used in both forms. The standard rainfall duration is $t_r = t_p/5.5$, and the peak discharge per mm of excess is

$$
Q_p = 0.275\,\frac{C_p\,A}{t_p} \qquad (\text{m}^3\,\text{s}^{-1} \text{ per mm},\ A \text{ in km}^2,\ t_p \text{ in h}),
$$ (eq-snyder-peak)

converted from the original $q_p = 640\,C_p/t_p$ (ft³ s⁻¹ per mi² per inch), with the peaking coefficient $C_p$ between about 0.4 and 0.8. For a duration $t_R$ other than $t_r$, the lag is adjusted to $t_{pR} = t_p + (t_R - t_r)/4$ and $t_{pR}$ replaces $t_p$ in Equation {eq}`eq-snyder-peak`. Snyder gave only the lag, the peak and empirical widths; the rest of the shape is sketched so that the volume is 1 mm. In the code below the shape is a gamma curve fitted to the Snyder peak and timing with unit volume, a common modern practice.

```{code-cell} python
mi = 1.609344
for L, Lc, Ct in [(22.0, 10.0, 2.0)]:
    tp_us = Ct * (L / mi * Lc / mi) ** 0.3
    tp_si = 0.75 * Ct * (L * Lc) ** 0.3
    print(f"Snyder lag: original {tp_us:.3f} h, SI form {tp_si:.3f} h  (1.609^-0.6 = {mi ** -0.6:.4f})")
cfs, mi2, inch = 0.3048 ** 3, mi ** 2, 25.4
print(f"Snyder peak constant: 640 ft³/s per mi² per in = {640 * cfs / mi2 / inch:.4f} m³/s per km² per mm")
print(f"NRCS peak constant:   484 ft³/s per mi² per in = {484 * cfs / mi2 / inch:.4f} m³/s per km² per mm")
```

### The NRCS unit hydrograph

The NRCS (formerly SCS) unit hydrograph is based on a dimensionless shape averaged from many catchments. In its **triangular** approximation, the time to peak, measured from the start of excess rainfall of duration $D$, is

$$
T_p = \frac{D}{2} + t_L, \qquad t_L \approx 0.6\,t_c ,
$$

where $t_L$ is the lag (Chapter 8), and the recession lasts $1.67\,T_p$, so the time base is $2.67\,T_p$. The peak follows from the volume condition alone. The area of the triangle must equal $A \times 1$ mm:

$$
\frac{1}{2}\,Q_p\,(2.67\,T_p \times 3600) = 1000\,A
\quad\Longrightarrow\quad
Q_p = \frac{2 \times 1000}{2.67 \times 3600}\,\frac{A}{T_p} = 0.208\,\frac{A}{T_p},
$$ (eq-nrcs-peak)

with $Q_p$ in m³ s⁻¹ per mm, $A$ in km² and $T_p$ in h. This is the SI equivalent of the familiar $q_p = 484\,A/T_p$ (ft³ s⁻¹ per inch, $A$ in mi²); the code above confirms it. The factor 0.208 says that 37.5% of the volume lies under the rising limb. The smooth **curvilinear** NRCS hydrograph has the same peak and time to peak; it is closely approximated by the gamma-type curve

$$
\frac{Q}{Q_p} = \left(\frac{t}{T_p}\right)^{m}\exp\!\left[m\left(1 - \frac{t}{T_p}\right)\right], \qquad m = 3.7,
$$

whose volume with $Q_p = 0.208\,A/T_p$ is also $A \times 1$ mm (checked in Example 9.5).

### Clark (1945)

Clark separated the two processes that shape a hydrograph: **translation**, the time water takes to travel to the outlet, and **attenuation**, the smoothing caused by storage. Excess rainfall is first translated with a **time–area histogram**, the fraction of the catchment area that lies within each band of travel time from the outlet, up to $t_c$. The translated inflow is then routed through a single linear reservoir with storage constant $R$. In discrete form, with time step $\Delta t$,

$$
O_n = c\,I_n + (1 - c)\,O_{n-1}, \qquad c = \frac{\Delta t}{R + 0.5\,\Delta t},
$$

where $I$ is the translated inflow and $O$ the outflow. A time–area curve can be measured from a map; when it is not, the curve $A_t/A = 1.414\,(t/t_c)^{1.5}$ for $t \le t_c/2$ and $1 - 1.414\,(1 - t/t_c)^{1.5}$ beyond, used as the default in the HEC-HMS program, describes a catchment that is widest in the middle. Clark's method needs only two parameters, $t_c$ and $R$, both of which can be related to catchment properties.

### The Nash cascade (1957)

James Nash represented the catchment as a series of $n$ identical linear reservoirs, each with storage constant $k$. The outflow of each reservoir is the inflow of the next, and the impulse response of the cascade is a **gamma distribution**:

$$
u(t) = \frac{1}{k\,\Gamma(n)}\left(\frac{t}{k}\right)^{n-1} e^{-t/k},
$$ (eq-nash)

where $\Gamma$ is the gamma function ($\Gamma(n) = (n - 1)!$ for integer $n$, but $n$ need not be an integer). With $n = 1$ this is the single reservoir of Section 9.2; as $n$ grows, the response becomes more delayed and more symmetric (Figure 9.5). The mean of the IUH, its **lag**, is $n\,k$, and its variance is $n\,k^2$. These moments give a direct way to fit $n$ and $k$: the lag between the centroids of excess rainfall and direct runoff estimates $nk$, and the difference between the second moments of the two estimates $nk^2$ (Example 9.6).

```{code-cell} python
:tags: [hide-input]
lag = 6.0
tt = np.linspace(0, 30, 601)
fig, ax = plt.subplots(figsize=(6.4, 3.9))
for n, col in [(1, ink2), (2, c3), (4, c1), (8, c2)]:
    kk = lag / n
    ax.plot(tt, gamma.pdf(tt, a=n, scale=kk), color=col, lw=2.2, label=f"n = {n}, k = {kk:.2f} h")
ax.axvline(lag, color=ink2, lw=0.8, ls=":"); ax.text(lag + 0.3, 0.215, "lag nk = 6 h", color=ink2, fontsize=9.5)
ax.set_xlabel("Time (h)"); ax.set_ylabel("u(t) (h⁻¹)"); ax.set_xlim(0, 30); ax.set_ylim(0, 0.235)
ax.legend(loc="upper right", fontsize=9.5)
plt.show()
```

*Figure 9.5. Instantaneous unit hydrographs of Nash cascades with the same lag $nk$ = 6 h. More reservoirs in series give a later, sharper peak; a single reservoir peaks at once.*

### Comparing the methods

Figure 9.6 applies the four synthetic methods to one catchment. They agree on the general timing, set by similar lag estimates, but their peaks differ by tens of percent. A synthetic unit hydrograph is a regional relation applied to a catchment that was not in the data set, so its parameters should be checked against nearby gauged catchments whenever possible.

```{code-cell} python
:tags: [hide-input]
A_s, L, Lc, Ct, Cp = 120.0, 22.0, 10.0, 1.5, 0.62
tc_s, R_c = 5.5, 4.5                                          # h, for NRCS and Clark
D, dts = 1.0, 0.25
tq = np.arange(0, 40 + dts, dts)

def gamma_uh(tpk, qpk_target=None, m=None):
    """Unit-volume gamma-shaped UH (m³/s per mm) with peak at tpk; either the shape m or the peak is given."""
    shape = lambda m: (np.clip(tq, 1e-9, None) / tpk) ** m * np.exp(m * (1 - tq / tpk))
    if m is None:
        from scipy.optimize import brentq
        m = brentq(lambda m: qpk_target * shape(m).sum() * dts * 3600 - 1000 * A_s, 0.5, 60)
    s = shape(m)
    return s / (s.sum() * dts * 3600) * 1000 * A_s

# Snyder: lag from centroid of 1-h excess, peak from Cp; shape fitted with unit volume
tp = 0.75 * Ct * (L * Lc) ** 0.3; tr = tp / 5.5
tpR = tp + (D - tr) / 4
U_sny = gamma_uh(tpR + D / 2, qpk_target=0.275 * Cp * A_s / tpR)
# NRCS curvilinear (gamma approximation, m = 3.7)
Tp = D / 2 + 0.6 * tc_s
U_nrcs = gamma_uh(Tp, m=3.7)
# Clark: time–area translation then linear reservoir, for 1 mm over D
ta_cum = lambda x: np.where(x <= tc_s / 2, 1.414 * (x / tc_s) ** 1.5, 1 - 1.414 * np.clip(1 - x / tc_s, 0, None) ** 1.5)
edges = np.arange(0, tc_s + dts, dts); frac = np.diff(np.clip(ta_cum(np.minimum(edges, tc_s)), 0, 1))
rain = np.where(tq < D, 1.0 / D, 0.0)                         # mm/h
inflow = np.convolve(rain, frac)[: len(tq)] * A_s / 3.6       # m³/s
c = dts / (R_c + 0.5 * dts)
U_clk = np.zeros(len(tq))
for n in range(1, len(tq)):
    U_clk[n] = c * inflow[n] + (1 - c) * U_clk[n - 1]
# Nash: n = 3, with the lag nk set to the mean centroid lag of the other three UHs
cen = lambda U: (U * tq).sum() / U.sum() - D / 2              # centroid lag after the centroid of excess
n_nash = 3.0
k_nash = np.mean([cen(U_sny), cen(U_nrcs), cen(U_clk)]) / n_nash
U_nash = (gamma.cdf(tq, n_nash, scale=k_nash) - gamma.cdf(np.clip(tq - D, 0, None), n_nash, scale=k_nash)) / D * A_s / 3.6

fig, ax = plt.subplots(figsize=(6.4, 4.1))
res = {}
for U, name, col, ls in [(U_sny, "Snyder", c1, "-"), (U_nrcs, "NRCS", c2, "-"),
                         (U_clk, "Clark", c3, "-"), (U_nash, "Nash (n = 3)", ink2, "--")]:
    ax.plot(tq, U, color=col, lw=2.2, ls=ls, label=f"{name}: peak {U.max():.1f} at {tq[U.argmax()]:.1f} h")
    res[name] = (U.sum() * dts * 3600) / (1000 * A_s)
ax.set_xlim(0, 30); ax.set_xlabel("Time from start of excess (h)"); ax.set_ylabel("U (m³ s⁻¹ per mm)")
ax.legend(loc="upper right", fontsize=9.5)
plt.show()
for name, v in res.items():
    assert abs(v - 1) < 0.01, (name, v)                       # every UH holds 1 mm
```

*Figure 9.6. One-hour unit hydrographs for a 120 km² catchment ($L$ = 22 km, $L_c$ = 10 km, $C_t$ = 1.5, $C_p$ = 0.62, $t_c$ = 5.5 h, Clark $R$ = 4.5 h; Nash $n$ = 3 with $nk$ equal to the mean centroid lag of the other three). Every curve holds exactly 1 mm over the catchment (checked in the code), but the peaks differ: the methods distribute the same volume differently in time.*

## 9.7 Applying a unit hydrograph

To compute a flood hydrograph from a storm:

1. compute the excess rainfall hyetograph with a loss method (Chapter 7), at a time step equal to the UH duration;
2. convolve it with the unit hydrograph (Equation {eq}`eq-discrete-convolution`);
3. add baseflow (Chapter 6);
4. check that the direct runoff depth equals the total excess.

The linearity assumption deserves a final word. Unit hydrographs derived from moderate floods tend to **underestimate** the peaks of extreme floods, because flow velocities, and therefore the speed of the response, increase with depth. Some guidelines increase the peak of a derived unit hydrograph, or shorten its lag, when it is used for very large design floods. The opposite nonlinearity appears when floods spill onto wide floodplains, whose storage slows the flood. Chapter 13 returns to the question of why linear models work as well as they do.

## 9.8 Worked examples

**Example 9.1 — A linear reservoir under a pulse of rain.** A 60 km² catchment behaves like a linear reservoir with $k$ = 5 h. Excess rainfall of 20 mm falls uniformly over 2 hours. Find the peak discharge and when it occurs, and the discharge 10 hours after the rain stops.

```{code-cell} python
A_c, k, D, Pe = 60.0, 5.0, 2.0, 20.0
i_rate = Pe / D                                    # mm/h
Q_peak = i_rate * (1 - np.exp(-D / k))             # mm/h at t = D
Q_later = Q_peak * np.exp(-10 / k)
print(f"Peak at t = {D:.0f} h: {Q_peak:.2f} mm/h = {Q_peak * A_c / 3.6:.1f} m³/s")
print(f"10 h after the rain stops: {Q_later:.3f} mm/h = {Q_later * A_c / 3.6:.2f} m³/s")
```

**Example 9.2 — A unit hydrograph from a single burst.** An isolated 2-hour storm on a 45 km² catchment produced the direct runoff below (after baseflow separation), at 1-hour intervals. Derive the 2-hour unit hydrograph and check its volume.

```{code-cell} python
A_e = 45.0
Qd_obs = np.array([0, 4.2, 15.8, 31.5, 38.0, 32.6, 23.9, 16.4, 10.8, 6.9, 4.3, 2.6, 1.5, 0.8, 0.3, 0.0])  # m³/s
V = Qd_obs.sum() * 3600                            # trapezoid rule with zero ends = simple sum
Pe = V / (A_e * 1e6) * 1000
U2 = Qd_obs / Pe
print(f"Direct runoff volume = {V / 1e6:.3f} million m³, excess depth = {Pe:.2f} mm")
print("2-h UH (m³/s per mm):", ", ".join(f"{u:.2f}" for u in U2))
print(f"Volume check: {U2.sum() * 3600 / (A_e * 1e6) * 1000:.3f} mm")
```

**Example 9.3 — Flood from a three-hour storm.** Using the 1-hour UH of the 60 km² catchment (Figure 9.2), compute the hydrograph of total flow for excess rainfall of 8, 20 and 6 mm in three successive hours, with a constant baseflow of 3.5 m³ s⁻¹.

```{code-cell} python
Qdir = np.convolve(Pex, U1)
Qtot = Qdir + 3.5
print(f"Peak total flow = {Qtot.max():.1f} m³/s at t = {Qtot.argmax() + 1} h")
print(f"Direct runoff depth = {Qdir.sum() * 3600 / (A * 1e6) * 1000:.2f} mm (excess = {Pex.sum():.0f} mm)")
```

**Example 9.4 — Changing the duration.** From the 1-hour UH, derive the 3-hour UH and compute the peak flow for 30 mm of excess falling uniformly over 3 hours. Compare with convolving the 1-hour UH with 10 mm in each of three hours.

```{code-cell} python
print(f"3-h UH: peak {U3.max():.2f} m³/s per mm -> 30 mm gives {30 * U3.max():.1f} m³/s")
print(f"1-h UH with 10, 10, 10 mm: peak {np.convolve([10, 10, 10], U1).max():.1f} m³/s")
```

The two answers are identical, as superposition requires: the 3-hour UH is simply a convenient package of the same calculation.

**Example 9.5 — NRCS unit hydrograph.** For the 120 km² catchment of Figure 9.6 ($t_c$ = 5.5 h), find $T_p$ and $Q_p$ for a 1-hour UH by Equation {eq}`eq-nrcs-peak`, and check that both the triangular and the gamma-shaped curves hold 1 mm.

```{code-cell} python
Tp = 1.0 / 2 + 0.6 * 5.5
Qp = 0.208 * 120.0 / Tp
tri_vol = 0.5 * Qp * 2.67 * Tp * 3600 / (120e6) * 1000
tg = np.linspace(0, 20 * Tp, 20001)
curv = Qp * (tg / Tp) ** 3.7 * np.exp(3.7 * (1 - tg / Tp))
curv_vol = np.trapezoid(curv, tg * 3600) / 120e6 * 1000
print(f"Tp = {Tp:.2f} h, Qp = {Qp:.2f} m³/s per mm, time base = {2.67 * Tp:.1f} h")
print(f"Volume: triangle {tri_vol:.4f} mm, curvilinear (m = 3.7) {curv_vol:.4f} mm")
```

**Example 9.6 — Fitting a Nash cascade by moments.** For the event of Example 9.2, the excess rainfall fell evenly over the first 2 hours. Use the first two moments of the excess and of the direct runoff to estimate $n$ and $k$.

```{code-cell} python
t_r = np.array([0.5, 1.5]); w_r = np.array([1.0, 1.0])            # excess: equal amounts in two hours
t_q = np.arange(len(Qd_obs), dtype=float)                         # direct runoff ordinates at 0, 1, 2, ... h
m1 = lambda t, w: (t * w).sum() / w.sum()
m2 = lambda t, w: ((t - m1(t, w)) ** 2 * w).sum() / w.sum()        # variance about the centroid
lag_nk = m1(t_q, Qd_obs) - m1(t_r, w_r)
var_nk2 = m2(t_q, Qd_obs) - m2(t_r, w_r)
k_fit = var_nk2 / lag_nk; n_fit = lag_nk / k_fit
print(f"Lag nk = {lag_nk:.2f} h, variance nk² = {var_nk2:.2f} h²  ->  k = {k_fit:.2f} h, n = {n_fit:.2f}")
```

## 9.9 Common mistakes

- **Using a UH with a hyetograph of a different time step.** Convert the UH with the S-curve first.
- **Forgetting the volume check.** Every UH must hold $A \times 1$ mm. A UH that does not is wrong, whatever its source.
- **Mixing inch-based and mm-based ordinates.** A UH per inch is 25.4 times a UH per mm.
- **Convolving total rainfall instead of excess rainfall.** The UH routes excess only; losses must be removed first.
- **Forward substitution on real data.** It amplifies errors; use least squares with non-negativity.
- **Measuring $T_p$ from the wrong origin.** In the NRCS method $T_p$ is measured from the start of excess rainfall; Snyder's $t_p$ from the centroid of excess.

## 9.10 Exercises

```{code-cell} python
:tags: [remove-cell]
glue("w1_Qp", 0.208 * 85.0 / (0.5 + 0.6 * 4.0), display=False); glue("w1_Tp", 0.5 + 0.6 * 4.0, display=False)
glue("w1_tb", 2.67 * (0.5 + 0.6 * 4.0), display=False)
U_ex = np.array([0, 3.0, 9.0, 12.0, 8.0, 4.5, 2.0, 0.5, 0.0])
glue("w2_A", float(U_ex.sum() * 3600 / 1000), display=False)
Q3 = np.convolve([12.0, 25.0], U_ex); glue("w3_Q", float(Q3.max()), display=False); glue("w3_t", int(Q3.argmax()), display=False)
glue("w4_k", float(-1 / np.log(0.82)), display=False)
glue("w5_tp", 0.75 * 1.6 * (35 * 16) ** 0.3, display=False); glue("w5_Qp", 0.275 * 0.6 * 210 / (0.75 * 1.6 * (35 * 16) ** 0.3), display=False)
```

1. A 1-hour NRCS unit hydrograph is needed for an 85 km² catchment with $t_c$ = 4.0 h. Find $T_p$, $Q_p$ and the time base of the triangular UH.
2. A 1-hour unit hydrograph has ordinates 0, 3.0, 9.0, 12.0, 8.0, 4.5, 2.0, 0.5 and 0 m³ s⁻¹ per mm at hourly intervals. What catchment area does it imply?
3. With the UH of Exercise 2, find the peak direct runoff and its time for excess rainfall of 12 mm in the first hour and 25 mm in the second.
4. The recession of direct runoff from a small catchment falls by a factor of 0.82 each hour. If the catchment is treated as a single linear reservoir, what is $k$?
5. A 210 km² catchment has $L$ = 35 km and $L_c$ = 16 km. With $C_t$ = 1.6 and $C_p$ = 0.6, find Snyder's basin lag and the peak of the unit hydrograph of standard duration.

```{admonition} Answers
:class: dropdown
1. $T_p$ = {glue:text}`w1_Tp:.1f` h, $Q_p$ = {glue:text}`w1_Qp:.2f` m³ s⁻¹ per mm, time base {glue:text}`w1_tb:.1f` h.
2. $A$ = {glue:text}`w2_A:.1f` km² (sum of ordinates × 3600 s ÷ 1000 m³ per km² per mm).
3. {glue:text}`w3_Q:.0f` m³ s⁻¹ at $t$ = {glue:text}`w3_t` h.
4. $k = -1/\ln 0.82$ = {glue:text}`w4_k:.2f` h.
5. $t_p$ = {glue:text}`w5_tp:.1f` h; $Q_p$ = {glue:text}`w5_Qp:.1f` m³ s⁻¹ per mm.
```

## 9.11 Further reading

- Dingman, S. L. (2015). *Physical Hydrology* (3rd ed.). Waveland Press. The section on unit hydrographs and event-response models.
- Chow, V. T., Maidment, D. R., and Mays, L. W. (1988). *Applied Hydrology*. McGraw-Hill. The chapters on unit hydrographs and synthetic unit hydrographs.
- 윤용남. *수문학* (in Korean). The chapter on the unit hydrograph.
- Sherman, L. K. (1932). Streamflow from rainfall by the unit-graph method. *Engineering News-Record*, 108, 501–505.
- Snyder, F. F. (1938). Synthetic unit-graphs. *Transactions, American Geophysical Union*, 19, 447–454.
- Clark, C. O. (1945). Storage and the unit hydrograph. *Transactions of the American Society of Civil Engineers*, 110, 1419–1446.
- Nash, J. E. (1957). The form of the instantaneous unit hydrograph. *International Association of Scientific Hydrology Publication*, 45(3), 114–121.
- USDA Natural Resources Conservation Service. *National Engineering Handbook*, Part 630 Hydrology, Chapter 16 (Hydrographs).

---

*Figures in this chapter are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
