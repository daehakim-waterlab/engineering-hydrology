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

# 6. Streamflow and the hydrograph

```{admonition} Core ideas
:class: tip
1. Rain reaches a stream by three main routes: **infiltration-excess** overland flow, **saturation-excess** overland flow from wet areas near streams, and **subsurface stormflow** through the soil. Which one dominates depends on soils, climate and land use.
2. Streamflow is rarely measured directly. A gauge records the water level (**stage**), and a **rating curve** $Q = a\,(h - h_0)^b$, fitted to occasional discharge measurements, converts it to discharge.
3. A **hydrograph** plots discharge against time. Its shape, with a rising limb, a peak and a falling limb, reflects both the storm and the catchment.
4. Engineers split the hydrograph into **direct runoff**, the quick response to the storm, and **baseflow**, the slow release from groundwater. The split can be drawn graphically or computed with a **recursive digital filter**.
5. During dry weather, flow declines along a **recession curve**, often close to exponential. The **baseflow index** and the **runoff ratio** summarize how a catchment turns rainfall into streamflow.
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

## 6.1 How rain becomes streamflow

Of the rain that reaches the ground during a storm, part is stored in surface depressions, part infiltrates, and part flows over the surface. Of the water that infiltrates, part is held in the soil, part moves laterally to the stream within hours to days, and part percolates to groundwater. Hydrologists distinguish three mechanisms that produce the quick response of a stream to a storm (Figure 6.1).

**Infiltration-excess overland flow.** When rainfall intensity exceeds the infiltration capacity of the soil (Chapter 5), water ponds and flows over the surface. Robert Horton made this mechanism the center of his theory of runoff, and it is often called **Hortonian overland flow**. It dominates where infiltration capacity is low: paved and compacted urban surfaces, crusted or bare soils, and semi-arid regions with intense convective storms. Because it depends on intensity, short bursts of heavy rain matter more than the storm total.

**Saturation-excess overland flow.** In humid regions with permeable soils and vegetation, measured infiltration capacities are often higher than almost any rainfall intensity, yet streams still rise quickly. Field studies in the 1960s and 1970s showed why. Near streams, in hollows and at the foot of slopes, the water table is shallow. During a storm it rises to the surface, and rain falling on these saturated areas cannot infiltrate whatever its intensity. Groundwater may also seep out of the soil there (**return flow**). The saturated area expands during a storm and shrinks afterward, so it is called a **variable source area**. Under this mechanism, runoff depends on the storm **depth** and the **antecedent wetness** of the catchment more than on intensity.

**Subsurface stormflow.** Water that infiltrates on a hillslope can move downslope through the soil above a less permeable layer, and especially through macropores such as root channels and animal burrows, reaching the stream within hours. Tracer studies show that much of the water in a storm hydrograph of a humid forested catchment is "old" water that was stored in the soil before the storm and was pushed out by the new rain.

Rain that falls directly on the channel (**channel precipitation**) is a small but immediate contribution. Between storms, streams are fed by **baseflow** from groundwater (Chapter 5).

```{code-cell} python
:tags: [hide-input]
x = np.linspace(0, 10, 300)
surface = 0.2 + 2.6 * (x / 10) ** 1.4                 # hillslope rising away from the stream
bedrock = surface - 1.4 - 0.4 * x / 10
wt_dry = np.maximum(bedrock + 0.25 + 0.35 * np.exp(-x / 2.0), bedrock)
wt_wet = np.minimum(np.maximum(bedrock + 0.6 + 1.05 * np.exp(-x / 3.2), bedrock), surface)

fig, ax = plt.subplots(figsize=(6.6, 3.9))
ax.fill_between(x, bedrock - 0.4, bedrock, color="#d9d8d3")
ax.fill_between(x, bedrock, surface, color="#efe9dc")
ax.plot(x, surface, color=ink, lw=1.6)
ax.plot(x, wt_dry, color=c1, lw=1.4, ls="--", label="Water table before storm")
ax.plot(x, wt_wet, color=c1, lw=2.2, label="Water table during storm")
sat = wt_wet >= surface - 1e-6
ax.fill_between(x[sat], surface[sat], surface[sat] + 0.06, color=c3)
ax.text(0.35, surface[sat].max() + 0.35, "saturated area\n(saturation excess)", fontsize=9, color=c3)
ax.annotate("", xy=(6.2, 0.2 + 2.6 * 0.62 ** 1.4 + 0.08), xytext=(8.4, 0.2 + 2.6 * 0.84 ** 1.4 + 0.08),
            arrowprops=dict(arrowstyle="-|>", color=c2, lw=2))
ax.text(6.0, 2.75, "infiltration-excess\noverland flow", fontsize=9, color=c2)
ax.annotate("", xy=(3.4, 0.55), xytext=(6.4, 1.15), arrowprops=dict(arrowstyle="-|>", color=ink2, lw=1.8))
ax.text(4.2, 0.35, "subsurface stormflow", fontsize=9, color=ink2)
ax.plot([0, 0.45], [0.2, 0.2], color=c1, lw=5); ax.text(0.05, -0.15, "stream", fontsize=9, color=c1)
for xr in np.linspace(1, 9.5, 12):
    ax.plot([xr, xr - 0.15], [3.6, 3.25], color=c1, lw=0.8)
ax.set_xlim(0, 10); ax.set_ylim(-0.9, 3.8); ax.axis("off")
ax.legend(loc="lower right", fontsize=9, bbox_to_anchor=(1.0, -0.02))
plt.show()
```

*Figure 6.1. Runoff mechanisms on a hillslope (schematic, vertical scale exaggerated). Near the stream the water table rises to the surface during a storm and creates a saturated area where all rain runs off. Upslope, overland flow occurs only where rainfall intensity exceeds the infiltration capacity. Water also reaches the stream through the soil.*

The mechanism matters for engineering because it determines which methods are appropriate. An intensity-based loss method (Chapter 7) suits a paved car park; a method that accounts for storm depth and antecedent wetness suits a forested mountain catchment. Land-use change can switch the mechanism: urbanization replaces subsurface pathways with fast surface flow and produces larger, earlier peaks.

## 6.2 Measuring streamflow

### Stage and the velocity–area method

Discharge cannot be measured continuously at most sites. What a gauging station records continuously is the **stage** $h$, the height of the water surface above a fixed datum, using a float, a pressure sensor or a radar sensor. Discharge is measured occasionally, typically a few to a few dozen times a year, with the **velocity–area method**: the cross-section is divided into vertical strips, the depth and mean velocity are measured in each, and

$$
Q = \sum_j v_j\, d_j\, w_j ,
$$

where $v_j$ is the mean velocity, $d_j$ the depth and $w_j$ the width of strip $j$. In each vertical, the mean velocity is approximately the point velocity at 0.6 of the depth below the surface, or the average of the velocities at 0.2 and 0.8 of the depth, a consequence of the roughly logarithmic velocity profile (Chapter 4 of the [Hydraulics](https://daehakim-waterlab.github.io/hydraulics/) book). Acoustic Doppler current profilers mounted on a boat now measure the whole section in one crossing. In small streams, a weir or flume with a known stage–discharge relation can be built instead (Chapter 6 of the Hydraulics book).

### The rating curve

Plotting the measured discharges against the stage at the time of measurement gives the **stage–discharge relation**, or **rating curve**. At a section controlled by a downstream riffle or by the channel itself, it is usually well described by a power function:

\begin{equation*}
Q = a\,(h - h_0)^b ,
\tag{6.1}
\end{equation*}

where $h_0$ is the stage of zero flow, and $a$ and $b$ are fitted constants. The exponent reflects the hydraulics of the control: about 1.5 for a section that behaves like a broad weir, about 5/3 for a wide channel in uniform flow (from the [Manning equation](https://daehakim-waterlab.github.io/hydraulics/chapters/ch06_flow_in_open_channels.html#uniform-flow-the-chezy-and-manning-equations)), and larger for V-shaped or trapezoidal sections. Taking logarithms, $\ln Q = \ln a + b \ln(h - h_0)$, which is a straight line, so $a$ and $b$ can be found by linear regression once $h_0$ is chosen. The value of $h_0$ can be surveyed at the control or chosen to make the plot straightest.

Rating curves are the weakest link in most streamflow records. Floods are rarely gauged, so the curve is often **extrapolated** far beyond the highest measurement. Flow spreads onto the floodplain at high stage, changing the hydraulics. Channel beds scour and fill, which **shifts** the curve. And on flat rivers the same stage carries more water on the rising limb than on the falling limb (**hysteresis**), because the water-surface slope is steeper as the flood wave arrives. Example 6.2 shows how extrapolation can go wrong.

## 6.3 The hydrograph

A **hydrograph** is a graph of discharge against time at a point on a stream. The response to a single storm, the **storm hydrograph**, has a characteristic shape (Figure 6.2):

- before the storm, the stream carries baseflow, slowly declining;
- the **rising limb** begins some time after the rain starts, when quickflow reaches the gauge, and is steep because contributions from more and more of the catchment arrive together;
- the **peak** occurs when the largest part of the catchment is contributing at once;
- the **falling limb** is longer and gentler, as the catchment drains;
- the **recession** at the end is fed by slowly draining soil and groundwater.

Several characteristic times are defined from the hydrograph and the hyetograph above it. The **time to peak** is measured from the start of effective rain (or the start of rise) to the peak. The **lag time** $t_L$ is the time from the centroid of effective rainfall to the peak (or to the centroid of the direct runoff). The **time base** is the duration of direct runoff. These times grow with catchment size and decrease with slope and with urbanization. They are used to build unit hydrographs in Chapter 9.

```{code-cell} python
:tags: [hide-input]
# A synthetic storm hydrograph: excess rain through a gamma-shaped response, plus slowly declining baseflow
dt = 1.0                                              # h
t = np.arange(0, 96 + dt, dt)
rain = np.zeros_like(t); rain[(t >= 6) & (t < 11)] = [6, 14, 18, 9, 4]   # mm per hour
A = 85.0                                              # km²
excess = np.zeros_like(t); excess[(t >= 7) & (t < 11)] = [5.5, 9.5, 4.0, 1.0]   # mm per hour
tau = np.arange(0, 60, dt)
u = tau ** 2.2 * np.exp(-tau / 3.2); u /= u.sum()     # response to 1 mm (dimensionless weights)
direct = np.convolve(excess, u)[: len(t)] * A / 3.6   # m³/s  (1 mm/h over A km² = A/3.6 m³/s)
base = 2.2 * np.exp(-t / 400) + 0.9 * np.clip(np.sin(np.pi * np.clip(t - 14, 0, 70) / 140), 0, None)
Q = direct + base

fig, (a0, a1) = plt.subplots(2, 1, figsize=(6.4, 5.4), sharex=True, gridspec_kw=dict(height_ratios=[1, 2.6]))
a0.bar(t + 0.5, rain, width=0.9, color=c1, alpha=0.5, label="Rainfall")
a0.bar(t + 0.5, excess, width=0.9, color=c1, label="Excess rainfall")
a0.invert_yaxis(); a0.set_ylabel("mm h⁻¹"); a0.legend(loc="lower right", fontsize=9, ncol=2)
a1.plot(t, Q, color=c1, lw=2.4, label="Streamflow")
a1.fill_between(t, base, Q, color=c1, alpha=0.15, label="Direct runoff")
a1.plot(t, base, color=c2, lw=1.8, ls="--", label="Baseflow")
ip = Q.argmax()
cent_rain = (excess * (t + 0.5)).sum() / excess.sum()
a1.annotate("peak", (t[ip], Q[ip]), xytext=(t[ip] + 6, Q[ip] - 2), fontsize=10,
            arrowprops=dict(arrowstyle="-", color=ink2, lw=0.8))
a1.text(t[ip] - 13, Q[ip] * 0.55, "rising\nlimb", fontsize=9.5, color=ink2)
a1.text(t[ip] + 12, Q[ip] * 0.45, "falling limb", fontsize=9.5, color=ink2)
a1.text(66, base[66] + 3.5, "recession", fontsize=9.5, color=ink2)
y_lag = Q[ip] * 1.06
a1.annotate("", xy=(t[ip], y_lag), xytext=(cent_rain, y_lag), arrowprops=dict(arrowstyle="<->", color=ink, lw=1))
a1.text((t[ip] + cent_rain) / 2, y_lag + 1.0, f"lag $t_L$ ≈ {t[ip] - cent_rain:.0f} h", ha="center", fontsize=9.5)
a1.set_ylim(0, Q.max() * 1.25); a1.set_xlim(0, 96)
a1.set_xlabel("Time (h)"); a1.set_ylabel("Q (m³ s⁻¹)"); a1.legend(loc="upper right", fontsize=9.5)
plt.show()
V_dir = direct.sum() * 3600
glue("hg_depth", float(V_dir / (A * 1e6) * 1000), display=False)
glue("hg_excess", float(excess.sum()), display=False)
glue("hg_peak", float(Q.max()), display=False)
```

*Figure 6.2. A synthetic storm hydrograph from an 85 km² catchment and the hyetograph above it. The direct runoff (shaded) has a depth of {glue:text}`hg_depth:.1f` mm over the catchment, equal to the {glue:text}`hg_excess:.0f` mm of excess rainfall, as conservation of mass requires. The peak, {glue:text}`hg_peak:.0f` m³ s⁻¹, arrives several hours after the most intense rain.*

## 6.4 Separating baseflow from direct runoff

To relate a storm to the flood it causes, engineers separate the hydrograph into **direct runoff** (quickflow), which is attributed to the storm, and **baseflow**, which would have been there anyway. The separation is a convention, not a measurement: water in the stream carries no label saying which path it took. What matters is to use a method consistently, especially when the same method was used to derive the unit hydrograph (Chapter 9) that will later be applied.

### Graphical methods

Three graphical methods are common (Figure 6.3).

1. **Constant discharge.** A horizontal line from the start of the rise. Simple, but it ignores the decline of baseflow and makes the direct runoff never end if the flow drops below the starting value.
2. **Straight line to a fixed time after the peak.** A straight line joins the start of the rise to the point on the falling limb $N$ days after the peak. An empirical formula for $N$ (Linsley, Kohler and Paulhus, 1975), originally written as $N = A^{0.2}$ with $A$ in square miles, becomes in SI units

\begin{equation*}
N = 0.827\,A^{0.2} \qquad (N \text{ in days},\ A \text{ in km}^2),
\tag{6.2}
\end{equation*}

since $(1/2.59)^{0.2} = 0.827$. The code below checks the conversion.

3. **Extended recession.** The pre-storm recession is extended under the peak, then joined by a straight line to the point where the falling limb meets the post-storm recession curve.

```{code-cell} python
for A_km2 in (10.0, 85.0, 1000.0):
    A_mi2 = A_km2 / 2.589988                        # 1 mi² = 2.589988 km²
    print(f"A = {A_km2:6.0f} km²: original N = {A_mi2 ** 0.2:.3f} d, SI form N = {0.827 * A_km2 ** 0.2:.3f} d")
```

```{code-cell} python
:tags: [hide-input]
i0 = np.argmax(Q > Q[0] * 1.02) - 1                  # start of rise
N_h = 0.827 * A ** 0.2 * 24                           # h
iN = int(round(ip + N_h / dt))
b_const = np.where((t >= t[i0]) & (Q > Q[i0]), Q[i0], Q)
b_line = Q.copy(); b_line[i0:iN + 1] = np.linspace(Q[i0], Q[iN], iN - i0 + 1)
fig, ax = plt.subplots(figsize=(6.4, 3.9))
ax.plot(t, Q, color=c1, lw=2.4, label="Streamflow")
ax.plot(t, b_const, color=c3, lw=1.8, ls="-.", label="Constant discharge")
ax.plot(t, b_line, color=c2, lw=1.8, ls="--", label=f"Straight line to N = {N_h / 24:.1f} d after peak")
ax.plot(t, base, color=ink2, lw=1.2, ls=":", label="Baseflow used to build the example")
ax.plot([t[i0], t[iN]], [Q[i0], Q[iN]], "o", color=c2, ms=5)
ax.set_xlim(0, 96); ax.set_ylim(0, 12)
ax.set_xlabel("Time (h)"); ax.set_ylabel("Q (m³ s⁻¹)"); ax.legend(loc="upper right", fontsize=9)
plt.show()
for name, b in [("const", b_const), ("line", b_line)]:
    glue(f"sep_{name}", float(((Q - b).clip(0)).sum() * 3600 / (A * 1e6) * 1000), display=False)
```

*Figure 6.3. Two graphical separations of the hydrograph of Figure 6.2 (vertical axis cut at 12 m³ s⁻¹ to show the base). They give direct runoff depths of {glue:text}`sep_const:.1f` mm (constant discharge) and {glue:text}`sep_line:.1f` mm (straight line). Here the straight line happens to lie close to the baseflow used to construct the example (dotted). With real data that baseflow is unknown, so any separation remains a convention.*

### Recursive digital filters

For long records, baseflow is separated automatically by treating the hydrograph as a signal: quickflow is the high-frequency part, baseflow the low-frequency part. The one-parameter filter of Lyne and Hollick (1979) computes the quickflow $q_f$ from the total flow $Q$ at successive time steps:

\begin{equation*}
q_{f,t} = \alpha\, q_{f,t-1} + \frac{1 + \alpha}{2}\,\bigl(Q_t - Q_{t-1}\bigr),
\qquad b_t = Q_t - q_{f,t},
\tag{6.3}
\end{equation*}

with the constraint $0 \le b_t \le Q_t$. The filter parameter $\alpha$ controls how slowly baseflow may change; $\alpha = 0.925$ for daily data was recommended by Nathan and McMahon (1990) after comparing with graphical methods. The filter is often passed three times (forward, backward, forward) to smooth the result.

Eckhardt (2005) derived a two-parameter filter from the assumption that the aquifer behaves as a linear reservoir (Chapter 5):

\begin{equation*}
b_t = \frac{(1 - \text{BFI}_{\max})\,a\,b_{t-1} + (1 - a)\,\text{BFI}_{\max}\,Q_t}{1 - a\,\text{BFI}_{\max}},
\qquad b_t \le Q_t ,
\tag{6.4}
\end{equation*}

where $a$ is the recession constant of baseflow (Section 6.5) and $\text{BFI}_{\max}$ is the maximum long-term baseflow index that the filter can produce, which reflects the aquifer (Eckhardt suggested about 0.80 for perennial streams on porous aquifers, 0.50 for ephemeral streams on porous aquifers and 0.25 for perennial streams on hard rock).

Figure 6.4 applies both filters to a year of synthetic daily flow generated by a model with a fast and a slow reservoir, so that the "true" slow component is known.

```{code-cell} python
:tags: [hide-input]
# One year of synthetic daily flow (mm/d) from a two-reservoir model with known slow flow
rng = np.random.default_rng(12)
nd = 365
doy = np.arange(nd)
wet = 0.15 + 0.35 * np.exp(-((doy - 200) / 40.0) ** 2)
P = np.where(rng.random(nd) < wet, rng.gamma(0.8, 18.0, nd), 0.0)
k_fast, k_slow, split = 2.0, 45.0, 0.35            # days; fraction of rain that becomes quickflow
Sf, Ss = 0.0, 60.0
Qf, Qs = np.zeros(nd), np.zeros(nd)
for d in range(nd):
    Sf += split * P[d]; Ss += 0.30 * P[d]           # the rest is lost to evapotranspiration
    Qf[d], Qs[d] = Sf / k_fast, Ss / k_slow
    Sf -= Qf[d]; Ss -= Qs[d]
Qd = Qf + Qs

def lyne_hollick(Q, alpha=0.925, passes=3):
    q = Q.copy()
    for p in range(passes):
        seq = q if p % 2 == 0 else q[::-1]
        qf = np.zeros_like(seq)
        for k in range(1, len(seq)):
            qf[k] = alpha * qf[k - 1] + (1 + alpha) / 2 * (seq[k] - seq[k - 1])
        b = np.clip(seq - np.clip(qf, 0, None), 0, seq)
        q = b if p % 2 == 0 else b[::-1]
    return q

def eckhardt(Q, a, bfimax):
    b = np.empty_like(Q); b[0] = Q[0] * bfimax
    for k in range(1, len(Q)):
        b[k] = min(((1 - bfimax) * a * b[k - 1] + (1 - a) * bfimax * Q[k]) / (1 - a * bfimax), Q[k])
    return b

a_true = np.exp(-1 / k_slow)
b_lh = lyne_hollick(Qd)
b_ek = eckhardt(Qd, a_true, 0.80)
fig, ax = plt.subplots(figsize=(6.4, 3.9))
ax.semilogy(doy, Qd, color=c1, lw=1.2, label="Streamflow")
ax.semilogy(doy, Qs, color=ink, lw=1.6, ls=":", label="True slow component")
ax.semilogy(doy, b_lh, color=c2, lw=1.8, label="Lyne–Hollick, α = 0.925")
ax.semilogy(doy, b_ek, color=c3, lw=1.8, ls="--", label="Eckhardt, BFI$_{max}$ = 0.80")
ax.set_xlim(0, 364); ax.set_ylim(0.1, None)
ax.set_xticks([0, 90, 181, 273, 364], ["Jan", "Apr", "Jul", "Oct", "Dec"])
ax.set_ylabel("Q (mm d⁻¹)"); ax.legend(loc="lower center", fontsize=9, ncol=2)
plt.show()
for name, b in [("true", Qs), ("lh", b_lh), ("ek", b_ek)]:
    glue(f"bfi_{name}", float(b.sum() / Qd.sum()), display=False)
```

*Figure 6.4. Baseflow from two recursive filters applied to a year of synthetic daily streamflow (logarithmic scale). The baseflow index is {glue:text}`bfi_lh:.2f` with the Lyne–Hollick filter and {glue:text}`bfi_ek:.2f` with the Eckhardt filter, against {glue:text}`bfi_true:.2f` for the slow component of the model that generated the data. Both filters follow the slow component during recessions; they differ most during the wet season.*

## 6.5 Recessions, the baseflow index and the runoff ratio

### The recession curve

During dry weather the flow declines as the catchment drains. If the catchment behaves like a linear reservoir (Chapter 5),

\begin{equation*}
Q_t = Q_0\, e^{-t/k} = Q_0\, K_r^{\,t}, \qquad K_r = e^{-\Delta t / k},
\tag{6.5}
\end{equation*}

where $K_r$ is the **recession constant** per time step $\Delta t$ (for daily data, typically 0.85–0.995: smaller for fast components, larger for groundwater). On a semi-logarithmic plot an exponential recession is a straight line, which is why hydrographs are often plotted with a logarithmic discharge axis. In practice the recession constant often increases as flow decreases: fast stores drain first, and the slow aquifer controls the late recession. Overlaying many recession segments gives a **master recession curve** that summarizes this behavior.

A more general way to look at recessions is to plot the rate of decline $-dQ/dt$ against $Q$. For a linear reservoir, $-dQ/dt = Q/k$ is a straight line through the origin; for real catchments the plot is often a power law, $-dQ/dt = a\,Q^{b}$, with $b$ between 1 and 2. Chapter 13 shows that this single relation can be used to model a whole catchment.

### The baseflow index

The **baseflow index** (BFI) is the ratio of the volume of baseflow to the total volume of streamflow over a long period. It ranges from below 0.2 for flashy streams on impermeable rock or clay to above 0.8 for streams fed by large aquifers. It is a useful single number for describing how much a river depends on groundwater, which matters for low-flow management and water supply.

### The runoff ratio

The runoff ratio, the fraction of precipitation that becomes streamflow (Chapter 1), can also be computed for a single event, as direct runoff depth divided by storm rainfall, or for a long period, as $\bar{Q}/\bar{P}$ (Chapter 1). Event runoff ratios vary widely from storm to storm in the same catchment, from a few percent after a dry spell to more than half during a wet season, because they depend on antecedent wetness. This variability is the subject of Chapter 7.

## 6.6 Worked examples

**Example 6.1 — Velocity–area discharge.** A stream 9 m wide was gauged at 1 m intervals. Depths and velocities at 0.6 of the depth were measured at the center of each strip. Find the discharge, the area and the mean velocity.

```{code-cell} python
w = np.ones(9)                                       # m
d = np.array([0.20, 0.45, 0.70, 0.85, 0.90, 0.80, 0.62, 0.40, 0.18])   # m
v = np.array([0.15, 0.32, 0.48, 0.56, 0.60, 0.54, 0.44, 0.30, 0.12])   # m/s at 0.6 depth
area = (w * d).sum(); Qm = (v * d * w).sum()
print(f"A = {area:.2f} m², Q = {Qm:.2f} m³/s, mean velocity = {Qm / area:.2f} m/s")
```

**Example 6.2 — Fitting and extrapolating a rating curve.** Twelve gaugings at a station span stages from 0.45 m to 1.9 m. The zero-flow stage is surveyed at $h_0$ = 0.20 m. Fit Equation (6.1), and estimate the discharge at a flood stage of 4.0 m. The synthetic "true" relation used to create the gaugings changes slope above 2.5 m, where water spills onto the floodplain.

```{code-cell} python
rng = np.random.default_rng(4)
h0 = 0.20
def true_Q(h):                                       # channel, plus floodplain flow above 2.5 m
    return 6.5 * (h - h0) ** 1.75 + 38.0 * np.clip(h - 2.5, 0, None) ** 1.6
h_obs = np.sort(rng.uniform(0.45, 1.9, 12))
Q_obs = true_Q(h_obs) * (1 + 0.05 * rng.standard_normal(12))
b_fit, ln_a = np.polyfit(np.log(h_obs - h0), np.log(Q_obs), 1)
a_fit = np.exp(ln_a)
print(f"Fitted rating: Q = {a_fit:.2f} (h - {h0})^{b_fit:.2f}")
for h in (1.5, 4.0):
    print(f"h = {h} m: rating gives {a_fit * (h - h0) ** b_fit:6.1f} m³/s; true value {true_Q(h):6.1f} m³/s")
```

```{code-cell} python
:tags: [hide-input]
hh = np.linspace(h0 + 0.01, 4.2, 300)
fig, ax = plt.subplots(figsize=(6.0, 4.0))
ax.plot(true_Q(hh), hh, color=ink2, lw=1.4, ls=":", label="True relation (synthetic)")
ax.plot(a_fit * (hh - h0) ** b_fit, hh, color=c1, lw=2.2, label="Fitted rating curve")
ax.plot(Q_obs, h_obs, "o", color=c2, ms=6, label="Gaugings")
ax.axhspan(1.9, 4.2, color=c2, alpha=0.06); ax.text(5, 3.9, "extrapolation", color=c2, fontsize=9.5)
ax.set_xlabel("Q (m³ s⁻¹)"); ax.set_ylabel("Stage h (m)"); ax.set_xlim(0, None); ax.set_ylim(0, 4.2)
ax.legend(loc="lower right", fontsize=9.5)
plt.show()
```

*Figure 6.5. A rating curve fitted to gaugings at low and medium stage. Extrapolated to a flood stage of 4 m it misses the extra flow carried by the floodplain.*

The fitted curve reproduces the gauged range within a few percent, but at 4 m it gives less than half of the true flood discharge. Flood peaks derived from extrapolated ratings should be checked by a hydraulic calculation of the floodplain section.

**Example 6.3 — Direct runoff from a storm.** For the hydrograph of Figure 6.2, use the straight-line method with $N$ from Equation (6.2) to compute the volume and depth of direct runoff, and the event runoff ratio for the 51 mm storm.

```{code-cell} python
dro = (Q - b_line).clip(0)
V = dro.sum() * dt * 3600                       # m³
depth = V / (A * 1e6) * 1000                    # mm
print(f"N = {N_h / 24:.2f} d; direct runoff volume = {V / 1e6:.2f} million m³ = {depth:.1f} mm")
print(f"Event runoff ratio = {depth / rain.sum():.2f} (rain = {rain.sum():.0f} mm)")
```

**Example 6.4 — Recession constant.** During a dry spell the flow drops from 4.8 m³ s⁻¹ to 3.1 m³ s⁻¹ in 12 days. Assuming an exponential recession, find the daily recession constant, the storage constant $k$, and the flow after another 20 days without rain.

```{code-cell} python
Q0, Q1, days = 4.8, 3.1, 12
Kr = (Q1 / Q0) ** (1 / days)
k = -1 / np.log(Kr)
print(f"K_r = {Kr:.4f} per day, k = {k:.1f} days, Q after 20 more days = {Q1 * Kr ** 20:.2f} m³/s")
print(f"Water still stored at the start (S = kQ) = {k * 86400 * Q1 / 1e6:.1f} million m³")
```

The last line uses $S = kQ$ to estimate how much water the catchment still holds for the dry season, a useful quantity for water-supply planning.

## 6.7 Common mistakes

- **Treating baseflow separation as a measurement.** It is a convention. Use the same method that was used to derive any unit hydrograph you apply.
- **Extrapolating a rating curve without checking the hydraulics.** Above the highest gauging, flow on the floodplain or a change of control can change the curve completely.
- **Forgetting that direct runoff depth equals excess rainfall depth.** If the two differ, the catchment area, the separation or the loss calculation is wrong.
- **Using the wrong area in Equation (6.2).** The original formula used square miles; the SI form has the factor 0.827.
- **Comparing BFI values from different methods or parameters.** The filter and its parameters must be reported with the index.
- **Reading a single event's runoff ratio as a catchment property.** It changes with antecedent wetness and storm size.

## 6.8 Exercises

```{code-cell} python
:tags: [remove-cell]
d1 = np.array([0.3, 0.8, 1.1, 0.9, 0.4]); v1 = np.array([0.2, 0.5, 0.7, 0.6, 0.25])
glue("x1_Q", float((d1 * v1 * 1.5).sum()), display=False)
glue("x2_Q", 9.2 * (2.4 - 0.35) ** 1.62, display=False); glue("x2_h", 0.35 + (40 / 9.2) ** (1 / 1.62), display=False)
glue("x3_N", 0.827 * 420 ** 0.2, display=False)
glue("x4_d", 1.85e6 / (64e6) * 1000, display=False); glue("x4_C", 1.85e6 / 64e6 * 1000 / 72, display=False)
Kr5 = (2.0 / 6.0) ** (1 / 25); glue("x5_K", float(Kr5), display=False); glue("x5_k", float(-1 / np.log(Kr5)), display=False)
glue("x5_t", float(np.log(1.0 / 2.0) / np.log(Kr5)), display=False)
```

1. A stream is divided into five strips 1.5 m wide. The depths are 0.3, 0.8, 1.1, 0.9 and 0.4 m and the mean velocities 0.20, 0.50, 0.70, 0.60 and 0.25 m s⁻¹. Find the discharge.
2. A rating curve is $Q = 9.2\,(h - 0.35)^{1.62}$. Find the discharge at $h$ = 2.4 m and the stage at which $Q$ = 40 m³ s⁻¹.
3. Find $N$ (days) for the straight-line separation method for a 420 km² catchment.
4. The direct runoff from a 64 km² catchment after a 72 mm storm has a volume of 1.85 million m³. Find the depth of direct runoff and the event runoff ratio.
5. The flow during a dry spell drops from 6.0 to 2.0 m³ s⁻¹ in 25 days. Find the daily recession constant and $k$. How many more days until the flow reaches 1.0 m³ s⁻¹?

```{admonition} Answers
:class: dropdown
1. {glue:text}`x1_Q:.2f` m³ s⁻¹.
2. {glue:text}`x2_Q:.1f` m³ s⁻¹; $h$ = {glue:text}`x2_h:.2f` m.
3. $N$ ≈ {glue:text}`x3_N:.1f` days.
4. {glue:text}`x4_d:.1f` mm; runoff ratio {glue:text}`x4_C:.2f`.
5. $K_r$ = {glue:text}`x5_K:.4f`, $k$ = {glue:text}`x5_k:.1f` days; about {glue:text}`x5_t:.0f` more days.
```

## 6.9 Further reading

- Dingman, S. L. (2015). *Physical Hydrology* (3rd ed.). Waveland Press. The chapters on streamflow generation and on streamflow measurement.
- Chow, V. T., Maidment, D. R., and Mays, L. W. (1988). *Applied Hydrology*. McGraw-Hill. The chapters on surface water and streamflow measurement.
- 윤용남. *수문학* (in Korean). The chapters on streamflow and hydrograph analysis.
- Eckhardt, K. (2005). How to construct recursive digital filters for baseflow separation. *Hydrological Processes*, 19, 507–515.
- Lyne, V., and Hollick, M. (1979). Stochastic time-variable rainfall-runoff modelling. *Institution of Engineers Australia National Conference*, Perth, 89–93.
- Nathan, R. J., and McMahon, T. A. (1990). Evaluation of automated techniques for base flow and recession analyses. *Water Resources Research*, 26, 1465–1473.

---

*Figures in this chapter are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
