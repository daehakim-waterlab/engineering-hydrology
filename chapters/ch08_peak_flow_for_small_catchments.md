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

# 8. Peak flow for small catchments

```{admonition} Core ideas
:class: tip
1. For a small catchment under steady rain, the outflow rises until the whole catchment contributes, after the **time of concentration** $t_c$. The peak then equals the rate of excess rainfall over the whole area. This is the **rational method**: $Q_p = 0.278\, C\, i\, A$ (m³ s⁻¹, with $i$ in mm h⁻¹ and $A$ in km²).
2. Because rainfall intensity falls with duration, the design storm duration is set equal to $t_c$, and the return period of the peak is taken to be that of the rainfall.
3. $t_c$ can be estimated from **empirical formulas** (Kirpich, Kerby, NRCS lag) or from **flow velocities** along the longest path. Different formulas often disagree by a factor of two, which is the largest source of uncertainty in the method.
4. Formulas first published in US customary units must be converted carefully. In SI, Kirpich becomes $t_c = 0.0195\,L^{0.77}S_0^{-0.385}$ (min, $L$ in m).
5. A **storm sewer** is designed pipe by pipe, from upstream to downstream. Each pipe carries the peak from all the area upstream, with $t_c$ grown by the travel time in the pipes, and is sized with the **Manning equation** for full flow.
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

## 8.1 The rational method

### The idea

Imagine rain falling at a constant intensity $i$ on a small catchment of area $A$, with a constant fraction $C$ of it becoming runoff (Chapter 7). Water from near the outlet arrives almost at once; water from the most remote point takes longest. The time it takes for water to travel from the hydraulically most distant point to the outlet is the **time of concentration** $t_c$. As the rain continues, more and more of the catchment contributes to the outflow. After $t_c$, the whole catchment contributes and the outflow becomes steady, equal to the rate at which excess rain falls on the catchment:

$$
Q_p = C\, i\, A .
$$

With $i$ in mm h⁻¹ and $A$ in km², the unit conversion of Chapter 1 (Equation (1.2) there) gives the working form

\begin{equation*}
Q_p = \frac{C\, i\, A}{3.6} = 0.278\, C\, i\, A \qquad (Q_p \text{ in m}^3\,\text{s}^{-1}).
\tag{8.1}
\end{equation*}

This is the **rational method**. Its basic idea, that the peak is reached when the whole catchment contributes, goes back to Thomas Mulvaney in Ireland in 1851, and the formula became standard practice for city drainage through the work of Emil Kuichling in the United States (1889) and David Lloyd-Davies in Britain (1906) (see [*A Brief History*](https://daehakim-waterlab.github.io/water-history/history-of-hydrology/#floods-from-formulas-to-the-unit-hydrograph)). It remains the most widely used method for the design of storm sewers, culverts and small drainage structures.

### Why the design duration equals $t_c$

Figure 8.1 shows the outflow from a catchment whose contributing area grows linearly with travel time, for storms of different durations $D$ taken from the same **intensity–duration–frequency (IDF)** curve (Chapter 11). Rainfall intensity decreases with duration, so:

- a storm **shorter** than $t_c$ is more intense, but it stops before the whole catchment contributes, so only part of the area adds to the peak;
- a storm **longer** than $t_c$ makes the whole catchment contribute, but at a lower intensity;
- the peak is largest when $D = t_c$: the whole area contributes at the highest intensity that lasts long enough to allow it.

```{code-cell} python
:tags: [hide-input]
def idf(D_min, T=10.0):
    """Illustrative IDF curve, intensity in mm/h for duration D (min) and return period T (years).
    Used in Chapters 8, 11 and 12; it does not describe any real station."""
    return 1450.0 * (1 + 0.20 * np.log(T / 10.0)) / (D_min + 12.0) ** 0.72

A, C, tc = 0.85, 0.55, 25.0                 # km², -, min
dt = 0.25                                    # min
tau = np.arange(0, tc, dt)
ta = np.full(len(tau), A / len(tau))         # time–area histogram: equal area per travel-time band
fig, ax = plt.subplots(figsize=(6.4, 4.0))
peaks = {}
for D, col in [(10.0, c2), (tc, c1), (60.0, c3)]:
    t = np.arange(0, 110, dt)
    rain = np.where(t < D, C * idf(D), 0.0)                       # excess intensity, mm/h
    Q = np.convolve(rain, ta)[: len(t)] / 3.6                     # m³/s
    peaks[D] = Q.max()
    lab = f"D = {D:.0f} min, i = {idf(D):.0f} mm h⁻¹" + (" (= $t_c$)" if D == tc else "")
    ax.plot(t, Q, color=col, lw=2.2, label=lab)
ax.axhline(peaks[tc], color=c1, lw=0.8, ls=":")
ax.set_xlabel("Time from start of rain (min)"); ax.set_ylabel("Q (m³ s⁻¹)")
ax.set_xlim(0, 110); ax.set_ylim(0, peaks[tc] * 1.25)
ax.legend(loc="upper right", fontsize=9.5)
plt.show()
for D in peaks: glue(f"pk_{int(D)}", float(peaks[D]), display=False)
```

*Figure 8.1. Outflow hydrographs from a 0.85 km² catchment ($C$ = 0.55, $t_c$ = 25 min, contributing area growing linearly with travel time) for storms of three durations from the same IDF curve. The peaks are {glue:text}`pk_10:.1f`, {glue:text}`pk_25:.1f` and {glue:text}`pk_60:.1f` m³ s⁻¹ for 10, 25 and 60 minutes. The storm lasting exactly $t_c$ gives the largest peak, equal to $0.278\,C\,i(t_c)\,A$.*

For catchments whose contributing area does not grow linearly (for example a catchment with a large, slow area far from the outlet), a storm shorter than $t_c$ falling on a fast, compact part near the outlet can produce a larger peak than the whole catchment does. This **partial area effect** is checked in careful designs by computing the peak from the fast sub-area alone.

### Assumptions and limits

The rational method assumes that:

1. rainfall is uniform in space and constant in time for at least $t_c$;
2. the runoff coefficient is constant during the storm and the same for all return periods;
3. the return period of the peak equals the return period of the rainfall intensity;
4. storage in the catchment and the drainage system is small.

These assumptions are reasonable for small, mostly impervious catchments, which is why many guidelines restrict the method to areas of a few square kilometers or less (limits vary between countries and agencies). For larger or more rural catchments the hydrograph methods of Chapter 9 are needed. Some guidelines increase $C$ for rare storms, since losses become relatively smaller as rainfall increases, with the product $C$ not allowed to exceed 1.

## 8.2 Time of concentration

The time of concentration is the only time parameter in the rational method, and through the IDF curve it controls the design intensity. It is also used to estimate the lag of synthetic unit hydrographs (Chapter 9). There are two families of methods: empirical formulas based on catchment geometry, and the velocity method.

### Empirical formulas

Most empirical formulas have the form $t_c \propto L^a S_0^{-b}$, where $L$ is a flow length and $S_0$ a slope (m m⁻¹): longer and flatter catchments respond more slowly. Many were derived from small samples of catchments and in US customary units. Each formula below is given in SI form, and the code checks it against the original.

**Kirpich (1940)**, from small agricultural catchments in Tennessee with well-defined channels:

\begin{equation*}
t_c = 0.0195\, L^{0.77}\, S_0^{-0.385} \qquad (t_c \text{ in min},\ L \text{ in m}),
\tag{8.2}
\end{equation*}

where $L$ is the length of the main channel from the outlet to the divide and $S_0$ the average slope along it (the elevation difference divided by $L$). The original form is $t_c = 0.0078\,L^{0.77}S_0^{-0.385}$ with $L$ in feet. Kirpich is often multiplied by 2 for overland flow on grass and by 0.4 for concrete or asphalt surfaces.

**Kerby (1959)**, for overland flow over short distances (less than about 300 m):

\begin{equation*}
t_c = 1.44\left(\frac{L\, N}{S_0^{0.5}}\right)^{0.467} \qquad (t_c \text{ in min},\ L \text{ in m}),
\tag{8.3}
\end{equation*}

where $N$ is a retardance coefficient (about 0.02 for smooth pavement, 0.1–0.3 for bare or sparsely grassed soil, 0.4 for average grass and 0.8 for dense grass or forest litter). The original constant is 0.83 with $L$ in feet.

**NRCS lag equation**, developed for agricultural catchments and widely applied to rural areas:

\begin{equation*}
t_L = \frac{(3.28\,L)^{0.8}\,\left(\dfrac{1000}{\text{CN}} - 9\right)^{0.7}}{1900\, Y^{0.5}} \quad (t_L \text{ in h},\ L \text{ in m}),
\qquad t_c = \frac{t_L}{0.6},
\tag{8.4}
\end{equation*}

where $L$ is the hydraulic length (m), CN the curve number (Chapter 7) and $Y$ the average land slope of the catchment in percent. The term $1000/\text{CN} - 9$ is $S_{\max} + 1$ with $S_{\max}$ in inches, which is kept because it is dimensionless here; only $L$ needs converting (1 m = 3.28 ft).

```{code-cell} python
ft = 0.3048                                              # m per foot
L_m, S0, N = 1500.0, 0.015, 0.4
kirpich_si = 0.0195 * L_m ** 0.77 * S0 ** -0.385
kirpich_us = 0.0078 * (L_m / ft) ** 0.77 * S0 ** -0.385
print(f"Kirpich: SI {kirpich_si:.2f} min, original {kirpich_us:.2f} min "
      f"(constant 0.0078 × 3.2808^0.77 = {0.0078 * (1 / ft) ** 0.77:.4f})")
L_ov = 90.0
kerby_si = 1.44 * (L_ov * N / S0 ** 0.5) ** 0.467
kerby_us = 0.83 * (L_ov / ft * N / S0 ** 0.5) ** 0.467
print(f"Kerby:   SI {kerby_si:.2f} min, original {kerby_us:.2f} min "
      f"(constant 0.83 × 3.2808^0.467 = {0.83 * (1 / ft) ** 0.467:.4f})")
CN, Y = 75.0, 6.0
lag_si = (3.28 * L_m) ** 0.8 * (1000 / CN - 9) ** 0.7 / (1900 * Y ** 0.5)
lag_us = (L_m / ft) ** 0.8 * ((1000 / CN - 10) + 1) ** 0.7 / (1900 * Y ** 0.5)
print(f"NRCS lag: SI {lag_si:.3f} h, original {lag_us:.3f} h")
```

The small differences in the last digit come from rounding the conversion constants.

### The velocity method

A more physical approach divides the longest flow path into segments with different flow types and adds the travel times:

$$
t_c = \sum_j \frac{L_j}{V_j} .
$$

- **Sheet flow** over the first tens of meters, as a thin film. The NRCS kinematic-wave approximation, converted to SI, is

\begin{equation*}
T_{\text{sheet}} = \frac{0.0913\,(n\,L)^{0.8}}{P_2^{0.5}\,S_0^{0.4}} \qquad (T \text{ in h},\ L \text{ in m},\ P_2 \text{ in mm}),
\tag{8.5}
\end{equation*}

where $n$ is a Manning roughness for sheet flow (0.011 for smooth pavement, 0.15 for short grass, 0.24 for dense grass, 0.4–0.8 for woods), $P_2$ is the 2-year, 24-hour rainfall and $L$ is limited to about 30 m in current NRCS guidance.

- **Shallow concentrated flow** in rills and gutters, with velocity $V = 4.918\,S_0^{0.5}$ m s⁻¹ on unpaved and $V = 6.196\,S_0^{0.5}$ m s⁻¹ on paved surfaces (converted from 16.13 and 20.33 ft s⁻¹).

- **Channel or pipe flow**, with the velocity from the [Manning equation](https://daehakim-waterlab.github.io/hydraulics/chapters/ch06_flow_in_open_channels.html#uniform-flow-the-chezy-and-manning-equations) at bankfull or full-pipe conditions, $V = R^{2/3}S_0^{1/2}/n$.

```{code-cell} python
c_sheet = 0.007 * (1 / ft) ** 0.8 * (25.4) ** 0.5            # converts 0.007 (ft, in) to SI (m, mm)
print(f"Sheet-flow constant: {c_sheet:.4f}")
n_s, L_s, P2, S_s = 0.24, 30.0, 110.0, 0.02
T_si = 0.0913 * (n_s * L_s) ** 0.8 / (P2 ** 0.5 * S_s ** 0.4)
T_us = 0.007 * (n_s * L_s / ft) ** 0.8 / ((P2 / 25.4) ** 0.5 * S_s ** 0.4)
print(f"Sheet flow over 30 m of dense grass: SI {T_si * 60:.2f} min, original {T_us * 60:.2f} min")
print(f"Shallow-flow velocity constants: {16.1345 * ft:.3f} and {20.3282 * ft:.3f} m/s")
```

### How much do the methods disagree?

Example 8.2 applies all the methods to one catchment. The spread is typical: estimates of $t_c$ commonly differ by a factor of two or more. Because the rational peak is proportional to $i(t_c)$, and intensity falls roughly as $t_c^{-0.6}$ to $t_c^{-0.75}$ at these durations, a factor of two in $t_c$ changes the peak by roughly a third to a half. The choice should follow the type of catchment for which a formula was developed, and, where possible, be checked against observed lag times.

## 8.3 Designing a storm sewer

A storm sewer system collects runoff from many small **subcatchments**, each draining to an inlet, and carries it in pipes to an outlet. The rational method is applied at the upstream end of each pipe, moving downstream:

1. **Time of concentration at a pipe.** It is the larger of (a) the inlet time of the local subcatchment (the time for surface flow to reach the inlet, typically 5–15 minutes in urban areas) and (b) the time of concentration of each upstream pipe plus the travel time through that pipe.
2. **Intensity.** Read $i$ from the IDF curve for the design return period at that time of concentration.
3. **Peak discharge.** $Q = 0.278\, i \sum C_j A_j$, where the sum covers all subcatchments draining to the pipe. Note that the intensity decreases downstream (longer $t_c$) while $\sum CA$ increases.
4. **Pipe size.** For a circular pipe flowing full, the Manning equation gives
\begin{equation*}
Q = \frac{1}{n}\,\frac{\pi D^2}{4}\left(\frac{D}{4}\right)^{2/3} S_0^{1/2}
\quad\Longrightarrow\quad
D = \left(\frac{3.208\, n\, Q}{S_0^{1/2}}\right)^{3/8},
\tag{8.6}
\end{equation*}
where $S_0$ is the pipe slope. The next larger commercial diameter is chosen.
5. **Velocity check.** The full-flow velocity should exceed a self-cleansing minimum (typically about 0.6–0.8 m s⁻¹, to keep sediment moving) and stay below a maximum (about 3 m s⁻¹, to limit abrasion).
6. **Travel time.** The travel time in the pipe, $L/V$, is added to the time of concentration for the next pipe downstream.

The design is usually laid out as a table, one row per pipe (Example 8.3). Modern practice checks the result with a hydrodynamic network model, especially for surcharge and for the overland flow paths that carry water when the pipes are full, but the rational-method table remains the standard first design.

## 8.4 Worked examples

**Example 8.1 — Peak discharge for a culvert.** A 0.85 km² suburban catchment has a composite runoff coefficient of 0.55 and a time of concentration of 25 minutes. Using the IDF curve of Figure 8.1, find the 10-year and 50-year peak discharges.

```{code-cell} python
A, C, tc = 0.85, 0.55, 25.0
for T in (10, 50):
    i = idf(tc, T)
    print(f"T = {T:3d} yr: i = {i:.1f} mm/h, Qp = 0.278 × {C} × {i:.1f} × {A} = {0.278 * C * i * A:.2f} m³/s")
```

**Example 8.2 — Comparing estimates of the time of concentration.** A 1.6 km² rural catchment has a main channel 1500 m long with an average slope of 0.015. Its longest flow path begins with 30 m of sheet flow over dense grass at a slope of 0.02, followed by 220 m of unpaved shallow concentrated flow at a slope of 0.03, then 1250 m in a grassed channel ($n$ = 0.035) with a bankfull hydraulic radius of 0.45 m and a slope of 0.012. The curve number is 75, the average land slope is 6%, and the 2-year 24-hour rainfall is 110 mm. Estimate $t_c$ with each method and the effect on the 10-year rational peak ($C$ = 0.35).

```{code-cell} python
L_ch, S_ch = 1500.0, 0.015
tc_kirpich = 0.0195 * L_ch ** 0.77 * S_ch ** -0.385                       # min
lag = (3.28 * L_ch) ** 0.8 * (1000 / 75 - 9) ** 0.7 / (1900 * 6.0 ** 0.5)
tc_nrcs = lag / 0.6 * 60                                                  # min
T1 = 0.0913 * (0.24 * 30) ** 0.8 / (110 ** 0.5 * 0.02 ** 0.4) * 60       # sheet flow, min
T2 = 220 / (4.918 * 0.03 ** 0.5) / 60                                     # shallow flow, min
V3 = 0.45 ** (2 / 3) * 0.012 ** 0.5 / 0.035
T3 = 1250 / V3 / 60                                                       # channel, min
tc_vel = T1 + T2 + T3
results = {"Kirpich": tc_kirpich, "NRCS lag": tc_nrcs, "Velocity method": tc_vel}
print(f"Velocity method: sheet {T1:.1f} + shallow {T2:.1f} + channel {T3:.1f} min (V = {V3:.2f} m/s)")
for name, t_ in results.items():
    i = idf(t_, 10)
    print(f"{name:16s} tc = {t_:5.1f} min, i10 = {i:5.1f} mm/h, Qp = {0.278 * 0.35 * i * 1.6:5.2f} m³/s")
```

```{code-cell} python
:tags: [hide-input]
fig, ax = plt.subplots(figsize=(6.0, 2.8))
names = list(results); vals = [results[k] for k in names]
ax.barh(names, vals, color=[c1, c2, c3], height=0.55)
for y, v in enumerate(vals):
    ax.text(v + 1, y, f"{v:.0f} min → {0.278 * 0.35 * idf(v, 10) * 1.6:.1f} m³ s⁻¹", va="center", fontsize=9.5)
ax.set_xlabel("Time of concentration (min)"); ax.set_xlim(0, max(vals) * 1.6)
ax.invert_yaxis(); ax.grid(axis="y", visible=False)
plt.show()
glue("tc_ratio", float(max(vals) / min(vals)), display=False)
qs = [0.278 * 0.35 * idf(v, 10) * 1.6 for v in vals]
glue("q_ratio", float(max(qs) / min(qs)), display=False)
```

*Figure 8.2. Three estimates of the time of concentration for the catchment of Example 8.2 and the resulting 10-year rational peak discharges. The longest estimate is {glue:text}`tc_ratio:.1f` times the shortest, and the peaks differ by a factor of {glue:text}`q_ratio:.2f`.*

The Kirpich formula, developed for steep agricultural catchments with defined channels, gives the shortest time; the NRCS lag equation the longest. The velocity method makes the assumptions visible, segment by segment, which makes it the easiest to defend and check.

**Example 8.3 — A storm sewer.** Three subcatchments drain to inlets 1, 2 and 3 along a street. Pipe 1 runs from inlet 1 to inlet 2, pipe 2 from inlet 2 to inlet 3, and pipe 3 from inlet 3 to the outfall. Design the pipes for a 10-year storm with $n$ = 0.013, using commercial diameters in steps of 75 mm up to 900 mm and 150 mm above.

```{code-cell} python
sizes = np.r_[np.arange(300, 901, 75), np.arange(1050, 1801, 150)] / 1000   # m
pipes = [  # local area (km²), C, inlet time (min), pipe length (m), pipe slope
    dict(A=0.032, C=0.75, t_in=8.0, L=120.0, S=0.010),
    dict(A=0.045, C=0.65, t_in=10.0, L=150.0, S=0.008),
    dict(A=0.060, C=0.60, t_in=12.0, L=180.0, S=0.006),
]
n = 0.013
sumCA, t_up = 0.0, None
print(" pipe  ΣCA(km²)  tc(min)  i(mm/h)  Q(m³/s)  D_req(mm)  D(mm)  V_full(m/s)  travel(min)")
for k, p in enumerate(pipes, start=1):
    sumCA += p["C"] * p["A"]
    tc = p["t_in"] if t_up is None else max(p["t_in"], t_up)
    i = idf(tc, 10)
    Q = 0.278 * i * sumCA
    D_req = (3.208 * n * Q / p["S"] ** 0.5) ** 0.375
    D = sizes[np.searchsorted(sizes, D_req)]
    V = (D / 4) ** (2 / 3) * p["S"] ** 0.5 / n
    t_travel = p["L"] / V / 60
    assert 0.6 <= V <= 3.0, "velocity outside the usual limits"
    print(f"{k:4d} {sumCA:9.4f} {tc:8.1f} {i:8.1f} {Q:8.3f} {D_req * 1000:10.0f} {D * 1000:6.0f} {V:11.2f} {t_travel:11.2f}")
    t_up = tc + t_travel
```

The intensity decreases from pipe to pipe as the time of concentration grows, but the contributing $\sum CA$ grows faster, so the discharge and the pipe size increase downstream. All velocities lie within the usual limits.

## 8.5 Common mistakes

- **Mixing units in $Q = CiA$.** With $i$ in mm h⁻¹ and $A$ in km², the factor is 0.278 (or 1/3.6). With $A$ in hectares it is 1/360.
- **Using a storm duration different from $t_c$.** The design intensity must be read from the IDF curve at the time of concentration.
- **Using a US-unit formula with SI inputs.** Kirpich with $L$ in meters but the constant 0.0078 underestimates $t_c$ by a factor of 2.5.
- **Adding pipe travel time incorrectly.** The time of concentration at a pipe is the longest of the upstream paths, not the sum of all inlet times.
- **Choosing a pipe smaller than computed.** Always round up to the next commercial size.
- **Applying the rational method to large rural catchments.** Storage, non-uniform rain and variable losses make it unreliable; use hydrograph methods.

## 8.6 Exercises

```{code-cell} python
:tags: [remove-cell]
glue("z1_Q", 0.278 * 0.7 * 95.0 * 0.4, display=False)
glue("z2_tc", 0.0195 * 2400 ** 0.77 * 0.008 ** -0.385, display=False)
tc3 = 0.0195 * 2400 ** 0.77 * 0.008 ** -0.385
glue("z3_i", float(idf(tc3, 25)), display=False); glue("z3_Q", float(0.278 * 0.45 * idf(tc3, 25) * 2.1), display=False)
glue("z4_wrong", 0.0078 * 2400 ** 0.77 * 0.008 ** -0.385, display=False)
D5 = (3.208 * 0.013 * 0.85 / 0.005 ** 0.5) ** 0.375
glue("z5_D", D5 * 1000, display=False)
D5c = float(sizes[np.searchsorted(sizes, D5)])
glue("z5_Dc", D5c * 1000, display=False)
glue("z5_V", (D5c / 4) ** (2 / 3) * 0.005 ** 0.5 / 0.013, display=False)
glue("z5_Qf", 1 / 0.013 * np.pi * D5c ** 2 / 4 * (D5c / 4) ** (2 / 3) * 0.005 ** 0.5, display=False)
```

1. A 0.4 km² commercial area with $C$ = 0.70 receives a design intensity of 95 mm h⁻¹. Find the peak discharge.
2. A catchment has a main channel 2400 m long with a slope of 0.008. Estimate $t_c$ with the Kirpich formula.
3. For the catchment of Exercise 2 ($A$ = 2.1 km², $C$ = 0.45), find the 25-year rational peak using the IDF curve of this chapter.
4. A student uses the original Kirpich constant 0.0078 with $L$ in meters for the catchment of Exercise 2. What $t_c$ does the student get, and would the resulting peak be too high or too low?
5. A concrete pipe ($n$ = 0.013) on a slope of 0.005 must carry 0.85 m³ s⁻¹. Find the required diameter, choose a commercial size from the list in Example 8.3, and find the full-flow velocity and capacity of the chosen pipe.

```{admonition} Answers
:class: dropdown
1. {glue:text}`z1_Q:.2f` m³ s⁻¹.
2. {glue:text}`z2_tc:.1f` min.
3. $i$ = {glue:text}`z3_i:.1f` mm h⁻¹, $Q_p$ = {glue:text}`z3_Q:.2f` m³ s⁻¹.
4. {glue:text}`z4_wrong:.1f` min. Too short a $t_c$ gives too high an intensity, so the peak would be overestimated.
5. $D_{\text{req}}$ = {glue:text}`z5_D:.0f` mm, so choose {glue:text}`z5_Dc:.0f` mm; $V$ = {glue:text}`z5_V:.2f` m s⁻¹, capacity {glue:text}`z5_Qf:.2f` m³ s⁻¹.
```

## 8.7 Further reading

- Dingman, S. L. (2015). *Physical Hydrology* (3rd ed.). Waveland Press. The section on the rational method and times of concentration.
- Chow, V. T., Maidment, D. R., and Mays, L. W. (1988). *Applied Hydrology*. McGraw-Hill. The chapter on hydrologic design, including storm sewer design.
- 윤용남. *수문학* (in Korean). The chapter on design floods for small catchments.
- Kirpich, Z. P. (1940). Time of concentration of small agricultural watersheds. *Civil Engineering*, 10(6), 362.
- Kerby, W. S. (1959). Time of concentration for overland flow. *Civil Engineering*, 29(3), 60.
- USDA Natural Resources Conservation Service. *National Engineering Handbook*, Part 630 Hydrology, Chapter 15 (Time of concentration).

---

*Figures in this chapter are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
