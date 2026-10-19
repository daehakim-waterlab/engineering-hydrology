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

# 11. Design rainfall

```{admonition} Core ideas
:class: tip
1. The **return period** $T$ of an event is the average interval between years in which it is equaled or exceeded. Its annual exceedance probability is $1/T$. It is an average, not a schedule.
2. Over a design life of $n$ years, the probability that the design event is exceeded at least once, the **risk**, is $R = 1 - (1 - 1/T)^n$. A 100-year flood has a 39% chance of occurring during a 50-year design life.
3. **Intensity–duration–frequency (IDF)** curves summarize point rainfall for many durations and return periods. They are the input to the rational method and to design storms.
4. Rain over a catchment averages less than the rain at a point. **Areal reduction factors** and **depth–area–duration** analysis convert point values to areal values.
5. A design storm also needs a **time distribution**. The **alternating block** method builds it from the IDF curve; **Huff** curves describe how observed storms accumulate.
6. For dams whose failure would be catastrophic, design uses the **probable maximum precipitation** (PMP) and the resulting **probable maximum flood** (PMF), not a return period.
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

## 11.1 Return period and exceedance probability

Design starts with a choice: how large an event should a structure withstand? Larger structures cost more but fail less often. The choice is expressed through the probability of the design event.

Consider the largest rainfall (or flood) in each year, the **annual maximum series**. Let $X$ be the annual maximum and $x_T$ a particular value. Its **annual exceedance probability** is

$$
p = P(X \ge x_T),
$$

and its **return period** (recurrence interval) is

$$
T = \frac{1}{p} .
$$ (eq-return-period)

The **non-exceedance probability** is $F = 1 - p = 1 - 1/T$. A "100-year rainfall" is a depth that has a 1% chance of being equaled or exceeded in any year.

The return period is an **average**: over a very long period, the years in which $x_T$ is exceeded are on average $T$ years apart. It does not mean the event happens once every $T$ years, and two 100-year floods can occur in successive years. Estimating $x_T$ for a given $T$ requires fitting a probability distribution to the annual maxima (Gumbel, generalized extreme value, log-Pearson type III, and others). **Frequency analysis is covered in a separate textbook on statistical hydrology**; this chapter starts from its results.

## 11.2 Risk over the design life

A structure is in service for many years, during which the design event may be exceeded more than once, or not at all. If annual maxima are independent and the climate does not change, the probability that $x_T$ is **not** exceeded in any of $n$ years is $(1 - 1/T)^n$. The probability of at least one exceedance, the **risk** (or hydrological risk), is therefore

$$
R = 1 - \left(1 - \frac{1}{T}\right)^{n}.
$$ (eq-risk)

Two consequences are worth remembering:

- if $n = T$, then $R \approx 1 - e^{-1} = 0.63$: an event with a $T$-year return period has a **63%** chance of occurring within $T$ years;
- to keep the risk to $R$ over $n$ years, the return period must be

$$
T = \frac{1}{1 - (1 - R)^{1/n}} ,
$$ (eq-T-from-risk)

  much longer than the design life. A 10% risk over 50 years needs $T \approx 475$ years.

```{code-cell} python
:tags: [hide-input]
n = np.arange(1, 201)
fig, ax = plt.subplots(figsize=(6.4, 4.0))
for T, col in [(10, c2), (50, c3), (100, c1), (500, ink2)]:
    ax.plot(n, 1 - (1 - 1 / T) ** n, color=col, lw=2.2, label=f"T = {T} years")
ax.axhline(1 - np.exp(-1), color=ink2, lw=0.8, ls=":"); ax.text(160, 0.66, "63%", color=ink2, fontsize=9.5)
ax.set_xlabel("Design life n (years)"); ax.set_ylabel("Risk R (probability of ≥ 1 exceedance)")
ax.set_xlim(0, 200); ax.set_ylim(0, 1.02)
ax.legend(loc="lower right", fontsize=9.5)
plt.show()
```

*Figure 11.1. Probability that the $T$-year event is equaled or exceeded at least once during a design life of $n$ years.*

The code below checks Equation {eq}`eq-risk` by simulation: 100,000 imaginary 50-year design lives, each year with a 1% chance of a 100-year event.

```{code-cell} python
rng = np.random.default_rng(2024)
T, n_life, trials = 100, 50, 100_000
hits = (rng.random((trials, n_life)) < 1 / T).any(axis=1)
print(f"Simulated risk = {hits.mean():.4f}; formula 1 - (1 - 1/T)^n = {1 - (1 - 1 / T) ** n_life:.4f}")
```

Because the risk over the life of a structure is so much larger than the annual probability, design standards specify return periods that grow with the consequences of failure: a few years for minor drains, 10–30 years for storm sewers, 50–200 years for levees and bridges on major rivers, and much longer, or the PMF of Section 11.6, for large dams. The values to use are set by national design guidelines.

## 11.3 Intensity–duration–frequency curves

For each rainfall duration $D$ (10 minutes, 1 hour, 24 hours, ...), the annual maxima of rainfall depth over that duration form a separate series. Frequency analysis of each series gives the depth $P(D, T)$ for each return period. Dividing by duration gives the average **intensity** $i = P/D$. Plotting intensity against duration for each return period gives the **intensity–duration–frequency (IDF)** curves (Figure 11.2). Intensity falls steeply with duration (Chapter 3) and rises with return period.

For computation, IDF curves are fitted with simple formulas. Common forms, with $i$ in mm h⁻¹ and $D$ in minutes, are

$$
i = \frac{a}{D + b} \ \ \text{(Talbot)}, \qquad
i = \frac{a}{D^{\,c}} \ \ \text{(Sherman)}, \qquad
i = \frac{a}{(D + b)^{c}} ,
$$ (eq-idf-forms)

where $a$, $b$ and $c$ are fitted separately for each return period, or $a$ is made a function of $T$. National agencies publish IDF curves or formulas for their stations, sometimes as maps for ungauged areas. This book uses the illustrative curve of Chapter 8,

$$
i(D, T) = \frac{1450\,\bigl[1 + 0.20 \ln (T/10)\bigr]}{(D + 12)^{0.72}} ,
$$

which has a realistic shape but does not describe any real station.

```{code-cell} python
def idf(D_min, T=10.0):
    """Illustrative IDF curve (Chapter 8): intensity in mm/h for duration D (min) and return period T (yr)."""
    return 1450.0 * (1 + 0.20 * np.log(T / 10.0)) / (D_min + 12.0) ** 0.72
```

```{code-cell} python
:tags: [hide-input]
D = np.logspace(np.log10(10), np.log10(1440), 200)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
cols = [ink2, c3, c1, c2, ink]
for T, col in zip([2, 10, 30, 50, 100], cols):
    ax.loglog(D, idf(D, T), color=col, lw=2, label=f"T = {T} yr")
ax.set_xticks([10, 30, 60, 120, 360, 720, 1440], ["10 min", "30 min", "1 h", "2 h", "6 h", "12 h", "24 h"])
ax.minorticks_off()
ax.set_yticks([2, 5, 10, 20, 50, 100, 200], ["2", "5", "10", "20", "50", "100", "200"])
ax.set_xlabel("Duration"); ax.set_ylabel("Intensity (mm h⁻¹)")
ax.legend(loc="upper right", fontsize=9.5)
plt.show()
```

*Figure 11.2. Illustrative intensity–duration–frequency curves (log scales). Intensity falls by a factor of about 20 between 10 minutes and 24 hours.*

## 11.4 From point to area: areal reduction and depth–area–duration

IDF curves describe rainfall at a **point**. A storm is never equally intense over a whole catchment, so the average rainfall over an area, for the same return period and duration, is smaller than the point value. The ratio is the **areal reduction factor** (ARF):

$$
P_{\text{areal}}(D, T) = \text{ARF}(A, D)\times P_{\text{point}}(D, T).
$$

ARF decreases as the area grows and increases with duration: short, intense bursts are small convective cells, while long-duration rain comes from large weather systems. A convenient fit to the classical US Weather Bureau curves (Leclerc and Schaake, 1972), converted to SI, is

$$
\text{ARF} = 1 - e^{-1.1\,D_h^{1/4}} + e^{-1.1\,D_h^{1/4} - 0.00386\,A},
$$ (eq-arf)

with $D_h$ the duration in hours and $A$ the area in km² (the original uses $0.01\,A$ with $A$ in mi²). Many countries have derived their own ARF curves from dense gauge networks, and those should be used where available.

**Depth–area–duration (DAD) analysis** looks at individual large storms instead. For a given storm and duration, the maximum rainfall accumulated over that duration is mapped. The area enclosed by each isohyet, measured from the storm center outward, is computed, together with the average depth over that area. Repeating for several durations gives a family of curves of maximum average depth against area. DAD curves of the largest observed storms are the starting point for estimating probable maximum precipitation (Section 11.6). Figure 11.3 performs a DAD analysis on a synthetic storm.

```{code-cell} python
:tags: [hide-input]
# A synthetic 24-hour storm: a slowly moving rain band with an embedded convective cell (rates in mm/h)
nx, ny, dx = 160, 120, 1.0                         # 1 km grid
x = (np.arange(nx) + 0.5) * dx; y = (np.arange(ny) + 0.5) * dx
X, Y = np.meshgrid(x, y)
hours = np.arange(24)
rate = np.zeros((24, ny, nx))
for h in hours:
    xc_band, xc_cell = 30 + 3.5 * h, 45 + 3.0 * h
    band = 9.0 * np.exp(-((X - xc_band) ** 2) / (2 * 22 ** 2)) * np.exp(-((Y - 60) ** 2) / (2 * 28 ** 2))
    cell = 45.0 * np.exp(-((X - xc_cell) ** 2 + (Y - 62) ** 2) / (2 * 6 ** 2)) * np.exp(-((h - 11) / 3.0) ** 2)
    rate[h] = band + cell

def dad_curve(depth_map, areas):
    """Average depth over the wettest area of each size (cells sorted from the maximum downward)."""
    srt = np.sort(depth_map.ravel())[::-1]
    csum = np.cumsum(srt)
    n_cells = (np.asarray(areas) / dx ** 2).astype(int)
    return csum[n_cells - 1] / n_cells

areas = np.logspace(0, np.log10(8000), 60)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
dad = {}
for Dh, col in [(1, ink2), (3, c3), (6, c1), (12, c2), (24, ink)]:
    run = np.cumsum(rate, axis=0)
    acc = run[Dh - 1:] - np.r_[np.zeros((1, ny, nx)), run[:-Dh]]   # all D-hour totals
    depth_max = acc.max(axis=0)                                   # max D-hour total at each point
    dad[Dh] = dad_curve(depth_max, areas)
    ax.semilogx(areas, dad[Dh], color=col, lw=2.2, label=f"{Dh} h")
ax.set_xlabel("Area (km²)"); ax.set_ylabel("Maximum average depth (mm)")
ax.set_xlim(1, 8000); ax.set_ylim(0, None)
ax.legend(loc="upper right", fontsize=9.5, title="Duration")
plt.show()
for Dh in dad: glue(f"dad_{Dh}_1", float(dad[Dh][0]), display=False); glue(f"dad_{Dh}_big", float(dad[Dh][-1]), display=False)
```

*Figure 11.3. Depth–area–duration curves of a synthetic 24-hour storm with a convective cell inside a rain band. For short durations the depth falls quickly with area, because the intense rain covers a small area: the maximum 1-hour depth is {glue:text}`dad_1_1:.0f` mm at a point but {glue:text}`dad_1_big:.0f` mm averaged over 8000 km². For 24 hours the curve is flatter ({glue:text}`dad_24_1:.0f` mm to {glue:text}`dad_24_big:.0f` mm).*

Strictly, a point value at the storm center is not the same as the $T$-year point value, and the "wettest area" of a DAD curve need not coincide with a particular catchment, so DAD curves are used for storm analysis and transposition, while ARFs are used with IDF curves.

## 11.5 The time distribution of design rainfall

The rational method needs only an intensity. Hydrograph methods (Chapters 9 and 12) need a **hyetograph**: how the design depth is distributed in time. Two families of methods are common.

### The alternating block method

The alternating block method builds a hyetograph directly from the IDF curve, so that **every** duration within the storm has the design intensity for that duration:

1. choose the storm duration $D$ and a time step $\Delta t$;
2. from the IDF curve, compute the cumulative depth $P_j = i(j\Delta t, T) \times j\Delta t$ for $j = 1, 2, \ldots, D/\Delta t$;
3. take differences, $\Delta P_j = P_j - P_{j-1}$, which decrease with $j$;
4. place the largest increment at the center of the storm and the following ones alternately to its right and left.

The result has a sharp central peak (Figure 11.4). Because it packs the maxima of all durations into a single storm, it is conservative: a real storm rarely contains the $T$-year depth for every duration at once.

### Huff curves

Huff (1967) analyzed many storms recorded in Illinois and expressed each as a **dimensionless mass curve**: cumulative fraction of the storm depth against fraction of the storm duration. He grouped the storms by the **quartile** (first, second, third or fourth quarter of the duration) in which the heaviest rain fell, and for each group gave curves for different probabilities. The curve for a 50% probability within the dominant quartile is commonly used for design. The method has been adopted in other countries, including Korea, with quartile curves derived from local records.

The code below demonstrates the procedure on a set of synthetic storms. Each storm's dimensionless mass curve is computed, the storm is assigned to the quartile with the largest share of its depth, and the median curve of each group is taken.

```{code-cell} python
:tags: [hide-input]
rng = np.random.default_rng(17)
nstorm, nstep = 600, 48
mass_curves, quartile = [], []
for _ in range(nstorm):
    # a storm = a few bursts of random timing, width and size on a weak background
    r = 0.05 + np.zeros(nstep)
    for _ in range(rng.integers(1, 4)):
        c, w, a = rng.uniform(0, nstep), rng.uniform(1.5, 8), rng.lognormal(0, 0.6)
        r += a * np.exp(-0.5 * ((np.arange(nstep) - c) / w) ** 2)
    cm = np.r_[0, np.cumsum(r)] / r.sum()
    mass_curves.append(cm)
    quartile.append(np.argmax([cm[12 * (q + 1)] - cm[12 * q] for q in range(4)]) + 1)
mass_curves, quartile = np.array(mass_curves), np.array(quartile)
tf = np.linspace(0, 1, nstep + 1)
huff = {q: np.median(mass_curves[quartile == q], axis=0) for q in range(1, 5)}

fig, ax = plt.subplots(figsize=(6.0, 4.4))
for q, col in zip(range(1, 5), [c1, c3, c2, ink2]):
    ax.plot(tf * 100, huff[q] * 100, color=col, lw=2.2,
            label=f"Q{q}, {(quartile == q).sum()} storms")
ax.plot([0, 100], [0, 100], color=ink2, lw=0.8, ls=":")
ax.set_xlabel("Cumulative storm time (%)"); ax.set_ylabel("Cumulative storm depth (%)")
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.legend(loc="lower right", fontsize=9.5)
plt.show()
```

*Figure 11.4. Huff-type dimensionless mass curves (median of each group) derived from 600 synthetic storms grouped by the quartile in which most rain fell. They illustrate the method only; design must use curves derived from local records.*

Other design hyetographs include the NRCS 24-hour distributions, the triangular hyetograph and the Chicago (Keifer–Chu) storm, which is the continuous-time analog of the alternating block method. Whatever method is used, the hyetograph must be applied with the areal depth (after the ARF) and checked for consistency with the IDF curve. Chapter 12 shows how much the choice affects the design peak.

## 11.6 Probable maximum precipitation and flood

For a large dam whose failure could cause loss of life on a large scale, no return period is long enough. Design is instead based on the physical upper limit of rainfall. The **probable maximum precipitation** (PMP) is the theoretically greatest depth of precipitation for a given duration that is physically possible over a given area at a given location at a certain time of year (WMO, 2009). The **probable maximum flood** (PMF) is the flood that results from the PMP falling on the catchment under the most severe reasonably possible conditions: wet soils, high baseflow and, where relevant, snowmelt.

**Storm maximization.** The main hydrometeorological method takes the largest observed storms in a region and asks how much more rain they would have produced if the atmosphere had contained the maximum possible moisture. The **precipitable water** $W$, the depth of water obtained by condensing all vapor in a column of air,

$$
W = \frac{1}{g}\int_{p_{\text{top}}}^{p_s} q\, dp ,
$$

is estimated from the surface dew point by assuming a saturated, pseudo-adiabatic atmosphere (Chapter 2). The storm depth is multiplied by the **moisture maximization ratio**

$$
P_{\max} = P_{\text{storm}} \times \frac{W_{\max}}{W_{\text{storm}}},
$$ (eq-moisture-max)

where $W_{\text{storm}}$ is the precipitable water of the inflowing air during the storm and $W_{\max}$ the maximum for that place and season, both found from persisting dew points. Storms are then **transposed** to the catchment from similar regions, and an **envelope** is drawn over the maximized DAD curves of all storms.

```{code-cell} python
:tags: [hide-input]
def precipitable_water(Td_surface, p_s=100.0, p_top=20.0, dp=0.05):
    """Precipitable water (mm) of a saturated pseudo-adiabatic column with surface temperature = dew point.
    Pressures in kPa; returns W = (1/g) ∫ q dp."""
    g, cp, Rd, eps, Lv = 9.81, 1005.0, 287.0, 0.622, 2.501e6
    es = lambda T: 0.6108 * np.exp(17.27 * T / (T + 237.3))
    T, p, W = Td_surface, p_s, 0.0
    while p > p_top:
        rs = eps * es(T) / (p - es(T)); q = rs / (1 + rs)
        TK = T + 273.15
        # moist adiabatic dT/dp (K per kPa) from the moist lapse rate and hydrostatics
        Gs = g * (1 + Lv * rs / (Rd * TK)) / (cp + Lv ** 2 * rs * eps / (Rd * TK ** 2))
        dTdp = Gs * Rd * TK / (g * p * 1000) * 1000
        W += q * dp * 1000 / g                                  # kg/m² = mm
        T -= dTdp * dp; p -= dp
    return W

Tds = np.arange(10, 31, 2)
Ws = [precipitable_water(Td) for Td in Tds]
fig, ax = plt.subplots(figsize=(6.0, 3.6))
ax.plot(Tds, Ws, "o-", color=c1, lw=2)
ax.set_xlabel("1000 hPa dew point (°C)"); ax.set_ylabel("Precipitable water W (mm)")
plt.show()
glue("pw_20", float(precipitable_water(20)), display=False); glue("pw_26", float(precipitable_water(26)), display=False)
```

*Figure 11.5. Precipitable water in a saturated pseudo-adiabatic column as a function of the surface dew point, computed by integrating the moist adiabat. It nearly triples between dew points of 14 and 26 °C ({glue:text}`pw_20:.0f` mm at 20 °C, {glue:text}`pw_26:.0f` mm at 26 °C), the Clausius–Clapeyron effect of Chapter 2 integrated over the column.*

**Statistical estimate.** Hershfield (1961) proposed a statistical shortcut for a first estimate where storm data are limited:

$$
P_{\text{PMP}} = \bar{X}_n + K_m\, s_n ,
$$

where $\bar{X}_n$ and $s_n$ are the mean and standard deviation of the annual maximum series for the duration of interest, and $K_m$ is a frequency factor that Hershfield found could reach about 15 in a large sample of US stations; later work made $K_m$ depend on the mean and the duration. The method is used as a check on hydrometeorological estimates.

PMP estimates carry large uncertainty, and the word "maximum" should not be read as a guarantee. Under a warming climate the maximum precipitable water increases (Chapter 13), and PMP studies are being revised in many countries.

## 11.7 Worked examples

**Example 11.1 — Risk and return period.** A culvert is designed for the 25-year flood and is expected to last 50 years. What is the probability that its capacity is exceeded at least once? What return period would keep that probability to 10%?

```{code-cell} python
T, n_life = 25, 50
R = 1 - (1 - 1 / T) ** n_life
T_needed = 1 / (1 - (1 - 0.10) ** (1 / n_life))
print(f"Risk with T = {T}: {R:.2f}")
print(f"T for a 10% risk over {n_life} years: {T_needed:.0f} years")
from math import comb
print("Probability of exactly k exceedances in 50 years (T = 25): " +
      ", ".join(f"k={k}: {comb(n_life, k) * (1 / T) ** k * (1 - 1 / T) ** (n_life - k):.3f}" for k in range(4)))
```

**Example 11.2 — Design depths from the IDF curve.** Find the 50-year intensity and depth for durations of 30 minutes, 2 hours and 24 hours.

```{code-cell} python
for Dm in (30, 120, 1440):
    i = idf(Dm, 50)
    print(f"D = {Dm:5d} min: i = {i:6.1f} mm/h, depth = {i * Dm / 60:6.1f} mm")
```

**Example 11.3 — Fitting an IDF formula.** Frequency analysis of a station gave the 20-year depths below. Fit $i = a/(D + b)^c$ by least squares.

```{code-cell} python
from scipy.optimize import curve_fit
D_tab = np.array([10, 20, 30, 60, 120, 180, 360, 720, 1440], float)     # min
P_tab = np.array([27.5, 41.0, 49.5, 66.0, 84.0, 95.5, 117.0, 141.0, 168.0])  # mm (synthetic)
i_tab = P_tab / D_tab * 60
form = lambda D, a, b, c: a / (D + b) ** c
(a, b, c), _ = curve_fit(form, D_tab, i_tab, p0=(1500, 10, 0.7), maxfev=20000)
rel = form(D_tab, a, b, c) / i_tab - 1
print(f"a = {a:.0f}, b = {b:.1f} min, c = {c:.3f}; largest relative error {np.abs(rel).max() * 100:.1f}%")
```

**Example 11.4 — Areal reduction.** The 50-year 3-hour point rainfall for a 250 km² catchment comes from the IDF curve. Find the ARF and the areal design depth, and check the SI form of Equation {eq}`eq-arf` against the original.

```{code-cell} python
Dh, A_km2 = 3.0, 250.0
arf_si = 1 - np.exp(-1.1 * Dh ** 0.25) + np.exp(-1.1 * Dh ** 0.25 - 0.00386 * A_km2)
arf_us = 1 - np.exp(-1.1 * Dh ** 0.25) + np.exp(-1.1 * Dh ** 0.25 - 0.01 * A_km2 / 2.589988)
P_point = idf(180, 50) * 3
print(f"ARF = {arf_si:.3f} (original form {arf_us:.3f}); point {P_point:.1f} mm -> areal {arf_si * P_point:.1f} mm")
```

**Example 11.5 — An alternating block hyetograph.** Build a 12-hour, 50-year design hyetograph with 1-hour blocks from the IDF curve.

```{code-cell} python
dt_h, D_h = 1.0, 12
j = np.arange(1, D_h + 1)
P_cum = idf(j * dt_h * 60, 50) * j * dt_h                     # cumulative depth for each duration
dP = np.diff(np.r_[0, P_cum])
order = np.argsort(dP)[::-1]
hyeto = np.zeros(D_h)
center = (D_h - 1) // 2                                        # largest block in the middle
positions = [center]
for m in range(1, D_h):                                        # then alternately right and left
    positions += [p for p in (center + m, center - m) if 0 <= p < D_h]
for blk, pos in zip(dP[order], positions[:D_h]):
    hyeto[pos] = blk
print("Hourly depths (mm):", ", ".join(f"{h:.1f}" for h in hyeto))
print(f"Total = {hyeto.sum():.1f} mm = 12-h IDF depth {P_cum[-1]:.1f} mm")
best3 = np.convolve(hyeto, np.ones(3), "valid").max()
print(f"Largest 3-h total in the storm = {best3:.1f} mm; 3-h IDF depth = {P_cum[2]:.1f} mm")
```

```{code-cell} python
:tags: [hide-input]
fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.bar(np.arange(D_h) + 0.5, hyeto, width=0.9, color=c1)
ax.set_xlabel("Time (h)"); ax.set_ylabel("Depth per hour (mm)"); ax.set_xlim(0, 12)
plt.show()
```

*Figure 11.6. A 12-hour, 50-year alternating block hyetograph with 1-hour blocks. Every window of 1, 2, 3, ... hours centered on the peak contains the IDF depth for that duration.*

**Example 11.6 — Moisture maximization.** A historical storm produced 410 mm in 24 hours over a catchment, while the persisting dew point of the inflowing air was 22 °C. The maximum persisting dew point for the season is 26 °C. Estimate the maximized storm depth.

```{code-cell} python
W_storm, W_max = precipitable_water(22), precipitable_water(26)
print(f"W_storm = {W_storm:.1f} mm, W_max = {W_max:.1f} mm, ratio = {W_max / W_storm:.2f}")
print(f"Maximized depth = {410 * W_max / W_storm:.0f} mm")
```

## 11.8 Common mistakes

- **Reading the return period as a schedule.** A 100-year event can occur in consecutive years; its probability in any year is 1%.
- **Choosing $T$ equal to the design life.** The risk is then 63%.
- **Using point IDF rainfall for a large catchment.** Apply an areal reduction factor.
- **Mixing durations.** The rational method needs the intensity for $t_c$; a hyetograph method needs a storm at least as long as the time to peak plus the time base of the response.
- **Using Huff curves from another climate.** Quartile curves depend on storm types and must come from local records.
- **Treating the PMP as a known physical constant.** It is an estimate with large uncertainty.

## 11.9 Exercises

```{code-cell} python
:tags: [remove-cell]
glue("u1", 1 - (1 - 1 / 100) ** 30, display=False)
glue("u2", 1 / (1 - (1 - 0.05) ** (1 / 100)), display=False)
glue("u3_i", float(idf(60, 30)), display=False); glue("u3_P", float(idf(1440, 30) * 24), display=False)
arf4 = 1 - np.exp(-1.1 * 24 ** 0.25) + np.exp(-1.1 * 24 ** 0.25 - 0.00386 * 800)
glue("u4_arf", float(arf4), display=False); glue("u4_P", float(arf4 * idf(1440, 100) * 24), display=False)
glue("u5", (1 - 1 / 50) ** 10, display=False)
```

1. What is the probability that the 100-year flood occurs at least once in 30 years?
2. What return period should be used if the risk of exceedance over a 100-year design life must not exceed 5%?
3. Using the illustrative IDF curve, find the 30-year 1-hour intensity and the 30-year 24-hour depth.
4. Find the ARF and the areal 100-year 24-hour depth for an 800 km² catchment.
5. What is the probability that a 50-year storm does **not** occur in a 10-year period?

```{admonition} Answers
:class: dropdown
1. {glue:text}`u1:.3f`.
2. About {glue:text}`u2:.0f` years.
3. {glue:text}`u3_i:.1f` mm h⁻¹; {glue:text}`u3_P:.0f` mm.
4. ARF = {glue:text}`u4_arf:.3f`; {glue:text}`u4_P:.0f` mm.
5. {glue:text}`u5:.3f`.
```

## 11.10 Further reading

- Dingman, S. L. (2015). *Physical Hydrology* (3rd ed.). Waveland Press. The material on hydrologic design and probability.
- Chow, V. T., Maidment, D. R., and Mays, L. W. (1988). *Applied Hydrology*. McGraw-Hill. The chapters on hydrologic design and design storms.
- 윤용남. *수문학* (in Korean). The chapters on design rainfall and probable maximum precipitation.
- Hershfield, D. M. (1961). Estimating the probable maximum precipitation. *Journal of the Hydraulics Division, ASCE*, 87(HY5), 99–116.
- Huff, F. A. (1967). Time distribution of rainfall in heavy storms. *Water Resources Research*, 3, 1007–1019.
- Leclerc, G., and Schaake, J. C. (1972). *Derivation of hydrologic frequency curves*. Report 142, Ralph M. Parsons Laboratory, Massachusetts Institute of Technology.
- World Meteorological Organization (2009). *Manual on Estimation of Probable Maximum Precipitation (PMP)*. WMO-No. 1045.

---

*Figures in this chapter are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
