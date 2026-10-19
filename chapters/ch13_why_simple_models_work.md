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

# 13. Why simple models work

```{admonition} Core ideas
:class: tip
1. Catchments are enormously heterogeneous, yet models with one or two parameters (a linear reservoir, a unit hydrograph, a Budyko curve) often describe them well. Averaging over many different parts produces **emergent simplicity** at the catchment scale.
2. Many catchment properties follow **power laws**, relations of the form $y = a x^b$, which look like straight lines on log–log plots and describe how quantities change with scale.
3. If streamflow depends mainly on the water stored in the catchment, the catchment behaves as a single nonlinear reservoir. Its **sensitivity function** $g(Q) = dQ/dS$ can be measured from **recession** data as $g(Q) = -(dQ/dt)/Q$, and the model can even be run backward to infer rainfall from streamflow (Kirchner, 2009).
4. Daily data smooth out flood peaks. Across events, **hourly peaks grow faster than daily peaks**, roughly as a power function of them, so flood risk assessed with daily indicators can underestimate the effect of wetter climates (Kim et al., 2018).
5. Under climate change, the same catchment model can be driven by systematically altered rainfall to map how flood indicators respond to **climatic stresses**, independent of any single climate projection.
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

## 13.1 Emergent simplicity

A catchment of 100 km² contains millions of hillslope elements, each with its own soil depth, slope, vegetation and distance to the stream. A complete physical description is out of reach, and the data to support one never exist. Yet the methods of this book, which describe the whole catchment with one or two parameters, work surprisingly well. This chapter asks why, and shows what simple models can tell us about floods in a changing climate.

Part of the answer is **averaging**. The outflow at the outlet is the sum of contributions from many parts of the catchment. When many independent, differently shaped responses are added together, the details of each cancel out and the sum takes a smooth, regular shape, in the same way that the sum of many random numbers tends towards a normal distribution. Figure 13.1 demonstrates this: 2000 hillslopes, each a linear reservoir with its own storage constant and its own travel time to the outlet, add up to a smooth catchment response that a two-parameter Nash cascade (Chapter 9) describes closely.

```{code-cell} python
:tags: [hide-input]
from scipy.stats import gamma
rng = np.random.default_rng(13)
n_hs = 2000
k_hs = rng.lognormal(np.log(2.0), 0.6, n_hs)                    # hillslope storage constants (h)
tau_hs = rng.gamma(3.0, 1.5, n_hs)                               # channel travel times to the outlet (h)
w_hs = rng.lognormal(0, 0.5, n_hs); w_hs /= w_hs.sum()           # hillslope areas (fractions)
t = np.linspace(0, 40, 801)
resp = np.array([np.where(t > tau, np.exp(-(t - tau) / k) / k, 0.0) for k, tau in zip(k_hs, tau_hs)])
catch = (w_hs[:, None] * resp).sum(axis=0)                       # catchment impulse response (1/h)
m1 = np.trapezoid(t * catch, t); var = np.trapezoid((t - m1) ** 2 * catch, t)
k_n, n_n = var / m1, m1 ** 2 / var                               # Nash cascade by moments
fit = gamma.pdf(t, n_n, scale=k_n)

fig, ax = plt.subplots(figsize=(6.4, 4.0))
for j in rng.choice(n_hs, 12, replace=False):
    ax.plot(t, resp[j] * 0.08, color=ink2, lw=0.6, alpha=0.6)
ax.plot([], [], color=ink2, lw=0.6, label="Individual hillslopes (scaled)")
ax.plot(t, catch, color=c1, lw=2.6, label=f"Sum of {n_hs} hillslopes")
ax.plot(t, fit, color=c2, lw=1.8, ls="--", label=f"Nash cascade, n = {n_n:.1f}, k = {k_n:.2f} h")
ax.set_xlabel("Time (h)"); ax.set_ylabel("Impulse response (h⁻¹)"); ax.set_xlim(0, 30); ax.set_ylim(0, None)
ax.legend(loc="upper right", fontsize=9.5)
plt.show()
glue("nash_err", float(np.abs(fit - catch).max() / catch.max() * 100), display=False)
```

*Figure 13.1. Emergent simplicity. Each of 2000 synthetic hillslopes responds like a linear reservoir delayed by its travel time to the outlet (a few are shown in gray). Their area-weighted sum, the catchment response, is matched by a Nash cascade fitted only by its first two moments, with a largest error of {glue:text}`nash_err:.0f`% of the peak.*

A second part of the answer is that catchments are not random collections of parts. Over long periods, water, soil, vegetation and the drainage network shape one another, and the result has **regular statistical structure**. Robert Horton (see [*A Brief History*](https://daehakim-waterlab.github.io/water-history/history-of-hydrology/#horton-and-the-physics-of-runoff)) found that the numbers, lengths and areas of streams of successive orders form geometric series, the **Horton laws** of drainage composition. Such regularities are **power laws**: relations of the form

\begin{equation}
y = a\,x^{b},
\tag{13.1}
\end{equation}

which become straight lines on log–log plots, $\ln y = \ln a + b \ln x$. A power law has no characteristic scale: multiplying $x$ by any factor multiplies $y$ by the same factor to the power $b$, whatever the starting size. Hack (1957) found that main-stream length grows with catchment area as $L \approx 1.4\,A^{0.6}$ (miles and square miles), or in SI

$$
L \approx 1.27\,A^{0.6} \qquad (L \text{ in km},\ A \text{ in km}^2),
$$

an exponent above 0.5 that means larger catchments are relatively more elongated (Example 13.5 checks the conversion). Mean annual floods and many other quantities scale with catchment area in a similar way, which is the basis of regional formulas for ungauged catchments. Power laws also describe processes, as the next section shows.

## 13.2 The catchment as a simple dynamical system

### One storage, one discharge

Suppose that the discharge of a catchment depends only on how much water it stores, $Q = f(S)$, as for the reservoirs of Chapters 9 and 10 but without assuming linearity. The water balance of Chapter 1 is $dS/dt = P - E - Q$. Differentiating $Q = f(S)$ with respect to time,

\begin{equation}
\frac{dQ}{dt} = \frac{dQ}{dS}\,\frac{dS}{dt} = g(Q)\,\bigl(P - E - Q\bigr),
\tag{13.2}
\end{equation}

where $g(Q) = dQ/dS$ is the **sensitivity function**: how strongly discharge responds to a change in storage. Kirchner (2009) pointed out that $g(Q)$ can be measured directly from the streamflow record. During periods when $P$ and $E$ are both negligible (rainless nights, when transpiration stops), Equation (13.2) reduces to

\begin{equation}
g(Q) = \frac{-\,dQ/dt}{Q}\bigg|_{P \approx 0,\ E \approx 0}.
\tag{13.3}
\end{equation}

This is the recession analysis of Chapter 6 in a new role. If the recession plot follows a power law, $-dQ/dt = a\,Q^{b}$, then $g(Q) = a\,Q^{b-1}$: a straight line on a log–log plot. For a linear reservoir $b = 1$ and $g = 1/k$ is constant. For $b > 1$ the catchment responds faster when it is wet, which is typical.

Once $g(Q)$ is known, Equation (13.2) is a complete rainfall–runoff model with no further calibration: given $P$ and $E$, it predicts $Q$. It can also be run **backward**: given $Q$, it estimates the catchment-average $P - E$,

\begin{equation}
P - E = \frac{dQ/dt}{g(Q)} + Q ,
\tag{13.4}
\end{equation}

which Kirchner called "doing hydrology backward". The storage–discharge relation itself follows by integrating $dS = dQ/g(Q)$.

### A synthetic test

To see whether this works, we need a catchment whose behavior is known exactly. The code below generates 40 years of synthetic hourly rainfall and drives a nonlinear storage model. Daily rainfall comes from a simple stochastic generator with a wet summer. It is disaggregated into hours with a **multiplicative random cascade**, which repeatedly splits each rainfall amount into random fractions (a day into three 8-hour blocks, then each block in halves three times), producing intermittent, bursty hourly rain with realistic intensities. The "true" catchment stores water $S$ (mm), loses it to evapotranspiration during daylight, and releases $Q = \alpha S^3$ (mm h⁻¹). We then forget the model and treat its output as observations.

```{code-cell} python
from scipy.stats import gamma as gamma_dist
nyr, A_c = 40, 100.0                                     # years, catchment area (km²)
nd = 365 * nyr
rng = np.random.default_rng(2018)
doy = np.arange(nd) % 365
season = np.exp(-((doy - 200) / 45.0) ** 2)              # wet summer
# Random numbers drawn once, so that the stress tests below change the climate, not the luck
u_occ, u_amt = rng.random(nd), rng.random(nd)
w3 = rng.dirichlet([1.2, 1.2, 1.2], nd)
levels = [(rng.uniform(-0.48, 0.48, nd * 3 * 2 ** k), rng.random(nd * 3 * 2 ** k)) for k in range(3)]

def daily_rain(mean_scale=1.0, cv_scale=1.0):
    """Daily rainfall (mm): Markov-chain occurrence, gamma amounts; scales change mean and variability."""
    p01, p11 = 0.15 + 0.25 * season, 0.35 + 0.35 * season
    wet = np.zeros(nd, bool)
    for d in range(1, nd):
        wet[d] = u_occ[d] < (p11[d] if wet[d - 1] else p01[d])
    mean = (6.0 + 14.0 * season) * mean_scale
    shape = 0.75 / cv_scale ** 2
    return np.where(wet, gamma_dist.ppf(u_amt, shape, scale=mean / shape), 0.0)

def cascade(Pd):
    """Multiplicative random cascade: 1 day -> 3 blocks of 8 h -> 3 binary splits -> 24 hours."""
    x = (Pd[:, None] * w3).reshape(-1)
    for dw, z in levels:
        W = np.clip(0.5 + dw, 0, 1)
        W = np.where(z < 0.12, 0.0, np.where(z > 0.88, 1.0, W))     # some halves stay dry
        x = np.column_stack([x * W, x * (1 - W)]).reshape(-1)
    return x

hours = np.arange(nd * 24)
hod, dh = hours % 24, (hours // 24) % 365
Ep = np.clip(np.sin(np.pi * (hod - 6) / 12), 0, None) * (0.33 + 0.25 * np.sin(2 * np.pi * (dh - 100) / 365))

def catchment_model(P, alpha=5e-6, S0=40.0, S_crit=40.0):
    """'True' catchment: storage S (mm), ET limited when dry, discharge Q = alpha S³ (mm/h)."""
    S = S0; Q = np.empty(len(P)); E = np.empty(len(P))
    for t in range(len(P)):
        S += P[t]
        E[t] = min(Ep[t] * min(1.0, S / S_crit), S); S -= E[t]
        Q[t] = min(alpha * S ** 3, S); S -= Q[t]
    return Q, E

P_h = cascade(daily_rain())
Q_h, E_h = catchment_model(P_h)
print(f"Annual means: P = {P_h.sum() / nyr:.0f} mm, E = {E_h.sum() / nyr:.0f} mm, Q = {Q_h.sum() / nyr:.0f} mm")
print(f"Largest hourly rain {P_h.max():.0f} mm; largest flow {Q_h.max():.1f} mm/h = {Q_h.max() * A_c / 3.6:.0f} m³/s")
```

Now the recession analysis. We keep hours with no rain in the hour itself or the two hours either side, and only night hours (no evapotranspiration), compute $-dQ/dt$ by central differences, average it within bins of $\ln Q$, and fit a straight line on log–log axes.

```{code-cell} python
dQdt = np.r_[np.nan, (Q_h[2:] - Q_h[:-2]) / 2, np.nan]
rainfree = np.convolve(P_h > 0, np.ones(5), "same") == 0
night = (hod < 5) | (hod > 20)
sel = rainfree & night & (dQdt < 0) & (Q_h > 1e-3)
lnQ, lnR = np.log(Q_h[sel]), np.log(-dQdt[sel])
edges = np.linspace(lnQ.min(), lnQ.max(), 25)
idx = np.digitize(lnQ, edges)
bins = [(lnQ[idx == j].mean(), lnR[idx == j].mean()) for j in range(1, len(edges)) if (idx == j).sum() >= 5]
bq, br = np.array(bins).T
b_fit, ln_a = np.polyfit(bq, br, 1); a_fit = np.exp(ln_a)
print(f"{sel.sum()} recession hours; fitted -dQ/dt = {a_fit:.4f} Q^{b_fit:.3f}")
print(f"True model: -dQ/dt = 3 alpha^(1/3) Q^(5/3) = {3 * 5e-6 ** (1 / 3):.4f} Q^{5 / 3:.3f}")
g = lambda Q: a_fit * Q ** (b_fit - 1)                       # sensitivity function (1/h)
```

```{code-cell} python
:tags: [hide-input]
fig, ax = plt.subplots(figsize=(6.0, 4.2))
ax.loglog(Q_h[sel], -dQdt[sel], ".", color=ink2, ms=2, alpha=0.25, label="Recession hours")
ax.loglog(np.exp(bq), np.exp(br), "o", color=c1, ms=6, label="Binned means")
qq = np.logspace(np.log10(np.exp(bq.min())), np.log10(np.exp(bq.max())), 50)
ax.loglog(qq, a_fit * qq ** b_fit, color=c2, lw=2, label=f"Fit: b = {b_fit:.2f}")
ax.set_xlabel("Q (mm h⁻¹)"); ax.set_ylabel("−dQ/dt (mm h⁻²)")
ax.legend(loc="upper left", fontsize=9.5)
plt.show()
glue("rec_b", float(b_fit), display=False)
```

*Figure 13.2. Recession plot of the synthetic catchment: the rate of decline of discharge against discharge, for rainless night hours. The binned means fall on a straight line with slope $b$ = {glue:text}`rec_b:.2f`, close to the true value of 5/3, so the sensitivity function $g(Q)$ is recovered from the streamflow record alone.*

With $g(Q)$ in hand, Equation (13.2) is run forward with the observed rainfall and evapotranspiration, and backward from the observed discharge. Integrating in $\ln Q$, $d(\ln Q)/dt = g(Q)\,(P - E - Q)/Q$, keeps the solution positive and stable.

```{code-cell} python
def simulate(P, E, Q0, sub=4):
    lnQ = np.log(Q0); out = np.empty(len(P)); h = 1.0 / sub
    for t in range(len(P)):
        for _ in range(sub):
            Q = np.exp(lnQ)
            lnQ += h * g(Q) * (P[t] - E[t] - Q) / Q
        out[t] = np.exp(lnQ)
    return out

yr = 20                                                       # look at one year of the record
w = slice(yr * 8760, (yr + 1) * 8760)
Q_sim = simulate(P_h[w], E_h[w], Q_h[w][0])
nse = 1 - ((Q_sim - Q_h[w]) ** 2).sum() / ((Q_h[w] - Q_h[w].mean()) ** 2).sum()
PE_back = np.gradient(Q_h[w]) / g(Q_h[w]) + Q_h[w]            # Eq. (13.4), mm/h
day_true = (P_h[w] - E_h[w]).reshape(-1, 24).sum(1)
day_back = PE_back.reshape(-1, 24).sum(1)
r_back = np.corrcoef(day_true, day_back)[0, 1]
print(f"Forward simulation, one year: Nash–Sutcliffe efficiency = {nse:.3f}")
print(f"Backward inference of daily P − E: correlation with truth = {r_back:.3f}")
```

```{code-cell} python
:tags: [hide-input]
days = np.arange(365)
win = slice(170, 245)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 6.0), sharex=True)
th = np.arange(8760) / 24
a1.plot(th, Q_h[w] * A_c / 3.6, color=ink, lw=1.6, label="'Observed' (true model)")
a1.plot(th, Q_sim * A_c / 3.6, color=c2, lw=1.4, ls="--", label="Simulated with g(Q) from recessions")
a1.set_ylabel("Q (m³ s⁻¹)"); a1.set_xlim(170, 245); a1.set_ylim(0, Q_h[w].max() * A_c / 3.6 * 1.45)
a1.legend(loc="upper center", fontsize=9, ncol=2)
a1.set_title("(a) Running the model forward")
a2.bar(days[win], day_true[win], width=0.9, color=c1, alpha=0.45, label="True daily P − E")
a2.plot(days[win], day_back[win], "o-", color=c2, ms=3, lw=1.2, label="Inferred from streamflow")
a2.set_xlabel("Day of year"); a2.set_ylabel("mm d⁻¹"); a2.set_ylim(None, day_true[win].max() * 1.45)
a2.legend(loc="upper center", fontsize=9, ncol=2)
a2.set_title("(b) Running it backward")
plt.show()
glue("nse", float(nse), display=False); glue("r_back", float(r_back), display=False)
```

*Figure 13.3. (a) A summer of hourly discharge from the synthetic catchment and its simulation by Equation (13.2) with $g(Q)$ estimated from recessions alone (Nash–Sutcliffe efficiency {glue:text}`nse:.3f` for the year). (b) Daily rainfall minus evapotranspiration inferred backward from the discharge, compared with the truth (correlation {glue:text}`r_back:.2f`).*

In real catchments the agreement is not this good: storage is not a single number, $Q$ is not a unique function of $S$, and rainfall and discharge are measured with error. Kirchner found that the approach works well in catchments whose behavior is dominated by one connected storage, and fails where it is not, which is itself a useful diagnosis. The broader lesson is that a catchment's dynamics can be **read from its streamflow record** and summarized in one function, which explains why simple storage models capture so much.

## 13.3 Hourly and daily peaks

Flood risk is often assessed with daily data, because long daily records and daily climate model outputs are much more common than hourly ones. But a daily mean discharge is an average over 24 hours, while the damage is done by the instantaneous peak. For a small or medium catchment that responds within hours, the daily mean can be a small fraction of the peak.

The relation between the two is not a constant ratio. Because a wet catchment responds faster ($g(Q)$ increases with $Q$ when $b > 1$), large floods are sharper than small ones: more of their volume passes in a few hours, and the hourly peak grows **faster** than the daily mean. Across events the relation is close to a power law,

\begin{equation}
Q_{\text{hourly peak}} \approx \alpha\, Q_{\text{daily peak}}^{\,\beta}, \qquad \beta > 1 .
\tag{13.5}
\end{equation}

Kim et al. (2018) examined this nonlinearity in a bottom-up assessment of flood risk for a mesoscale catchment. They disaggregated daily rainfall into hours with a multiplicative random cascade, as in Section 13.2, and simulated hourly flows with a simple first-order catchment model of the kind described above. They found that a daily flow indicator could considerably **underestimate** the impact of climate change on flood risk, because the nonlinearity between daily and hourly peaks amplifies the effect of increases in the mean and variability of daily precipitation. Figure 13.4 shows the relation for the synthetic catchment of this chapter.

```{code-cell} python
:tags: [hide-input]
Qd_h = Q_h.reshape(-1, 24).mean(axis=1)                       # daily mean flow (mm/h)
amax_h = Q_h.reshape(nyr, -1).max(axis=1) * A_c / 3.6         # annual maximum hourly peak (m³/s)
amax_d = Qd_h.reshape(nyr, -1).max(axis=1) * A_c / 3.6        # annual maximum daily mean (m³/s)
beta_hd, ln_alpha = np.polyfit(np.log(amax_d), np.log(amax_h), 1)
alpha_hd = np.exp(ln_alpha)
fig, ax = plt.subplots(figsize=(6.0, 4.2))
ax.loglog(amax_d, amax_h, "o", color=c1, ms=6, label="Annual maxima (40 years)")
xx = np.logspace(np.log10(amax_d.min() * 0.9), np.log10(amax_d.max() * 1.1), 50)
ax.loglog(xx, alpha_hd * xx ** beta_hd, color=c2, lw=2, label=f"Power law, β = {beta_hd:.2f}")
ax.loglog(xx, xx * np.median(amax_h / amax_d), color=ink2, lw=1, ls=":", label="Constant ratio (β = 1)")
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
for axis, ticks in [(ax.xaxis, [10, 20, 50, 100, 200]), (ax.yaxis, [20, 50, 100, 200, 500])]:
    axis.set_major_locator(FixedLocator(ticks)); axis.set_major_formatter(FixedFormatter([str(v) for v in ticks]))
    axis.set_minor_locator(NullLocator())
ax.set_xlabel("Annual max. daily mean flow (m³ s⁻¹)"); ax.set_ylabel("Annual max. hourly flow (m³ s⁻¹)")
ax.legend(loc="upper left", fontsize=9.5)
plt.show()
glue("beta_hd", float(beta_hd), display=False)
```

*Figure 13.4. Annual maximum hourly discharge against annual maximum daily mean discharge for the synthetic 100 km² catchment. The points follow a power law with exponent β = {glue:text}`beta_hd:.2f`: larger floods have proportionally sharper peaks than a constant ratio (dotted) would suggest.*

## 13.4 Climatic stress and flood risk

A common way to assess the effect of climate change on floods is **top-down**: take the output of climate models, downscale it, and drive a hydrological model. The result depends heavily on which climate models and scenarios are chosen, and their projections of rainfall extremes differ widely. A complementary **bottom-up** (scenario-neutral) approach starts from the system instead. The rainfall input is changed systematically (more or less rain on average, more or less variable), the catchment model is run for each change, and the response of a flood indicator is mapped over the whole range of plausible **climatic stresses**. Climate projections can later be placed on this map, but the map itself does not depend on them, and it shows directly which changes the system is most sensitive to and where thresholds lie.

The physics of Chapter 2 tells us which direction the stress is likely to take. Saturation vapor pressure rises by about 6–7% per degree of warming, so the moisture available to the heaviest storms rises too. Observations in many regions show hourly rainfall extremes increasing at about this Clausius–Clapeyron rate, and in some cases faster (Lenderink and van Meijgaard, 2008). Changes in the frequency and organization of storms add to, or offset, this thermodynamic effect.

Figure 13.5 applies a simple stress test to the synthetic catchment. The mean and the variability of daily rainfall on wet days are scaled separately, the same random numbers are used every time so that only the climate changes, and the change in the mean annual maximum flood is computed with an hourly and with a daily indicator.

```{code-cell} python
:tags: [hide-input]
def flood_indicators(mean_scale=1.0, cv_scale=1.0):
    Q, _ = catchment_model(cascade(daily_rain(mean_scale, cv_scale)))
    Qd = Q.reshape(-1, 24).mean(axis=1)
    return Q.reshape(nyr, -1).max(axis=1).mean(), Qd.reshape(nyr, -1).max(axis=1).mean()

base_h, base_d = flood_indicators()
factors = np.array([0.9, 1.0, 1.1, 1.2, 1.3])
res = {}
for kind in ("mean", "cv"):
    out = [flood_indicators(f, 1.0) if kind == "mean" else flood_indicators(1.0, f) for f in factors]
    res[kind] = (np.array([100 * (o[0] / base_h - 1) for o in out]), np.array([100 * (o[1] / base_d - 1) for o in out]))

fig, axes = plt.subplots(2, 1, figsize=(6.0, 6.6), sharey=True)
for ax, kind, title in [(axes[0], "mean", "(a) Change in mean wet-day rainfall"),
                        (axes[1], "cv", "(b) Change in variability of wet-day rainfall")]:
    ax.plot((factors - 1) * 100, res[kind][0], "o-", color=c2, lw=2.2, label="Hourly peak indicator")
    ax.plot((factors - 1) * 100, res[kind][1], "s-", color=c1, lw=2.2, label="Daily mean indicator")
    ax.axhline(0, color=ink2, lw=0.8); ax.axvline(0, color=ink2, lw=0.8)
    ax.set_title(title); ax.set_ylabel("Change in mean annual\nmaximum flow (%)")
    ax.legend(loc="upper left", fontsize=9.5)
axes[1].set_xlabel("Change in the rainfall statistic (%)")
plt.show()
glue("st_h_mean", float(res["mean"][0][-1]), display=False); glue("st_d_mean", float(res["mean"][1][-1]), display=False)
glue("st_h_cv", float(res["cv"][0][-1]), display=False); glue("st_d_cv", float(res["cv"][1][-1]), display=False)
```

*Figure 13.5. A stress test of the synthetic catchment. A 30% increase in mean wet-day rainfall raises the mean annual maximum hourly flow by {glue:text}`st_h_mean:.0f`% but the daily indicator by only {glue:text}`st_d_mean:.0f`%; a 30% increase in variability raises them by {glue:text}`st_h_cv:.0f`% and {glue:text}`st_d_cv:.0f`%. A daily indicator understates the sensitivity of flood peaks to climatic stress, the effect reported by Kim et al. (2018).*

Three cautions apply to this example. The catchment and its rainfall are synthetic, so the numbers illustrate a mechanism, not a prediction. The model has one storage, so it cannot represent changes in which runoff processes are active (Chapter 6). And stress tests explore the sensitivity of the system; deciding how likely each stress is remains a question for climate science.

## 13.5 When simple models fail

Simple models work when one process, or one well-connected store, dominates the response, and when the conditions of use resemble those of calibration. They fail, or need care, when:

- the catchment has **thresholds**, such as soils that suddenly connect hillslopes to the stream when wet, or floodplains that start to store water above bankfull;
- several **stores with very different time scales** matter at once, such as snow, deep groundwater and fast surface flow;
- **land use or climate changes** move the catchment into conditions never seen in the calibration data;
- many different parameter sets fit the data equally well (**equifinality**), so the parameters cannot be interpreted physically and extrapolation is uncertain.

The engineering response is not to abandon simple models but to use them with understanding: check their assumptions against field knowledge, test them on events not used for calibration, compare several model structures, and report the range of results. That has been the approach of this whole book.

## 13.6 Worked examples

**Example 13.1 — Sensitivity function and storage.** Using the fitted recession relation, find $g(Q)$ at flows of 0.05, 0.5 and 5 mm h⁻¹, the equivalent "storage constant" $1/g$ at each, and the storage released as the flow recedes from 5 to 0.05 mm h⁻¹.

```{code-cell} python
from scipy.integrate import quad
for Q in (0.05, 0.5, 5.0):
    print(f"Q = {Q:5.2f} mm/h: g = {g(Q):.4f} 1/h, 1/g = {1 / g(Q):6.1f} h")
dS, _ = quad(lambda Q: 1 / g(Q), 0.05, 5.0)
print(f"Storage released between Q = 5 and Q = 0.05 mm/h: {dS:.0f} mm")
```

The catchment responds within hours when it is very wet but takes about a week when it is dry. A single linear reservoir, with one value of $k$, cannot do both.

**Example 13.2 — Inferring rainfall from a hydrograph.** Over one hour at night the discharge rises from 0.80 to 1.10 mm h⁻¹. With the fitted $g(Q)$, estimate the catchment-average rainfall during that hour.

```{code-cell} python
Q1, Q2 = 0.80, 1.10
Qm = np.sqrt(Q1 * Q2)                                       # geometric mean flow in the hour
P_est = (Q2 - Q1) / g(Qm) + Qm                              # Eq. (13.4), E ≈ 0 at night
print(f"g({Qm:.2f}) = {g(Qm):.4f} 1/h -> estimated rainfall ≈ {P_est:.1f} mm in the hour")
```

**Example 13.3 — From a daily peak to an hourly peak.** A daily model predicts an annual maximum daily mean flow of 80 m³ s⁻¹ for this catchment. Estimate the hourly peak with the power law of Figure 13.4, and compare with a constant-ratio estimate.

```{code-cell} python
Qd = 80.0
print(f"Power law: {alpha_hd * Qd ** beta_hd:.0f} m³/s; constant median ratio: {Qd * np.median(amax_h / amax_d):.0f} m³/s")
print(f"If the daily peak rises 20% to {1.2 * Qd:.0f} m³/s, the hourly peak rises "
      f"{100 * (1.2 ** beta_hd - 1):.0f}% (power law) instead of 20%")
```

**Example 13.4 — A Clausius–Clapeyron estimate.** If the heaviest rainfalls scale with saturation vapor pressure, by how much would they increase for 2 °C of warming at 22 °C? Use Equation (2.4).

```{code-cell} python
es = lambda T: 0.6108 * np.exp(17.27 * T / (T + 237.3))
print(f"Increase for +2 °C at 22 °C: {100 * (es(24) / es(22) - 1):.1f}%")
```

**Example 13.5 — Hack's law in SI.** Convert $L = 1.4\,A^{0.6}$ (miles, square miles) to kilometers and square kilometers, and estimate the main-stream length of a 145 km² catchment.

```{code-cell} python
mi = 1.609344
c_si = 1.4 * mi / (mi ** 2) ** 0.6
print(f"L = {c_si:.3f} A^0.6 (km, km²); for A = 145 km²: L ≈ {c_si * 145 ** 0.6:.0f} km")
print(f"check: original form gives {1.4 * (145 / mi ** 2) ** 0.6 * mi:.1f} km")
```

The 24 km main channel of the example catchment in Chapter 12 is close to this estimate.

## 13.7 Common mistakes

- **Fitting recession relations to all data.** Rainfall and evapotranspiration contaminate $dQ/dt$. Use rainless periods, preferably at night, and bin the data before fitting.
- **Fitting a power law to a narrow range.** Any smooth curve looks straight over a small range; a power law is meaningful only over at least an order of magnitude.
- **Converting daily flood indicators to hourly peaks with a constant ratio.** The ratio grows with flood size.
- **Reading a stress test as a forecast.** It maps sensitivity; the likelihood of each stress comes from climate information.
- **Assuming a model that fits the past will fit a changed future.** Check whether the processes that dominate may change.

## 13.8 Exercises

```{code-cell} python
:tags: [remove-cell]
glue("f1_b", float(np.log(0.9 / 0.1) / np.log(3.0)), display=False)
glue("f2_g", 0.02 * 2.0 ** 0.6, display=False); glue("f2_k", 1 / (0.02 * 2.0 ** 0.6), display=False)
glue("f3_ratio", 3.0 ** 1.25, display=False)
glue("f4", float(100 * (es(27) / es(25) - 1) * 1.5), display=False)
glue("f5_L", float(c_si * 600 ** 0.6), display=False)
```

1. Two points on a log–log recession plot are ($Q$ = 0.2, $-dQ/dt$ = 0.1) and ($Q$ = 0.6, $-dQ/dt$ = 0.9), in mm h⁻¹ and mm h⁻². Find the exponent $b$ of the power law through them. Is the catchment more or less responsive when wet?
2. A catchment has $g(Q) = 0.02\,Q^{0.6}$ (h⁻¹, $Q$ in mm h⁻¹). What is $g$ at $Q$ = 2 mm h⁻¹, and what linear-reservoir storage constant would give the same response at that flow?
3. If annual maximum hourly peaks scale with daily peaks as $Q_h \propto Q_d^{1.25}$, by what factor does the hourly peak grow when the daily peak triples?
4. Heavy hourly rainfall is observed to increase at 1.5 times the Clausius–Clapeyron rate in a region. Estimate the increase for warming from 25 to 27 °C.
5. Use Hack's law in SI form to estimate the main-stream length of a 600 km² catchment.

```{admonition} Answers
:class: dropdown
1. $b$ = {glue:text}`f1_b:.2f`; more responsive when wet ($b > 1$).
2. $g$ = {glue:text}`f2_g:.4f` h⁻¹, equivalent to $k$ = {glue:text}`f2_k:.1f` h.
3. {glue:text}`f3_ratio:.2f` times.
4. About {glue:text}`f4:.0f`%.
5. About {glue:text}`f5_L:.0f` km.
```

## 13.9 Further reading

- Dingman, S. L. (2015). *Physical Hydrology* (3rd ed.). Waveland Press. The material on catchment-scale hydrological modeling.
- Chow, V. T., Maidment, D. R., and Mays, L. W. (1988). *Applied Hydrology*. McGraw-Hill. The introduction to hydrologic systems.
- 윤용남. *수문학* (in Korean).
- Beven, K. (2006). A manifesto for the equifinality thesis. *Journal of Hydrology*, 320, 18–36.
- Brutsaert, W., and Nieber, J. L. (1977). Regionalized drought flow hydrographs from a mature glaciated plateau. *Water Resources Research*, 13, 637–643.
- Hack, J. T. (1957). Studies of longitudinal stream profiles in Virginia and Maryland. *US Geological Survey Professional Paper* 294-B.
- Kim, D., Chun, J. A., and Aikins, C. M. (2018). An hourly-scale scenario-neutral flood risk assessment in a mesoscale catchment under climate change. *Hydrological Processes*, 32, 3416–3430. [https://doi.org/10.1002/hyp.13273](https://doi.org/10.1002/hyp.13273)
- Kirchner, J. W. (2009). Catchments as simple dynamical systems: catchment characterization, rainfall-runoff modeling, and doing hydrology backward. *Water Resources Research*, 45, W02429. [https://doi.org/10.1029/2008WR006912](https://doi.org/10.1029/2008WR006912)
- Lenderink, G., and van Meijgaard, E. (2008). Increase in hourly precipitation extremes beyond expectations from temperature changes. *Nature Geoscience*, 1, 511–514.
- Prudhomme, C., Wilby, R. L., Crooks, S., Kay, A. L., and Reynard, N. S. (2010). Scenario-neutral approach to climate change impact studies: application to flood risk. *Journal of Hydrology*, 390, 198–209.

---

*Figures in this chapter are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
