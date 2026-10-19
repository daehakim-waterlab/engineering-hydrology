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

# 12. Design floods: putting it together

```{admonition} Core ideas
:class: tip
1. A design flood hydrograph is built in a chain: **design rainfall → excess rainfall → direct runoff → plus baseflow → routing to the design point**. Each link was developed in an earlier chapter.
2. Because the storm duration that produces the largest peak is not known in advance, the chain is run for several durations, and the **critical duration** is the one that gives the largest peak at the design point.
3. Every link involves a choice of method and parameters. Changing **one** choice at a time shows which links matter most; in this example the unit hydrograph method and the antecedent moisture assumed in the loss model each matter more than doubling or halving the return period.
4. Varying **all** uncertain choices together gives a range of plausible peaks. A design value should be reported with that range and with the assumptions that produced it.
5. Mass balance checks at each step (rain ≥ excess = direct runoff depth; routed volume = inflow volume) catch most errors.
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

## 12.1 The design flood chain

Chapters 3 to 11 each solved one part of a problem. This chapter puts them together to answer the question an engineer is actually asked: *what flood hydrograph should a structure at this point on the river be designed for?* The calculation for a medium-sized catchment usually follows the chain in Figure 12.1.

```{code-cell} python
:tags: [hide-input]
from matplotlib.patches import FancyBboxPatch
steps = [("Design rainfall", "IDF curve, return period,\nareal reduction, time pattern", "Ch. 11"),
         ("Excess rainfall", "loss model (curve number)", "Ch. 7"),
         ("Direct runoff", "unit hydrograph, convolution", "Ch. 9"),
         ("Total flow", "add baseflow", "Ch. 6"),
         ("Design point", "route along the river", "Ch. 10")]
fig, ax = plt.subplots(figsize=(5.6, 6.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 10.4); ax.axis("off")
for k, (title, sub, ch) in enumerate(steps):
    y = 9.0 - k * 2.0
    ax.add_patch(FancyBboxPatch((0.8, y - 0.6), 8.4, 1.25, boxstyle="round,pad=0.08",
                                fc=[c1, c3, c1, c3, c2][k], ec="none", alpha=0.16))
    ax.text(1.2, y + 0.2, title, fontsize=11.5, color=ink, va="center", weight="bold")
    ax.text(1.2, y - 0.28, sub, fontsize=9.5, color=ink2, va="center")
    ax.text(8.9, y, ch, fontsize=10, color=ink2, ha="right", va="center")
    if k < len(steps) - 1:
        ax.annotate("", xy=(5, y - 1.3), xytext=(5, y - 0.72), arrowprops=dict(arrowstyle="-|>", color=ink2, lw=1.4))
plt.show()
```

*Figure 12.1. The chain of calculations for a design flood hydrograph, and the chapters where each step is explained. The chain is repeated for several storm durations to find the critical one.*

The example below uses a hypothetical catchment, with all numbers invented for illustration. The code collects the methods of earlier chapters as short functions, so every number in the chapter can be traced and changed.

## 12.2 The example catchment and the toolbox

The catchment drains 145 km² of hilly, partly forested land. A town lies 18 km downstream of the catchment outlet, and a new bridge there must be designed for the 100-year flood.

```{list-table} Properties of the example catchment (hypothetical)
:header-rows: 1
:widths: 50 25 25

* - Property
  - Value
  - Used for
* - Catchment area $A$
  - 145 km²
  - all steps
* - Main channel length $L$, length to centroid $L_c$
  - 24 km, 11 km
  - Snyder lag
* - Time of concentration $t_c$ (mean of several formulas)
  - 4.5 h
  - NRCS and Clark UHs
* - Clark storage coefficient $R$
  - 4.0 h
  - Clark UH
* - Curve number, average conditions (CN$_{\text{II}}$)
  - 70
  - losses
* - Baseflow during floods
  - 0.04 m³ s⁻¹ km⁻²
  - total flow
* - Reach to the town: Muskingum $K$, $x$
  - 4.0 h, 0.10
  - routing
```

*Table 12.1. The hypothetical catchment. All values are invented for this example.*

```{code-cell} python
# ---- Toolbox: methods from Chapters 7–11, written compactly ----
from scipy.stats import gamma

def idf(D_min, T):                        # Chapter 8/11 illustrative IDF, mm/h
    return 1450.0 * (1 + 0.20 * np.log(T / 10.0)) / (D_min + 12.0) ** 0.72

def arf(D_h, A_km2):                      # Chapter 11, Eq. (11.5)
    return 1 - np.exp(-1.1 * D_h ** 0.25) + np.exp(-1.1 * D_h ** 0.25 - 0.00386 * A_km2)

from scipy.stats import beta

def design_hyetograph(T, D_h, A_km2, peak_pos=0.5, dt=1.0, use_arf=True, pattern="huff"):
    """Design hyetograph (mm per step) for return period T and duration D_h (h).
    pattern="huff":  areal IDF depth spread with a fixed dimensionless mass curve (a beta CDF whose
                     steepest part lies at fraction peak_pos of the storm), as in Huff-type methods;
    pattern="block": alternating blocks from the IDF curve, largest block at fraction peak_pos."""
    n = int(round(D_h / dt)); j = np.arange(1, n + 1)
    red = arf(D_h, A_km2) if use_arf else 1.0
    if pattern == "huff":
        a_, b_ = 1 + 6 * peak_pos, 1 + 6 * (1 - peak_pos)
        mass = beta.cdf(j / n, a_, b_)
        return idf(D_h * 60, T) * D_h * red * np.diff(np.r_[0, mass])
    P_cum = idf(j * dt * 60, T) * j * dt * red
    blocks = np.sort(np.diff(np.r_[0, P_cum]))[::-1]
    c = int(round(peak_pos * (n - 1)))
    pos = [c]
    for m in range(1, n):
        pos += [p for p in (c + m, c - m) if 0 <= p < n]
    hy = np.zeros(n); hy[pos[:n]] = blocks
    return hy

def cn_excess(P, CN):                     # Chapter 7: cumulative CN method, mm per step
    S = 25400.0 / CN - 254.0; Ia = 0.2 * S
    cum = np.cumsum(P)
    Pe = np.where(cum > Ia, (cum - Ia) ** 2 / (cum - Ia + S), 0.0)
    return np.diff(np.r_[0.0, Pe])

cn_amc3 = lambda cn: 23 * cn / (10 + 0.13 * cn)

def uh_nrcs(A, tc, dt=1.0, D=1.0, n=72):  # Chapter 9: curvilinear NRCS UH (gamma form, m = 3.7)
    Tp = D / 2 + 0.6 * tc
    t = np.arange(1, n + 1) * dt
    q = (t / Tp) ** 3.7 * np.exp(3.7 * (1 - t / Tp))
    return q / (q.sum() * dt * 3600) * 1000 * A          # rescaled to exactly 1 mm

def uh_clark(A, tc, R, dt=1.0, D=1.0, n=72, sub=8):
    h = dt / sub; t = np.arange(0, n * dt + h / 2, h)
    tac = lambda x: np.where(x <= tc / 2, 1.414 * (x / tc) ** 1.5, 1 - 1.414 * np.clip(1 - x / tc, 0, None) ** 1.5)
    edges = np.arange(0, tc + h, h); frac = np.diff(np.clip(tac(np.minimum(edges, tc)), 0, 1))
    inflow = np.convolve(np.where(t < D, 1.0 / D, 0.0), frac)[: len(t)] * A / 3.6
    c = h / (R + 0.5 * h); O = np.zeros(len(t))
    for k in range(1, len(t)):
        O[k] = c * inflow[k] + (1 - c) * O[k - 1]
    U = O[sub::sub][:n]
    return U / (U.sum() * dt * 3600) * 1000 * A

def uh_snyder(A, L, Lc, Ct=1.6, Cp=0.6, dt=1.0, D=1.0, n=72):
    tp = 0.75 * Ct * (L * Lc) ** 0.3; tpR = tp + (D - tp / 5.5) / 4
    Qp = 0.275 * Cp * A / tpR; tpk = tpR + D / 2
    t = np.arange(1, n + 1) * dt
    from scipy.optimize import brentq
    shape = lambda m: (t / tpk) ** m * np.exp(m * (1 - t / tpk))
    m = brentq(lambda m: Qp * shape(m).sum() * dt * 3600 - 1000 * A, 0.5, 60)
    return Qp * shape(m)

def muskingum(I, K, x, dt):
    den = 2 * K * (1 - x) + dt
    C0, C1, C2 = (dt - 2 * K * x) / den, (dt + 2 * K * x) / den, (2 * K * (1 - x) - dt) / den
    O = np.empty_like(I); O[0] = I[0]
    for k in range(len(I) - 1):
        O[k + 1] = C0 * I[k + 1] + C1 * I[k] + C2 * O[k]
    return O

catchment = dict(A=145.0, L=24.0, Lc=11.0, tc=4.5, R=4.0, CN=70.0, qb=0.04, K=4.0, x=0.10)
```

```{code-cell} python
def design_flood(T=100, D_h=12, peak_pos=0.5, amc=3, uh="NRCS", tc_factor=1.0, use_arf=True,
                 pattern="huff", c=catchment, n_out=120):
    """Run the whole chain; returns a dictionary of the intermediate series (1-hour steps)."""
    A = c["A"]
    P = design_hyetograph(T, D_h, A, peak_pos, use_arf=use_arf, pattern=pattern)
    CN = cn_amc3(c["CN"]) if amc == 3 else c["CN"]
    Pe = cn_excess(P, CN)
    tc = c["tc"] * tc_factor
    U = {"NRCS": lambda: uh_nrcs(A, tc), "Clark": lambda: uh_clark(A, tc, c["R"] * tc_factor),
         "Snyder": lambda: uh_snyder(A, c["L"], c["Lc"])}[uh]()
    Qd = np.zeros(n_out); conv = np.convolve(Pe, U)[:n_out]; Qd[: len(conv)] = conv
    Qb = c["qb"] * A
    Q_out = Qd + Qb                                         # at the catchment outlet
    Q_town = muskingum(Q_out, c["K"], c["x"], 1.0)          # at the design point
    # mass-balance checks
    assert np.isclose(Qd.sum() * 3600 / (A * 1e6) * 1000, Pe.sum(), rtol=0.02)
    return dict(P=P, Pe=Pe, CN=CN, U=U, Qd=Qd, Qb=Qb, Q_out=Q_out, Q_town=Q_town)
```

## 12.3 Step by step for a 12-hour storm

### Design rainfall

For a 100-year, 12-hour storm, the point depth from the IDF curve is reduced by the areal reduction factor for 145 km² (Chapter 11). It is distributed in time with a fixed dimensionless mass curve of the Huff type, whose steepest part lies in the middle of the storm. In practice the curve would come from local storm records (Chapter 11); here a smooth beta-shaped curve stands in for it. Section 12.4 explains why the alternating block method is not used for this purpose.

### Excess rainfall

Design guidelines often assume wet antecedent conditions for rare floods; here CN$_{\text{II}}$ = 70 is converted to AMC III with Equation (7.5), giving CN$_{\text{III}}$ ≈ 84. The cumulative curve number method (Chapter 7) gives the excess hyetograph.

### Direct runoff, baseflow and routing

The excess is convolved with the 1-hour NRCS unit hydrograph (Chapter 9), a constant baseflow of 0.04 m³ s⁻¹ km⁻² is added (Chapter 6), and the result is routed 18 km downstream with the Muskingum method (Chapter 10, $\Delta t$ = 1 h, which satisfies $2Kx \le \Delta t \le 2K(1-x)$).

```{code-cell} python
r = design_flood(T=100, D_h=12)
A = catchment["A"]
print(f"Point 12-h depth = {idf(720, 100) * 12:.1f} mm, ARF = {arf(12, A):.3f}, areal depth = {r['P'].sum():.1f} mm")
print(f"CN_III = {r['CN']:.1f}; excess = {r['Pe'].sum():.1f} mm (runoff ratio {r['Pe'].sum() / r['P'].sum():.2f})")
print(f"UH volume = {r['U'].sum() * 3600 / (A * 1e6) * 1000:.3f} mm; peak UH = {r['U'].max():.2f} m³/s per mm")
print(f"Direct runoff depth = {r['Qd'].sum() * 3600 / (A * 1e6) * 1000:.1f} mm; baseflow = {r['Qb']:.1f} m³/s")
print(f"Peak at catchment outlet = {r['Q_out'].max():.0f} m³/s at {r['Q_out'].argmax()} h")
print(f"Peak at the town        = {r['Q_town'].max():.0f} m³/s at {r['Q_town'].argmax()} h")
```

```{code-cell} python
:tags: [hide-input]
t = np.arange(120)
fig, (a0, a1) = plt.subplots(2, 1, figsize=(6.4, 5.8), sharex=True, gridspec_kw=dict(height_ratios=[1, 2.4]))
a0.bar(np.arange(len(r["P"])) + 0.5, r["P"], width=0.9, color=c1, alpha=0.35, label="Design rainfall")
a0.bar(np.arange(len(r["Pe"])) + 0.5, r["Pe"], width=0.9, color=c1, label="Excess rainfall")
a0.invert_yaxis(); a0.set_ylabel("mm h⁻¹"); a0.legend(loc="lower right", fontsize=9.5)
a1.plot(t, r["Q_out"], color=c1, lw=2.2, label="Catchment outlet")
a1.plot(t, r["Q_town"], color=c2, lw=2.2, label="Town (routed 18 km)")
a1.axhline(r["Qb"], color=ink2, lw=0.9, ls=":", label="Baseflow")
a1.set_xlabel("Time from start of storm (h)"); a1.set_ylabel("Q (m³ s⁻¹)")
a1.set_xlim(0, 60); a1.legend(loc="upper right", fontsize=9.5)
plt.show()
glue("b_out", float(r["Q_out"].max()), display=False); glue("b_town", float(r["Q_town"].max()), display=False)
glue("b_P", float(r["P"].sum()), display=False); glue("b_Pe", float(r["Pe"].sum()), display=False)
```

*Figure 12.2. The 100-year, 12-hour design storm, its excess rainfall, and the resulting flood at the catchment outlet and at the town. The {glue:text}`b_P:.0f` mm storm produces {glue:text}`b_Pe:.0f` mm of excess; the peak of {glue:text}`b_out:.0f` m³ s⁻¹ at the outlet is attenuated to {glue:text}`b_town:.0f` m³ s⁻¹ at the town.*

## 12.4 The critical duration

A short storm is intense but small; a long storm is large but less intense; and a long storm also wets the catchment so that losses become relatively smaller. Which duration gives the largest peak depends on the catchment's response time, its losses and the routing. The chain is therefore run for a range of durations (Figure 12.3), and the **critical duration** is the one with the largest peak at the design point.

```{code-cell} python
:tags: [hide-input]
durations = [2, 3, 4, 9, 12, 15, 18, 24, 36, 48]   # hours
peaks_out = [design_flood(D_h=d)["Q_out"].max() for d in durations]
peaks_town = [design_flood(D_h=d)["Q_town"].max() for d in durations]
d_crit = durations[int(np.argmax(peaks_town))]
fig, ax = plt.subplots(figsize=(6.4, 3.9))
ax.plot(durations, peaks_out, "o-", color=c1, lw=2, label="Catchment outlet")
ax.plot(durations, peaks_town, "s-", color=c2, lw=2, label="Town")
ax.axvline(d_crit, color=ink2, lw=0.8, ls=":")
ax.text(d_crit + 0.8, min(peaks_town) * 1.02, f"critical duration\nat the town: {d_crit} h", color=ink2, fontsize=9.5)
ax.set_xlabel("Storm duration (h)"); ax.set_ylabel("100-year peak (m³ s⁻¹)"); ax.set_xlim(0, 50)
ax.legend(loc="upper right", fontsize=9.5)
plt.show()
glue("d_crit", d_crit, display=False); glue("q_crit", float(max(peaks_town)), display=False)
glue("d_crit_out", durations[int(np.argmax(peaks_out))], display=False)
```

*Figure 12.3. Peak discharge of the 100-year design flood as a function of the storm duration. The critical duration at the town is {glue:text}`d_crit` h, with a peak of {glue:text}`q_crit:.0f` m³ s⁻¹. At the catchment outlet it is {glue:text}`d_crit_out` h; routing favors longer storms, whose floods have more volume and are attenuated less.*

The peak changes little around the critical duration: the curve is flat near its maximum, so an error of a few hours in the duration matters less than the other choices discussed next.

**Why not alternating blocks?** An alternating block storm of 24 hours contains, by construction, the design depth for every shorter duration, including the most intense hour of the 1-hour storm. A longer block storm is therefore a shorter one with more rain added before and after, and with a loss model like the curve number, whose losses shrink as rain accumulates, its peak keeps growing with duration. The search for a critical duration then has no maximum (Example 12.4). Methods that spread each duration's depth with a fixed dimensionless pattern, such as Huff curves, avoid this nesting, which is why they are paired with critical-duration searches in practice.

## 12.5 Sensitivity: which choices matter?

Every link in the chain involves a choice, and each choice is uncertain. The simplest way to see which matter is to change **one at a time** from a baseline and record the change in the design peak (Figure 12.4). The baseline is the critical duration with the choices above.

```{code-cell} python
:tags: [hide-input]
base = design_flood(D_h=d_crit)["Q_town"].max()
cases = [
    ("Antecedent moisture AMC II instead of III", dict(amc=2)),
    ("Storm peak early (25%)", dict(peak_pos=0.25)),
    ("Storm peak late (75%)", dict(peak_pos=0.75)),
    ("Clark unit hydrograph", dict(uh="Clark")),
    ("Snyder unit hydrograph", dict(uh="Snyder")),
    ("tc 30% shorter", dict(tc_factor=0.7)),
    ("tc 30% longer", dict(tc_factor=1.3)),
    ("No areal reduction", dict(use_arf=False)),
    ("Return period 50 years", dict(T=50)),
    ("Return period 200 years", dict(T=200)),
]
changes = [(name, 100 * (design_flood(D_h=d_crit, **kw)["Q_town"].max() / base - 1)) for name, kw in cases]
changes.sort(key=lambda z: abs(z[1]))
fig, ax = plt.subplots(figsize=(6.4, 4.8))
names = [c_[0] for c_ in changes]; vals = [c_[1] for c_ in changes]
ax.barh(names, vals, color=[c2 if v > 0 else c1 for v in vals], height=0.6)
for y, v in enumerate(vals):
    ax.text(v + (1 if v >= 0 else -1), y, f"{v:+.0f}%", va="center", ha="left" if v >= 0 else "right", fontsize=9)
ax.axvline(0, color=ink, lw=0.8)
ax.set_xlabel("Change in the design peak at the town (%)")
lim = max(abs(min(vals)), max(vals)) * 1.35
ax.set_xlim(-lim, lim); ax.grid(axis="y", visible=False); ax.tick_params(axis="y", labelsize=9)
plt.show()
glue("base_peak", float(base), display=False)
for name, v in changes:
    key = {"Antecedent moisture AMC II instead of III": "s_amc", "Clark unit hydrograph": "s_clark",
           "Snyder unit hydrograph": "s_snyder", "Storm peak late (75%)": "s_late"}.get(name)
    if key: glue(key, float(v), display=False)
```

*Figure 12.4. Change in the 100-year peak at the town (baseline {glue:text}`base_peak:.0f` m³ s⁻¹) when one choice at a time is changed. Replacing the NRCS unit hydrograph changes the peak by {glue:text}`s_clark:.0f`% (Clark) and {glue:text}`s_snyder:.0f`% (Snyder); assuming average instead of wet antecedent conditions changes it by {glue:text}`s_amc:.0f`%. Each of these is larger than the effect of halving or doubling the return period.*

Three lessons stand out. First, the **unit hydrograph method** matters a great deal, although every method holds exactly the same volume: the methods disagree about how sharply that volume is released, and the peak is what they disagree about most. Second, the **loss model**, here through the antecedent moisture condition, decides how much of the storm becomes runoff at all. Third, a factor of two in the **return period** changes the peak by less than either of these. The temporal pattern and the time of concentration matter less in this example, but not negligibly. Calibrating the response and the losses against observed floods, in the catchment or in similar gauged catchments, usually pays off more than refining the frequency analysis of rainfall.

## 12.6 Uncertainty: varying everything at once

One-at-a-time changes ignore combinations. A Monte Carlo analysis draws all uncertain inputs together from plausible ranges, runs the chain many times and looks at the spread of the results. The ranges below are judgments, as they always are in practice; the point is the method, not the particular numbers.

```{code-cell} python
rng = np.random.default_rng(12)
n_mc = 1500
peaks_mc = np.empty(n_mc)
for k in range(n_mc):
    c_k = dict(catchment, CN=rng.uniform(65, 75))                 # curve number
    peaks_mc[k] = design_flood(
        D_h=d_crit, c=c_k,
        amc=3 if rng.random() < 0.7 else 2,                       # usually wet, sometimes average
        peak_pos=rng.uniform(0.25, 0.75),                         # timing of the storm peak
        uh=rng.choice(["NRCS", "Clark", "Snyder"]),               # unit hydrograph method
        tc_factor=rng.lognormal(0, 0.2),                          # time of concentration
    )["Q_town"].max()
p5, p50, p95 = np.percentile(peaks_mc, [5, 50, 95])
print(f"Baseline peak: {base:.0f} m³/s")
print(f"Monte Carlo: median {p50:.0f}, 90% range {p5:.0f}–{p95:.0f} m³/s")
```

```{code-cell} python
:tags: [hide-input]
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.hist(peaks_mc, bins=40, color=c1, alpha=0.7)
for v, lab, ls in [(p5, "5%", ":"), (p50, "median", "--"), (p95, "95%", ":")]:
    ax.axvline(v, color=ink, lw=1.2, ls=ls); ax.text(v, ax.get_ylim()[1] * 0.93, f" {lab}", fontsize=9.5)
ax.axvline(base, color=c2, lw=2); ax.text(base, ax.get_ylim()[1] * 0.75, " baseline", color=c2, fontsize=9.5)
ax.set_xlabel("100-year peak at the town (m³ s⁻¹)"); ax.set_ylabel("Number of runs")
plt.show()
glue("mc5", float(p5), display=False); glue("mc50", float(p50), display=False); glue("mc95", float(p95), display=False)
glue("mc_ratio", float(p95 / p5), display=False)
```

*Figure 12.5. Distribution of the 100-year peak at the town from 1500 runs in which the curve number, antecedent condition, storm pattern, unit hydrograph method and time of concentration were varied together. The 90% range, {glue:text}`mc5:.0f`–{glue:text}`mc95:.0f` m³ s⁻¹, spans a factor of {glue:text}`mc_ratio:.1f`.*

The baseline lies near the upper end of the range because it combines wet antecedent conditions with the NRCS unit hydrograph, which has the sharpest peak of the three. The histogram has several humps because two of the inputs are discrete choices.

A range this wide is normal. It does not mean the design is arbitrary: it means the design value is a decision, made with knowledge of the uncertainty. Common responses are to calibrate the most influential parameters against observed floods in the catchment or in similar gauged catchments, to compare the result with a regional flood frequency estimate (from statistical hydrology), to use the upper part of the range where the consequences of failure are severe, and to add freeboard. Whatever is chosen, the report should state the method, every parameter and the range of plausible values.

## 12.7 Worked examples

**Example 12.1 — Checking a design hydrograph by hand.** For the baseline 12-hour run, check (a) that the direct runoff volume equals the excess depth over the catchment, (b) that routing conserves volume, and (c) that the peak lies within a rough bound: no peak can exceed the maximum hourly excess times $A/3.6$ times the peak unit-hydrograph ordinate per mm of excess, summed over hours.

```{code-cell} python
r = design_flood(D_h=12)
V_d = r["Qd"].sum() * 3600
print(f"(a) direct runoff {V_d / 1e6:.2f} million m³ = {V_d / (A * 1e6) * 1000:.1f} mm; excess {r['Pe'].sum():.1f} mm")
V_out, V_town = r["Q_out"].sum() * 3600, r["Q_town"].sum() * 3600
print(f"(b) volume at outlet {V_out / 1e6:.2f}, at town {V_town / 1e6:.2f} million m³ (difference {100 * (V_town / V_out - 1):+.2f}%)")
bound = r["Pe"].sum() * r["U"].max() + r["Qb"]
print(f"(c) peak {r['Q_out'].max():.0f} m³/s <= bound (total excess × peak UH + baseflow) = {bound:.0f} m³/s")
```

The small volume difference in (b) is the water still in the reach at the end of the 96-hour window.

**Example 12.2 — Effect of urbanization.** Part of the catchment is to be developed, raising CN$_{\text{II}}$ from 70 to 76 and shortening $t_c$ by 20%. Estimate the change in the 100-year peak at the town, at the critical duration found above.

```{code-cell} python
before = design_flood(D_h=d_crit)["Q_town"].max()
after = design_flood(D_h=d_crit, c=dict(catchment, CN=76.0), tc_factor=0.8)["Q_town"].max()
print(f"Before {before:.0f} m³/s, after {after:.0f} m³/s ({100 * (after / before - 1):+.0f}%)")
```

**Example 12.4 — The nesting problem of alternating blocks.** Repeat the critical-duration search at the town with alternating block storms (Chapter 11) instead of the Huff-type pattern.

```{code-cell} python
for d in (3, 12, 24, 48):
    hb = design_flood(D_h=d, pattern="block"); hh = design_flood(D_h=d)
    print(f"D = {d:2d} h: block storm peak {hb['Q_town'].max():5.0f} m³/s (max hourly rain {hb['P'].max():5.1f} mm); "
          f"Huff-type {hh['Q_town'].max():5.0f} m³/s (max hourly rain {hh['P'].max():5.1f} mm)")
```

With alternating blocks the peak rises steadily with duration, because every longer storm keeps the same intense core. With the fixed pattern, longer storms are less intense at their peak, and the peak passes through a maximum.

**Example 12.3 — A different return period.** The bridge's approach road is designed for the 50-year flood. Find the 50-year peak at the town at the same critical duration, and the probability that it is exceeded at least once during the bridge's 75-year design life.

```{code-cell} python
q50 = design_flood(T=50, D_h=d_crit)["Q_town"].max()
print(f"50-year peak at the town = {q50:.0f} m³/s; risk over 75 years = {1 - (1 - 1 / 50) ** 75:.2f}")
```

## 12.8 Common mistakes

- **Mixing time steps.** The hyetograph, the unit hydrograph and the routing step must share the same $\Delta t$ (or be converted consistently, Chapter 9).
- **Running only one storm duration.** The critical duration must be searched for, and it may differ between points on the river.
- **Applying point rainfall to a large catchment.** Use areal reduction (Chapter 11).
- **Forgetting baseflow, or adding it twice.** If the unit hydrograph was derived from data, baseflow must be separated with the same method used to derive it.
- **Reporting a single number.** A design flood should be reported with its assumptions and a range.
- **Treating a calibrated parameter set as universal.** Parameters tuned to moderate floods may not hold for the 100-year event (nonlinearity, Chapter 9).

## 12.9 Exercises

```{code-cell} python
:tags: [remove-cell]
glue("e1_q", float(design_flood(D_h=24)["Q_town"].max()), display=False)
glue("e2_q", float(design_flood(D_h=d_crit, amc=2, uh="Clark")["Q_town"].max()), display=False)
glue("e3_pe", float(design_flood(D_h=24)["Pe"].sum()), display=False); glue("e3_p", float(design_flood(D_h=24)["P"].sum()), display=False)
r4 = design_flood(D_h=d_crit, c=dict(catchment, K=6.0, x=0.10))
glue("e4_q", float(r4["Q_town"].max()), display=False)
glue("e5_q", float(design_flood(T=200, D_h=d_crit)["Q_town"].max()), display=False)
```

1. Using the functions of this chapter, find the 100-year peak at the town for a 24-hour storm.
2. Find the peak at the critical duration with AMC II and the Clark unit hydrograph together. Compare with the sum of the two separate changes in Figure 12.4. Why are they not additive?
3. For the 24-hour, 100-year storm, what are the areal rainfall depth and the excess depth? What is the event runoff ratio?
4. If the reach to the town were longer, with $K$ = 6 h ($x$ unchanged), what would the 100-year peak at the town be at the same storm duration? Check that the Muskingum coefficients are non-negative.
5. Find the 200-year peak at the town at the critical duration, and its ratio to the 100-year peak.

```{admonition} Answers
:class: dropdown
1. About {glue:text}`e1_q:.0f` m³ s⁻¹.
2. About {glue:text}`e2_q:.0f` m³ s⁻¹. The effects interact: the loss model changes the excess hyetograph that the unit hydrograph then shapes, so the changes multiply rather than add.
3. {glue:text}`e3_p:.0f` mm of rain, {glue:text}`e3_pe:.0f` mm of excess.
4. About {glue:text}`e4_q:.0f` m³ s⁻¹; with $\Delta t$ = 1 h, $2Kx$ = 1.2 h, so $C_0$ is slightly negative: use $\Delta t$ ≥ 1.2 h or a smaller $x$.
5. {glue:text}`e5_q:.0f` m³ s⁻¹.
```

## 12.10 Further reading

- Chow, V. T., Maidment, D. R., and Mays, L. W. (1988). *Applied Hydrology*. McGraw-Hill. The chapters on hydrologic design and design flows.
- Dingman, S. L. (2015). *Physical Hydrology* (3rd ed.). Waveland Press. The sections on hydrologic modeling and uncertainty.
- 윤용남. *수문학* (in Korean). The chapter on the estimation of design floods.
- US Army Corps of Engineers, Hydrologic Engineering Center. *HEC-HMS Technical Reference Manual*. Davis, California.

---

*Figures in this chapter are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
