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

# 7. Excess rainfall

```{admonition} Core ideas
:class: tip
1. Only part of a storm becomes direct runoff. The rest is **abstracted** by interception, depression storage and infiltration. The part that remains is the **excess (effective) rainfall** $P_e$, and its depth equals the depth of direct runoff.
2. The simplest loss models are a **runoff coefficient** (a constant fraction of rain runs off) and the **φ-index** (a constant loss rate). Both are easy to fit to an observed event but say nothing about how losses change from storm to storm.
3. The **NRCS curve number** method relates storm runoff to storm rainfall through a single parameter, the curve number CN, which combines soil type, land cover and antecedent wetness. In SI units the potential retention is $S_{\max} = 25400/\text{CN} - 254$ mm.
4. **Infiltration equations** (Chapter 5) compute losses from soil properties and rainfall intensity, and they distribute the excess correctly in time.
5. Different loss methods can give the same total excess but very different **timing**, and timing controls the flood peak.
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

## 7.1 Losses and excess rainfall

Part III of this book follows a storm through a catchment to the flood it produces. The first step is to decide how much of the rain becomes direct runoff. Hydrologists call the rain that does **not** become direct runoff the **losses** or **abstractions**, although the water is not lost from the catchment: it is stored, evaporated later, or released slowly as baseflow. The losses are:

- **interception**: rain caught by leaves and branches and evaporated from them, typically 1–3 mm per storm for crops and grass and 2–5 mm for forest canopies;
- **depression storage**: water held in surface hollows, typically 1–5 mm on natural surfaces and less than 2 mm on pavement;
- **infiltration**: water entering the soil (Chapter 5), usually by far the largest loss.

What is left is the **excess rainfall** (also called **effective rainfall**) $P_e$. Excess rainfall is the input to the unit hydrograph and other runoff-routing methods (Chapter 9), and its total depth must equal the depth of direct runoff:

$$
P_e = \frac{V_{\text{direct}}}{A} .
$$

This identity is how loss parameters are calibrated: the direct runoff volume is measured at a gauge (after baseflow separation, Chapter 6), divided by the catchment area, and the loss parameter is adjusted until the excess rainfall has the same depth.

A useful way to think about losses is in two parts (Figure 7.1). At the start of a storm, interception, depression storage and the high initial infiltration capacity absorb all the rain; this is the **initial loss**. Afterward, the soil keeps absorbing water at a lower rate, the **continuing loss**. All the methods in this chapter are different ways of describing these two parts.

```{code-cell} python
:tags: [hide-input]
dt = 0.5
t = np.arange(0, 10, dt) + dt / 2
i = np.array([2, 4, 6, 10, 16, 24, 30, 22, 14, 10, 8, 12, 18, 12, 6, 4, 3, 2, 1, 1], float)   # mm/h
IL, CL = 14.0, 7.0                                         # initial loss (mm), continuing loss (mm/h)
P = i * dt
cum = np.cumsum(P)
loss = np.zeros_like(P); excess = np.zeros_like(P); left = IL
for k in range(len(P)):
    take = min(left, P[k]); left -= take
    rest = P[k] - take
    cl = min(rest, CL * dt)
    loss[k] = take + cl; excess[k] = rest - cl
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.bar(t, i, width=dt * 0.92, color=c1, alpha=0.30, label="Rainfall")
ax.bar(t, excess / dt, width=dt * 0.92, color=c1, label="Excess rainfall")
il_end = t[np.argmax(np.cumsum(P) >= IL)]
ax.axvspan(0, il_end + dt / 2, color=c2, alpha=0.08)
ax.text(0.15, 27, "initial loss\n(all rain absorbed)", color=c2, fontsize=9.5)
ax.axhline(CL, color=c2, lw=1.6, ls="--")
ax.text(6.9, CL + 1.0, "continuing loss rate", color=c2, fontsize=9.5)
ax.set_xlabel("Time (h)"); ax.set_ylabel("mm h⁻¹"); ax.set_xlim(0, 10); ax.set_ylim(0, 34)
ax.legend(loc="upper right", fontsize=9.5)
plt.show()
glue("il_P", float(P.sum()), display=False); glue("il_Pe", float(excess.sum()), display=False)
```

*Figure 7.1. An initial loss of 14 mm followed by a continuing loss of 7 mm h⁻¹. Of {glue:text}`il_P:.0f` mm of rain, {glue:text}`il_Pe:.1f` mm becomes excess rainfall.*

## 7.2 The runoff coefficient

The simplest model takes the excess rainfall as a fixed fraction of the rain:

$$
P_e = C\,P ,
$$

where $C$ is the **runoff coefficient**. It is the event runoff ratio of Chapter 6 used as a parameter. It is appropriate for small, highly impervious catchments, where losses are small and nearly proportional to rainfall, and it is the loss model of the rational method (Chapter 8). Typical values range from 0.7–0.95 for roofs and pavement, through 0.3–0.6 for suburban areas, to 0.05–0.35 for lawns, parks and forests, increasing with slope and with soil fineness. A catchment with several land uses has a **composite** coefficient equal to the area-weighted average:

\begin{equation*}
C = \frac{\sum_j C_j A_j}{\sum_j A_j} .
\tag{7.1}
\end{equation*}

The weakness of the runoff coefficient is that real losses are not proportional to rainfall: a small storm on dry soil may produce almost no runoff, while a large storm on wet soil produces much more than the same fraction. For larger and less impervious catchments, $C$ must therefore be chosen for the storm size and the antecedent wetness of interest.

## 7.3 The φ-index

The **φ-index** is the constant loss rate (mm h⁻¹) that, subtracted from the rainfall hyetograph, leaves an excess rainfall equal to the observed direct runoff:

\begin{equation*}
P_e = \sum_k \max\bigl(i_k - \phi,\ 0\bigr)\,\Delta t .
\tag{7.2}
\end{equation*}

Intervals with rainfall intensity below $\phi$ contribute no excess, so $\phi$ cannot be computed in one step: assume which intervals produce excess, solve for $\phi$, and check that the assumption holds. A simple and safe way is to note that the right side of Equation (7.2) decreases steadily as $\phi$ increases, so the root can be found by bisection (Example 7.2).

The φ-index is the infiltration capacity averaged over the storm. It ignores the initial loss and the decline of infiltration capacity during the storm, so it puts too little excess at the beginning and too much at the end. A common refinement, the **initial loss–continuing loss** model of Figure 7.1, removes a fixed depth first and then a constant rate.

## 7.4 The NRCS curve number method

### The runoff equation

The US Soil Conservation Service, now the Natural Resources Conservation Service (NRCS), developed a method in the 1950s to estimate storm runoff from small agricultural catchments where only daily rainfall was available. It rests on one hypothesis, written in terms of depths over the whole storm. After an **initial abstraction** $I_a$ has been satisfied, the ratio of actual to potential retention equals the ratio of actual to potential runoff:

$$
\frac{F}{S_{\max}} = \frac{P_e}{P - I_a},
$$

where $F = P - I_a - P_e$ is the water retained after runoff begins and $S_{\max}$ is the **potential maximum retention**, the most water the catchment could retain once runoff starts. Substituting $F$ and solving for $P_e$:

\begin{equation*}
P_e = \begin{cases}\dfrac{(P - I_a)^2}{P - I_a + S_{\max}} & P > I_a \\[8pt] 0 & P \le I_a\end{cases}
\tag{7.3}
\end{equation*}

From data on small catchments the SCS took $I_a = 0.2\,S_{\max}$, so that

$$
P_e = \frac{(P - 0.2\,S_{\max})^2}{P + 0.8\,S_{\max}} \qquad (P > 0.2\,S_{\max}).
$$

As $P$ grows, $P_e/P \to 1$ and the retention approaches $S_{\max}$: the catchment "fills up". This behavior, which a constant runoff coefficient lacks, is the main strength of the method.

### The curve number

To make $S_{\max}$ easier to tabulate, it is expressed through a dimensionless **curve number** CN between 0 and 100. The original definition in inches is $S_{\max} = 1000/\text{CN} - 10$. In millimeters (1 in = 25.4 mm):

\begin{equation*}
S_{\max} = \frac{25400}{\text{CN}} - 254 \qquad (\text{mm}).
\tag{7.4}
\end{equation*}

CN = 100 means no retention (an impervious or water surface), and lower values mean more retention: CN = 50 gives $S_{\max}$ = 254 mm. The code below checks that the SI form gives exactly the same runoff as the original equation in inches.

```{code-cell} python
def cn_runoff(P, CN, ratio=0.2):
    """NRCS runoff depth (mm) from storm rainfall P (mm); Ia = ratio * Smax."""
    S = 25400.0 / CN - 254.0
    Ia = ratio * S
    P = np.asarray(P, float)
    return np.where(P > Ia, (P - Ia) ** 2 / (P - Ia + S), 0.0)

def cn_runoff_inches(P_in, CN):                    # the original US customary form
    S = 1000.0 / CN - 10.0
    return np.where(P_in > 0.2 * S, (P_in - 0.2 * S) ** 2 / (P_in + 0.8 * S), 0.0)

for P_mm, CN in [(50, 65), (120, 72), (250, 85), (80, 95)]:
    si = cn_runoff(P_mm, CN)
    us = cn_runoff_inches(P_mm / 25.4, CN) * 25.4
    print(f"P = {P_mm:3d} mm, CN = {CN}: SI form {si:7.3f} mm, original form {us:7.3f} mm")
    assert np.isclose(si, us)
```

```{code-cell} python
:tags: [hide-input]
Pg = np.linspace(0, 300, 301)
fig, ax = plt.subplots(figsize=(6.2, 4.4))
cols = [c3, c1, c2, ink2, ink]
for CN, col in zip([50, 65, 80, 90, 98], cols):
    Pe = cn_runoff(Pg, CN)
    ax.plot(Pg, Pe, color=col, lw=2)
    ax.text(302, Pe[-1], f"CN {CN}", va="center", fontsize=9.5, color=col)
ax.plot([0, 300], [0, 300], color=ink2, lw=0.8, ls=":")
ax.text(95, 150, "$P_e = P$", color=ink2, fontsize=9.5, ha="right")
ax.set_xlabel("Storm rainfall P (mm)"); ax.set_ylabel("Direct runoff $P_e$ (mm)")
ax.set_xlim(0, 335); ax.set_ylim(0, 300)
plt.show()
```

*Figure 7.2. Direct runoff from the curve number equation with $I_a = 0.2\,S_{\max}$. No runoff occurs until the rain exceeds the initial abstraction; afterward each curve bends toward a slope of one.*

### Choosing the curve number

CN depends on three things.

1. **Hydrologic soil group.** Soils are classified into four groups by their minimum infiltration rate when wet: group A (sands and gravels, high infiltration) to group D (clays and shallow soils over rock, very low infiltration).
2. **Land cover and treatment.** Impervious surfaces, row crops, pasture, forest, and their condition (for example good or poor grass cover).
3. **Antecedent moisture condition (AMC).** The same catchment produces more runoff when wet.

Tables of CN for combinations of soil group and land cover, for average antecedent conditions (AMC II), are given in the NRCS National Engineering Handbook (Part 630, Chapter 9) and in national design guidelines, including Korea's. Values range from about 30 for good forest on group A soils to 98 for pavement. For catchments with several land uses, it is more accurate to compute the runoff for each land use and add the volumes than to average the curve numbers first, because Equation (7.3) is nonlinear (Example 7.3).

**Antecedent moisture.** In the traditional procedure, the 5-day rainfall before the storm determines the antecedent moisture class. Converted from the original thresholds in inches, AMC I (dry) applies when the 5-day total is below 13 mm in the dormant season or 36 mm in the growing season, and AMC III (wet) when it exceeds 28 mm or 53 mm respectively; AMC II lies between. Curve numbers for AMC II are converted to the other classes with the widely used approximations

\begin{equation*}
\text{CN}_{\text{I}} = \frac{4.2\,\text{CN}_{\text{II}}}{10 - 0.058\,\text{CN}_{\text{II}}},
\qquad
\text{CN}_{\text{III}} = \frac{23\,\text{CN}_{\text{II}}}{10 + 0.13\,\text{CN}_{\text{II}}} .
\tag{7.5}
\end{equation*}

```{code-cell} python
thresholds_in = {"dormant": (0.5, 1.1), "growing": (1.4, 2.1)}     # AMC I upper / AMC III lower, inches
for season, (lo, hi) in thresholds_in.items():
    print(f"{season:8s}: AMC I if 5-day rain < {lo * 25.4:.1f} mm, AMC III if > {hi * 25.4:.1f} mm")
cn1 = lambda cn: 4.2 * cn / (10 - 0.058 * cn)
cn3 = lambda cn: 23 * cn / (10 + 0.13 * cn)
for cn in (60, 72, 85):
    print(f"CN_II = {cn}: CN_I = {cn1(cn):.1f}, CN_III = {cn3(cn):.1f}")
```

NRCS guidance has since moved away from tying AMC to the 5-day rainfall, because that rule explains only part of the event-to-event variation. Studies of measured rainfall–runoff data (for example Woodward et al., 2003) also found that $I_a \approx 0.05\,S_{\max}$ fits better than $0.2\,S_{\max}$, with correspondingly adjusted curve numbers. For design, follow the version required by the governing guideline and use it consistently.

### Excess rainfall hyetograph

Equation (7.3) gives the storm total. To distribute the excess in time, apply it to the **cumulative** rainfall at the end of each interval and take differences:

$$
P_{e,k} = P_e\!\left(\sum_{j \le k} P_j\right) - P_e\!\left(\sum_{j \le k-1} P_j\right).
$$

This produces no excess until the cumulative rainfall exceeds $I_a$, and then an increasing fraction of each interval's rain, since the loss rate implied by the method decreases with time. Example 7.4 does the calculation.

## 7.5 Losses from infiltration equations

The infiltration equations of Chapter 5 can be applied directly to a hyetograph. For each short time step, the infiltration capacity is computed from the water already infiltrated; the actual infiltration is the smaller of the rainfall and the capacity; and the remainder is excess. With the Green–Ampt model:

$$
f_p = K_s\left(1 + \frac{\psi_f\,\Delta\theta}{F}\right),
\qquad
f = \min(i,\ f_p),
\qquad
i_e = i - f ,
$$

with $F$ updated at each step. When rainfall pauses, capacity does not recover in this simple form, which is acceptable within a single storm. Because the infiltration capacity depends on intensity relative to the soil, this approach puts the excess where the rain is most intense, which is important for small catchments with infiltration-excess runoff. Its parameters come from soil data, so it can be applied to ungauged catchments and to land-use scenarios, but they vary greatly in space, and catchment-scale values usually need calibration.

Figure 7.3 applies four loss methods to the same storm, each calibrated to produce the same total excess.

```{code-cell} python
:tags: [hide-input]
from scipy.optimize import brentq

def phi_excess(i, dt, phi):
    return np.clip(i - phi, 0, None) * dt

def cn_excess(P, CN):
    cum = cn_runoff(np.cumsum(P), CN)
    return np.diff(np.r_[0.0, cum])

def ga_excess(i, dt, Ks, psi_f, dth, sub=30):
    """Green–Ampt excess per interval, with sub-steps inside each interval."""
    a, F, out = psi_f * dth, 1e-6, []
    h = dt / sub
    for ik in i:
        e = 0.0
        for _ in range(sub):
            fp = Ks * (1 + a / F)
            f = min(ik, fp)
            F += f * h; e += (ik - f) * h
        out.append(e)
    return np.array(out)

# The storm of Figure 7.1
target = 40.0                                              # mm of direct runoff to match
phi = brentq(lambda p: phi_excess(i, dt, p).sum() - target, 0, i.max())
CN_fit = brentq(lambda c: cn_excess(P, c).sum() - target, 30, 99.9)
Ks_fit = brentq(lambda k: ga_excess(i, dt, k, 110.0, 0.30).sum() - target, 0.1, 40)
C_fit = target / P.sum()
methods = [("Runoff coefficient", C_fit * P, f"C = {C_fit:.2f}"),
           ("φ-index", phi_excess(i, dt, phi), f"φ = {phi:.1f} mm h⁻¹"),
           ("Curve number", cn_excess(P, CN_fit), f"CN = {CN_fit:.1f}"),
           ("Green–Ampt", ga_excess(i, dt, Ks_fit, 110.0, 0.30), f"$K_s$ = {Ks_fit:.1f} mm h⁻¹")]
fig, axes = plt.subplots(4, 1, figsize=(6.2, 7.6), sharex=True, sharey=True)
for ax, (name, ex, par) in zip(axes, methods):
    ax.bar(t, i, width=dt * 0.92, color=c1, alpha=0.25)
    ax.bar(t, ex / dt, width=dt * 0.92, color=c1)
    cen = (ex * t).sum() / ex.sum()
    ax.axvline(cen, color=c2, lw=1.4, ls="--")
    ax.set_title(f"{name} ({par}); centroid at {cen:.2f} h", fontsize=10)
    ax.set_ylabel("mm h⁻¹")
axes[-1].set_xlabel("Time (h)"); axes[-1].set_xlim(0, 10)
plt.show()
glue("m_phi", float(phi), display=False); glue("m_cn", float(CN_fit), display=False)
glue("m_ks", float(Ks_fit), display=False); glue("m_C", float(C_fit), display=False)
glue("m_maxC", float((C_fit * P).max() / dt), display=False)
glue("m_maxGA", float(methods[3][1].max() / dt), display=False)
```

*Figure 7.3. Four loss methods applied to the storm of Figure 7.1, each adjusted so that the total excess is 40 mm. The light bars are rainfall, the dark bars excess rainfall, and the dashed line the centroid of the excess. The runoff coefficient spreads excess over the whole storm (its maximum is {glue:text}`m_maxC:.1f` mm h⁻¹), whereas the φ-index and Green–Ampt concentrate it in the most intense half-hours ({glue:text}`m_maxGA:.1f` mm h⁻¹ for Green–Ampt). The curve number method delays the excess toward the end of the storm.*

The same 40 mm of runoff, routed through the same catchment (Chapter 9), would produce different peaks because the excess has different intensities and timing. For design, the loss method is as important a choice as the unit hydrograph, a point that Chapter 12 examines quantitatively.

```{list-table} Summary of loss methods
:header-rows: 1
:widths: 22 28 25 25

* - Method
  - Parameters
  - Strengths
  - Weaknesses
* - Runoff coefficient
  - $C$
  - Simple; suits impervious urban areas
  - Ignores storm size and timing of losses
* - φ-index
  - $\phi$ (often with an initial loss)
  - Easy to fit to one event
  - Constant rate; varies from event to event
* - NRCS curve number
  - CN (soil, land cover, AMC)
  - Widely tabulated; nonlinear with storm depth
  - Empirical; timing of excess is a by-product
* - Green–Ampt
  - $K_s$, $\psi_f$, $\Delta\theta$
  - Physical; responds to intensity and wetness
  - Parameters vary greatly in space
```

*Table 7.1. Loss methods compared.*

## 7.6 Worked examples

**Example 7.1 — Composite runoff coefficient.** A 2.4 km² drainage area consists of 0.6 km² of roofs and roads ($C$ = 0.90), 1.1 km² of residential gardens ($C$ = 0.35) and 0.7 km² of park ($C$ = 0.15). Find the composite runoff coefficient and the direct runoff from 45 mm of rain.

```{code-cell} python
Aj = np.array([0.6, 1.1, 0.7]); Cj = np.array([0.90, 0.35, 0.15])
C = (Cj * Aj).sum() / Aj.sum()
print(f"C = {C:.3f}; direct runoff = {C * 45:.1f} mm = {C * 45 / 1000 * Aj.sum() * 1e6:,.0f} m³")
```

**Example 7.2 — φ-index from an observed event.** A storm gave the hourly rainfall below on a 36 km² catchment. After baseflow separation, the direct runoff volume was 0.79 million m³. Find the φ-index.

```{code-cell} python
ih = np.array([4.0, 11.0, 23.0, 17.0, 9.0, 5.0])   # mm in each hour (= mm/h)
Pe_obs = 0.79e6 / 36e6 * 1000                    # mm
phi = brentq(lambda p: phi_excess(ih, 1.0, p).sum() - Pe_obs, 0, ih.max())
print(f"Observed direct runoff = {Pe_obs:.2f} mm")
print(f"φ = {phi:.2f} mm/h; excess by hour = {np.round(phi_excess(ih, 1.0, phi), 2)}")
print(f"check: total excess = {phi_excess(ih, 1.0, phi).sum():.2f} mm")
```

Only the three wettest hours produce excess, so the first and last hours were rightly treated as fully absorbed.

**Example 7.3 — Curve number runoff and antecedent moisture.** A catchment has CN = 72 under average conditions. Find the direct runoff from a 120 mm storm for AMC II and AMC III. Then suppose the catchment is 30% paved (CN = 98) and 70% woodland (CN = 60); compare the runoff computed from the area-weighted CN with the area-weighted runoff.

```{code-cell} python
P_storm = 120.0
for label, cn in [("AMC II", 72.0), ("AMC III", cn3(72.0))]:
    S = 25400 / cn - 254
    print(f"{label}: CN = {cn:.1f}, S = {S:.0f} mm, Ia = {0.2 * S:.0f} mm, Pe = {cn_runoff(P_storm, cn):.1f} mm")
w = np.array([0.3, 0.7]); cns = np.array([98.0, 60.0])
cn_avg = (w * cns).sum()
print(f"Weighted CN = {cn_avg:.1f} -> Pe = {cn_runoff(P_storm, cn_avg):.1f} mm")
print(f"Weighted runoff = {(w * cn_runoff(P_storm, cns)).sum():.1f} mm")
```

The wet antecedent condition raises the runoff by more than half. Averaging the curve numbers underestimates runoff from the mixed catchment, because the paved area produces runoff from the very start of the storm while the woodland does not.

**Example 7.4 — Excess rainfall hyetograph by the curve number method.** Distribute the excess for the hourly storm of Example 7.2 with CN = 80.

```{code-cell} python
cum_P = np.cumsum(ih)
cum_Pe = cn_runoff(cum_P, 80.0)
inc = np.diff(np.r_[0, cum_Pe])
print("hour  P    cum P   cum Pe   Pe in hour")
for k in range(len(ih)):
    print(f"{k + 1:3d} {ih[k]:5.1f} {cum_P[k]:7.1f} {cum_Pe[k]:8.2f} {inc[k]:9.2f}")
print(f"S = {25400 / 80 - 254:.1f} mm, Ia = {0.2 * (25400 / 80 - 254):.1f} mm")
```

The first two hours satisfy most of the initial abstraction. The excess in hour 5 is larger than in hour 2 although the rainfall is similar, because the retention is filling up.

**Example 7.5 — Green–Ampt excess for the same storm.** Compute the excess for the storm of Example 7.2 on a soil with $K_s$ = 6 mm h⁻¹, $\psi_f$ = 90 mm and $\Delta\theta$ = 0.25.

```{code-cell} python
ex_ga = ga_excess(ih, 1.0, 6.0, 90.0, 0.25, sub=120)
print("excess by hour:", ", ".join(f"{e:.2f}" for e in ex_ga), f"| total = {ex_ga.sum():.1f} mm of {ih.sum():.0f} mm")
```

## 7.7 Common mistakes

- **Applying the curve number equation to each interval separately.** The equation relates cumulative quantities; apply it to cumulative rainfall and take differences.
- **Mixing inch and millimeter forms.** Use $S_{\max} = 25400/\text{CN} - 254$ with $P$ in mm, or the original inch form throughout, never a mixture.
- **Averaging curve numbers across very different land uses.** Compute runoff for each and add volumes.
- **Treating a calibrated φ or $C$ as a catchment constant.** Both change with antecedent wetness and storm size.
- **Forgetting that excess depth must equal direct runoff depth.** Check every calibration with this identity.
- **Using the curve number method for very small storms or for timing.** It was built for daily storm totals; its implied loss rates within a storm are a by-product, not a calibrated feature.

## 7.8 Exercises

```{code-cell} python
:tags: [remove-cell]
glue("y1_C", (0.85 * 0.4 + 0.30 * 1.2 + 0.10 * 0.9) / 2.5, display=False)
ih2 = np.array([6.0, 18.0, 26.0, 12.0, 4.0])
glue("y2_phi", brentq(lambda p: phi_excess(ih2, 1.0, p).sum() - 22.0, 0, ih2.max()), display=False)
glue("y3_S", 25400 / 68 - 254, display=False); glue("y3_Pe", float(cn_runoff(95.0, 68.0)), display=False)
glue("y3_Pe3", float(cn_runoff(95.0, cn3(68.0))), display=False); glue("y3_cn3", float(cn3(68.0)), display=False)
glue("y4_Pe", float(cn_runoff(30.0, 85.0)), display=False); glue("y4_Ia", 0.2 * (25400 / 85 - 254), display=False)
glue("y5_cn", float(brentq(lambda c: cn_runoff(150.0, c) - 60.0, 40, 99.9)), display=False)
```

1. A 2.5 km² area has 0.4 km² of pavement ($C$ = 0.85), 1.2 km² of residential land ($C$ = 0.30) and 0.9 km² of forest ($C$ = 0.10). Find the composite runoff coefficient.
2. Hourly rainfalls of 6, 18, 26, 12 and 4 mm produced 22 mm of direct runoff. Find the φ-index.
3. For CN = 68, find $S_{\max}$ and the direct runoff from a 95 mm storm under AMC II, and the runoff under AMC III.
4. A storm of 30 mm falls on an area with CN = 85. How much direct runoff is produced? What is the initial abstraction?
5. A 150 mm storm produced 60 mm of direct runoff on a gauged catchment. What curve number does this event imply?

```{admonition} Answers
:class: dropdown
1. $C$ = {glue:text}`y1_C:.3f`.
2. φ = {glue:text}`y2_phi:.2f` mm h⁻¹.
3. $S_{\max}$ = {glue:text}`y3_S:.1f` mm; $P_e$ = {glue:text}`y3_Pe:.1f` mm (AMC II); CN$_{\text{III}}$ = {glue:text}`y3_cn3:.1f`, $P_e$ = {glue:text}`y3_Pe3:.1f` mm.
4. $I_a$ = {glue:text}`y4_Ia:.1f` mm; $P_e$ = {glue:text}`y4_Pe:.1f` mm.
5. CN ≈ {glue:text}`y5_cn:.1f`.
```

## 7.9 Further reading

- Dingman, S. L. (2015). *Physical Hydrology* (3rd ed.). Waveland Press. The sections on rainfall–runoff relations and event-response models.
- Chow, V. T., Maidment, D. R., and Mays, L. W. (1988). *Applied Hydrology*. McGraw-Hill. The chapter on surface water, including abstractions and the SCS method.
- 윤용남. *수문학* (in Korean). The chapter on effective rainfall and losses.
- USDA Natural Resources Conservation Service. *National Engineering Handbook*, Part 630 Hydrology, Chapters 9 (Hydrologic soil-cover complexes) and 10 (Estimation of direct runoff from storm rainfall).
- Woodward, D. E., Hawkins, R. H., Jiang, R., Hjelmfelt, A. T., Van Mullem, J. A., and Quan, Q. D. (2003). Runoff curve number method: examination of the initial abstraction ratio. *World Water & Environmental Resources Congress 2003*, ASCE.

---

*Figures in this chapter are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
