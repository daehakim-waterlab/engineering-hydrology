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

# 1. Hydrology as an engineering science

```{admonition} Core ideas
:class: tip
1. Hydrology is the study of how water moves through the land, the atmosphere and the oceans, and how much of it is stored where. **Engineering hydrology** asks three practical questions: *how much* water, *when*, and *how often*.
2. The **catchment** (watershed) is the natural control volume of hydrology: all rain that falls inside its divide and becomes streamflow leaves through one outlet.
3. Every hydrological calculation rests on **conservation of mass**, written as the water balance $dS/dt = P - E - Q$. Energy conservation (Chapter 2) controls how much water evaporates.
4. Hydrologists express volumes as **depths over the catchment** (mm). Converting between m³ s⁻¹ and mm d⁻¹ needs only the catchment area: $1$ m³ s⁻¹ from $1$ km² is $86.4$ mm d⁻¹.
5. Storage in a catchment cannot grow or shrink forever, so over many years **the storage change averages to almost zero** and $\bar{Q} \approx \bar{P} - \bar{E}$.
6. Hydrological processes span seconds to millennia and centimeters to continents. Choosing the right **time and space scale** is the first decision in any analysis.
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

## 1.1 What hydrology is about

**Hydrology** is the science of water on and under the land surface: where it comes from, where it goes, how long it stays, and what it carries. A hydrologist follows a raindrop from the cloud to the ground, into the soil or across the surface, into a stream or back to the air as vapor, and eventually to the sea.

Civil engineers meet hydrology whenever a structure must deal with water that it does not control. A culvert under a road, a storm sewer, a levee, a dam spillway and a water-supply reservoir all have to be sized for water that arrives from a catchment, often in amounts that nobody has measured at that site. **Engineering hydrology** is the part of hydrology that turns physical understanding into numbers for design and operation. It asks three questions:

- **How much?** The volume of a flood, the mean annual flow of a river, the yield of a reservoir.
- **When?** The time at which a flood peaks, how fast it rises and falls, the season of low flows.
- **How often?** The probability that a given flood or drought will be equaled or exceeded in any year.

The third question needs probability and statistics. This book uses the results of frequency analysis (Chapter 11) but does not teach how to fit frequency distributions; that is covered in a separate textbook on statistical hydrology.

Hydrology is younger than hydraulics as a quantitative science. The first measured water balance of a river basin, Pierre Perrault's study of the upper Seine published in 1674, showed that rainfall was more than enough to supply the river (see [*A Brief History of Fluid Mechanics and Water Sciences*](https://daehakim-waterlab.github.io/water-history/history-of-hydrology/#the-birth-of-quantitative-hydrology-perrault-mariotte-and-halley)). Most of the methods in this book were developed in the 19th and 20th centuries by engineers who needed answers before the science was complete. As a result, engineering hydrology mixes physics (conservation of mass and energy, Darcy's law) with empirical formulas whose coefficients come from data. Knowing which is which is an important skill.

## 1.2 The water cycle and the main stores

Water moves continuously between the oceans, the atmosphere and the land. This circulation is the **hydrological cycle** (water cycle). The sun supplies the energy that evaporates water; gravity brings it back down as precipitation and drives it downhill as streamflow and groundwater flow.

It helps to think of the cycle as a set of **stores** (reservoirs) connected by **fluxes** (flows). A store holds a volume of water; a flux moves water from one store to another at a certain rate. Table 1.1 lists the main stores. Almost all of the world's water is in the oceans. Of the fresh water, most is locked in ice sheets and deep groundwater. The water that engineers manage day to day, in rivers, lakes and soils, is a tiny fraction of the total.

```{code-cell} python
:tags: [remove-input]
from IPython.display import Markdown
# Approximate global stores (km³), after Shiklomanov (1993).
stores = [("Oceans", 1_338_000_000), ("Ice sheets, glaciers and permanent snow", 24_064_000),
          ("Groundwater", 23_400_000), ("Ground ice and permafrost", 300_000),
          ("Lakes (fresh and saline)", 176_400), ("Soil moisture", 16_500),
          ("Atmosphere", 12_900), ("Swamps and marshes", 11_470),
          ("Rivers", 2_120), ("Living organisms", 1_120)]
total = sum(v for _, v in stores)
area_earth = 510e6  # km²
rows = [f"| {n} | {v:,.0f} | {100 * v / total:.4g} | {v / area_earth * 1e6:,.3g} |" for n, v in stores]
Markdown("| Store | Volume (km³) | Share of total (%) | Equivalent depth over the Earth (mm) |\n"
         "|---|---:|---:|---:|\n" + "\n".join(rows) +
         f"\n| **Total** | **{total:,.0f}** | **100** | **{total / area_earth * 1e6:,.3g}** |")
```

*Table 1.1. Approximate volumes of water in the main global stores, after Shiklomanov (1993). The last column spreads each volume evenly over the whole surface of the Earth (510 million km²). The atmosphere holds only about 25 mm of water.*

The atmosphere holds little water at any moment, but it moves a great deal of it. Global precipitation is roughly 500,000 km³ per year. Figure 1.1 shows the annual fluxes between the land, the oceans and the atmosphere, rounded from the global estimates of Oki and Kanae (2006). The cycle balances: the oceans lose more by evaporation than they gain by precipitation, and the difference is carried over the land by the atmosphere and returned by rivers and groundwater.

```{code-cell} python
:tags: [hide-input]
from matplotlib.patches import FancyBboxPatch

# Rounded annual fluxes (thousand km³ per year), after Oki and Kanae (2006).
P_land, E_land = 110, 65
P_ocean, E_ocean = 390, 435
R = P_land - E_land                       # runoff from land to ocean
transport = E_ocean - P_ocean             # vapor carried from ocean to land
assert R == transport                     # the cycle closes

fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
def box(x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                fc=color, ec="none", alpha=0.18))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11, color=ink)
box(0.4, 5.5, 9.2, 1.0, "Atmosphere", c1)
box(0.4, 0.4, 3.8, 1.3, "Land", c3)
box(5.8, 0.4, 3.8, 1.3, "Ocean", c1)

def arrow(x0, y0, x1, y1, color, label, dx=0.12, ha="left"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2, mutation_scale=16))
    ax.text((x0 + x1) / 2 + dx, (y0 + y1) / 2, label, color=ink, fontsize=10, ha=ha, va="center")
arrow(1.3, 5.4, 1.3, 1.85, c1, f"P = {P_land}", dx=0.12)
arrow(3.2, 1.85, 3.2, 5.4, c2, f"E = {E_land}", dx=0.12)
arrow(6.8, 5.4, 6.8, 1.85, c1, f"P = {P_ocean}", dx=-0.12, ha="right")
arrow(8.7, 1.85, 8.7, 5.4, c2, f"E = {E_ocean}", dx=-0.12, ha="right")
ax.annotate("", xy=(4.4, 1.05), xytext=(5.6, 1.05),
            arrowprops=dict(arrowstyle="<|-", color=c3, lw=2.2, mutation_scale=16))
ax.text(5.0, 2.05, f"Runoff\n{R}", ha="center", color=ink, fontsize=10)
ax.annotate("", xy=(4.2, 4.85), xytext=(5.8, 4.85),
            arrowprops=dict(arrowstyle="-|>", color=ink2, lw=1.6, ls="--", mutation_scale=14))
ax.text(5.0, 4.2, f"Vapor transport\n{transport}", ha="center", color=ink2, fontsize=10)
ax.text(5.0, 0.0, "Fluxes in thousand km³ per year", ha="center", color=ink2, fontsize=9.5)
plt.show()
```

*Figure 1.1. The global water cycle as three stores connected by annual fluxes (rounded, after Oki and Kanae, 2006). Over the land, precipitation exceeds evaporation; the surplus returns to the ocean as runoff. Over the ocean the reverse holds, and the atmosphere carries the difference to the land.*

A useful number for any store is its **mean residence time**, the average time a water molecule spends in it:

$$
T_r = \frac{S}{F},
$$

where $S$ is the volume in the store and $F$ the flux through it (inflow or outflow, which are equal on average). Water stays in the atmosphere for about nine days (Example 1.4) and in the oceans for thousands of years. Short residence times mean fast response: rivers and the atmosphere change within days, which is why floods are a problem of hours to days, while depleted aquifers recover over decades.

## 1.3 The catchment

A **catchment** (also called a **watershed**, **drainage basin** or **river basin**) is the area of land that drains to a given point on a stream, called the **outlet**. Its boundary is the **drainage divide**, the line of high ground that separates water flowing to the outlet from water flowing elsewhere. The catchment is defined by the outlet: move the outlet downstream and the catchment grows.

On a topographic map, the divide follows ridges and crosses contours at right angles. Streams flow along valley bottoms, where contours form V shapes that point **upstream** (Figure 1.2). The **catchment area** $A$, the plan area inside the divide, is the single most important property of a catchment: a larger area collects more rain and produces more flow.

```{code-cell} python
:tags: [hide-input]
# A synthetic catchment: a teardrop-shaped divide around a valley that drains south.
Lc = 9.0                                            # length of the catchment, km
W = lambda y: 2.3 * np.sin(np.pi * np.clip(y, 0, Lc) / Lc) ** 0.7   # half-width, km
yy = np.linspace(0, Lc, 400)
A_syn = np.trapz(2 * W(yy), yy)                    # catchment area, km²
glue("A_syn", float(A_syn), display=False)

x = np.linspace(-2.5, 2.5, 300); y = np.linspace(0, Lc, 400)
X, Y = np.meshgrid(x, y)
Wg = np.maximum(W(Y), 1e-6)
Z = 100 + 25 * Y + 120 * (np.abs(X) / Wg) ** 1.5    # elevation, m: valley floor rises upstream
Z = np.ma.masked_where(np.abs(X) > W(Y), Z)

fig, ax = plt.subplots(figsize=(5.2, 6.0))
cs = ax.contour(X, Y, Z, levels=np.arange(120, 360, 20), colors=ink2, linewidths=0.6, alpha=0.7)
ax.clabel(cs, levels=cs.levels[::3], fontsize=8, fmt="%d m")
ax.plot(W(yy), yy, color=c2, lw=2.2); ax.plot(-W(yy), yy, color=c2, lw=2.2, label="Drainage divide")
ax.plot(np.zeros(2), [0, 7.6], color=c1, lw=2.4, label="Stream")
for y0, side, length in [(2.0, 1, 1.2), (3.8, -1, 1.4), (5.0, 1, 1.3), (6.3, -1, 0.9)]:
    t = np.linspace(0, 1, 30)
    ax.plot(side * length * (1 - t), y0 + 1.2 * (1 - t) ** 0.5 * 0.9 - 0.2 * t, color=c1, lw=1.5)
ax.plot(0, 0, "o", color=ink, ms=7)
ax.annotate("Outlet (stream gauge)", (0, 0), xytext=(0.35, -0.55), fontsize=10, color=ink)
ax.set_aspect("equal"); ax.set_xlim(-3, 3); ax.set_ylim(-0.9, 9.4)
ax.set_xlabel("Distance (km)"); ax.set_ylabel("Distance (km)")
ax.legend(loc="upper right", fontsize=9.5)
plt.show()
```

*Figure 1.2. A synthetic catchment. The drainage divide (orange) follows the ridges and meets the stream only at the outlet. Contours bend into V shapes that point upstream where they cross the valleys. The area inside the divide is {glue:text}`A_syn:.0f` km².*

Two cautions apply. First, the **topographic divide** drawn from surface elevation need not match the **groundwater divide**, so some water can enter or leave a catchment underground. This matters most in karst (limestone) terrain and in small catchments on permeable rock. Second, in cities, sewers and drainage channels can move water across natural divides, so an urban catchment must be traced from the drainage network as well as from the topography.

## 1.4 The water balance

### Conservation of mass

Take the catchment as a **control volume** that extends from the top of the vegetation down to an impermeable layer below the soil and groundwater. Water enters as precipitation and leaves as evaporation and streamflow. Water is incompressible for our purposes (see Chapter 1 of the [Hydraulics](https://daehakim-waterlab.github.io/hydraulics/) book), so conservation of mass is the same as conservation of volume:

$$
\frac{dS}{dt} = P - E - Q + Q_{g,\mathrm{in}} - Q_{g,\mathrm{out}},
$$ (eq-water-balance)

where

- $S$ is the water stored in the catchment (in the soil, in aquifers, in lakes and channels, as snow and on vegetation);
- $P$ is the precipitation rate;
- $E$ is the **evapotranspiration** rate, the sum of evaporation from wet surfaces and soil and **transpiration** (water taken up by roots and released through leaves);
- $Q$ is the streamflow leaving through the outlet;
- $Q_{g,\mathrm{in}}$ and $Q_{g,\mathrm{out}}$ are groundwater flows across the catchment boundary.

In words: **the rate of change of storage equals inflow minus outflow**. For most catchments the groundwater terms are small and are dropped, giving

$$
\frac{dS}{dt} = P - E - Q.
$$

Over a finite period $\Delta t$ the balance becomes $\Delta S = P - E - Q$, with each term now a total over the period. All terms must be in the same units. Hydrologists usually express them as **depths** over the catchment area (mm) or depth rates (mm d⁻¹), which is the subject of the next section.

The same equation describes any control volume: a reservoir, a lake, a soil column, a city block. Chapter 9 uses it for a "linear reservoir" model of the whole catchment, and Chapter 10 uses it to route a flood through a reservoir.

### Conservation of energy

Evaporation needs energy: about 2.45 MJ to evaporate 1 kg of water at 20 °C. This energy comes mostly from the sun. The **energy balance** of the land surface (Chapter 2) therefore sets a ceiling on $E$, and it couples the water cycle to the climate. Mass and energy conservation are the two physical laws on which the whole of hydrology rests.

## 1.5 Depths, volumes and discharges

Streamflow is measured as a **discharge** in m³ s⁻¹, while precipitation is measured as a depth in mm. To put them in the same equation, divide the volume of flow by the catchment area. The result is the **runoff depth** (or specific discharge), the depth of water that would cover the whole catchment if the flow volume were spread evenly over it.

The conversion follows from the units alone. A flow of $Q$ m³ s⁻¹ leaving a catchment of $A$ km² $= A \times 10^6$ m² corresponds to a depth rate

$$
q = \frac{Q}{A \times 10^{6}}\ \ \text{m s}^{-1}
= \frac{Q}{A \times 10^{6}} \times 1000\ \frac{\text{mm}}{\text{m}} \times 86\,400\ \frac{\text{s}}{\text{d}}
= 86.4\,\frac{Q}{A}\ \ \text{mm d}^{-1}.
$$

The same argument with 3600 s gives the hourly form,

$$
q\ (\text{mm h}^{-1}) = 3.6\,\frac{Q}{A}
\qquad \Longleftrightarrow \qquad
Q\ (\text{m}^3\,\text{s}^{-1}) = \frac{q\,A}{3.6} = 0.278\, q\, A ,
$$ (eq-mmh-to-cms)

with $q$ in mm h⁻¹ and $A$ in km². Equation {eq}`eq-mmh-to-cms` will reappear in Chapter 8 as the heart of the rational method. Remember the key number: **1 mm of water over 1 km² is 1000 m³**.

```{code-cell} python
:tags: [remove-input]
# Conversion factors for a discharge of 1 m³/s from an area of 1 km²
Q, A = 1.0, 1.0
per_s = Q / (A * 1e6) * 1000                     # mm per second
facs = [("mm h⁻¹", per_s * 3600), ("mm d⁻¹", per_s * 86400), ("mm yr⁻¹ (365 d)", per_s * 86400 * 365)]
rows = [f"| {u} | {f:,.4g} | $Q = q A / {f:,.4g}$ |" for u, f in facs]
Markdown("| Depth rate $q$ in | $q$ for 1 m³ s⁻¹ from 1 km² | Inverse ($Q$ in m³ s⁻¹, $A$ in km²) |\n"
         "|---|---:|---|\n" + "\n".join(rows))
```

*Table 1.2. Factors for converting a discharge from a catchment into a depth rate over that catchment. Multiply $Q/A$ (m³ s⁻¹ per km²) by the factor in the second column.*

Depths make catchments of different sizes comparable. A mean flow of 20 m³ s⁻¹ is large for a 200 km² catchment (8.6 mm d⁻¹, a very wet climate) but small for a 5000 km² one (0.35 mm d⁻¹, a dry one). In depth units the comparison is immediate.

## 1.6 Why storage change vanishes over the long term

Storage in a catchment is **bounded**. The soil can hold only so much water before it drains, and it cannot hold less than nothing. Aquifers, lakes and snowpacks behave the same way. Integrate the water balance over $N$ years and divide by $N$:

$$
\frac{S(N) - S(0)}{N} = \bar{P} - \bar{E} - \bar{Q},
$$

where the bars denote annual means. The left side is a bounded number divided by $N$, so it shrinks towards zero as the record grows. For a long record,

$$
\bar{Q} \approx \bar{P} - \bar{E}.
$$ (eq-long-term-balance)

This simple result is very useful. If precipitation and streamflow are measured, Equation {eq}`eq-long-term-balance` gives the long-term evapotranspiration, which is hard to measure directly (Chapter 4). The ratio $\bar{Q}/\bar{P}$ is the long-term **runoff ratio**, a first summary of a catchment's climate and soils.

For a single month, or a single storm, the storage term cannot be ignored. During a summer storm most of the rain may go into storage (wetting the soil) and come out weeks later as baseflow or evapotranspiration. Figure 1.3 shows this with a simple "bucket" model driven by 30 years of synthetic daily weather: storage swings by up to {glue:text}`b_Srange:.0f` mm, but the average imbalance $\overline{P - E - Q}$ shrinks steadily as more years are included.

```{code-cell} python
:tags: [hide-input]
# Synthetic daily weather and a single-bucket catchment, 30 years
rng = np.random.default_rng(42)
nyr = 30; nd = 365 * nyr
doy = np.arange(nd) % 365
season = np.clip(np.sin(2 * np.pi * (doy - 100) / 365), 0, None)  # wet summer
p_wet = 0.18 + 0.30 * season                                       # chance of a wet day
P = np.where(rng.random(nd) < p_wet, rng.exponential(11.0, nd), 0.0)   # mm/d
Ep = 2.4 + 2.0 * np.sin(2 * np.pi * (doy - 100) / 365)             # potential ET, mm/d

Smax, kb = 300.0, 60.0          # bucket capacity (mm) and drainage time constant (d)
S = np.empty(nd + 1); S[0] = 150.0
E = np.empty(nd); Q = np.empty(nd)
for t in range(nd):
    s = S[t] + P[t]
    E[t] = min(Ep[t] * min(1.0, s / (0.6 * Smax)), s)   # ET falls when the soil dries
    s -= E[t]
    over = max(s - Smax, 0.0)                 # spill when the bucket is full
    s -= over
    base = s / kb                             # slow drainage to the stream
    Q[t] = over + base
    S[t + 1] = s - base

closure = abs(S[-1] - S[0] - (P.sum() - E.sum() - Q.sum()))
assert closure < 1e-6, closure                # water balance closes exactly

yr = lambda v: v.reshape(nyr, 365).sum(axis=1)
imb = yr(P) - yr(E) - yr(Q)                   # annual storage change, mm
running = np.cumsum(imb) / np.arange(1, nyr + 1)

fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 5.6))
a1.plot(np.arange(nd + 1) / 365, S, color=c1, lw=0.8)
a1.set_ylabel("Storage S (mm)"); a1.set_xlabel("Year")
a1.set_title("(a) Storage rises and falls every year but stays bounded")
a1.set_xlim(0, nyr)
a2.bar(np.arange(1, nyr + 1), imb, color=c2, alpha=0.45, label="Annual ΔS = P − E − Q")
a2.plot(np.arange(1, nyr + 1), running, color=ink, lw=2, label="Mean over the first N years")
a2.axhline(0, color=ink2, lw=0.8)
a2.set_xlabel("Number of years N"); a2.set_ylabel("mm per year")
a2.set_title("(b) The long-term mean storage change shrinks towards zero")
a2.legend(loc="upper right", fontsize=9.5); a2.set_xlim(0.3, nyr + 0.7)
plt.show()

glue("b_P", float(yr(P).mean()), display=False); glue("b_E", float(yr(E).mean()), display=False)
glue("b_Q", float(yr(Q).mean()), display=False); glue("b_dS", float(abs(running[-1])), display=False)
glue("b_maxdS", float(np.abs(imb).max()), display=False)
glue("b_Srange", float(S.max() - S.min()), display=False)
```

*Figure 1.3. A bucket model of a catchment driven by synthetic daily rainfall with a wet summer. Over 30 years the mean annual precipitation is {glue:text}`b_P:.0f` mm, evapotranspiration {glue:text}`b_E:.0f` mm and streamflow {glue:text}`b_Q:.0f` mm. Single years have storage changes of up to {glue:text}`b_maxdS:.0f` mm, but the 30-year mean storage change is only {glue:text}`b_dS:.1f` mm per year in magnitude.*

How long is "long enough"? It depends on the slowest store. For a catchment on shallow soils, a few years may be enough. For one with a large aquifer, a glacier or a big lake, storage can drift for decades and the long-term balance must be applied with care.

## 1.7 Scales in time and space

Hydrological processes operate over an enormous range of scales (Figure 1.4). Infiltration into a patch of soil changes over minutes and centimeters. A flash flood from a small urban catchment rises and falls within an hour. A flood on a large river takes weeks to pass. Groundwater in a deep aquifer may be thousands of years old.

```{code-cell} python
:tags: [hide-input]
from matplotlib.patches import Rectangle
mn, hr, day, yrs = 60, 3600, 86400, 365.25 * 86400
# (label, time range in s, length range in m, color, label position (time, length))
procs = [
    ("Infiltration\nat a point", (1 * mn, 1 * day), (0.05, 5), c3, (8 * mn, 0.5)),
    ("Overland flow\non a hillslope", (10 * mn, 12 * hr), (10, 500), c1, (1.2 * hr, 70)),
    ("Flash flood,\nsmall catchment", (30 * mn, 1 * day), (1e3, 3e4), c2, (2.7 * hr, 5e3)),
    ("Large-river\nflood", (3 * day, 60 * day), (1e5, 2e6), c2, (14 * day, 4.5e5)),
    ("Seasonal soil\nwater storage", (20 * day, 1 * yrs), (1e2, 3e4), c3, (85 * day, 1.8e3)),
    ("Groundwater\nflow", (1 * yrs, 1e4 * yrs), (1e2, 1e5), c1, (100 * yrs, 3e3)),
]
fig, ax = plt.subplots(figsize=(6.4, 4.6))
for lab, (t0, t1), (l0, l1), col, (tx, ly) in procs:
    ax.add_patch(Rectangle((t0, l0), t1 - t0, l1 - l0, fc=col, ec=col, alpha=0.18, lw=1.2))
    ax.text(tx, ly, lab, ha="center", va="center", fontsize=9, color=ink)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(30, 3e12); ax.set_ylim(0.02, 5e6)
ticks_t = [mn, hr, day, 30 * day, yrs, 100 * yrs, 1e4 * yrs]
ax.set_xticks(ticks_t, ["1 min", "1 h", "1 d", "1 mo", "1 yr", "100 yr", "10⁴ yr"])
ax.set_yticks([0.1, 10, 1e3, 1e5], ["10 cm", "10 m", "1 km", "100 km"])
ax.minorticks_off()
ax.set_xlabel("Time scale"); ax.set_ylabel("Length scale")
plt.show()
```

*Figure 1.4. Typical time and length scales of some hydrological processes (indicative ranges). Fast processes act over small distances and slow processes over large ones, so most processes lie near a diagonal band.*

The choice of scale shapes the analysis. A storm sewer is designed with rainfall intensities over 10 minutes to an hour (Chapter 8). A dam spillway needs a flood hydrograph over days (Chapter 12). A water-supply reservoir is sized with monthly or annual flows over many years. Using daily data to design for a process that happens within an hour leads to serious errors, a theme we return to in Chapter 13.

## 1.8 Where engineers use hydrology

```{list-table} Typical engineering problems and the hydrological questions behind them
:header-rows: 1
:widths: 28 42 30

* - Problem
  - Hydrological question
  - Chapters
* - Culvert, storm sewer, small bridge
  - Peak discharge from a small catchment for a chosen return period
  - 7, 8, 11
* - Levee, river channel, dam spillway
  - Full flood hydrograph (peak, volume, timing) for a design storm, and how it changes downstream or through a reservoir
  - 9, 10, 12
* - Water-supply reservoir
  - Long-term mean flow, its seasonal pattern, evaporation losses
  - 1, 4, 6
* - Irrigation scheme
  - Crop water demand and the rainfall that meets part of it
  - 4, 5
* - Urban and agricultural drainage
  - Excess rainfall that cannot infiltrate, and how fast it must be removed
  - 5, 7, 8
* - Climate adaptation
  - How floods and water availability respond to changes in rainfall and temperature
  - 13
```

*Table 1.3. Where the chapters of this book are used in practice.*

All of these problems share one feature: the engineer must make a decision with incomplete data about a future that is uncertain. A design discharge is not a prediction of what will happen; it is a carefully chosen number that balances the cost of a larger structure against the risk of failure (Chapter 11). Understanding the physical processes behind the number, the subject of Parts I and II, is what allows an engineer to judge whether that number is sensible.

## 1.9 Worked examples

**Example 1.1 — From discharge to depth.** A river gauge records a mean flow of 6.2 m³ s⁻¹ from a catchment of 250 km². Express the flow as a depth rate in mm d⁻¹ and as an annual depth in mm.

```{code-cell} python
Q, A = 6.2, 250.0                  # m³/s, km²
q_day = 86.4 * Q / A               # mm/d
q_year = q_day * 365               # mm/yr
print(f"q = {q_day:.2f} mm/d = {q_year:.0f} mm/yr")
```

About 780 mm of runoff per year. If the catchment receives around 1300 mm of precipitation, more than half of it leaves as streamflow.

**Example 1.2 — Long-term evapotranspiration from a water balance.** A 320 km² catchment has a mean annual precipitation of 1310 mm and a mean flow of 7.1 m³ s⁻¹, both from a 25-year record. Estimate the mean annual evapotranspiration and the runoff ratio.

```{code-cell} python
A, P, Q = 320.0, 1310.0, 7.1
Q_mm = Q * 365.25 * 86400 / (A * 1e6) * 1000   # mm/yr
E = P - Q_mm                                   # long-term balance, dS ≈ 0
print(f"Q = {Q_mm:.0f} mm/yr,  E = P - Q = {E:.0f} mm/yr,  runoff ratio Q/P = {Q_mm / P:.2f}")
```

About 610 mm of evapotranspiration per year. The result depends entirely on the accuracy of $P$ and $Q$: a 5% error in catchment-average precipitation (66 mm) becomes an 11% error in $E$.

**Example 1.3 — A reservoir over ten days.** A reservoir with a surface area of 3.2 km² (assume vertical sides) receives an average inflow of 18 m³ s⁻¹ for 10 days and releases 22 m³ s⁻¹. In the same period 35 mm of rain falls directly on the lake and 28 mm evaporates. By how much does the water level change?

```{code-cell} python
area = 3.2e6                        # m²
dt = 10 * 86400                     # s
I, O = 18.0, 22.0                   # m³/s
P_lake, E_lake = 0.035, 0.028       # m
dV = (I - O) * dt + (P_lake - E_lake) * area     # m³
dh = dV / area
print(f"ΔV = {dV/1e6:.2f} million m³,  Δh = {dh*100:.1f} cm")
```

The level falls by about one meter. Rain and evaporation on the lake surface nearly cancel here, but in a dry summer the evaporation term alone can lower a shallow reservoir by more than 10 cm a month.

**Example 1.4 — Residence times.** Using Table 1.1 and Figure 1.1, estimate the mean residence time of water in the atmosphere, in rivers and in the oceans.

```{code-cell} python
S_atm, S_riv, S_oce = 12_900, 2_120, 1_338_000_000    # km³
P_global = (110 + 390) * 1e3      # km³/yr, total precipitation = outflow from the atmosphere
R_rivers = 45e3                   # km³/yr, runoff from land
E_ocean = 435e3                   # km³/yr
print(f"Atmosphere: {S_atm / P_global * 365:.0f} days")
print(f"Rivers:     {S_riv / R_rivers * 365:.0f} days")
print(f"Oceans:     {S_oce / E_ocean:,.0f} years")
```

Water passes through the atmosphere in about nine days and through river channels in a few weeks. This is why rainfall and streamflow respond so quickly to weather, and why a river's flow during a dry spell must come from slower stores (soil and groundwater, Chapters 5 and 6).

## 1.10 Common mistakes

- **Mixing depth and volume.** A depth (mm) means nothing without the area it covers. Always state "mm over the catchment".
- **Forgetting the factor 10⁶ in km².** 1 km² = 10⁶ m², not 10³ m². The factors 86.4 and 3.6 already include it.
- **Using 365 days for one quantity and 365.25 for another.** The difference is small (0.07%) but can make a balance that should close appear not to. Use one convention throughout.
- **Assuming ΔS = 0 for short periods.** The long-term balance holds only over many years. For a month, a season or a storm, storage change can be the largest term.
- **Confusing the catchment with the stream.** Area, slope and land use refer to the whole area inside the divide, not just the channel.
- **Ignoring the outlet.** A catchment is defined by its outlet. Two gauges on the same river have different catchments.

## 1.11 Exercises

```{code-cell} python
:tags: [remove-cell]
# Answers to the exercises (glued into the dropdown below)
glue("ex1_mm", 3.4 * 86.4 / 85 * 365, display=False)
glue("ex2_Q", (1250 - 560) / 1000 * 540e6 / (365 * 86400), display=False)
q3 = 2.8 * 86.4 / 120 * 30
glue("ex3_Q", q3, display=False)
glue("ex3_dS", 210 - 95 - q3, display=False)
glue("ex3_V", (210 - 95 - q3) / 1000 * 120e6 / 1e6, display=False)
dV4 = (12 - 15) * 10 * 86400 + (0.020 - 0.030) * 2.5e6
glue("ex4_dh", float(abs(dV4 / 2.5e6 * 100)), display=False)
glue("ex5_depth", 12_900 / 510e6 * 1e6, display=False)
glue("ex5_P", 500e3 / 510e6 * 1e6, display=False)
glue("ex5_Tr", (12_900 / 510e6 * 1e6) / (500e3 / 510e6 * 1e6) * 365, display=False)
```

1. A stream drains 85 km² and has a mean flow of 3.4 m³ s⁻¹. What is the mean annual runoff in mm (use 365 days)?
2. A catchment of 540 km² receives 1250 mm of precipitation per year, and its evapotranspiration is estimated at 560 mm per year. What mean discharge (m³ s⁻¹) do you expect at the outlet?
3. During a 30-day month a 120 km² catchment receives 210 mm of rain and loses 95 mm to evapotranspiration, while the mean streamflow is 2.8 m³ s⁻¹. Find the runoff depth and the change in storage, in mm and in million m³. Where might that water be stored?
4. A reservoir with vertical sides and a surface area of 2.5 km² receives 12 m³ s⁻¹ and releases 15 m³ s⁻¹ for 10 days. During this time 20 mm of rain falls on the lake and 30 mm evaporates. Find the change in water level.
5. Spread the water in the atmosphere (Table 1.1) evenly over the Earth's surface and find its depth. Global precipitation is about 500,000 km³ per year; express it as a depth per year over the Earth and estimate the residence time of atmospheric water.

```{admonition} Answers
:class: dropdown
1. About {glue:text}`ex1_mm:.0f` mm per year.
2. About {glue:text}`ex2_Q:.1f` m³ s⁻¹.
3. Runoff {glue:text}`ex3_Q:.0f` mm; storage increases by {glue:text}`ex3_dS:.0f` mm, or {glue:text}`ex3_V:.1f` million m³ (mostly in the soil and groundwater).
4. The level falls by {glue:text}`ex4_dh:.0f` cm.
5. About {glue:text}`ex5_depth:.0f` mm of water; precipitation is about {glue:text}`ex5_P:.0f` mm per year; residence time about {glue:text}`ex5_Tr:.0f` days.
```

## 1.12 Further reading

- Dingman, S. L. (2015). *Physical Hydrology* (3rd ed.). Waveland Press. The opening chapters on the water cycle, catchments and the water balance.
- Chow, V. T., Maidment, D. R., and Mays, L. W. (1988). *Applied Hydrology*. McGraw-Hill. The introductory chapter on the hydrologic cycle and hydrologic systems.
- 윤용남. *수문학* (in Korean). The introductory chapter on the hydrological cycle.
- Oki, T., and Kanae, S. (2006). Global hydrological cycles and world water resources. *Science*, 313, 1068–1072.
- Shiklomanov, I. A. (1993). World fresh water resources. In P. H. Gleick (Ed.), *Water in Crisis: A Guide to the World's Fresh Water Resources*. Oxford University Press.

---

*Figures and tables in this chapter are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
