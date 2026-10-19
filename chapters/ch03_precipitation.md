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

# 3. Precipitation

```{admonition} Core ideas
:class: tip
1. Precipitation needs four things: **moist air, cooling to saturation, nuclei on which vapor can condense, and a way for tiny cloud droplets to grow** into drops heavy enough to fall.
2. Air is cooled mostly by being **lifted**: along fronts, by convection, over mountains, or in the spiraling inflow of a tropical cyclone. The lifting mechanism sets the intensity, duration and extent of the rain.
3. A rain gauge samples a few hundred square centimeters of a catchment. Its record contains **systematic errors** (mostly wind-induced undercatch) and **inconsistencies** (gauge moves, new instruments) that must be checked, for example with a **double-mass curve**.
4. Missing values can be estimated from neighboring gauges, most simply by **inverse distance weighting**.
5. Hydrology needs the **areal** average rainfall over a catchment. The Thiessen and isohyetal methods weight each gauge by the area it represents.
6. Snow is stored on the ground and released later. Its water content is the **snow water equivalent**, and a simple **degree-day** model estimates how fast it melts.
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

## 3.1 How precipitation forms

**Precipitation** is water in liquid or solid form that falls from the atmosphere and reaches the ground. Clouds are common, but most clouds do not produce precipitation. Four conditions must be met at the same time.

1. **A supply of moist air.** The atmosphere over a region holds only a few tens of millimeters of water (Chapter 1), far less than a large storm delivers. Heavy rain therefore needs a continuous inflow of moist air, usually from a warm ocean.
2. **Cooling to saturation.** Air becomes saturated when it cools to its dew point (Chapter 2). Radiative cooling at night produces dew and fog but very little precipitation. Large amounts of cooling come from **lifting**: rising air expands and cools at about 10 °C per km.
3. **Condensation nuclei.** Vapor does not easily condense in perfectly clean air. It condenses on tiny particles of sea salt, dust, smoke and sulfate, between about 0.1 and 1 µm across. The atmosphere almost always contains enough of them.
4. **Growth of droplets to falling size.** A cloud droplet is about 10 µm across and falls at about 1 cm s⁻¹, so it evaporates before it can reach the ground. A raindrop of 1–2 mm contains about a million times more water. Droplets grow fast enough in two ways. In warm clouds, larger droplets fall faster, collide with smaller ones and merge (**collision–coalescence**). In clouds colder than 0 °C, ice crystals and supercooled droplets coexist; because the saturation vapor pressure over ice is lower than over water, vapor moves from the droplets to the crystals, which grow quickly into snowflakes (the **ice-crystal process**). The snowflakes may melt into rain on the way down.

Condition 4 is why the details of cloud physics matter little to engineers once rain is falling, while conditions 1 and 2 control how much falls, where and for how long. They depend on the weather systems described next.

## 3.2 Types of precipitation

Hydrologists classify precipitation by the mechanism that lifts the air. Each type has a typical intensity, duration and areal extent, and therefore a typical effect on floods.

**Frontal (cyclonic) precipitation.** A **front** is the boundary between two air masses of different temperature and humidity. Along a **warm front**, warm air glides up over a wedge of colder air on a gentle slope; the result is widespread, moderate, long-lasting rain or snow. Along a **cold front**, cold air pushes under warm air and forces it up steeply; the rain is heavier, shorter and confined to a narrower band. Mid-latitude cyclones carry both kinds of front and produce most of the rain of mid-latitude winters.

**Convective precipitation.** When the ground is heated strongly, the air near it becomes warmer and lighter than the air above and rises in plumes. If the air is humid, it becomes saturated, condensation releases latent heat, and the cloud can grow explosively into a **cumulonimbus** (thunderstorm). Convective rain is intense (often above 50 mm h⁻¹), brief (less than an hour or two at a point) and local (a few kilometers to a few tens of kilometers across). It produces the flash floods that small urban catchments are designed for (Chapter 8).

**Orographic precipitation.** Mountains force moist air to rise on their windward side (Figure 2.3). Precipitation increases with elevation on that side and decreases sharply on the lee side, in the **rain shadow**. Orography also enhances frontal and convective rain, so mountain catchments typically receive more precipitation than the lowland gauges around them suggest.

**Tropical cyclones.** Tropical cyclones (called **typhoons** in the western North Pacific) form over oceans warmer than about 26 °C. Warm, humid air spirals in toward a low-pressure center and rises in a ring of towering clouds around the calm **eye**, releasing enormous amounts of latent heat that sustain the storm. They combine strong wind, storm surge and very heavy rain over hundreds of kilometers. When a typhoon meets a mountain range, the orographic lift can produce several hundred millimeters of rain in a day. In Korea, typhoons arrive mainly in late summer and early autumn and cause many of the largest floods.

**The East Asian summer rainy season.** In early summer, a nearly stationary front forms between warm, humid air on the western flank of the North Pacific subtropical high and cooler air to the north. Along this front, called *Changma* in Korea (*Meiyu* in China and *Baiu* in Japan), a long belt of cloud brings repeated heavy rain for several weeks. The front moves north as the subtropical high expands, so the rainy period begins earlier in the south. Its rain falls on soils that are progressively wetter, so floods during Changma are often caused by several days of rain rather than by a single short burst. Embedded convective cells can still produce very intense rain locally.

**Forms of precipitation.** Liquid forms are **rain** (drops larger than 0.5 mm) and **drizzle** (smaller drops falling slowly from low clouds). Solid forms are **snow** (ice crystals, often joined into flakes), **graupel** (soft pellets of rimed snow) and **hail** (hard balls of ice grown in the strong updrafts of thunderstorms). **Sleet** and **freezing rain** form when snow melts in a warm layer aloft and refreezes, or arrives supercooled, near the ground. For most hydrological purposes all forms are converted to the depth of liquid water they contain.

## 3.3 Measuring precipitation at a point

### Rain gauges

A rain gauge collects the precipitation falling on a horizontal opening of known area and measures its depth. The Korean *cheugugi* of 1441 was among the earliest standardized gauges used in a network (see [*A Brief History of Fluid Mechanics and Water Sciences*](https://daehakim-waterlab.github.io/water-history/history-of-hydrology/#koreas-rain-gauge-and-river-gauge)). Today's types include:

- **Non-recording (manual) gauges**, read once a day, which give daily totals;
- **Tipping-bucket gauges**, in which a small two-sided bucket tips each time it collects a fixed depth (often 0.5 mm) and the time of each tip is logged;
- **Weighing gauges**, which record the accumulated weight of the catch and work for both rain and snow;
- **Optical and acoustic disdrometers**, which measure the size and speed of individual drops.

Recording gauges give **rainfall intensity**, the rate in mm h⁻¹, which is needed for the design of urban drainage (Chapters 8 and 11).

### Errors in gauge records

A gauge measures the rain that enters its opening, not the rain that would have reached the ground. The main errors are systematic and make gauges read low:

- **Wind-induced undercatch.** The gauge itself disturbs the airflow, which speeds up over the opening and carries drops past it. The loss grows with wind speed and is much larger for snow than for rain: typically a few percent for rain, but tens of percent for snow in strong winds. Wind shields and gauges set in pits reduce it.
- **Wetting and evaporation losses.** Water that wets the funnel and container, or evaporates before being read, is lost; typically a fraction of a millimeter per event.
- **Tipping-bucket losses at high intensity.** Water keeps arriving while the bucket is tipping, so a tipping bucket reads low in very heavy rain unless it is calibrated dynamically.
- **Splash and siting.** Drops can splash in or out, and a gauge too close to trees or buildings is sheltered from part of the rain.

There are also random and human errors: misreadings, blocked funnels, wrongly recorded dates. And a record can be **inconsistent** even if each reading is correct: if the gauge is moved, re-exposed to wind by the removal of a nearby building, or replaced by a different type, the record before and after the change measures slightly different things. Section 3.4 shows how to detect this.

### Radar and satellites

A **weather radar** sends out microwave pulses and measures the power reflected by raindrops, the **reflectivity** $Z$ (mm⁶ m⁻³, usually expressed in decibels, $\text{dBZ} = 10 \log_{10} Z$). Since reflectivity depends on the sixth power of drop diameter while rainfall rate depends on about the third to fourth power, the two are related empirically by a **Z–R relationship**, $Z = a R^b$. The classical values $a = 200$ and $b = 1.6$, with $R$ in mm h⁻¹, are due to Marshall and Palmer (1948). Radar sees the spatial pattern of a storm every few minutes, which no gauge network can do, but its rainfall amounts must be adjusted to gauges. Satellites estimate rainfall from cloud-top temperature and from microwave emission, which is valuable over oceans and in sparsely gauged regions.

## 3.4 Checking consistency: the double-mass curve

If two gauges are exposed to the same climate, their cumulative totals should grow in proportion. The **double-mass curve** plots the cumulative precipitation at the station being tested against the cumulative mean of several nearby **base stations** that are known to be reliable. Year-to-year weather affects both, so a consistent record plots as a straight line. A **break in slope** marks a change at the tested station.

To make the record consistent, the data from the period before (or after) the break are multiplied by the ratio of the slopes. If the current conditions are the reference,

$$
P_{\text{adjusted}} = \frac{m_{\text{current}}}{m_{\text{earlier}}}\, P_{\text{earlier}},
$$

where $m$ is the slope of each segment. A break should be accepted only if it is persistent (at least five years or so) and, ideally, explained by the station history. Random scatter of single years is not a reason to adjust.

```{code-cell} python
:tags: [hide-input]
# 30 years of synthetic annual totals: a regional signal plus local noise.
rng = np.random.default_rng(7)
years = np.arange(1995, 2025)
regional = 1300 * (1 + 0.18 * rng.standard_normal(len(years)))      # mm, shared by all stations
base = np.array([regional * f * (1 + 0.05 * rng.standard_normal(len(years)))
                 for f in (0.95, 1.05, 1.00, 0.92)])                  # four base stations
test = regional * 1.08 * (1 + 0.05 * rng.standard_normal(len(years)))
change = 2010                                                         # gauge moved to a windier site
test = np.where(years >= change, 0.86 * test, test)

cum_base, cum_test = np.cumsum(base.mean(axis=0)), np.cumsum(test)
k = np.searchsorted(years, change)
m1 = np.polyfit(cum_base[:k], cum_test[:k], 1)[0]                     # slope before the change
m2 = np.polyfit(cum_base[k - 1:], cum_test[k - 1:], 1)[0]             # slope after
factor = m2 / m1                                                      # adjust the earlier years

fig, ax = plt.subplots(figsize=(6.0, 4.4))
ax.plot(cum_base / 1000, cum_test / 1000, "o-", color=c1, ms=4, lw=1.4, label="Tested station")
x = np.array([0, cum_base[k - 1]])
ax.plot(x / 1000, m1 * x / 1000, color=ink2, lw=1, ls="--")
ax.plot(cum_base[k - 1:] / 1000, (cum_test[k - 1] + m1 * (cum_base[k - 1:] - cum_base[k - 1])) / 1000,
        color=ink2, lw=1, ls="--", label="Extension of the earlier slope")
ax.annotate(f"break in {change}", (cum_base[k - 1] / 1000, cum_test[k - 1] / 1000),
            xytext=(cum_base[k - 1] / 1000 - 13, cum_test[k - 1] / 1000 + 6), fontsize=10,
            arrowprops=dict(arrowstyle="-|>", color=c2, lw=1.2), color=c2)
ax.text(5, 30, f"slope {m1:.2f}", color=ink, fontsize=10)
ax.text(28, 26, f"slope {m2:.2f}", color=ink, fontsize=10)
ax.set_xlabel("Cumulative mean of base stations (m)")
ax.set_ylabel("Cumulative tested station (m)")
ax.set_xlim(0, None); ax.set_ylim(0, None)
ax.legend(loc="upper left", fontsize=9.5)
plt.show()
glue("dm_m1", float(m1), display=False); glue("dm_m2", float(m2), display=False)
glue("dm_f", float(factor), display=False)
```

*Figure 3.1. A double-mass curve for synthetic data. The tested gauge was moved to a windier site in 2010 and has caught about 14% less since then. The slope drops from {glue:text}`dm_m1:.2f` to {glue:text}`dm_m2:.2f`, so the totals before 2010 must be multiplied by {glue:text}`dm_f:.3f` to be consistent with the present site.*

## 3.5 Estimating missing data

Gauges fail, and records have gaps. A missing value at station $x$ is estimated from nearby stations $i = 1, \dots, n$ that have data for the same period.

**Inverse distance weighting (IDW)** gives each neighbor a weight that falls with its distance $d_i$ from the station:

\begin{equation*}
\hat{P}_x = \frac{\sum_{i=1}^{n} w_i P_i}{\sum_{i=1}^{n} w_i}, \qquad w_i = \frac{1}{d_i^{\,b}} ,
\tag{3.1}
\end{equation*}

where the exponent $b$ is usually 2. Close stations dominate the estimate, and the estimate always lies between the smallest and largest neighboring values. IDW ignores differences in climate between stations, which is a problem in mountainous terrain.

**The normal-ratio method** handles that problem when the stations have quite different long-term means. Each neighbor's value is scaled by the ratio of the long-term (normal) annual precipitation $N$ at the two sites:

$$
\hat{P}_x = \frac{1}{n} \sum_{i=1}^{n} \frac{N_x}{N_i}\, P_i .
$$

A common rule is to use the simple average of neighbors when the normals differ by less than about 10%, and the normal-ratio method otherwise. Both methods can be combined: scale by the normal ratio, then weight by inverse distance. An estimated value should always be flagged as such in the record.

## 3.6 Areal precipitation

A catchment responds to the rain falling on its whole area, not on a gauge. The **areal (mean) precipitation** is

$$
\bar{P} = \frac{1}{A} \int_A P(x, y)\, dA ,
$$

and in practice it is estimated as a weighted average of the gauge values, $\bar{P} = \sum w_i P_i$ with $\sum w_i = 1$. The methods differ in how the weights are chosen.

**Arithmetic mean.** All gauges in or near the catchment get the same weight. This is adequate only for flat catchments with an even gauge network.

**Thiessen polygons.** Each point of the catchment is assigned to its nearest gauge. The lines that bisect the segments joining neighboring gauges divide the catchment into polygons, and each gauge is weighted by the fraction of the catchment area in its polygon (Figure 3.2a). Thiessen (1911) proposed the method for large regions. It is objective and easy to automate, and the weights are computed once for a fixed network. It ignores topography, and the weights must be recomputed whenever a gauge is missing.

**Isohyetal method.** Lines of equal precipitation (**isohyets**) are drawn between the gauges, using the hydrologist's knowledge of the storm and of orographic effects. The area between each pair of adjacent isohyets is measured and multiplied by the mean of the two isohyet values:

$$
\bar{P} = \frac{1}{A}\sum_j A_j \, \frac{P_j + P_{j+1}}{2} ,
$$

where $A_j$ is the area between isohyets $P_j$ and $P_{j+1}$ inside the catchment (Figure 3.2b). Done well, it is the most accurate of the three, but it depends on judgment. Today the isohyets are usually produced by a spatial interpolation method on a computer, which is the same idea in automatic form.

Figure 3.2 applies all three methods to a synthetic storm whose true areal mean is known.

```{code-cell} python
:tags: [hide-input]
from matplotlib.path import Path
from scipy.interpolate import RBFInterpolator

# Catchment outline (km) and gauges
outline = np.array([[2, 3], [7, 1], [13, 2], [18, 5], [19, 10], [15, 14], [9, 15], [4, 12], [1, 8], [2, 3]])
gauges = np.array([[4.0, 4.5], [10.5, 3.0], [16.5, 7.0], [12.0, 9.0], [6.0, 10.5], [13.5, 13.5], [21.0, 12.0]])
names = list("ABCDEFG")

# Synthetic storm: a moving convective cell over a widespread background (mm)
storm = lambda x, y: 30 + 95 * np.exp(-((x - 14.5) ** 2 / 30 + (y - 10.5) ** 2 / 18))
P_g = storm(gauges[:, 0], gauges[:, 1])

xg, yg = np.meshgrid(np.linspace(0, 22, 441), np.linspace(0, 16, 321))
pts = np.c_[xg.ravel(), yg.ravel()]
inside = Path(outline).contains_points(pts).reshape(xg.shape)
cell = (22 / 440) * (16 / 320)                                   # km² per grid cell
A_tot = inside.sum() * cell

P_true = storm(xg, yg)[inside].mean()
in_gauges = Path(outline).contains_points(gauges)
P_arith = P_g[in_gauges].mean()

# Thiessen: nearest gauge for every grid point inside the catchment
d2 = ((pts[:, None, :] - gauges[None, :, :]) ** 2).sum(axis=2)
nearest = d2.argmin(axis=1).reshape(xg.shape)
w_th = np.array([(nearest[inside] == i).sum() for i in range(len(gauges))]) / inside.sum()
P_thiessen = (w_th * P_g).sum()

# Isohyets from a smooth interpolation of the gauge values, then the band-area formula
P_int = RBFInterpolator(gauges, P_g, kernel="thin_plate_spline")(pts).reshape(xg.shape)
levels = np.arange(0, 160, 10)
band = np.digitize(P_int[inside], levels) - 1                    # index of the band each cell is in
A_j = np.bincount(band, minlength=len(levels)) * cell
mid = levels + 5.0
P_isohyetal = (A_j * mid).sum() / A_tot

fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.6, 8.6))
for ax in (a1, a2):
    ax.plot(outline[:, 0], outline[:, 1], color=ink, lw=1.8)
    ax.set_aspect("equal"); ax.set_xlim(0, 22.5); ax.set_ylim(0, 16.5)
    ax.set_xlabel("km"); ax.set_ylabel("km"); ax.grid(False)
shade = np.ma.masked_where(~inside, nearest % 3)
a1.pcolormesh(xg, yg, shade, cmap=plt.matplotlib.colors.ListedColormap([c1, c2, c3]),
              alpha=0.16, shading="auto")
a1.contour(xg, yg, np.ma.masked_where(~inside, nearest), levels=np.arange(-0.5, 7), colors=ink2,
           linewidths=0.9)
for (x, y), n, p, w in zip(gauges, names, P_g, w_th):
    a1.plot(x, y, "o", color=ink, ms=5)
    a1.text(x + 0.35, y + 0.35, f"{n}: {p:.0f} mm" + (f"\n{w*100:.0f}% of area" if w > 0 else "\n(outside)"),
            fontsize=8.5, color=ink)
a1.set_title("(a) Thiessen polygons")
cs = a2.contour(xg, yg, np.ma.masked_where(~inside, P_int), levels=levels, colors=c1, linewidths=1.1)
a2.clabel(cs, fontsize=8.5, fmt="%d")
for (x, y), p in zip(gauges, P_g):
    a2.plot(x, y, "o", color=ink, ms=5); a2.text(x + 0.35, y + 0.35, f"{p:.0f}", fontsize=8.5, color=ink)
a2.set_title("(b) Isohyets (mm)")
plt.show()

for k, v in dict(P_true=P_true, P_arith=P_arith, P_thiessen=P_thiessen, P_isohyetal=P_isohyetal,
                 A_tot=A_tot).items():
    glue(k, float(v), display=False)
```

*Figure 3.2. Areal rainfall over a synthetic {glue:text}`A_tot:.0f` km² catchment with seven gauges, one of them (G) outside the divide. The storm is known exactly, so the methods can be checked: the true areal mean is {glue:text}`P_true:.1f` mm; the arithmetic mean of the six gauges inside gives {glue:text}`P_arith:.1f` mm, the Thiessen method {glue:text}`P_thiessen:.1f` mm and the isohyetal method {glue:text}`P_isohyetal:.1f` mm.*

In this example the arithmetic mean is pulled up because two of the six gauges inside the catchment happen to sit near the storm center. The Thiessen and isohyetal methods account for how much area each gauge represents, and come closer to the truth. No method can recover rain that falls between gauges, however: a convective cell smaller than the gauge spacing can be missed entirely. This is the main reason radar is used together with gauges.

## 3.7 Rainfall intensity and the hyetograph

For flood studies, the total depth of a storm is not enough; its distribution in time matters. A **hyetograph** is a bar chart of rainfall depth (or intensity) in successive time intervals. The **cumulative rainfall curve** (mass curve) adds these up; its slope at any time is the intensity. The average intensity over a duration $D$ is $i = P/D$, and the **maximum** average intensity over a given duration, taken from the steepest part of the mass curve, is what design methods use (Chapter 11).

```{code-cell} python
:tags: [hide-input]
# A synthetic 12-hour storm in 30-minute intervals
dt = 0.5                                                              # h
t_end = np.arange(dt, 12 + dt, dt)
rng = np.random.default_rng(11)
shape = np.exp(-0.5 * ((t_end - 5.0) / 1.6) ** 2) + 0.25 * np.exp(-0.5 * ((t_end - 9.0) / 1.0) ** 2)
depth = shape / shape.sum() * 96.0 * (1 + 0.15 * rng.standard_normal(len(t_end)))
depth = np.clip(depth, 0, None); depth *= 96.0 / depth.sum()          # 96 mm in total
cum = np.cumsum(depth)

def max_intensity(depth, dt, D):
    n = int(round(D / dt))
    return np.convolve(depth, np.ones(n), mode="valid").max() / D

fig, a1 = plt.subplots(figsize=(6.4, 3.9))
a1.bar(t_end - dt / 2, depth / dt, width=dt * 0.92, color=c1, alpha=0.8, label="Intensity (left axis)")
a1.set_xlabel("Time from start of storm (h)"); a1.set_ylabel("Intensity (mm h⁻¹)")
a1.set_xlim(0, 12); a1.set_xticks(range(0, 13, 2))
a2 = a1.twinx()
a2.plot(np.r_[0, t_end], np.r_[0, cum], color=c2, lw=2.2, label="Cumulative depth (right axis)")
a2.set_ylabel("Cumulative depth (mm)", color=c2); a2.set_ylim(0, 105); a2.grid(False)
a2.spines["right"].set_visible(True); a2.spines["top"].set_visible(False)
h1, l1 = a1.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
a1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9.5)
plt.show()
for D in (0.5, 1, 3, 6):
    glue(f"imax_{str(D).replace('.', '_')}", float(max_intensity(depth, dt, D)), display=False)
```

*Figure 3.3. Hyetograph and cumulative rainfall of a synthetic 12-hour storm of 96 mm. The average intensity over the whole storm is 8 mm h⁻¹, but the maximum intensities over shorter windows are much higher: {glue:text}`imax_0_5:.1f` mm h⁻¹ over 30 minutes, {glue:text}`imax_1:.1f` mm h⁻¹ over 1 hour and {glue:text}`imax_3:.1f` mm h⁻¹ over 3 hours.*

The shorter the duration, the higher the maximum average intensity. This inverse relation between intensity and duration is the basis of the intensity–duration–frequency curves of Chapter 11.

## 3.8 Snow and snowmelt

In cold regions and mountains, part of the precipitation falls as snow and is stored on the ground for weeks or months. The snowpack delays runoff, and rapid melt, especially with rain on snow, can cause floods in spring.

**Snow water equivalent.** The depth of liquid water that a snowpack would produce if melted is its **snow water equivalent** (SWE):

$$
\text{SWE} = h_s \frac{\rho_s}{\rho_w} ,
$$

where $h_s$ is the snow depth and $\rho_s$ the snow density. Fresh snow has a density of about 50–150 kg m⁻³, so 10 cm of new snow often contains only about 1 cm of water. As the pack ages it settles and its density rises to 300–500 kg m⁻³.

**Degree-day melt.** Melting a snowpack takes energy: 0.334 MJ per kilogram of ice (Chapter 2), on top of warming the pack to 0 °C. A full energy balance needs radiation, wind and humidity data. The simplest practical model uses air temperature alone as an index of the available energy:

\begin{equation*}
M = \begin{cases} C_m\,(T_a - T_b) & T_a > T_b \\ 0 & T_a \le T_b \end{cases}
\tag{3.2}
\end{equation*}

where $M$ is the daily melt (mm d⁻¹), $T_a$ the daily mean air temperature, $T_b$ a base temperature (usually 0 °C) and $C_m$ the **degree-day factor**, typically 2–6 mm °C⁻¹ d⁻¹ for snow. The factor is larger late in the season, when the snow is older and darker and the sun is stronger, and in open terrain than under forest. It must be calibrated for each region. The method works because, over a day, air temperature is correlated with the main energy inputs (incoming longwave radiation and sensible heat).

```{code-cell} python
:tags: [hide-input]
# Spring melt of a snowpack with a degree-day model and synthetic daily temperature
rng = np.random.default_rng(3)
days = np.arange(0, 75)
Ta = -4 + 0.22 * days + 2.5 * rng.standard_normal(len(days))          # °C, warming through spring
Cm, Tb = 3.5, 0.0
swe = np.empty(len(days) + 1); swe[0] = 230.0                          # mm
melt = np.zeros(len(days))
for d in days:
    melt[d] = min(Cm * max(Ta[d] - Tb, 0.0), swe[d])
    swe[d + 1] = swe[d] - melt[d]
day_gone = int(np.argmax(swe <= 0)) if (swe <= 0).any() else None

fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 5.4), sharex=True)
a1.plot(days, Ta, color=c2, lw=1.4)
a1.axhline(Tb, color=ink2, lw=0.9, ls="--"); a1.set_ylabel("$T_a$ (°C)")
a1.set_title("(a) Daily mean air temperature")
a2.plot(np.r_[days, days[-1] + 1], swe, color=c1, lw=2.2, label="Snow water equivalent")
a2.bar(days + 0.5, melt, width=0.9, color=c3, alpha=0.7, label="Daily melt")
a2.set_ylabel("mm"); a2.set_xlabel("Day"); a2.set_title("(b) Snowpack and melt")
a2.legend(loc="upper right", fontsize=9.5); a2.set_xlim(0, 75)
plt.show()
glue("snow_day", day_gone, display=False); glue("snow_maxmelt", float(melt.max()), display=False)
```

*Figure 3.4. A degree-day model ($C_m$ = 3.5 mm °C⁻¹ d⁻¹) applied to a 230 mm snowpack in a synthetic spring. Melt begins once the air is regularly above freezing, reaches {glue:text}`snow_maxmelt:.0f` mm on the warmest day, and the snow is gone by day {glue:text}`snow_day`.*

## 3.9 Worked examples

**Example 3.1 — Filling a gap by inverse distance weighting.** Station X missed the storm of 14 August. Four neighbors recorded the rainfall below. Estimate the storm rainfall at X with $b = 2$.

```{code-cell} python
d = np.array([5.2, 8.1, 11.4, 14.0])        # km from X
P = np.array([62.0, 48.0, 75.0, 39.0])      # mm
w = 1 / d ** 2
P_x = (w * P).sum() / w.sum()
print("weights:", np.round(w / w.sum(), 3))
print(f"Estimated rainfall at X = {P_x:.1f} mm  (simple mean {P.mean():.1f} mm)")
```

The nearest station receives more than half of the weight, so the estimate is close to its value.

**Example 3.2 — Adjusting an inconsistent record.** In Figure 3.1, the tested station recorded 1384 mm in 2003, before the break. What value is consistent with the present site?

```{code-cell} python
print(f"slopes: before {m1:.3f}, after {m2:.3f}; factor = {factor:.3f}")
print(f"Adjusted 2003 total = {1384 * factor:.0f} mm")
```

**Example 3.3 — Thiessen weights by hand.** A 128 km² catchment has four gauges. The Thiessen polygons inside the catchment have the areas below. Compare the Thiessen and arithmetic means.

```{code-cell} python
area = np.array([22.0, 41.0, 37.0, 28.0])   # km²
P = np.array([84.0, 112.0, 131.0, 95.0])    # mm
assert np.isclose(area.sum(), 128.0)        # polygons must cover the whole catchment
P_th = (area * P).sum() / area.sum()
print(f"Thiessen mean = {P_th:.1f} mm, arithmetic mean = {P.mean():.1f} mm")
```

**Example 3.4 — Radar rainfall rate.** A radar measures a reflectivity of 45 dBZ over a catchment. Estimate the rainfall rate with the Marshall–Palmer relation, and the change if the reflectivity were 3 dB higher.

```{code-cell} python
for dBZ in (45, 48):
    Z = 10 ** (dBZ / 10)                    # mm⁶ m⁻³
    R = (Z / 200) ** (1 / 1.6)              # mm/h
    print(f"{dBZ} dBZ -> Z = {Z:,.0f} mm⁶/m³ -> R = {R:.1f} mm/h")
```

A 3 dB error, which is common for an uncalibrated radar, changes the estimated rainfall rate by about 50%. This is why radar rainfall is adjusted with gauge data.

**Example 3.5 — How long will the snow last?** A snowpack is 75 cm deep with a mean density of 300 kg m⁻³. With a degree-day factor of 4 mm °C⁻¹ d⁻¹ and a base temperature of 0 °C, how many days of 5 °C mean air temperature will melt it?

```{code-cell} python
h_s, rho_s, Cm = 0.75, 300.0, 4.0
SWE = h_s * rho_s / 1000 * 1000             # mm
M = Cm * (5.0 - 0.0)                        # mm/d
print(f"SWE = {SWE:.0f} mm, melt = {M:.0f} mm/d, time to melt = {SWE / M:.1f} days")
```

The snowpack holds 225 mm of water, more than a large rainstorm, and releases it over about 11 days. If 30 mm of rain falls on the pack during that time, both the rain and the melt it causes reach the soil together, which is why rain-on-snow events produce the largest snowmelt floods.

## 3.10 Common mistakes

- **Using point rainfall as areal rainfall.** The rainfall at the wettest gauge exceeds the catchment average, often by a large margin for convective storms. Chapter 11 introduces areal reduction factors for design.
- **Weighting by gauges outside the catchment without checking their polygons.** A gauge outside the divide may still deserve Thiessen weight if it is the nearest gauge to part of the catchment (gauge G in Figure 3.2 represents a small corner of it).
- **Adjusting for every wiggle in a double-mass curve.** Only persistent, explainable breaks should be corrected.
- **Treating estimated values as measurements.** Infilled data must be flagged so that later users know which values are estimates.
- **Forgetting snow density.** Snow depth is not water depth. Convert to SWE before using it in a water balance.
- **Mixing intensity and depth.** 20 mm in 30 minutes is an intensity of 40 mm h⁻¹, not 20 mm h⁻¹.

## 3.11 Exercises

```{code-cell} python
:tags: [remove-cell]
d1 = np.array([3.0, 6.5, 9.0]); P1 = np.array([28.0, 41.0, 35.0]); w1 = 1 / d1 ** 2
glue("e1", float((w1 * P1).sum() / w1.sum()), display=False)
glue("e2_f", 1.12 / 0.97, display=False); glue("e2_P", 1120 * 0.97 / 1.12, display=False)
a3 = np.array([18.0, 25.0, 31.0, 12.0]); p3 = np.array([45.0, 60.0, 72.0, 55.0])
glue("e3_th", float((a3 * p3).sum() / a3.sum()), display=False); glue("e3_ar", float(p3.mean()), display=False)
a4 = np.array([8.0, 26.0, 34.0, 19.0, 6.0]); pm4 = np.array([[30, 40], [40, 50], [50, 60], [60, 70], [70, 80]]).mean(1)
glue("e4", float((a4 * pm4).sum() / a4.sum()), display=False); glue("e4_A", float(a4.sum()), display=False)
glue("e5_swe", 1.2 * 350 / 1000 * 1000, display=False)
glue("e5_days", 1.2 * 350 / (3.0 * 6.0), display=False)
```

1. Estimate the missing daily rainfall at a station from three neighbors at 3.0, 6.5 and 9.0 km that recorded 28, 41 and 35 mm, using inverse distance weighting with $b = 2$.
2. The slope of a double-mass curve is 0.97 for 1990–2004 and 1.12 for 2005–2024, after a gauge was replaced. Which period should be adjusted if the present gauge is taken as correct, by what factor, and what does a 1120 mm year in the later period become if instead the earlier gauge is taken as the reference?
3. Four Thiessen polygons cover 18, 25, 31 and 12 km² of a catchment, and their gauges recorded 45, 60, 72 and 55 mm. Find the areal mean by the Thiessen method and by the arithmetic mean.
4. The areas between isohyets in a catchment are: 30–40 mm, 8 km²; 40–50 mm, 26 km²; 50–60 mm, 34 km²; 60–70 mm, 19 km²; 70–80 mm, 6 km². Find the areal precipitation.
5. A snowpack is 1.2 m deep with a density of 350 kg m⁻³. Find its SWE, and the number of days needed to melt it if the mean air temperature is 6 °C and $C_m$ = 3 mm °C⁻¹ d⁻¹.

```{admonition} Answers
:class: dropdown
1. About {glue:text}`e1:.1f` mm.
2. Adjust 1990–2004 by multiplying by {glue:text}`e2_f:.3f`; with the earlier gauge as reference, 1120 mm becomes {glue:text}`e2_P:.0f` mm.
3. Thiessen {glue:text}`e3_th:.1f` mm; arithmetic {glue:text}`e3_ar:.1f` mm.
4. {glue:text}`e4:.1f` mm over {glue:text}`e4_A:.0f` km².
5. SWE = {glue:text}`e5_swe:.0f` mm; about {glue:text}`e5_days:.0f` days.
```

## 3.12 Further reading

- Dingman, S. L. (2015). *Physical Hydrology* (3rd ed.). Waveland Press. The chapters on precipitation and on snow.
- Chow, V. T., Maidment, D. R., and Mays, L. W. (1988). *Applied Hydrology*. McGraw-Hill. The chapter on atmospheric water and precipitation.
- 윤용남. *수문학* (in Korean). The chapter on precipitation.
- Marshall, J. S., and Palmer, W. McK. (1948). The distribution of raindrops with size. *Journal of Meteorology*, 5, 165–166.
- Thiessen, A. H. (1911). Precipitation averages for large areas. *Monthly Weather Review*, 39, 1082–1084.

---

*Figures in this chapter are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
