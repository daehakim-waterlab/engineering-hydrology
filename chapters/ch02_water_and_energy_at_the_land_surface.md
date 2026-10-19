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

# 2. Water and energy at the land surface

```{admonition} Core ideas
:class: tip
1. The sun powers the water cycle. The energy that reaches the ground and stays there is the **net radiation** $R_n$: incoming solar radiation minus the reflected part, plus the longwave exchange between the surface and the sky.
2. The surface cannot store much energy, so net radiation is shared out at once: $R_n - G = \lambda E + H$. Energy used to evaporate water ($\lambda E$) is not available to warm the air ($H$).
3. Evaporating **1 mm of water takes about 2.45 MJ per square meter**. This single number links the energy balance to the water balance.
4. Rising air cools by expansion, by about 10 °C per km while unsaturated. When it cools to its **dew point**, vapor condenses into cloud. This is the first step toward precipitation.
5. The amount of vapor that air can hold, the **saturation vapor pressure**, rises by about 6–7% for every degree of warming (the Clausius–Clapeyron relation).
6. Humidity can be expressed as vapor pressure, relative humidity, specific humidity or dew point. They are different views of the same quantity, and each is easy to compute from the others.
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

## 2.1 Why hydrologists study energy

Chapter 1 showed that evapotranspiration returns more than half of the precipitation on land to the atmosphere. To evaporate water, molecules must be given enough energy to break free from the liquid. In nature that energy comes from the sun, either directly as radiation or indirectly as heat carried by warm air. A wet surface in a cold, cloudy climate evaporates little because it lacks energy; a desert surface in strong sunshine evaporates little because it lacks water. Estimating evaporation (Chapter 4) is therefore an exercise in accounting for **both water and energy**.

The atmosphere also has to deliver the water in the first place. Precipitation forms when moist air rises and cools. This chapter explains the energy budget of the land surface, how air cools as it rises, and how to describe the water vapor in air. These are the tools needed for precipitation (Chapter 3) and evaporation (Chapter 4).

## 2.2 Radiation at the land surface

All bodies emit electromagnetic radiation. The **Stefan–Boltzmann law** gives the energy flux emitted by a surface at absolute temperature $T_s$ (K):

$$
L = \varepsilon\, \sigma\, T_s^{4},
$$

where $\sigma = 5.67 \times 10^{-8}$ W m⁻² K⁻⁴ is the Stefan–Boltzmann constant and $\varepsilon$ is the **emissivity** (1 for a perfect emitter; 0.95–0.99 for water, vegetation and moist soil). The wavelength at which a body emits most strongly is inversely proportional to its temperature. The sun (about 5800 K) emits mainly at wavelengths shorter than 4 µm, called **shortwave** radiation. The Earth's surface and atmosphere (about 250–310 K) emit at wavelengths longer than 4 µm, called **longwave** radiation. Because the two bands hardly overlap, it is convenient to treat them separately.

**Shortwave radiation.** At the top of the atmosphere, a surface facing the sun receives about 1361 W m⁻² (the **solar constant**). The daily total that reaches a horizontal surface at the top of the atmosphere, the **extraterrestrial radiation** $R_a$, depends only on latitude and the day of the year, and can be calculated exactly from astronomy (Figure 2.1). Clouds, water vapor and aerosols absorb and scatter part of it, so the **incoming shortwave radiation** at the ground, $K_\downarrow$, is typically 40–75% of $R_a$. A fraction $\alpha$ of $K_\downarrow$, the **albedo**, is reflected: about 0.06 for deep water, 0.15–0.25 for forests and crops, 0.3–0.4 for dry sand and up to 0.9 for fresh snow.

**Longwave radiation.** The surface emits $L_\uparrow = \varepsilon \sigma T_s^4$ and receives $L_\downarrow$ from water vapor, carbon dioxide and clouds in the atmosphere. Because the atmosphere is usually colder than the ground, $L_\uparrow > L_\downarrow$ and the **net longwave radiation** is a loss, typically 50–100 W m⁻² under clear skies and much less under cloud.

Adding the four terms gives the **net radiation**:

\begin{equation*}
R_n = (1 - \alpha)\,K_\downarrow + L_\downarrow - L_\uparrow .
\tag{2.1}
\end{equation*}

```{code-cell} python
:tags: [hide-input]
def extraterrestrial_radiation(lat_deg, doy):
    """Daily extraterrestrial radiation Ra (MJ m⁻² d⁻¹) on a horizontal surface (FAO-56, Eqs. 21–25)."""
    Gsc = 0.0820                                      # solar constant, MJ m⁻² min⁻¹
    phi = np.radians(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)    # inverse relative Earth–sun distance
    delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)   # solar declination, rad
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1, 1))  # sunset hour angle, rad
    return 24 * 60 / np.pi * Gsc * dr * (ws * np.sin(phi) * np.sin(delta)
                                         + np.cos(phi) * np.cos(delta) * np.sin(ws))

doy = np.arange(1, 366)
fig, ax = plt.subplots(figsize=(6.4, 3.8))
for lat, col in [(0, c2), (36, c1), (60, c3)]:
    ax.plot(doy, extraterrestrial_radiation(lat, doy), color=col, lw=2, label=f"{lat}° N")
ax.set_xlabel("Day of year"); ax.set_ylabel("$R_a$ (MJ m⁻² d⁻¹)")
ax.set_xlim(1, 365); ax.set_ylim(0, 47)
ax.set_xticks([1, 91, 182, 274, 365], ["Jan 1", "Apr 1", "Jul 1", "Oct 1", "Dec 31"])
ax.legend(loc="lower center", ncol=3, fontsize=10)
plt.show()
```

*Figure 2.1. Daily extraterrestrial radiation on a horizontal surface at three latitudes, computed from solar geometry. At the equator it varies little through the year. At 60° N it ranges from almost nothing in winter to more than at the equator in midsummer, because summer days there are long.*

It is convenient to know two conversions. A flux of 1 W m⁻² sustained for a day delivers $86\,400$ J m⁻² $= 0.0864$ MJ m⁻² d⁻¹. And since 1 mm of water over 1 m² has a mass of 1 kg, an energy flux in MJ m⁻² d⁻¹ divided by the latent heat (in MJ kg⁻¹, Section 2.4) gives the depth of water it could evaporate, in mm d⁻¹.

## 2.3 The surface energy balance

The land surface is a thin layer with almost no capacity to store heat. Energy arriving as net radiation must therefore leave at once through three routes:

\begin{equation*}
R_n = \lambda E + H + G ,
\tag{2.2}
\end{equation*}

or, written the way it is used in this book,

$$
R_n - G = \lambda E + H .
$$

- $\lambda E$ is the **latent heat flux**: energy used to evaporate water at the rate $E$, which is carried away by the vapor and released later when it condenses in a cloud;
- $H$ is the **sensible heat flux**: energy that warms the air by conduction and turbulent mixing, which we can "sense" as a change of temperature;
- $G$ is the **ground heat flux** into the soil (or into a lake).

All terms are fluxes per unit area (W m⁻² or MJ m⁻² d⁻¹), taken as positive when $R_n$ is directed toward the surface and the other three are directed away from it. The term $R_n - G$ is called the **available energy**. Over a full day $G$ is small, because heat that enters the soil in the morning comes out again at night; over a year it is close to zero. For daily calculations it is often set to zero (Chapter 4).

Figure 2.2 shows a typical clear summer day over a well-watered crop. Net radiation follows the sun and becomes slightly negative at night, when the surface loses longwave radiation to the sky. Most of the daytime energy goes into evaporation. Over a dry surface the same net radiation would go mostly into $H$, which is why the air over a dry field or a city is hotter than over an irrigated field.

```{code-cell} python
:tags: [hide-input]
# Synthetic half-hourly energy balance of a well-watered crop on a clear summer day
t = np.linspace(0, 24, 49)
sun = np.clip(np.sin(np.pi * (t - 5.5) / 14), 0, None)          # daylight 05:30–19:30
Rn = 640 * sun ** 1.2 - 55 * (sun == 0) - 25 * (sun > 0) * (1 - sun)   # W m⁻²
G = np.where(Rn > 0, 0.10 * Rn, 0.45 * Rn)                       # into soil by day, out at night
LE = np.where(Rn > 0, 0.78 * (Rn - G), 0.0)                       # evaporation by day only
H = Rn - G - LE                                                   # closes the balance
assert np.allclose(Rn, LE + H + G)

fig, ax = plt.subplots(figsize=(6.4, 3.9))
ax.plot(t, Rn, color=ink, lw=2, label="$R_n$ net radiation")
ax.plot(t, LE, color=c1, lw=2, label="$\\lambda E$ latent heat")
ax.plot(t, H, color=c2, lw=2, label="$H$ sensible heat")
ax.plot(t, G, color=c3, lw=2, label="$G$ ground heat")
ax.axhline(0, color=ink2, lw=0.8)
ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 6))
ax.set_xlabel("Hour of day"); ax.set_ylabel("Energy flux (W m⁻²)")
ax.legend(loc="upper left", fontsize=9.5)
plt.show()

MJ = lambda f: np.trapezoid(f, t * 3600) / 1e6                         # daily total, MJ m⁻²
glue("d_Rn", float(MJ(Rn)), display=False); glue("d_LE", float(MJ(LE)), display=False)
glue("d_G", float(MJ(G)), display=False); glue("d_E", float(MJ(LE) / 2.45), display=False)
```

*Figure 2.2. A synthetic but typical energy balance of a well-watered crop on a clear summer day. The daily totals are $R_n$ = {glue:text}`d_Rn:.1f` MJ m⁻², $\lambda E$ = {glue:text}`d_LE:.1f` MJ m⁻² (about {glue:text}`d_E:.1f` mm of evaporation) and $G$ = {glue:text}`d_G:.1f` MJ m⁻²: the soil takes in heat by day and returns part of it at night, so its daily total is small.*

How the available energy is divided between $\lambda E$ and $H$ depends on how wet the surface is, how much the plants restrict transpiration, and how dry and windy the air is. Chapter 4 develops the equations that make this split quantitative.

## 2.4 Latent heat

The **latent heat of vaporization** $\lambda$ is the energy needed to turn a unit mass of liquid water into vapor at the same temperature. It decreases slightly with temperature:

$$
\lambda = 2.501 - 0.002361\, T_a \quad \text{(MJ kg}^{-1}\text{, with } T_a \text{ in °C)},
$$

which gives 2.454 MJ kg⁻¹ at 20 °C. For most hydrological work the constant value $\lambda = 2.45$ MJ kg⁻¹ is accurate enough. The same energy is released when vapor condenses. Melting ice needs a further 0.334 MJ kg⁻¹ (the latent heat of fusion), which matters for snowmelt (Chapter 3).

Because 1 mm of water on 1 m² weighs 1 kg, the evaporation rate in mm d⁻¹ is

$$
E = \frac{\lambda E}{\lambda} \qquad (E \text{ in mm d}^{-1},\ \lambda E \text{ in MJ m}^{-2}\text{ d}^{-1},\ \lambda \text{ in MJ kg}^{-1}).
$$

So **1 mm d⁻¹ of evaporation uses 2.45 MJ m⁻² d⁻¹, or a steady 28.4 W m⁻²**. A summer day with 15 MJ m⁻² of available energy can evaporate at most about 6 mm if all of it goes into $\lambda E$. This upper bound is a useful check on any evaporation estimate.

## 2.5 Rising air and the lapse rate

Air temperature normally falls with height in the lower atmosphere. The rate of decrease is the **lapse rate** $\Gamma = -dT/dz$.

**The dry adiabatic lapse rate.** Imagine a parcel of unsaturated air rising quickly enough that it exchanges no heat with its surroundings (an **adiabatic** process). As it rises the surrounding pressure falls, the parcel expands and does work on its surroundings, and its temperature drops. The first law of thermodynamics for a unit mass of air at constant pressure heat capacity $c_p$ is $c_p\, dT = \frac{1}{\rho} dp$ when no heat is added. The pressure change with height is hydrostatic, $dp = -\rho g\, dz$ (Chapter 2 of the [Hydraulics](https://hydroclimatol.com/hydraulics/) book). Combining the two,

$$
\Gamma_d = -\frac{dT}{dz} = \frac{g}{c_p} = \frac{9.81}{1005} \approx 9.8\ \text{K km}^{-1}.
$$

**The saturated lapse rate.** Once the parcel cools to its dew point, vapor condenses and releases latent heat, which offsets part of the cooling. The **saturated (moist) adiabatic lapse rate** $\Gamma_s$ is therefore smaller, about 4 K km⁻¹ in warm, humid air and close to $\Gamma_d$ in very cold air, where little vapor is left to condense.

**The environmental lapse rate.** The actual temperature profile of the atmosphere averages about 6.5 K km⁻¹. When a rising parcel stays warmer than its surroundings it keeps rising (the atmosphere is **unstable**); when it becomes colder it sinks back (**stable**). Since $\Gamma_s < 6.5 < \Gamma_d$ K km⁻¹, saturated air that is lifted tends to keep rising while unsaturated air does not. This is why thunderstorms grow from humid air (Chapter 3).

The height at which a lifted parcel first becomes saturated is the **lifting condensation level**, the base of the clouds that form. Figure 2.3 follows a parcel of air forced over a 2000 m mountain range. It cools at $\Gamma_d$ up to the cloud base and at the slower $\Gamma_s$ above it, losing its condensed water as rain on the windward side. On the lee side it descends and warms at $\Gamma_d$ all the way down, arriving warmer and much drier than it started. This warm, dry downslope wind is the **foehn**, and the dry lee side is the **rain shadow**.

```{code-cell} python
:tags: [hide-input]
g, cp, Rd, Rv, eps, Lv = 9.81, 1005.0, 287.0, 461.5, 0.622, 2.501e6

def es_kPa(T):            # saturation vapor pressure (kPa), T in °C (Tetens)
    return 0.6108 * np.exp(17.27 * T / (T + 237.3))

def dewpoint(e):          # dew point (°C) from vapor pressure (kPa), inverse of Tetens
    x = np.log(e / 0.6108)
    return 237.3 * x / (17.27 - x)

def pressure(z):          # standard-atmosphere pressure (kPa) at height z (m)
    return 101.3 * ((293 - 0.0065 * z) / 293) ** 5.26

def lift_parcel(T0, Td0, z_top, dz=5.0):
    """Lift a parcel from sea level to z_top: dry adiabat to saturation, then saturated adiabat."""
    z = np.arange(0, z_top + dz, dz)
    T = np.empty_like(z); T[0] = T0
    r = eps * es_kPa(Td0) / (pressure(0) - es_kPa(Td0))      # mixing ratio, kg/kg (conserved while dry)
    z_lcl = None
    for k in range(1, len(z)):
        p = pressure(z[k - 1])
        rs = eps * es_kPa(T[k - 1]) / (p - es_kPa(T[k - 1]))
        if rs > r:                                           # unsaturated: dry adiabatic
            Gamma = g / cp
        else:                                                # saturated: moist adiabatic
            if z_lcl is None: z_lcl = z[k - 1]
            TK = T[k - 1] + 273.15
            Gamma = g * (1 + Lv * rs / (Rd * TK)) / (cp + Lv ** 2 * rs * eps / (Rd * TK ** 2))
            r = rs                                           # excess vapor condenses and rains out
        T[k] = T[k - 1] - Gamma * dz
    return z, T, z_lcl, r

T0, Td0, z_top = 22.0, 16.0, 2000.0
z_up, T_up, z_lcl, r_top = lift_parcel(T0, Td0, z_top)
T_down = T_up[-1] + g / cp * (z_top - z_up[::-1])               # descend dry adiabatically
e_lee = r_top * pressure(0) / (eps + r_top)
RH0, RH_lee = es_kPa(Td0) / es_kPa(T0), e_lee / es_kPa(T_down[-1])

fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.plot(T_up, z_up / 1000, color=c1, lw=2.2, label="Rising (windward side)")
ax.plot(T_down, z_up[::-1] / 1000, color=c2, lw=2.2, ls="--", label="Descending (lee side)")
ax.axhline(z_lcl / 1000, color=ink2, lw=0.9, ls=":")
ax.text(T0 + 0.6, z_lcl / 1000 + 0.05, f"cloud base ≈ {z_lcl:.0f} m", color=ink2, fontsize=9.5)
ax.annotate(f"{T0:.0f} °C, RH {RH0*100:.0f}%", (T0, 0), xytext=(T0 - 9.8, 0.12), fontsize=9.5,
            color=c1, arrowprops=dict(arrowstyle="-", color=c1, lw=0.8))
ax.annotate(f"{T_down[-1]:.1f} °C, RH {RH_lee*100:.0f}%", (T_down[-1], 0),
            xytext=(T_down[-1] - 3.0, 0.45), fontsize=9.5, color=c2,
            arrowprops=dict(arrowstyle="-", color=c2, lw=0.8))
ax.set_xlabel("Parcel temperature (°C)"); ax.set_ylabel("Height (km)")
ax.set_ylim(0, 2.1); ax.set_xlim(0, 32)
ax.legend(loc="upper right", fontsize=9.5)
plt.show()
glue("z_lcl", float(z_lcl), display=False); glue("T_lee", float(T_down[-1]), display=False)
glue("T_dif", float(T_down[-1] - T0), display=False)
```

*Figure 2.3. A parcel of air at 22 °C with a dew point of 16 °C forced over a 2000 m ridge. Above the cloud base (about {glue:text}`z_lcl:.0f` m) it cools at the saturated rate and loses water as rain. It descends dry on the lee side and arrives {glue:text}`T_dif:.1f` °C warmer than it started.*

### The general circulation, briefly

The same physics, on the scale of the planet, organizes the global pattern of rainfall. The tropics receive more solar energy than they radiate back to space, and the polar regions less (Figure 2.4). The atmosphere and the oceans carry the surplus poleward. Near the equator, heated humid air rises in a band of deep clouds and heavy rain called the **Intertropical Convergence Zone**. The air spreads poleward at high altitude, cools, and sinks near 20–30° latitude, where it warms by compression and dries out; most of the world's great deserts lie under these **subtropical highs**. This overturning loop is the **Hadley cell**. In the mid-latitudes, heat is carried mainly by large weather systems, the cyclones and anticyclones that travel eastward in the **westerlies**, steered by the jet stream. Fronts in these cyclones bring much of the mid-latitude rain.

On top of this zonal pattern, the contrast between land and sea creates **monsoons**: in summer the continents heat faster than the oceans, and moist ocean air is drawn inland. East Asia, including Korea, lies in a monsoon region with a wet summer and a dry winter, the setting for the rainy season discussed in Chapter 3.

```{code-cell} python
:tags: [hide-input]
lats = np.linspace(-89.5, 89.5, 180)
Ra_annual = np.array([extraterrestrial_radiation(l, doy).mean() for l in lats]) / 0.0864  # W m⁻²
S0 = 1361.0
absorbed = 0.70 * Ra_annual                       # planetary albedo about 0.3
olr = 0.70 * S0 / 4                               # outgoing longwave that would balance the global mean
global_mean = np.average(Ra_annual, weights=np.cos(np.radians(lats)))

fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.plot(lats, Ra_annual, color=ink2, lw=1.4, ls="--", label="Solar radiation at the top of the atmosphere")
ax.plot(lats, absorbed, color=c2, lw=2.2, label="Absorbed (70%)")
ax.axhline(olr, color=c1, lw=2.2, label=f"Global-mean emission ≈ {olr:.0f} W m⁻²")
ax.fill_between(lats, absorbed, olr, where=absorbed > olr, color=c2, alpha=0.15)
ax.fill_between(lats, absorbed, olr, where=absorbed < olr, color=c1, alpha=0.15)
ax.text(0, 270, "surplus", ha="center", color=c2, fontsize=10)
ax.text(-70, 160, "deficit", ha="center", color=c1, fontsize=10)
ax.text(70, 160, "deficit", ha="center", color=c1, fontsize=10)
ax.set_xlim(-90, 90); ax.set_ylim(0, 520)
ax.set_xticks([-90, -60, -30, 0, 30, 60, 90], ["90° S", "60° S", "30° S", "0°", "30° N", "60° N", "90° N"])
ax.set_ylabel("Annual mean (W m⁻²)")
ax.legend(loc="upper center", fontsize=9, ncol=1)
plt.show()
glue("gm", float(global_mean), display=False)
```

*Figure 2.4. Annual-mean solar radiation at the top of the atmosphere by latitude, computed from solar geometry (its global mean is {glue:text}`gm:.0f` W m⁻², one quarter of the solar constant), and the part absorbed with a planetary albedo of 0.3. If the Earth emitted the same amount at every latitude, the tropics would gain energy and high latitudes would lose it. The real emission varies less with latitude than absorption, and the difference is carried poleward by winds and ocean currents.*

## 2.6 Water vapor and humidity

### Vapor pressure and saturation

Air is a mixture of gases, and each gas contributes a **partial pressure** to the total. The partial pressure of water vapor is the **vapor pressure** $e$. Above a flat water surface, molecules continually escape into the air and return to the water. When the two rates balance the air is **saturated**, and the vapor pressure has reached the **saturation vapor pressure** $e_s$, which depends only on temperature.

### The Clausius–Clapeyron relation

The thermodynamics of phase change gives the slope of the saturation curve:

$$
\frac{de_s}{dT} = \frac{\lambda}{T\,(v_v - v_l)},
$$

where $T$ is in kelvin and $v_v$, $v_l$ are the specific volumes of vapor and liquid. The liquid volume is about a thousand times smaller than the vapor volume and can be ignored. Treating vapor as an ideal gas, $v_v = R_v T / e_s$ with $R_v = 461.5$ J kg⁻¹ K⁻¹, gives the **Clausius–Clapeyron equation**:

\begin{equation*}
\frac{1}{e_s}\frac{de_s}{dT} = \frac{\lambda}{R_v T^{2}} .
\tag{2.3}
\end{equation*}

At 20 °C (293 K) the right side is $2.45\times10^{6} / (461.5 \times 293^2) \approx 0.062$ K⁻¹: **the saturation vapor pressure increases by about 6% per degree**. This exponential sensitivity has large consequences. Warm air can carry much more water than cold air, which is why the heaviest rainfalls occur in summer and in the tropics, and why a warmer climate is expected to intensify extreme rainfall (Chapter 13).

For practical work, an empirical fit to measured values is used. This book uses the **Tetens formula** in the form adopted by FAO-56:

\begin{equation*}
e_s(T_a) = 0.6108 \exp\!\left(\frac{17.27\, T_a}{T_a + 237.3}\right) \quad (\text{kPa},\ T_a \text{ in °C}).
\tag{2.4}
\end{equation*}

Its derivative, the **slope of the saturation vapor pressure curve**, appears in every combination equation for evaporation (Chapter 4):

\begin{equation*}
\Delta = \frac{de_s}{dT_a} = \frac{4098\, e_s(T_a)}{(T_a + 237.3)^{2}} \quad (\text{kPa °C}^{-1}).
\tag{2.5}
\end{equation*}

The code below checks that the Tetens formula and its derivative agree with the integrated Clausius–Clapeyron equation.

```{code-cell} python
T = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
TK, T0K, e0 = T + 273.15, 273.15, 0.6108
lam = 2.501e6 - 2361 * T                                # J/kg
es_tetens = es_kPa(T)
es_cc = e0 * np.exp(2.50e6 / Rv * (1 / T0K - 1 / TK))   # Clausius–Clapeyron, constant λ
delta_formula = 4098 * es_tetens / (T + 237.3) ** 2
delta_numeric = (es_kPa(T + 0.01) - es_kPa(T - 0.01)) / 0.02
rate_cc = lam / (Rv * TK ** 2) * 100                    # % per K
for row in zip(T, es_tetens, es_cc, delta_formula, delta_numeric, rate_cc):
    print("T = {:4.0f} °C  es: Tetens {:.3f}, C-C {:.3f} kPa   Δ: {:.4f} vs {:.4f} kPa/°C   {:.1f} %/K".format(*row))
assert np.allclose(delta_formula, delta_numeric, rtol=1e-4)
```

The two forms agree to within about 2.5% between 0 and 30 °C, and the rate of increase falls from about 7% K⁻¹ near freezing to about 5% K⁻¹ at 40 °C.

### Measures of humidity

The same moisture content can be described in several ways. All of them follow from $e$, $T_a$ and the air pressure $p$:

- **Relative humidity** is the ratio of actual to saturation vapor pressure, $RH = e / e_s(T_a)$, usually given in percent.
- **Vapor pressure deficit** is $e_s(T_a) - e$, the "drying power" of the air. It drives evaporation (Chapter 4).
- **Dew point** $T_d$ is the temperature to which air must be cooled at constant pressure and moisture content to become saturated, so that $e = e_s(T_d)$. It is found by inverting Equation (2.4):

$$
T_d = \frac{237.3\, \ln(e/0.6108)}{17.27 - \ln(e/0.6108)} .
$$

- **Specific humidity** is the mass of vapor per unit mass of moist air. Using the ideal-gas law for both vapor and dry air, with $\varepsilon = R_d/R_v = 0.622$,

$$
q = \frac{\varepsilon\, e}{p - (1-\varepsilon)\, e} \approx 0.622\,\frac{e}{p}.
$$

Unlike relative humidity, specific humidity and dew point do not change when air is warmed or cooled without adding or removing water. That makes them the right variables for following air masses. Relative humidity, by contrast, typically drops from near 100% at dawn to 50% in the afternoon on a day when the moisture content hardly changes, simply because $e_s$ rises with temperature (Figure 2.5).

```{code-cell} python
:tags: [hide-input]
Ta, e_air = 26.0, 1.9                       # an example air sample
Td = dewpoint(e_air)
Tg = np.linspace(-5, 40, 300)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(Tg, es_kPa(Tg), color=c1, lw=2.2, label="Saturation vapor pressure $e_s(T_a)$")
ax.plot([Td, Ta], [e_air, e_air], color=c3, lw=1.6)
ax.plot([Ta, Ta], [e_air, es_kPa(Ta)], color=c2, lw=2.2)
ax.plot(Ta, e_air, "o", color=ink, ms=7, zorder=3)
ax.plot(Td, e_air, "o", color=c3, ms=6, zorder=3)
ax.text(Ta + 0.8, e_air - 0.25, f"air sample\n$T_a$ = {Ta:.0f} °C, $e$ = {e_air} kPa", fontsize=9.5)
ax.text(Ta + 0.8, (e_air + es_kPa(Ta)) / 2, f"vapor pressure\ndeficit = {es_kPa(Ta) - e_air:.2f} kPa",
        color=c2, fontsize=9.5, va="center")
ax.text(Td - 0.6, e_air + 0.2, f"dew point\n{Td:.1f} °C", color=c3, fontsize=9.5, ha="right")
ax.set_xlabel("Temperature (°C)"); ax.set_ylabel("Vapor pressure (kPa)")
ax.set_xlim(-5, 40); ax.set_ylim(0, 7.5)
ax.legend(loc="upper left", fontsize=9.5)
plt.show()
glue("rh_ex", float(e_air / es_kPa(Ta) * 100), display=False)
```

*Figure 2.5. The saturation vapor pressure curve and one air sample (black dot). Moving left at constant vapor pressure until the curve is reached gives the dew point; the vertical gap up to the curve is the vapor pressure deficit. The relative humidity of this sample is {glue:text}`rh_ex:.0f`%.*

### Measuring humidity: the psychrometer

A **psychrometer** has two thermometers. One measures the air temperature (the dry bulb). The other is wrapped in a wet wick (the wet bulb). Evaporation from the wick cools it until the heat drawn from the passing air balances the latent heat used, at the **wet-bulb temperature** $T_w$. The drier the air, the faster the evaporation and the larger the wet-bulb depression $T_a - T_w$. An energy balance on the wick gives

\begin{equation*}
e = e_s(T_w) - \gamma_{\text{psy}}\, (T_a - T_w),
\tag{2.6}
\end{equation*}

where $\gamma_{\text{psy}} = a\, p$ is the psychrometer constant, with $a \approx 6.62 \times 10^{-4}$ °C⁻¹ for a ventilated psychrometer (FAO-56). The same group of terms, the **psychrometric constant**

\begin{equation*}
\gamma = \frac{c_p\, p}{\varepsilon\, \lambda} \approx 0.665 \times 10^{-3}\, p \quad (\text{kPa °C}^{-1}),
\tag{2.7}
\end{equation*}

appears in the Penman equation (Chapter 4). It depends on air pressure, which falls with elevation $z$ (m) roughly as

$$
p = 101.3 \left(\frac{293 - 0.0065\, z}{293}\right)^{5.26} \quad (\text{kPa}).
$$

Today most weather stations measure relative humidity electronically, but the psychrometer remains the reference instrument and its equation shows clearly how evaporation and cooling are linked.

## 2.7 Worked examples

**Example 2.1 — Net radiation and its evaporation equivalent.** On a summer day a grass surface receives a mean incoming shortwave radiation of 250 W m⁻² and incoming longwave radiation of 340 W m⁻². The albedo is 0.23, the emissivity 0.97 and the mean surface temperature 24 °C. Find the net radiation in W m⁻² and MJ m⁻² d⁻¹, and the depth of water it could evaporate if all of it were used for evaporation.

```{code-cell} python
sigma = 5.67e-8
K_in, L_in, albedo, emis, Ts = 250.0, 340.0, 0.23, 0.97, 24.0
L_out = emis * sigma * (Ts + 273.15) ** 4
Rn = (1 - albedo) * K_in + L_in - L_out
Rn_MJ = Rn * 0.0864
E_max = Rn_MJ / 2.45
print(f"L_out = {L_out:.0f} W/m²,  Rn = {Rn:.0f} W/m² = {Rn_MJ:.1f} MJ/m²/d,  E_max = {E_max:.1f} mm/d")
```

The net longwave loss is about 90 W m⁻², nearly half of the absorbed shortwave radiation. At most about 3.7 mm of water could evaporate in a day, and in practice less, because part of the energy heats the air and the ground.

**Example 2.2 — Humidity from a psychrometer.** A ventilated psychrometer at a station 150 m above sea level reads 28.0 °C (dry bulb) and 22.0 °C (wet bulb). Find the vapor pressure, relative humidity, dew point, specific humidity and vapor pressure deficit.

```{code-cell} python
z, Ta, Tw = 150.0, 28.0, 22.0
p = pressure(z)
e = es_kPa(Tw) - 6.62e-4 * p * (Ta - Tw)
RH = e / es_kPa(Ta)
Td = dewpoint(e)
q = 0.622 * e / (p - 0.378 * e)
print(f"p = {p:.1f} kPa,  e = {e:.2f} kPa,  RH = {RH*100:.0f}%,  Td = {Td:.1f} °C")
print(f"q = {q*1000:.1f} g/kg,  VPD = {es_kPa(Ta) - e:.2f} kPa")
```

**Example 2.3 — Where will clouds form?** Using the parcel model of Figure 2.3, find the cloud-base height and the lee-side temperature and relative humidity for air at 22 °C with a dew point of 16 °C crossing a 2000 m ridge.

```{code-cell} python
print(f"Cloud base ≈ {z_lcl:.0f} m")
print(f"Lee side at sea level: {T_down[-1]:.1f} °C (warmer by {T_down[-1] - T0:.1f} °C), "
      f"RH = {RH_lee*100:.0f}% (was {RH0*100:.0f}%)")
print(f"Rule of thumb, 125 m per °C of dew-point depression: {125 * (T0 - Td0):.0f} m")
```

The rule of thumb in the last line comes from the difference between the dry adiabatic lapse rate and the slower decrease of the dew point in a rising parcel (about 1.8 °C per km): $1000/(9.8 - 1.8) \approx 125$ m per °C.

**Example 2.4 — Night-time cooling.** At 3 p.m. the air is at 31 °C with a relative humidity of 45%. Overnight the temperature drops to 19 °C without any change in moisture. What is the relative humidity at dawn? Will dew form?

```{code-cell} python
e = 0.45 * es_kPa(31.0)
RH_dawn = e / es_kPa(19.0)
print(f"e = {e:.2f} kPa, dew point = {dewpoint(e):.1f} °C, RH at 19 °C = {RH_dawn*100:.0f}%")
```

The relative humidity rises to over 90% although no water has been added. Dew forms only if surfaces cool below the dew point, about 17.7 °C, which often happens on grass under a clear sky because grass loses longwave radiation faster than the air.

## 2.8 Common mistakes

- **Using °C in the Stefan–Boltzmann law.** Temperature must be in kelvin. Using 24 instead of 297 makes $L_\uparrow$ wrong by a factor of more than 20,000.
- **Confusing W m⁻² and MJ m⁻² d⁻¹.** Multiply W m⁻² by 0.0864 to get MJ m⁻² d⁻¹.
- **Treating relative humidity as a measure of water content.** Relative humidity changes with temperature alone. Use vapor pressure, specific humidity or dew point to compare moisture content.
- **Forgetting the sign convention.** In $R_n - G = \lambda E + H$, $H$ and $\lambda E$ are positive away from the surface. A negative $H$ means the air is heating the surface, as happens over an irrigated field surrounded by dry land.
- **Mixing kPa and hPa.** Vapor pressure is often given in hPa (millibars) in meteorology; 1 kPa = 10 hPa.

## 2.9 Exercises

```{code-cell} python
:tags: [remove-cell]
glue("x1_MJ", 180 * 0.0864, display=False)
glue("x1_mm", 180 * 0.0864 / (2.501 - 0.002361 * 20), display=False)
glue("x2_L", float(0.95 * 5.67e-8 * (30 + 273.15) ** 4), display=False)
e3 = float(0.55 * es_kPa(30.0))
glue("x3_e", e3, display=False); glue("x3_Td", float(dewpoint(e3)), display=False)
glue("x3_q", 1000 * 0.622 * e3 / (101.3 - 0.378 * e3), display=False)
glue("x3_vpd", float(es_kPa(30.0)) - e3, display=False)
p4 = float(pressure(800.0)); e4 = float(es_kPa(15.0)) - 6.62e-4 * p4 * 5.0
glue("x4_p", p4, display=False); glue("x4_e", e4, display=False)
glue("x4_RH", 100 * e4 / float(es_kPa(20.0)), display=False)
glue("x5_z", 125 * (26 - 18.0), display=False); glue("x5_T", 26 - 9.8 * 125 * 8 / 1000, display=False)
```

1. A lake receives a mean net radiation of 180 W m⁻² in July, and the heat stored in the water does not change. Express this in MJ m⁻² d⁻¹, and find the maximum evaporation rate (mm d⁻¹) if the water is at 20 °C.
2. How much longwave radiation (W m⁻²) does an asphalt surface at 30 °C with an emissivity of 0.95 emit?
3. Air at 30 °C has a relative humidity of 55% at sea level ($p$ = 101.3 kPa). Find its vapor pressure, dew point, specific humidity (g kg⁻¹) and vapor pressure deficit.
4. A ventilated psychrometer at 800 m elevation reads 20 °C dry bulb and 15 °C wet bulb. Find the air pressure, the vapor pressure and the relative humidity.
5. Air at sea level has a temperature of 26 °C and a dew point of 18 °C. Use the 125 m per °C rule to estimate the cloud-base height, and the dry adiabatic lapse rate to estimate the temperature there.

```{admonition} Answers
:class: dropdown
1. {glue:text}`x1_MJ:.1f` MJ m⁻² d⁻¹; at most {glue:text}`x1_mm:.1f` mm d⁻¹.
2. About {glue:text}`x2_L:.0f` W m⁻².
3. $e$ = {glue:text}`x3_e:.2f` kPa, $T_d$ = {glue:text}`x3_Td:.1f` °C, $q$ = {glue:text}`x3_q:.1f` g kg⁻¹, deficit = {glue:text}`x3_vpd:.2f` kPa.
4. $p$ = {glue:text}`x4_p:.1f` kPa, $e$ = {glue:text}`x4_e:.2f` kPa, RH = {glue:text}`x4_RH:.0f`%.
5. About {glue:text}`x5_z:.0f` m; about {glue:text}`x5_T:.1f` °C (equal to the dew point there, as it must be at saturation, to within the accuracy of the rule).
```

## 2.10 Further reading

- Dingman, S. L. (2015). *Physical Hydrology* (3rd ed.). Waveland Press. The chapters on climate, the atmosphere and the surface energy balance.
- Chow, V. T., Maidment, D. R., and Mays, L. W. (1988). *Applied Hydrology*. McGraw-Hill. The chapter on atmospheric water.
- 윤용남 (2007). *수문학: 기초와 응용* (in Korean). 청문각. The chapter on hydrometeorology.
- Allen, R. G., Pereira, L. S., Raes, D., and Smith, M. (1998). *Crop Evapotranspiration: Guidelines for Computing Crop Water Requirements*. FAO Irrigation and Drainage Paper 56. FAO, Rome. Chapter 3 gives the formulas for radiation, humidity and pressure used here.
- Monteith, J. L., and Unsworth, M. H. (2013). *Principles of Environmental Physics* (4th ed.). Academic Press.

---

*Figures in this chapter are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
