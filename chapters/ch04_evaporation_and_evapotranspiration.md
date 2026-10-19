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

# 4. Evaporation and evapotranspiration

```{admonition} Core ideas
:class: tip
1. Evaporation needs three things: **energy** to change liquid into vapor, a **vapor pressure deficit** and turbulence to carry the vapor away, and a **supply of water** at the surface. The scarcest of the three controls the rate.
2. **Mass-transfer** (Dalton-type) equations describe the transport of vapor; the **energy balance** describes the supply of energy. Each alone needs the surface temperature, which is rarely known.
3. **Penman** combined the two and eliminated the surface temperature. The result needs only standard weather data: net radiation, air temperature, humidity and wind speed.
4. **Penman–Monteith** adds a **surface resistance** for the stomata of plants. With fixed resistances for a reference grass it becomes the **FAO-56 reference evapotranspiration** $E_0$, the world standard. Crop coefficients $K_c$ and a water-stress coefficient $K_w$ adapt it to real crops and drying soils.
5. **Priestley–Taylor** keeps only the energy term, multiplied by 1.26. It works well for wet surfaces in humid climates.
6. **Actual** evapotranspiration $E$ is usually less than **potential** evapotranspiration $E_p$. Over many years it can be found from the catchment water balance as $E = P - Q$, and the ratios $E/P$ and $E/E_p$ summarize whether a catchment is limited by water or by energy.
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

## 4.1 Evaporation, transpiration and their controls

**Evaporation** is the change of liquid water into vapor and its transfer to the atmosphere. It happens from open water (lakes, reservoirs, rivers), from bare soil, and from rain held on leaves and branches (**interception**). **Transpiration** is evaporation from inside plant leaves: roots take water from the soil, it flows up the stem, evaporates in small cavities inside the leaf and escapes through pores called **stomata**. Plants open their stomata to take in carbon dioxide and partly close them when they are short of water or when the air is very dry. Over land these processes are hard to separate, and their sum is called **evapotranspiration**. In this book $E$ denotes evaporation or evapotranspiration, depending on the surface, always as a depth rate (mm d⁻¹).

Three conditions control the rate:

- **Energy.** Each millimeter evaporated uses about 2.45 MJ m⁻² (Chapter 2). The energy comes from net radiation and, when the air is warmer than the surface, from sensible heat drawn from the air.
- **Transport.** Vapor must be carried away, or the air next to the surface saturates and evaporation stops. This needs a vapor pressure deficit and turbulent mixing by wind.
- **Water supply.** The surface must be wet, or water must reach it from below through the soil or the plants.

Two reference rates are used to separate the climate from the surface. **Potential evapotranspiration** $E_p$ is the rate from an extensive surface that is never short of water, under the given weather. It measures the "evaporative demand" of the climate. **Actual evapotranspiration** $E$ is what really happens; $E \le E_p$, with equality only when the surface is wet. Because "an extensive wet surface" can mean open water, a wet forest or a grass lawn, each of which evaporates at a different rate, engineers now use a precisely defined grass surface, the **reference crop evapotranspiration** $E_0$ (Section 4.6).

**Measuring evaporation.** An **evaporation pan** is a shallow open tank filled with water whose level is read daily. The widely used Class A pan is 1.21 m in diameter and 25.4 cm deep. Pans evaporate more than lakes, because their thin water and metal walls heat up quickly, so pan readings are multiplied by a **pan coefficient** (typically 0.6–0.8) to estimate lake evaporation. A **weighing lysimeter**, a large container of soil and vegetation set flush with the ground and weighed continuously, measures evapotranspiration directly from the water balance. The **eddy covariance** method measures the turbulent fluxes of vapor and heat above a surface from fast measurements of vertical wind speed and humidity. Each method is expensive or local, which is why most evapotranspiration estimates come from the equations in this chapter.

## 4.2 Mass transfer

### From Fick's law to the aerodynamic resistance

In still air, vapor diffuses from high to low concentration according to **Fick's law**:

$$
F = -D_v \frac{d\rho_v}{dz},
$$

where $F$ is the vapor flux (kg m⁻² s⁻¹), $\rho_v$ the vapor density (kg m⁻³) and $D_v \approx 2.5\times10^{-5}$ m² s⁻¹ the molecular diffusivity of vapor in air. Molecular diffusion alone is far too slow to explain observed evaporation. Over a field, wind creates turbulent eddies that mix the air thousands of times faster. The flux keeps the same form with the molecular diffusivity replaced by a much larger **eddy diffusivity** $K_E$, which grows with wind speed and height.

Integrating over the height between the evaporating surface and a measurement height $z$ gives the flux in terms of the difference of concentrations, divided by a **resistance**:

$$
F = \frac{\rho_{v,s} - \rho_{v,a}}{r_a}, \qquad r_a = \int_{\text{surface}}^{z} \frac{dz'}{K_E} ,
$$

just as an electric current equals a voltage difference divided by a resistance. The **aerodynamic resistance** $r_a$ (s m⁻¹) falls as wind speed rises and as the surface gets rougher. Above a uniform crop of height $h$, the logarithmic wind profile gives

\begin{equation*}
r_a = \frac{\ln\!\left[\dfrac{z_m - d}{z_{om}}\right]\ln\!\left[\dfrac{z_h - d}{z_{oh}}\right]}{k^2\, u_z},
\tag{4.1}
\end{equation*}

where $z_m$ and $z_h$ are the heights of the wind and humidity measurements, $d \approx \tfrac{2}{3}h$ is the zero-plane displacement, $z_{om} \approx 0.123\,h$ and $z_{oh} \approx 0.1\, z_{om}$ are roughness lengths for momentum and vapor, $k = 0.41$ is von Kármán's constant and $u_z$ is the wind speed at height $z_m$. A tall forest has a resistance about ten times smaller than short grass at the same wind speed, so it exchanges vapor and heat with the air much more easily.

### Dalton-type equations

Writing vapor density in terms of vapor pressure (Chapter 2) and multiplying by $\lambda$ gives the latent heat flux:

\begin{equation*}
\lambda E = \frac{\rho_a c_p}{\gamma}\, \frac{e_s(T_s) - e_a}{r_a} ,
\tag{4.2}
\end{equation*}

where $\rho_a$ is the air density, $c_p = 1.013\times10^{-3}$ MJ kg⁻¹ °C⁻¹ the specific heat of air, $\gamma$ the psychrometric constant, $e_s(T_s)$ the saturation vapor pressure at the temperature of the wet surface and $e_a$ the actual vapor pressure of the air. The equation says the evaporation rate is proportional to the vapor pressure difference and to a function of wind speed. John Dalton reached this conclusion experimentally in 1802 (see [*A Brief History*](https://hydroclimatol.com/water-history/history-of-hydrology/#evaporation-and-the-atmosphere)), and many empirical lake-evaporation formulas have the form

$$
E = f(u)\,\bigl[e_s(T_s) - e_a\bigr], \qquad f(u) = a + b\,u .
$$

The weakness of all mass-transfer equations is the **surface temperature** $T_s$. It is not measured at ordinary weather stations, and it depends on the evaporation itself: a surface that evaporates fast cools down. Replacing $T_s$ by the air temperature can give large errors.

## 4.3 The energy balance and the Bowen ratio

The other route to $E$ is the surface energy balance of Chapter 2, $R_n - G = \lambda E + H$. The sensible heat flux has the same resistance form as the vapor flux, driven by a temperature difference:

\begin{equation*}
H = \rho_a c_p \frac{T_s - T_a}{r_a}.
\tag{4.3}
\end{equation*}

The ratio of sensible to latent heat flux is the **Bowen ratio**, $\beta = H / \lambda E$. If heat and vapor are carried by the same eddies, the resistances cancel and

$$
\beta = \gamma\,\frac{T_s - T_a}{e_s(T_s) - e_a} \approx \gamma\,\frac{\Delta T}{\Delta e} ,
$$

where $\Delta T$ and $\Delta e$ are the differences of temperature and vapor pressure measured between two heights above the surface. Substituting $H = \beta \lambda E$ into the energy balance gives

\begin{equation*}
E = \frac{R_n - G}{\lambda\,(1 + \beta)} .
\tag{4.4}
\end{equation*}

The **Bowen ratio energy balance** method uses this equation with measured $R_n$, $G$ and the two gradients. Typical Bowen ratios are about 0.1 over oceans, 0.2–0.5 over well-watered crops and forests, and 2–10 over dry land. A negative value means the air is warmer than the surface and supplies heat for evaporation, which happens over irrigated fields in a dry region.

## 4.4 The Penman equation

In 1948 Howard Penman found a way around the unknown surface temperature (see [*A Brief History*](https://hydroclimatol.com/water-history/history-of-hydrology/#evaporation-from-energy-penman-thornthwaite-and-budyko)). The idea is to use **both** the mass-transfer equation and the energy balance, and to eliminate $T_s$ between them.

**Step 1: linearize the saturation curve.** Over a small temperature range,

$$
e_s(T_s) \approx e_s(T_a) + \Delta\,(T_s - T_a),
$$

where $\Delta$ is the slope of the saturation vapor pressure curve at the air temperature (Equation (2.5) in Chapter 2).

**Step 2: rewrite the latent heat flux.** Substituting into Equation (4.2), and writing $\delta e = e_s(T_a) - e_a$ for the vapor pressure deficit of the air,

$$
\lambda E = \frac{\rho_a c_p}{\gamma\, r_a}\bigl[\Delta (T_s - T_a) + \delta e\bigr]
= \frac{\Delta}{\gamma}\, H + \frac{\rho_a c_p\, \delta e}{\gamma\, r_a},
$$

using Equation (4.3) for $H$. The unknown surface temperature is now hidden in $H$.

**Step 3: use the energy balance.** Replace $H$ by $R_n - G - \lambda E$ and solve for $\lambda E$:

\begin{equation*}
\lambda E = \frac{\Delta\,(R_n - G) + \rho_a c_p\, \delta e / r_a}{\Delta + \gamma} .
\tag{4.5}
\end{equation*}

This is the **Penman equation** for a wet surface. It contains only quantities measured at a weather station. Its two terms have a clear meaning. The first, weighted by $\Delta/(\Delta + \gamma)$, is the evaporation driven by **available energy**. The second, weighted by $\gamma/(\Delta + \gamma)$, is the evaporation driven by the **drying power of the air**. Because $\Delta$ grows with temperature (about 0.145 kPa °C⁻¹ at 20 °C against $\gamma \approx 0.067$ kPa °C⁻¹), the energy term dominates in warm weather.

For open water, Penman fitted the aerodynamic term with an empirical wind function. In SI units the commonly used form (Shuttleworth, 1993) is

\begin{equation*}
E = \frac{1}{\lambda}\left[\frac{\Delta}{\Delta + \gamma}\, R_n + \frac{\gamma}{\Delta + \gamma}\; 6.43\,(1 + 0.536\,u_2)\, \delta e\right],
\tag{4.6}
\end{equation*}

with $E$ in mm d⁻¹, $R_n$ in MJ m⁻² d⁻¹ (computed with the albedo of water, about 0.06), $u_2$ the wind speed at 2 m in m s⁻¹, $\delta e$ in kPa and $\lambda$ in MJ kg⁻¹. Heat storage in a deep lake ($G$) shifts evaporation from spring to autumn but nearly cancels over a year.

## 4.5 The Penman–Monteith equation

Leaves are not wet surfaces. Vapor inside a leaf is saturated, but to reach the air it must pass through the stomata, which adds a second resistance in series with $r_a$. John Monteith (1965) treated the whole canopy as a single **big leaf** with a **surface (canopy) resistance** $r_s$ (see [*A Brief History*](https://hydroclimatol.com/water-history/history-of-hydrology/#evaporation-from-energy-penman-thornthwaite-and-budyko)). The vapor flux becomes $(\rho_a c_p / \gamma)\,[e_s(T_s) - e_a]/(r_a + r_s)$, while heat still leaves through $r_a$ alone. Repeating the three steps of Section 4.4 gives the **Penman–Monteith equation**:

\begin{equation*}
\lambda E = \frac{\Delta\,(R_n - G) + \rho_a c_p\, \delta e / r_a}{\Delta + \gamma\,\bigl(1 + r_s / r_a\bigr)} .
\tag{4.7}
\end{equation*}

With $r_s = 0$ it reduces to the Penman equation. The surface resistance depends on how many leaves there are and how open their stomata are. A rough estimate is $r_s \approx r_l / \text{LAI}_{\text{active}}$, where $r_l$ (about 100 s m⁻¹ for a well-watered leaf) is the resistance of a single leaf and $\text{LAI}_{\text{active}}$ is the area of sunlit leaves per unit ground area. Typical values of $r_s$ are 50–100 s m⁻¹ for well-watered crops, 100–200 s m⁻¹ for forests, and several hundred or more for plants under water stress.

Figure 4.1 shows how strongly the surface resistance controls evapotranspiration under the same weather. A tall forest, with its small $r_a$, is especially sensitive to $r_s$: its evaporation is controlled more by its stomata than by the energy supply.

```{code-cell} python
:tags: [hide-input]
def es(T):                     # saturation vapor pressure, kPa (Tetens, Chapter 2)
    return 0.6108 * np.exp(17.27 * T / (T + 237.3))

def slope(T):                  # Δ, kPa/°C
    return 4098 * es(T) / (T + 237.3) ** 2

def r_a(h, u_z, z_m=2.0, z_h=2.0, k=0.41):
    """Aerodynamic resistance (s/m) above a crop of height h (m), Eq. (4.1)."""
    d, zom = 2 / 3 * h, 0.123 * h
    zoh = 0.1 * zom
    return np.log((z_m - d) / zom) * np.log((z_h - d) / zoh) / (k ** 2 * u_z)

def penman_monteith(Rn_G, T, D, ra, rs, p=101.3):
    """Evapotranspiration (mm/d) from the Penman–Monteith equation; Rn_G in MJ m⁻² d⁻¹."""
    lam, cp = 2.45, 1.013e-3
    gam = cp * p / (0.622 * lam)
    rho_a = p / (1.01 * (T + 273.0) * 0.287)
    num = slope(T) * Rn_G + rho_a * cp * D * 86400 / ra
    return num / (slope(T) + gam * (1 + rs / ra)) / lam

T, Rn_G = 25.0, 14.0
rs = np.linspace(0, 400, 201)
fig, ax = plt.subplots(figsize=(6.4, 4.0))
for D, col in [(0.6, c3), (1.5, c1), (2.5, c2)]:
    ax.plot(rs, penman_monteith(Rn_G, T, D, r_a(0.12, 2.0), rs), color=col, lw=2.2,
            label=f"grass, $\\delta e$ = {D} kPa")
ax.plot(rs, penman_monteith(Rn_G, T, 1.5, r_a(15.0, 2.0, z_m=20, z_h=20), rs), color=c1, lw=1.8,
        ls="--", label="forest, $\\delta e$ = 1.5 kPa")
ax.axvline(70, color=ink2, lw=0.8, ls=":")
ax.text(74, 0.4, "$r_s$ = 70 s m⁻¹\n(reference grass)", color=ink2, fontsize=9)
ax.set_xlabel("Surface resistance $r_s$ (s m⁻¹)"); ax.set_ylabel("$E$ (mm d⁻¹)")
ax.set_xlim(0, 400); ax.set_ylim(0, None)
ax.legend(loc="upper right", fontsize=9.5)
plt.show()
glue("ra_grass", float(r_a(0.12, 2.0)), display=False)
glue("ra_forest", float(r_a(15.0, 2.0, z_m=20, z_h=20)), display=False)
```

*Figure 4.1. Penman–Monteith evapotranspiration for $R_n - G$ = 14 MJ m⁻² d⁻¹, $T_a$ = 25 °C and a wind speed of 2 m s⁻¹, as a function of the surface resistance. For 12 cm grass $r_a$ = {glue:text}`ra_grass:.0f` s m⁻¹; for a 15 m forest (wind measured at 20 m) $r_a$ = {glue:text}`ra_forest:.0f` s m⁻¹. Drier air raises evapotranspiration, but closing stomata lowers it, and the forest responds much more steeply to its stomata.*

## 4.6 Reference evapotranspiration and crop coefficients

### The FAO-56 reference crop

To make evapotranspiration estimates comparable worldwide, the Food and Agriculture Organization (Allen et al., 1998) defined a **reference crop**: a hypothetical grass, 0.12 m tall, with a surface resistance of 70 s m⁻¹ and an albedo of 0.23, actively growing and well watered. For this surface Equation (4.1) gives $r_a = 208/u_2$ s m⁻¹, and substituting the fixed values into the Penman–Monteith equation gives the **FAO-56 reference evapotranspiration**:

\begin{equation*}
E_0 = \frac{0.408\,\Delta\,(R_n - G) + \gamma\,\dfrac{900}{T_a + 273}\,u_2\,(e_s - e_a)}{\Delta + \gamma\,(1 + 0.34\,u_2)} ,
\tag{4.8}
\end{equation*}

with $E_0$ in mm d⁻¹, $R_n$ and $G$ in MJ m⁻² d⁻¹, $T_a$ the daily mean air temperature (°C), $u_2$ the wind speed at 2 m (m s⁻¹), and $e_s$, $e_a$, $\Delta$, $\gamma$ in kPa or kPa °C⁻¹. The factor 0.408 is $1/\lambda$, the term $1 + 0.34\,u_2$ is $1 + r_s/r_a = 1 + 70\,u_2/208$, and 900 collects the constants of the aerodynamic term. For daily steps $G \approx 0$. FAO-56 also recommends averaging the saturation vapor pressure at the daily maximum and minimum temperatures, $e_s = [e_s(T_{\max}) + e_s(T_{\min})]/2$, because $e_s$ is a convex function of temperature.

**Net radiation from routine data.** Most stations measure incoming shortwave radiation $R_s$ (or sunshine hours) and temperature but not net radiation. FAO-56 estimates

$$
R_n = (1 - 0.23)\,R_s - R_{nl}, \qquad
R_{nl} = \sigma\,\frac{T_{\max,K}^4 + T_{\min,K}^4}{2}\,\bigl(0.34 - 0.14\sqrt{e_a}\bigr)\left(1.35\,\frac{R_s}{R_{so}} - 0.35\right),
$$

where $R_{nl}$ is the net longwave loss, $\sigma = 4.903\times10^{-9}$ MJ K⁻⁴ m⁻² d⁻¹, the temperatures are in kelvin, and $R_{so} = (0.75 + 2\times10^{-5} z)\,R_a$ is the clear-sky radiation at elevation $z$ (m), with $R_a$ the extraterrestrial radiation of Figure 2.1. The humidity factor accounts for longwave emission by water vapor and the ratio $R_s/R_{so}$ for cloudiness. Example 4.3 carries out the full calculation and checks that Equation (4.8) gives the same result as the general Penman–Monteith equation with $r_s$ = 70 s m⁻¹.

### Crop coefficients

Real crops differ from the reference grass in height, roughness, albedo, leaf area and stomatal behavior. FAO-56 handles this with a **crop coefficient**:

$$
E_c = K_c\, E_0 ,
$$

where $E_c$ is the evapotranspiration of the crop under good conditions. $K_c$ changes over the growing season: it is low after planting, when the soil is mostly bare; rises as the canopy develops; peaks at mid-season, often slightly above 1 for tall crops that are rougher than grass; and declines as the crop ripens (Figure 4.2). Values for many crops are tabulated in FAO-56 and should be looked up there and adjusted to local climate.

```{code-cell} python
:tags: [hide-input]
# Crop coefficient curve for a hypothetical field crop, and a synthetic summer series of E0
L = dict(ini=25, dev=35, mid=50, late=30)            # stage lengths, days
Kc_ini, Kc_mid, Kc_end = 0.35, 1.15, 0.55
t1 = L["ini"]; t2 = t1 + L["dev"]; t3 = t2 + L["mid"]; t4 = t3 + L["late"]
day = np.arange(t4 + 1)
Kc = np.interp(day, [0, t1, t2, t3, t4], [Kc_ini, Kc_ini, Kc_mid, Kc_mid, Kc_end])

rng = np.random.default_rng(5)
E0 = 4.2 + 1.0 * np.sin(np.pi * day / t4) + 0.8 * rng.standard_normal(len(day))  # mm/d
E0 = np.clip(E0, 1.0, None)
Ec = Kc * E0

fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 5.6), sharex=True)
a1.plot(day, Kc, color=c3, lw=2.4)
for x, lab in [(t1 / 2, "initial"), ((t1 + t2) / 2, "development"), ((t2 + t3) / 2, "mid-season"),
               ((t3 + t4) / 2, "late")]:
    a1.text(x, 1.3, lab, ha="center", fontsize=9.5, color=ink2)
for x in (t1, t2, t3): a1.axvline(x, color=ink2, lw=0.7, ls=":")
a1.set_ylabel("$K_c$"); a1.set_ylim(0, 1.45); a1.set_title("(a) Crop coefficient")
a2.plot(day, E0, color=ink2, lw=1.2, label="$E_0$ reference")
a2.plot(day, Ec, color=c1, lw=2.0, label="$E_c = K_c E_0$ crop")
a2.set_xlabel("Days after planting"); a2.set_ylabel("mm d⁻¹"); a2.set_xlim(0, t4)
a2.set_title("(b) Reference and crop evapotranspiration"); a2.legend(loc="upper left", fontsize=9.5, ncol=2)
a2.set_ylim(0, 10.5)
plt.show()
glue("season_E0", float(E0.sum()), display=False); glue("season_Ec", float(Ec.sum()), display=False)
```

*Figure 4.2. The four growth stages of a hypothetical field crop and its crop coefficient (a), applied to a synthetic series of daily reference evapotranspiration (b). Over the {glue:text}`season_E0:.0f` mm of reference evapotranspiration in the season, the crop uses {glue:text}`season_Ec:.0f` mm.*

### Water stress

When the soil dries, roots cannot extract water fast enough and the plants close their stomata. FAO-56 reduces the crop evapotranspiration by a **water stress coefficient**, written $K_s$ in FAO-56 and $K_w$ in this book, which reserves $K_s$ for the saturated hydraulic conductivity of Chapter 5:

$$
E_{c,\text{adj}} = K_w\,K_c\,E_0 .
$$

The water that the root zone can hold between field capacity and the wilting point (Chapter 5) is the **total available water**, $\text{TAW} = 1000\,(\theta_{fc} - \theta_{wp})\,Z_r$ in mm, where $\theta$ is the volumetric water content and $Z_r$ the rooting depth in m. Plants can extract a fraction $p$ of it (typically 0.4–0.6) without stress; this is the **readily available water** $\text{RAW} = p\,\text{TAW}$. If $D_r$ is the root-zone depletion (the water deficit below field capacity), then

\begin{equation*}
K_w = \begin{cases} 1 & D_r \le \text{RAW} \\[4pt] \dfrac{\text{TAW} - D_r}{(1 - p)\,\text{TAW}} & D_r > \text{RAW} \end{cases}
\tag{4.9}
\end{equation*}

so $K_w$ falls linearly from 1 to 0 as the soil dries from the readily available limit to the wilting point (Figure 4.3). The same idea, a reduction of evapotranspiration as soil water falls, appears in almost every hydrological model, including the bucket model of Chapter 1.

```{code-cell} python
:tags: [hide-input]
TAW, p_frac = 108.0, 0.5
Dr = np.linspace(0, TAW, 200)
Ks = np.where(Dr <= p_frac * TAW, 1.0, (TAW - Dr) / ((1 - p_frac) * TAW))
fig, ax = plt.subplots(figsize=(6.0, 3.4))
ax.plot(Dr, Ks, color=c2, lw=2.4)
ax.axvline(p_frac * TAW, color=ink2, lw=0.8, ls=":")
ax.text(p_frac * TAW + 2, 0.2, "RAW = p · TAW", color=ink2, fontsize=9.5)
ax.set_xlabel("Root-zone depletion $D_r$ (mm)"); ax.set_ylabel("$K_w$")
ax.set_xlim(0, TAW); ax.set_ylim(0, 1.1)
ax.set_xticks([0, 27, 54, 81, 108], ["0\n(field capacity)", "27", "54", "81", "108\n(wilting point)"])
plt.show()
```

*Figure 4.3. The FAO-56 water stress coefficient for a root zone with 108 mm of total available water and $p$ = 0.5. Evapotranspiration is unrestricted until half of the available water has been used, then falls linearly to zero.*

## 4.7 The Priestley–Taylor equation

Over a large, wet region the air that flows over the surface has itself been moistened by it, and the vapor pressure deficit becomes small. In the limit $\delta e \to 0$, the Penman equation keeps only its energy term. This **equilibrium evaporation** is

$$
E_{\text{eq}} = \frac{\Delta}{\Delta + \gamma}\,\frac{R_n - G}{\lambda} .
$$

Priestley and Taylor (1972) analyzed measurements over oceans, lakes and well-watered land and found that evaporation was consistently larger than $E_{\text{eq}}$ by a nearly constant factor (see [*A Brief History*](https://hydroclimatol.com/water-history/history-of-hydrology/#evaporation-from-energy-penman-thornthwaite-and-budyko)):

\begin{equation*}
E_{PT} = \alpha\,\frac{\Delta}{\Delta + \gamma}\,\frac{R_n - G}{\lambda}, \qquad \alpha \approx 1.26 .
\tag{4.10}
\end{equation*}

The **Priestley–Taylor equation** needs only net radiation and temperature, which is a great advantage where humidity and wind data are poor or where they come from a climate model. Its price is that it ignores the drying power of the air. It underestimates evaporation in dry, windy climates, where $\alpha$ of 1.5–1.7 or more may be needed, and it overestimates for surfaces whose stomata restrict transpiration.

Figure 4.4 compares the methods over a synthetic year at a mid-latitude site with a humid summer.

```{code-cell} python
:tags: [hide-input]
def Ra_daily(lat_deg, doy):                          # MJ m⁻² d⁻¹, FAO-56 (as in Chapter 2)
    phi = np.radians(lat_deg); dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    dl = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dl), -1, 1))
    return 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(phi) * np.sin(dl) + np.cos(phi) * np.cos(dl) * np.sin(ws))

def fao56_E0(Tmax, Tmin, ea, Rs, u2, Ra, z=50.0):
    """FAO-56 reference evapotranspiration (mm/d) and the net radiation used (MJ m⁻² d⁻¹)."""
    T = (Tmax + Tmin) / 2
    p = 101.3 * ((293 - 0.0065 * z) / 293) ** 5.26
    gam = 0.665e-3 * p
    es_ = (es(Tmax) + es(Tmin)) / 2
    Rso = (0.75 + 2e-5 * z) * Ra
    Rnl = 4.903e-9 * ((Tmax + 273.16) ** 4 + (Tmin + 273.16) ** 4) / 2 \
          * (0.34 - 0.14 * np.sqrt(ea)) * (1.35 * np.clip(Rs / Rso, 0.25, 1.0) - 0.35)
    Rn = 0.77 * Rs - Rnl
    E0 = (0.408 * slope(T) * Rn + gam * 900 / (T + 273) * u2 * (es_ - ea)) / (slope(T) + gam * (1 + 0.34 * u2))
    return E0, Rn

# A synthetic year at 36° N with a cloudy, humid summer
rng = np.random.default_rng(21)
doy = np.arange(1, 366)
season = np.sin(2 * np.pi * (doy - 105) / 365)
Tmax = 18 + 13 * season + 2.0 * rng.standard_normal(365)
Tmin = Tmax - 9 + 2 * np.clip(season, 0, None) + 1.0 * rng.standard_normal(365)
ea = es(Tmin - 2.0 + 2.0 * np.clip(season, 0, None))                 # humid summer
cloud = 0.30 + 0.25 * np.exp(-((doy - 195) / 25.0) ** 2) + 0.15 * rng.random(365)
Ra = Ra_daily(36.0, doy)
Rs = (0.25 + 0.50 * (1 - cloud)) * Ra
u2 = np.clip(2.0 + 0.6 * rng.standard_normal(365), 0.5, None)

E0, Rn_grass = fao56_E0(Tmax, Tmin, ea, Rs, u2, Ra)
T = (Tmax + Tmin) / 2
gam = 0.665e-3 * 100.7
E_pt = 1.26 * slope(T) / (slope(T) + gam) * Rn_grass / 2.45
Rn_water = 0.94 * Rs - (0.77 * Rs - Rn_grass)                         # albedo 0.06, same longwave loss
D = (es(Tmax) + es(Tmin)) / 2 - ea
E_pen = (slope(T) / (slope(T) + gam) * Rn_water + gam / (slope(T) + gam) * 6.43 * (1 + 0.536 * u2) * D) / 2.45

smooth = lambda x: np.convolve(np.r_[x[-15:], x, x[:15]], np.ones(31) / 31, mode="valid")
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.plot(doy, smooth(E_pen), color=c1, lw=2.2, label="Penman, open water")
ax.plot(doy, smooth(E0), color=c2, lw=2.2, label="FAO-56 reference $E_0$")
ax.plot(doy, smooth(E_pt), color=c3, lw=2.2, label="Priestley–Taylor (grass $R_n$)")
ax.set_xticks([1, 91, 182, 274, 365], ["Jan", "Apr", "Jul", "Oct", "Dec"])
ax.set_xlim(1, 365); ax.set_ylim(0, None)
ax.set_ylabel("mm d⁻¹ (31-day running mean)")
ax.legend(loc="upper left", fontsize=9.5)
plt.show()
for k, v in dict(yr_E0=E0.sum(), yr_pen=E_pen.sum(), yr_pt=E_pt.sum()).items():
    glue(k, float(v), display=False)
```

*Figure 4.4. Three estimates of potential evaporation over a synthetic year at 36° N with a cloudy, humid summer. Annual totals: open-water Penman {glue:text}`yr_pen:.0f` mm, FAO-56 reference {glue:text}`yr_E0:.0f` mm, Priestley–Taylor {glue:text}`yr_pt:.0f` mm. Open water absorbs more radiation (lower albedo) and has no stomatal resistance, so it evaporates most. The cloudy rainy season flattens the summer peak.*

## 4.8 Actual evapotranspiration from the water balance

The equations above estimate potential rates. **Actual** evapotranspiration over a catchment is limited by water as well as energy, and the most reliable estimate of its long-term mean comes from the water balance of Chapter 1:

$$
\bar{E} = \bar{P} - \bar{Q} ,
$$

valid over many years when storage change and groundwater exchange are negligible. Comparing $\bar{E}$ with $\bar{E}_p$ and $\bar{P}$ shows what limits it:

- the **evaporative ratio** $\bar{E}/\bar{P}$ is the fraction of precipitation returned to the air (and $1 - \bar{E}/\bar{P}$ is the runoff ratio);
- the ratio $\bar{E}/\bar{E}_p$ shows how close the catchment comes to its potential rate;
- the **aridity index** $\phi = \bar{E}_p / \bar{P}$ compares the energy supply with the water supply.

Two limits are obvious: $\bar{E}$ cannot exceed $\bar{P}$ (the **water limit**) or $\bar{E}_p$ (the **energy limit**). Mikhail Budyko observed that catchments around the world lie close to a single curve beneath these limits (see [*A Brief History*](https://hydroclimatol.com/water-history/history-of-hydrology/#evaporation-from-energy-penman-thornthwaite-and-budyko)). A one-parameter form of this curve, due to Fu (1981), is

\begin{equation*}
\frac{\bar{E}}{\bar{P}} = 1 + \phi - \bigl(1 + \phi^{\,w}\bigr)^{1/w} ,
\tag{4.11}
\end{equation*}

where $w > 1$ reflects catchment properties such as soil water storage capacity and vegetation; a larger $w$ means more evapotranspiration for the same climate. In humid catchments ($\phi < 1$) evapotranspiration is limited mainly by energy and $\bar{E}/\bar{E}_p$ is close to 1. In arid catchments ($\phi > 1$) it is limited by water, $\bar{E}/\bar{P}$ approaches 1 and almost no runoff is produced (Figure 4.5).

```{code-cell} python
:tags: [hide-input]
fu = lambda phi, w: 1 + phi - (1 + phi ** w) ** (1 / w)
phi = np.linspace(0.01, 4, 400)
rng = np.random.default_rng(8)
phi_c = np.exp(rng.uniform(np.log(0.3), np.log(3.5), 40))      # 40 synthetic catchments
w_c = rng.uniform(1.8, 3.5, 40)
EP_c = fu(phi_c, w_c) * (1 + 0.04 * rng.standard_normal(40))

fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot([0, 1, 4], [0, 1, 1], color=ink, lw=1.4, ls="--")
ax.text(0.05, 0.78, "energy limit\n$E = E_p$", color=ink2, fontsize=9.5)
ax.text(2.8, 1.03, "water limit $E = P$", color=ink2, fontsize=9.5)
for w, col in [(1.8, c2), (2.6, c1), (3.5, c3)]:
    ax.plot(phi, fu(phi, w), color=col, lw=2, label=f"Fu, w = {w}")
ax.scatter(phi_c, EP_c, s=18, color=ink2, zorder=3, label="Synthetic catchments")
ax.set_xlabel("Aridity index $\\phi = E_p / P$"); ax.set_ylabel("Evaporative ratio $E / P$")
ax.set_xlim(0, 4); ax.set_ylim(0, 1.15)
ax.legend(loc="lower right", fontsize=9.5)
plt.show()
```

*Figure 4.5. The Budyko framework. Long-term evapotranspiration is bounded by the energy limit and the water limit, and catchments lie beneath them along curves such as Fu's equation. The points are synthetic catchments with different climates and values of $w$.*

## 4.9 Worked examples

**Example 4.1 — Bowen ratio energy balance.** Over a soybean field, the available energy on a summer day is $R_n - G$ = 14.0 MJ m⁻² d⁻¹. Between 0.5 m and 2.0 m above the canopy, the mean air temperature decreases by 0.60 °C and the vapor pressure by 0.25 kPa. The station is near sea level. Find the Bowen ratio, the latent heat flux and the evapotranspiration.

```{code-cell} python
Rn_G, dT, de = 14.0, 0.60, 0.25
gam = 0.665e-3 * 101.3                      # kPa/°C
beta = gam * dT / de
LE = Rn_G / (1 + beta)
print(f"γ = {gam:.4f} kPa/°C, β = {beta:.3f}, λE = {LE:.2f} MJ/m²/d, E = {LE / 2.45:.2f} mm/d")
```

Most of the available energy goes into evapotranspiration, as expected for a well-watered crop.

**Example 4.2 — Evaporation from a reservoir.** On a July day a reservoir near sea level receives $R_s$ = 21 MJ m⁻² d⁻¹ of shortwave radiation, and its net longwave loss is 4.0 MJ m⁻² d⁻¹. The mean air temperature is 26 °C, the vapor pressure deficit 1.1 kPa and $u_2$ = 2.5 m s⁻¹. Estimate the evaporation with Equation (4.6), and the volume lost from a 4.5 km² reservoir.

```{code-cell} python
Rs, Rnl, T, D, u2, area = 21.0, 4.0, 26.0, 1.1, 2.5, 4.5e6
Rn = (1 - 0.06) * Rs - Rnl
gam = 0.665e-3 * 101.3
w_rad, w_aero = slope(T) / (slope(T) + gam), gam / (slope(T) + gam)
E_rad = w_rad * Rn / 2.45
E_aero = w_aero * 6.43 * (1 + 0.536 * u2) * D / 2.45
E = E_rad + E_aero
print(f"Rn = {Rn:.1f} MJ/m²/d; weights {w_rad:.2f} / {w_aero:.2f}")
print(f"E = {E_rad:.2f} (radiation) + {E_aero:.2f} (aerodynamic) = {E:.2f} mm/d")
print(f"Volume lost = {E / 1000 * area:,.0f} m³/d  ({E / 1000 * area / 86400:.2f} m³/s)")
```

About a quarter of the evaporation is driven by the drying power of the air, even on a sunny day. A steady loss of a third of a cubic meter per second is significant for a small water-supply reservoir in a dry month.

**Example 4.3 — FAO-56 reference evapotranspiration, step by step.** A station at 36° N and 50 m elevation records on 15 July (day 196): $T_{\max}$ = 31.0 °C, $T_{\min}$ = 22.5 °C, actual vapor pressure 2.45 kPa, incoming shortwave radiation 19.5 MJ m⁻² d⁻¹ and $u_2$ = 1.8 m s⁻¹. Compute $E_0$, and check it against the general Penman–Monteith equation with $r_s$ = 70 s m⁻¹ and $r_a$ from Equation (4.1).

```{code-cell} python
lat, z, J = 36.0, 50.0, 196
Tmax, Tmin, ea, Rs, u2 = 31.0, 22.5, 2.45, 19.5, 1.8
Ra = Ra_daily(lat, J)
E0, Rn = fao56_E0(Tmax, Tmin, ea, Rs, u2, Ra, z)
T = (Tmax + Tmin) / 2
es_ = (es(Tmax) + es(Tmin)) / 2
p = 101.3 * ((293 - 0.0065 * z) / 293) ** 5.26
print(f"Ra = {Ra:.1f}, Rso = {(0.75 + 2e-5 * z) * Ra:.1f}, Rn = {Rn:.2f} MJ/m²/d")
print(f"T = {T:.2f} °C, Δ = {slope(T):.4f} kPa/°C, γ = {0.665e-3 * p:.4f} kPa/°C, es - ea = {es_ - ea:.3f} kPa")
print(f"FAO-56:          E0 = {E0:.2f} mm/d")
E_pm = penman_monteith(Rn, T, es_ - ea, r_a(0.12, u2), 70.0, p)
print(f"General P–M:     E  = {E_pm:.2f} mm/d  (r_a = {r_a(0.12, u2):.0f} s/m, 208/u2 = {208 / u2:.0f} s/m)")
```

The two forms agree to within a few hundredths of a millimeter; the small difference comes from the rounded constants in Equation (4.8).

**Example 4.4 — A crop under water stress.** In mid-season a crop has $K_c$ = 1.15. The root zone holds TAW = 108 mm with $p$ = 0.5, and 70 mm has been depleted since the last rain. With $E_0$ = 5.4 mm d⁻¹, find the crop evapotranspiration with and without stress.

```{code-cell} python
Kc, TAW, p_frac, Dr, E0_mid = 1.15, 108.0, 0.5, 70.0, 5.4
Ks = 1.0 if Dr <= p_frac * TAW else (TAW - Dr) / ((1 - p_frac) * TAW)
print(f"RAW = {p_frac * TAW:.0f} mm, K_w = {Ks:.2f}")
print(f"E_c = {Kc * E0_mid:.2f} mm/d unstressed, {Ks * Kc * E0_mid:.2f} mm/d with stress")
```

**Example 4.5 — Priestley–Taylor versus Penman–Monteith.** For the weather of Example 4.3, compute the Priestley–Taylor estimate with the same net radiation and compare it with $E_0$.

```{code-cell} python
gam = 0.665e-3 * p
E_eq = slope(T) / (slope(T) + gam) * Rn / 2.45
print(f"Equilibrium E = {E_eq:.2f} mm/d, Priestley–Taylor = {1.26 * E_eq:.2f} mm/d, FAO-56 E0 = {E0:.2f} mm/d")
print(f"α that would match E0: {E0 / E_eq:.2f}")
```

On this humid summer day the two methods are close. On a dry, windy day the aerodynamic term would make $E_0$ larger and the Priestley–Taylor coefficient would need to be higher than 1.26.

**Example 4.6 — Catchment water balance.** A 410 km² catchment has a 30-year mean precipitation of 1350 mm, a mean flow of 8.9 m³ s⁻¹ and a mean annual reference evapotranspiration of 880 mm. Find the mean actual evapotranspiration, the ratios $E/P$ and $E/E_p$, the aridity index, and the value of $w$ in Fu's equation that matches this catchment.

```{code-cell} python
from scipy.optimize import brentq
A, P, Qm3, Ep = 410.0, 1350.0, 8.9, 880.0
Q = Qm3 * 365.25 * 86400 / (A * 1e6) * 1000          # mm/yr
E = P - Q
phi = Ep / P
w = brentq(lambda w: fu(phi, w) - E / P, 1.01, 20)
print(f"Q = {Q:.0f} mm, E = {E:.0f} mm")
print(f"E/P = {E / P:.2f}, E/Ep = {E / Ep:.2f}, aridity index = {phi:.2f}, Fu w = {w:.2f}")
```

The catchment is humid ($\phi < 1$), yet its actual evapotranspiration is only about three quarters of the reference rate. Part of the gap is real (forests on steep slopes with thin soils dry out in short dry spells), and part comes from uncertainty in catchment precipitation, which is usually underestimated in mountains (Chapter 3).

## 4.10 Common mistakes

- **Using air temperature as the surface temperature** in a mass-transfer equation. The Penman approach exists precisely to avoid this.
- **Mixing time steps.** Equation (4.8) is written for daily data with $R_n$ in MJ m⁻² d⁻¹. For hourly data the constants change (900 becomes 37) and $G$ cannot be neglected.
- **Computing $e_s$ from the mean temperature.** Because $e_s(T)$ is convex, $e_s(\bar{T})$ underestimates the mean saturation vapor pressure and therefore the deficit. Average $e_s(T_{\max})$ and $e_s(T_{\min})$.
- **Treating $E_0$ as actual evapotranspiration.** $E_0$ is a climatic demand for a reference grass. Actual evapotranspiration needs $K_c$, $K_w$ or a water balance.
- **Using a pan reading as lake evaporation.** Apply a pan coefficient.
- **Applying $\alpha$ = 1.26 everywhere.** The Priestley–Taylor coefficient is for wet surfaces in humid air.

## 4.11 Exercises

```{code-cell} python
:tags: [remove-cell]
g1 = 0.665e-3 * 101.3; b1 = g1 * 1.2 / 0.15
glue("q1_beta", b1, display=False); glue("q1_E", 11.0 / (1 + b1) / 2.45, display=False)
g2 = 0.665e-3 * 101.3; T2 = 20.0
w2 = float(slope(T2) / (slope(T2) + g2))
E2 = w2 * 12.0 / 2.45 + (1 - w2) * 6.43 * (1 + 0.536 * 3.0) * 0.9 / 2.45
glue("q2_E", E2, display=False); glue("q2_mm", E2 * 31, display=False)
E3, _ = fao56_E0(28.0, 18.0, 1.8, 22.0, 2.5, Ra_daily(37.5, 152), 120.0)
glue("q3_E0", float(E3), display=False)
glue("q4_Ks", (120 - 84) / (0.45 * 120), display=False); glue("q4_E", (120 - 84) / (0.45 * 120) * 1.05 * 4.8, display=False)
Q5 = 1.8 * 365 * 86400 / 95e6 * 1000
glue("q5_Q", Q5, display=False); glue("q5_E", 1180 - Q5, display=False)
glue("q5_EP", (1180 - Q5) / 1180, display=False); glue("q5_fu", float(fu(780 / 1180, 2.6) * 1180), display=False)
```

1. Above a wheat field, temperature decreases by 1.2 °C and vapor pressure by 0.15 kPa between two heights, and $R_n - G$ = 11 MJ m⁻² d⁻¹ at sea level. Find the Bowen ratio and the evapotranspiration. What does the large Bowen ratio suggest about the field?
2. A lake at sea level has a mean net radiation of 12 MJ m⁻² d⁻¹ in a month with a mean air temperature of 20 °C, a vapor pressure deficit of 0.9 kPa and $u_2$ = 3.0 m s⁻¹. Use Equation (4.6) to find the evaporation rate and the total for a 31-day month.
3. A station at 37.5° N and 120 m elevation records on 1 June (day 152): $T_{\max}$ = 28 °C, $T_{\min}$ = 18 °C, $e_a$ = 1.8 kPa, $R_s$ = 22 MJ m⁻² d⁻¹, $u_2$ = 2.5 m s⁻¹. Compute $E_0$.
4. A crop with $K_c$ = 1.05 grows in a root zone with TAW = 120 mm and $p$ = 0.55. The depletion is 84 mm and $E_0$ = 4.8 mm d⁻¹. Find $K_w$ and the actual crop evapotranspiration.
5. A 95 km² catchment receives 1180 mm of precipitation a year, its mean flow is 1.8 m³ s⁻¹, and its potential evapotranspiration is 780 mm a year. Find the mean actual evapotranspiration and $E/P$, and compare it with the prediction of Fu's equation with $w$ = 2.6.

```{admonition} Answers
:class: dropdown
1. $\beta$ = {glue:text}`q1_beta:.2f`; $E$ = {glue:text}`q1_E:.2f` mm d⁻¹. About a third of the available energy heats the air: the crop is probably short of water.
2. {glue:text}`q2_E:.2f` mm d⁻¹, about {glue:text}`q2_mm:.0f` mm in the month.
3. $E_0$ ≈ {glue:text}`q3_E0:.2f` mm d⁻¹.
4. $K_w$ = {glue:text}`q4_Ks:.2f`; $E_{c,\text{adj}}$ = {glue:text}`q4_E:.2f` mm d⁻¹.
5. $Q$ = {glue:text}`q5_Q:.0f` mm, $E$ = {glue:text}`q5_E:.0f` mm, $E/P$ = {glue:text}`q5_EP:.2f`; Fu's equation gives {glue:text}`q5_fu:.0f` mm.
```

## 4.12 Further reading

- Dingman, S. L. (2015). *Physical Hydrology* (3rd ed.). Waveland Press. The chapter on evapotranspiration.
- Chow, V. T., Maidment, D. R., and Mays, L. W. (1988). *Applied Hydrology*. McGraw-Hill. The sections on evaporation and evapotranspiration.
- 윤용남 (2007). *수문학: 기초와 응용* (in Korean). 청문각. The chapter on evaporation and transpiration.
- Allen, R. G., Pereira, L. S., Raes, D., and Smith, M. (1998). *Crop Evapotranspiration: Guidelines for Computing Crop Water Requirements*. FAO Irrigation and Drainage Paper 56. FAO, Rome.
- Shuttleworth, W. J. (1993). Evaporation. In D. R. Maidment (Ed.), *Handbook of Hydrology* (pp. 4.1–4.53). McGraw-Hill.
- Penman, H. L. (1948). Natural evaporation from open water, bare soil and grass. *Proceedings of the Royal Society of London A*, 193, 120–145.
- Monteith, J. L. (1965). Evaporation and environment. *Symposia of the Society for Experimental Biology*, 19, 205–234.
- Priestley, C. H. B., and Taylor, R. J. (1972). On the assessment of surface heat flux and evaporation using large-scale parameters. *Monthly Weather Review*, 100, 81–92.
- Fu, B. P. (1981). On the calculation of the evaporation from land surface. *Scientia Atmospherica Sinica*, 5, 23–31 (in Chinese).

---

*Figures in this chapter are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
