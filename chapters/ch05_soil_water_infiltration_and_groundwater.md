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

# 5. Soil water, infiltration and groundwater

```{admonition} Core ideas
:class: tip
1. Soil is a porous medium. Its **volumetric water content** $\theta$ ranges from a small residual value to the **porosity**; between **field capacity** and the **wilting point** lies the water that plants can use.
2. Water in unsaturated soil is held by capillary forces, so its pressure is below atmospheric. This negative pressure, the **matric potential** $\psi$, and the elevation together make up the **hydraulic head** that drives flow.
3. **Darcy's law** says the flux is proportional to the gradient of head. In unsaturated soil the **hydraulic conductivity** falls steeply as the soil dries. Combining Darcy's law with conservation of mass gives the **Richards equation**.
4. The **infiltration capacity** of a soil is high when it is dry and decreases toward a steady value during a storm. **Horton**, **Philip** and **Green–Ampt** describe this decrease; Green–Ampt does so from physical soil properties.
5. Under steady rain, runoff starts at the **ponding time**, when the infiltration capacity has fallen to the rainfall intensity.
6. Water that drains below the root zone **recharges** groundwater. Aquifers release it slowly to streams as **baseflow**, which keeps rivers flowing between storms.
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

## 5.1 Soil as a porous medium

Soil consists of mineral grains, organic matter and the **pores** between them, which hold air and water. Particle size defines the **texture**: sand (0.05–2 mm), silt (0.002–0.05 mm) and clay (smaller than 0.002 mm). Coarse soils have large pores that drain quickly; fine soils have many small pores that hold water tightly. Roots, worms and cracks add **macropores** that can carry water much faster than the soil matrix.

A soil sample of total volume $V$ contains a pore volume $V_v$ and a water volume $V_w$. Three quantities describe it:

- the **porosity** $n = V_v / V$, typically 0.3–0.5 for mineral soils (a soil with porosity 0.45 can hold 450 mm of water per meter of depth when saturated);
- the **volumetric water content** $\theta = V_w / V$, which ranges from a small **residual** value $\theta_r$ to the saturated value $\theta_s \approx n$;
- the **degree of saturation** $\theta / n$, or the **effective saturation** $S_e = (\theta - \theta_r)/(\theta_s - \theta_r)$, which ranges from 0 to 1.

The depth of water stored in a soil layer of thickness $Z$ is simply $\theta Z$. This is how soil water enters the water balance.

After heavy rain, water drains out of the large pores under gravity within a day or two. The water content that remains is called **field capacity** $\theta_{fc}$. As plants extract water the soil dries further until roots can no longer pull water out and the plants wilt permanently; this is the **wilting point** $\theta_{wp}$. The difference $\theta_{fc} - \theta_{wp}$ is the **plant-available water**, the quantity used for the total available water TAW in Chapter 4. Both limits are defined more precisely in terms of matric potential in the next section.

## 5.2 Soil water potential

### Hydraulic head

Water flows from points of high to low energy. In soil, velocities are tiny and kinetic energy is negligible, so the energy per unit weight of water, the **hydraulic head**, has two parts:

$$
h = \psi + z ,
$$

where $z$ is the elevation above a datum and $\psi$ is the **pressure head**. Below the water table, $\psi > 0$ and equals the depth below the water table, as in hydrostatics. Above it, water is held in the pores by surface tension, the same force that lifts water in a capillary tube (Chapter 1 of the [Hydraulics](https://daehakim-waterlab.github.io/hydraulics/) book). Its pressure is below atmospheric, so $\psi < 0$. The negative pressure head is called the **matric potential** (or **suction**, $|\psi|$). It is expressed as a head in meters or centimeters of water, or as a pressure in kPa (1 m of head ≈ 9.81 kPa).

### The water retention curve

As a soil dries, the remaining water retreats into smaller pores, where capillary forces are stronger, so the suction increases. The relation between $\theta$ and $\psi$ is the **water retention curve** (soil moisture characteristic). A widely used formula is that of van Genuchten (1980):

\begin{equation*}
\theta(\psi) = \theta_r + \frac{\theta_s - \theta_r}{\bigl[1 + (\alpha |\psi|)^{n_v}\bigr]^{m}}, \qquad m = 1 - \frac{1}{n_v},
\tag{5.1}
\end{equation*}

where $\alpha$ (m⁻¹) is related to the inverse of the suction at which the largest pores start to drain and $n_v > 1$ to the spread of pore sizes. Field capacity and wilting point are conventionally taken as the water contents at $\psi = -3.4$ m (−33 kPa) and $\psi = -153$ m (−1500 kPa).

```{code-cell} python
:tags: [hide-input]
# Illustrative van Genuchten parameters (not from a specific soil survey)
soils = {
    "sandy soil": dict(tr=0.04, ts=0.40, a=3.0, n=1.9, Ks=250.0, col=c2),
    "loam":       dict(tr=0.05, ts=0.45, a=1.5, n=1.45, Ks=12.0, col=c1),
    "clay":       dict(tr=0.10, ts=0.48, a=0.8, n=1.15, Ks=1.5, col=c3),
}
def theta_vg(psi, tr, ts, a, n, **_):
    m = 1 - 1 / n
    return tr + (ts - tr) / (1 + (a * np.abs(psi)) ** n) ** m

def K_vg(psi, tr, ts, a, n, Ks, **_):          # van Genuchten–Mualem conductivity, mm/h
    m = 1 - 1 / n
    Se = (theta_vg(psi, tr, ts, a, n) - tr) / (ts - tr)
    return Ks * Se ** 0.5 * (1 - (1 - Se ** (1 / m)) ** m) ** 2

suction = np.logspace(-2, 3, 400)               # |ψ| in m
fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 7.2))
avail = {}
for name, s in soils.items():
    a1.semilogy(theta_vg(-suction, **s), suction, color=s["col"], lw=2.2, label=name)
    a2.loglog(suction, K_vg(-suction, **s), color=s["col"], lw=2.2, label=name)
    fc, wp = theta_vg(-3.4, **s), theta_vg(-153.0, **s)
    avail[name] = (fc, wp)
for y, lab, xr in [(3.4, "field capacity (−33 kPa)", 0.385), (153.0, "wilting point (−1500 kPa)", 0.27)]:
    a1.axhline(y, color=ink2, lw=0.8, ls=":")
    a1.text(xr, y * 1.35, lab, color=ink2, fontsize=8.5, ha="right")
a1.set_xlabel("Volumetric water content θ"); a1.set_ylabel("Suction |ψ| (m)")
a1.set_xlim(0, 0.52); a1.set_ylim(0.01, 1000)
a1.set_title("(a) Water retention curves"); a1.legend(loc="lower left", fontsize=9.5)
a2.set_xlabel("Suction |ψ| (m)"); a2.set_ylabel("K (mm h⁻¹)")
a2.set_ylim(1e-6, 1e3); a2.set_xlim(0.01, 1000)
a2.set_title("(b) Unsaturated hydraulic conductivity"); a2.legend(loc="lower left", fontsize=9.5)
plt.show()
for name, (fc, wp) in avail.items():
    key = name.split()[0]
    glue(f"fc_{key}", float(fc), display=False); glue(f"wp_{key}", float(wp), display=False)
    glue(f"aw_{key}", float((fc - wp) * 1000), display=False)
```

*Figure 5.1. Water retention curves (a) and unsaturated hydraulic conductivity (b) for three illustrative soils. The sandy soil drains at low suction and holds little water at field capacity; the clay holds water even at high suction. Plant-available water per meter of soil: sandy soil {glue:text}`aw_sandy:.0f` mm, loam {glue:text}`aw_loam:.0f` mm, clay {glue:text}`aw_clay:.0f` mm. Conductivity falls by many orders of magnitude as the soil dries, and the sandy soil, most conductive when wet, becomes the least conductive when dry.*

## 5.3 Darcy's law

### Saturated flow

In 1856 Henry Darcy measured the flow of water through columns of sand for the fountains of Dijon and found that the discharge per unit area was proportional to the drop in head per unit length (see [*A Brief History*](https://daehakim-waterlab.github.io/water-history/history-of-hydrology/#groundwater-from-darcy-to-theis)). In one dimension:

\begin{equation*}
q = -K_s \frac{dh}{dx},
\tag{5.2}
\end{equation*}

where $q$ is the **Darcy flux** (volume per unit total area per unit time, with units of velocity), $K_s$ the **saturated hydraulic conductivity** and $dh/dx$ the **hydraulic gradient**. The minus sign means water flows down the gradient. The Darcy flux is not the speed of the water: water moves only through the pores, so the average **pore velocity** is $q/n_e$, where $n_e$ is the effective porosity. Saturated conductivity ranges over many orders of magnitude: roughly 10–1000 mm h⁻¹ for sands, 1–50 mm h⁻¹ for loams and less than 1 mm h⁻¹ for clays, before the effect of macropores is considered.

Darcy's law is the porous-medium version of laminar pipe flow: at the low velocities and tiny pore sizes in soils and most aquifers, the Reynolds number is small and the head loss is proportional to velocity (Chapter 4 of the [Hydraulics](https://daehakim-waterlab.github.io/hydraulics/) book).

### Unsaturated flow

Edgar Buckingham (1907) extended Darcy's law to unsaturated soil. The flux is still driven by the gradient of total head, but the conductivity now depends on the water content:

\begin{equation*}
q = -K(\psi)\,\frac{\partial h}{\partial z} = -K(\psi)\left(\frac{\partial \psi}{\partial z} + 1\right),
\tag{5.3}
\end{equation*}

with $z$ taken positive upward. When a soil dries, its water occupies only the smaller pores, and the paths through the soil become thin and tortuous. Conductivity therefore falls very steeply, by several orders of magnitude over the normal range of water contents (Figure 5.1b). This nonlinearity is the main reason soil water physics is difficult.

### The Richards equation

Conservation of mass for a thin horizontal slice of soil says that the water content increases at the rate at which more water flows in from above than leaves below: $\partial\theta/\partial t = -\partial q / \partial z$. Substituting Equation (5.3) gives the **Richards equation** (Richards, 1931; see [*A Brief History*](https://daehakim-waterlab.github.io/water-history/history-of-hydrology/#horton-and-the-physics-of-runoff)) for vertical flow:

\begin{equation*}
\frac{\partial \theta}{\partial t} = \frac{\partial}{\partial z}\left[K(\psi)\left(\frac{\partial \psi}{\partial z} + 1\right)\right].
\tag{5.4}
\end{equation*}

Together with the retention curve $\theta(\psi)$ and the conductivity function $K(\psi)$, it describes infiltration, redistribution, drainage and evaporation from the soil. It is strongly nonlinear and must usually be solved numerically. For engineering, its main role is as the physical basis for the simpler infiltration equations that follow.

## 5.4 Infiltration

**Infiltration** is the entry of water into the soil through its surface. Its rate $f$ (mm h⁻¹) depends on the supply of water and on the soil's ability to absorb it. The maximum rate at which a soil can absorb water at a given moment is its **infiltration capacity** $f_p$. If rain falls at intensity $i$,

- when $i \le f_p$, all the rain infiltrates and $f = i$;
- when $i > f_p$, water accumulates at the surface (**ponding**), $f = f_p$, and the excess $i - f_p$ becomes available for surface runoff (Chapter 6) or fills depressions.

The **cumulative infiltration** $F(t) = \int_0^t f\, dt$ is the depth that has entered the soil.

At the start of a storm a dry soil absorbs water quickly: the suction gradient at the surface pulls water in, in addition to gravity. As the wetted zone deepens, the suction gradient spreads out and weakens, and the infiltration capacity falls toward a steady value close to the saturated conductivity, at which only gravity drives the flow. All the equations below describe this decay.

### Horton's equation

Robert Horton (1933, 1940) proposed an empirical exponential decay (see [*A Brief History*](https://daehakim-waterlab.github.io/water-history/history-of-hydrology/#horton-and-the-physics-of-runoff)):

\begin{equation*}
f_p(t) = f_c + (f_0 - f_c)\,e^{-k_H t},
\qquad
F(t) = f_c\,t + \frac{f_0 - f_c}{k_H}\bigl(1 - e^{-k_H t}\bigr),
\tag{5.5}
\end{equation*}

where $f_0$ is the initial capacity, $f_c$ the final (steady) capacity and $k_H$ a decay constant (h⁻¹). The three parameters are fitted to measurements, typically from a ring infiltrometer. Horton's equation is easy to use, but its parameters have no direct physical meaning and depend on the initial wetness of the soil.

### Philip's equation

John Philip (1957) solved the Richards equation for infiltration from a ponded surface as a series in powers of $t^{1/2}$. Its first two terms give

\begin{equation*}
F(t) = S_p\,t^{1/2} + A_p\,t,
\qquad
f_p(t) = \tfrac{1}{2}\,S_p\,t^{-1/2} + A_p ,
\tag{5.6}
\end{equation*}

where $S_p$ is the **sorptivity** (mm h⁻¹ᐟ²), which measures the capillary pull of the dry soil, and $A_p$ is a constant of the order of $K_s$ (usually between $K_s/3$ and $2K_s/3$). At early times the $t^{1/2}$ term dominates, as in horizontal absorption by a sponge; at late times gravity takes over.

### The Green–Ampt model

W. H. Green and G. A. Ampt (1911) (see [*A Brief History*](https://daehakim-waterlab.github.io/water-history/history-of-hydrology/#horton-and-the-physics-of-runoff)) idealized infiltration as a sharp **wetting front** moving down into the soil: behind the front the soil is saturated, ahead of it the soil keeps its initial water content $\theta_i$ (Figure 5.2). This "piston" picture is a fair description of infiltration into dry, coarse soils and gives an equation whose parameters are physical.

```{code-cell} python
:tags: [hide-input]
fig, ax = plt.subplots(figsize=(6.0, 4.0))
theta_i, theta_s, Lf = 0.15, 0.45, 0.35
z = np.linspace(0, 0.8, 300)
# A smooth "real" profile and the Green–Ampt idealization
theta_real = theta_i + (theta_s - 0.02 - theta_i) / (1 + np.exp((z - Lf) / 0.035))
ax.plot(theta_real, z, color=c1, lw=2.2, label="Observed-type profile")
ax.plot([theta_s, theta_s, theta_i, theta_i], [0, Lf, Lf, 0.8], color=c2, lw=2.2, ls="--",
        label="Green–Ampt idealization")
ax.fill_betweenx([0, Lf], theta_i, theta_s, color=c2, alpha=0.10)
ax.text((theta_i + theta_s) / 2, Lf / 2, "$F = L\\,\\Delta\\theta$", ha="center", fontsize=11)
ax.annotate("", xy=(0.11, Lf), xytext=(0.11, 0), arrowprops=dict(arrowstyle="<->", color=ink2))
ax.text(0.10, Lf / 2, "$L$", ha="right", va="center", color=ink2, fontsize=11)
ax.text(theta_s + 0.005, Lf + 0.05, "wetting front\n(suction head $\\psi_f$)", fontsize=9.5, color=ink2)
ax.annotate("", xy=(theta_s, 0.62), xytext=(theta_i, 0.62), arrowprops=dict(arrowstyle="<->", color=ink2))
ax.text((theta_i + theta_s) / 2, 0.66, "$\\Delta\\theta = \\theta_s - \\theta_i$", ha="center", fontsize=10.5)
ax.set_ylim(0.8, -0.05); ax.set_xlim(0.05, 0.6)
ax.set_xlabel("Volumetric water content θ"); ax.set_ylabel("Depth (m)")
ax.legend(loc="lower right", fontsize=9.5)
plt.show()
```

*Figure 5.2. Soil water profile during infiltration. The Green–Ampt model replaces the gradual wetting front by a sharp step at depth $L$; the water that has infiltrated fills the shaded rectangle.*

**Derivation.** Let the surface be ponded with negligible depth, so $\psi = 0$ at $z = 0$. At the wetting front, at depth $L$, the soil water is under a suction $\psi_f$ (a positive number, the **wetting-front suction head**). Taking depth positive downward, the head at the surface is $0$ and the head at the front is $-\psi_f - L$. Darcy's law across the saturated zone gives the infiltration rate:

$$
f = K_s\,\frac{0 - (-\psi_f - L)}{L} = K_s\,\frac{\psi_f + L}{L}.
$$

The cumulative infiltration fills the wetted zone, $F = L\,\Delta\theta$ with $\Delta\theta = \theta_s - \theta_i$. Eliminating $L$:

\begin{equation*}
f_p = K_s\left(1 + \frac{\psi_f\,\Delta\theta}{F}\right).
\tag{5.7}
\end{equation*}

Since $f = dF/dt$, this is a differential equation for $F$. Separating variables and integrating from $F = 0$ at $t = 0$ gives

\begin{equation*}
K_s\,t = F - \psi_f\,\Delta\theta\,\ln\!\left(1 + \frac{F}{\psi_f\,\Delta\theta}\right).
\tag{5.8}
\end{equation*}

**Solving for $F$.** Equation (5.8) gives $t$ for a known $F$ but cannot be solved for $F$ algebraically: it is **implicit**. Newton's method converges in a few iterations. Write $a = \psi_f \Delta\theta$ and

$$
g(F) = F - a\ln\!\left(1 + \frac{F}{a}\right) - K_s t, \qquad g'(F) = \frac{F}{a + F},
$$

then iterate $F_{k+1} = F_k - g(F_k)/g'(F_k)$, starting from $F_0 = K_s t$ (or any positive value). The function below does this; Example 5.4 uses it.

```{code-cell} python
def green_ampt_F(t, Ks, psi_f, dtheta, tol=1e-10):
    """Cumulative infiltration F (mm) after t hours of ponding, Green–Ampt (Newton's method)."""
    a = psi_f * dtheta
    F = max(Ks * t, 1e-6) + np.sqrt(2 * a * Ks * t)       # a good first guess
    for _ in range(50):
        g = F - a * np.log(1 + F / a) - Ks * t
        F_new = F - g / (F / (a + F))
        if abs(F_new - F) < tol:
            return F_new
        F = max(F_new, 1e-9)
    raise RuntimeError("Newton iteration did not converge")
```

**Parameters.** The model needs $K_s$, $\psi_f$ and $\Delta\theta$. They have been tabulated by soil texture from large data sets (for example by Rawls, Brakensiek and Miller, 1983), and $\Delta\theta$ depends on how wet the soil is before the storm: a wet soil has a small $\Delta\theta$ and its infiltration capacity falls quickly to $K_s$. This dependence on antecedent wetness is physically built in, which is the main advantage of Green–Ampt over Horton.

```{code-cell} python
:tags: [hide-input]
from scipy.optimize import curve_fit
Ks, psi_f, dth = 10.0, 110.0, 0.30                     # mm/h, mm, -
t = np.linspace(0.02, 4, 200)                          # h
F_ga = np.array([green_ampt_F(tt, Ks, psi_f, dth) for tt in t])
f_ga = Ks * (1 + psi_f * dth / F_ga)

S = np.sqrt(2 * Ks * psi_f * dth)                      # sorptivity consistent with Green–Ampt
A = Ks / 2
f_ph = 0.5 * S / np.sqrt(t) + A

horton_f = lambda t, f0, fc, k: fc + (f0 - fc) * np.exp(-k * t)
(f0, fc, kH), _ = curve_fit(horton_f, t, f_ga, p0=(60, 10, 3))

fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.plot(t, f_ga, color=c1, lw=2.4, label="Green–Ampt")
ax.plot(t, f_ph, color=c2, lw=2.0, ls="--", label=f"Philip, $S_p$ = {S:.0f} mm h⁻¹ᐟ², $A_p$ = $K_s$/2")
ax.plot(t, horton_f(t, f0, fc, kH), color=c3, lw=2.0, ls="-.", label="Horton, fitted")
ax.axhline(Ks, color=ink2, lw=0.8, ls=":"); ax.text(0.05, Ks - 4.0, "$K_s$", color=ink2)
ax.set_xlabel("Time since ponding (h)"); ax.set_ylabel("Infiltration capacity $f_p$ (mm h⁻¹)")
ax.set_xlim(0, 4); ax.set_ylim(0, 60)
ax.legend(loc="upper right", fontsize=9.5)
plt.show()
glue("h_f0", float(f0), display=False); glue("h_fc", float(fc), display=False); glue("h_k", float(kH), display=False)
```

*Figure 5.3. Infiltration capacity of a soil with $K_s$ = 10 mm h⁻¹, $\psi_f$ = 110 mm and $\Delta\theta$ = 0.30 under continuous ponding. Philip's equation uses the sorptivity implied by the Green–Ampt parameters; Horton's equation was fitted to the Green–Ampt curve ($f_0$ = {glue:text}`h_f0:.0f`, $f_c$ = {glue:text}`h_fc:.1f` mm h⁻¹, $k_H$ = {glue:text}`h_k:.2f` h⁻¹). The three shapes are similar. After four hours the Green–Ampt capacity is still above $K_s$ and falling slowly, so the fitted Horton $f_c$ is larger than $K_s$: a "final" rate measured in a short test overestimates the saturated conductivity.*

## 5.5 Ponding under rainfall

The equations above assume the surface is ponded from the start. Under rain of intensity $i$, the surface ponds only when the infiltration capacity has fallen to $i$. Before that, all the rain infiltrates.

With the Green–Ampt model and constant $i > K_s$, set $f_p = i$ in Equation (5.7). The cumulative infiltration at ponding is

\begin{equation*}
F_p = \frac{K_s\,\psi_f\,\Delta\theta}{i - K_s},
\qquad
t_p = \frac{F_p}{i} .
\tag{5.9}
\end{equation*}

If $i \le K_s$, the soil can absorb the rain indefinitely and ponding never occurs. After ponding, infiltration proceeds at capacity. Because the soil already holds $F_p$, the cumulative infiltration follows Equation (5.8) written from the ponding point:

$$
K_s\,(t - t_p) = F - F_p - \psi_f\,\Delta\theta \,\ln\!\left(\frac{\psi_f\,\Delta\theta + F}{\psi_f\,\Delta\theta + F_p}\right),
$$

and the **excess rainfall** rate is $i - f_p$. Ponding time is shorter for wetter soils (smaller $\Delta\theta$), for less permeable soils and for more intense rain. This is the mechanism of **infiltration-excess** runoff (Chapter 6) and the basis of the infiltration-based loss methods of Chapter 7.

```{code-cell} python
:tags: [hide-input]
def ga_rain(i, D, Ks, psi_f, dth, dt=1 / 60):
    """Infiltration and excess under constant rain i (mm/h) for D hours (Green–Ampt with ponding)."""
    a = psi_f * dth
    Fp = Ks * a / (i - Ks) if i > Ks else np.inf
    tp = Fp / i
    t = np.arange(0, D + dt / 2, dt)
    F = np.empty_like(t)
    for k, tt in enumerate(t):
        if tt <= tp:
            F[k] = i * tt
        else:
            F[k] = F[k - 1] if k else Fp
            for _ in range(50):                        # Newton from the ponding point
                g = F[k] - Fp - a * np.log((a + F[k]) / (a + Fp)) - Ks * (tt - tp)
                F[k] -= g / (1 - a / (a + F[k]))
    return t, F, tp, Fp

i, D = 30.0, 2.0
t, F, tp, Fp = ga_rain(i, D, Ks, psi_f, dth)
f = np.gradient(F, t)
fig, ax = plt.subplots(figsize=(6.4, 3.9))
ax.axhline(i, color=c1, lw=2, label=f"Rainfall intensity {i:.0f} mm h⁻¹")
ax.plot(t, f, color=c2, lw=2.4, label="Infiltration rate")
ax.fill_between(t, f, i, where=t >= tp, color=c1, alpha=0.18, label="Excess rainfall")
ax.axvline(tp, color=ink2, lw=0.8, ls=":")
ax.text(tp + 0.03, 4, f"ponding at\n$t_p$ = {tp * 60:.0f} min", color=ink2, fontsize=9.5)
ax.set_xlabel("Time (h)"); ax.set_ylabel("mm h⁻¹"); ax.set_xlim(0, D); ax.set_ylim(0, 36)
ax.legend(loc="lower left", fontsize=9.5)
plt.show()
glue("tp_min", float(tp * 60), display=False); glue("Fp", float(Fp), display=False)
glue("F_end", float(F[-1]), display=False); glue("Pe_end", float(i * D - F[-1]), display=False)
```

*Figure 5.4. Green–Ampt infiltration under rain of 30 mm h⁻¹ for 2 hours on the soil of Figure 5.3. All the rain infiltrates until ponding at {glue:text}`tp_min:.0f` minutes, when {glue:text}`Fp:.1f` mm has entered the soil. Afterward the infiltration rate falls and the shaded excess grows. Of the 60 mm of rain, {glue:text}`F_end:.1f` mm infiltrates and {glue:text}`Pe_end:.1f` mm is excess.*

## 5.6 Groundwater

### Aquifers

Below the **water table** the pores are saturated and the pressure head is positive. A geological formation that stores groundwater and transmits it in useful amounts is an **aquifer**: sand and gravel deposits, fractured or porous rock. Layers of clay or unfractured rock that transmit water very slowly are **aquitards**. An **unconfined aquifer** has the water table as its upper boundary. A **confined aquifer** lies between aquitards and is under pressure, so water in a well rises above the top of the aquifer, to the **potentiometric surface**.

Aquifers release water from storage in different ways:

- an unconfined aquifer drains pores as the water table falls; the volume released per unit area per unit fall is the **specific yield** $S_y$, typically 0.05–0.3;
- a confined aquifer stays saturated and releases water only by compression of the skeleton and expansion of water; its **storativity** $S_c$ (storage coefficient) is tiny, typically $10^{-5}$–$10^{-3}$.

The ability of an aquifer of thickness $b$ to transmit water is its **transmissivity** $K_s\,b$ (m² d⁻¹), the product of conductivity and thickness.

### The groundwater flow equation

Applying conservation of mass to a vertical column of aquifer and using Darcy's law for horizontal flow gives, for a confined aquifer with recharge $R$ (m d⁻¹),

\begin{equation*}
S_c\,\frac{\partial h}{\partial t} = K_s\,b\left(\frac{\partial^2 h}{\partial x^2} + \frac{\partial^2 h}{\partial y^2}\right) + R .
\tag{5.10}
\end{equation*}

In words: the change of storage in the column equals the net lateral inflow plus the recharge. For an unconfined aquifer, $S_c$ becomes $S_y$ and the transmissivity is $K_s h$, which changes with the saturated thickness (the **Boussinesq equation**); if the water table varies little, Equation (5.10) is still a good approximation. The **Dupuit assumptions** (nearly horizontal flow, head constant with depth) are what make this two-dimensional form possible.

### Recharge and baseflow

**Recharge** is the water that drains below the root zone and reaches the water table. It is the part of the precipitation that is neither evaporated nor runs off quickly, typically from a few percent of annual precipitation in dry climates to a third or more in humid regions with permeable soils.

Where an aquifer meets a stream whose level is below the water table, groundwater seeps into the channel. This **baseflow** sustains streamflow during dry weather. Consider a strip of unconfined aquifer between two parallel streams $2L$ apart, both at water level $h_0$ above an impermeable base, with uniform steady recharge $R$. The Dupuit approximation gives the water table

\begin{equation*}
h^2(x) = h_0^2 + \frac{R}{K_s}\,x\,(2L - x),
\tag{5.11}
\end{equation*}

a mound highest midway between the streams, and each stream receives a baseflow of $R\,L$ per unit length of bank: in steady state, everything that recharges the aquifer comes out as baseflow (Example 5.6).

When recharge stops, the aquifer drains and baseflow declines. If the outflow is proportional to the water stored, $Q = S/k$, the water balance $dS/dt = -Q$ gives an exponential **recession**:

\begin{equation*}
Q(t) = Q_0\, e^{-t/k} .
\tag{5.12}
\end{equation*}

The aquifer then behaves as a **linear reservoir** with storage constant $k$. This simple model reappears in the analysis of recessions (Chapter 6) and of the whole catchment (Chapter 9).

## 5.7 Worked examples

**Example 5.1 — Water stored in the root zone.** A 0.8 m root zone in loam has a water content of 0.16 after a dry spell. Using the field capacity and wilting point of the loam in Figure 5.1, find the water stored, the water available to plants, and the rainfall needed to bring the root zone back to field capacity.

```{code-cell} python
Zr, theta = 0.8, 0.16
fc, wp = avail["loam"]
print(f"θ_fc = {fc:.3f}, θ_wp = {wp:.3f}")
print(f"Stored: {theta * Zr * 1000:.0f} mm; available: {(theta - wp) * Zr * 1000:.0f} mm; "
      f"deficit to field capacity: {(fc - theta) * Zr * 1000:.0f} mm")
glue("ex51_def", float((fc - theta) * Zr * 1000), display=False)
```

The first {glue:text}`ex51_def:.0f` mm or so of the next storm will be absorbed by the root zone before much water drains to groundwater, a quantity that plays the role of an initial loss in Chapter 7.

**Example 5.2 — Darcy flux through a sand column.** A vertical column of sand 0.50 m long with $K_s$ = 120 mm h⁻¹ and porosity 0.38 is kept saturated, with water ponded 0.20 m deep on top and free drainage at the bottom (pressure head zero). Find the Darcy flux, the pore velocity and the time for water to cross the column.

```{code-cell} python
Ks_s, Lc, pond, n_e = 120.0, 0.50, 0.20, 0.38
h_top, h_bot = pond + Lc, 0.0 + 0.0          # head = pressure head + elevation (datum at the bottom)
q = Ks_s * (h_top - h_bot) / Lc              # mm/h, downward
v = q / n_e
print(f"gradient = {(h_top - h_bot) / Lc:.2f}, q = {q:.0f} mm/h, pore velocity = {v:.0f} mm/h, "
      f"travel time = {Lc * 1000 / v:.1f} h")
```

**Example 5.3 — Horton infiltration.** A ring infiltrometer test gives $f_0$ = 75 mm h⁻¹, $f_c$ = 8 mm h⁻¹ and $k_H$ = 2.2 h⁻¹. Assuming continuous ponding, find the infiltration capacity after 30 minutes and the cumulative infiltration after 1 and 2 hours.

```{code-cell} python
f0, fc, kH = 75.0, 8.0, 2.2
fH = lambda t: fc + (f0 - fc) * np.exp(-kH * t)
FH = lambda t: fc * t + (f0 - fc) / kH * (1 - np.exp(-kH * t))
print(f"f(0.5 h) = {fH(0.5):.1f} mm/h,  F(1 h) = {FH(1):.1f} mm,  F(2 h) = {FH(2):.1f} mm")
```

**Example 5.4 — Green–Ampt with a ponded surface.** For the soil of Figure 5.3 ($K_s$ = 10 mm h⁻¹, $\psi_f$ = 110 mm, $\Delta\theta$ = 0.30), find the cumulative infiltration and infiltration rate after 1 h and 3 h of ponding, and the depth of the wetting front. Check the answer by putting $F$ back into Equation (5.8).

```{code-cell} python
a = psi_f * dth
for tt in (1.0, 3.0):
    F = green_ampt_F(tt, Ks, psi_f, dth)
    f = Ks * (1 + a / F)
    t_back = (F - a * np.log(1 + F / a)) / Ks
    print(f"t = {tt:.0f} h: F = {F:.1f} mm, f = {f:.1f} mm/h, front at {F / dth:.0f} mm; check t = {t_back:.6f} h")
```

**Example 5.5 — Ponding time and excess rainfall.** Rain falls at 30 mm h⁻¹ for 2 hours on the same soil. Find the ponding time and the depth of excess rainfall. Repeat for a wetter soil with $\Delta\theta$ = 0.10.

```{code-cell} python
for d in (0.30, 0.10):
    t, F, tp, Fp = ga_rain(30.0, 2.0, Ks, psi_f, d)
    print(f"Δθ = {d:.2f}: ponding after {tp * 60:.0f} min (F_p = {Fp:.1f} mm); "
          f"infiltration {F[-1]:.1f} mm, excess {60 - F[-1]:.1f} mm")
```

The same storm produces about twice as much excess rainfall on the wetter soil. Antecedent wetness is one of the main reasons why identical storms can cause very different floods.

**Example 5.6 — A groundwater mound between two streams.** Two parallel streams are 600 m apart and their water levels are 8 m above the base of an unconfined sand aquifer with $K_s$ = 6 m d⁻¹. Recharge is 450 mm per year. Find the height of the water table midway between the streams and the baseflow into each stream per kilometer of channel.

```{code-cell} python
Ks_aq, h0, L = 6.0, 8.0, 300.0                 # m/d, m, m (half the spacing)
R = 0.450 / 365                                 # m/d
x = L
h_mid = np.sqrt(h0 ** 2 + R / Ks_aq * x * (2 * L - x))
q_bank = R * L                                  # m³/d per m of bank, one side
print(f"Water table midway: {h_mid:.2f} m (mound {h_mid - h0:.2f} m above the streams)")
print(f"Baseflow per km of channel (from one side): {q_bank * 1000:,.0f} m³/d = {q_bank * 1000 / 86400 * 1000:.1f} L/s")
```

## 5.8 Common mistakes

- **Confusing Darcy flux with pore velocity.** Contaminants and tracers move at $q/n_e$, several times faster than $q$.
- **Forgetting the elevation head.** Flow follows the gradient of $\psi + z$, not of $\psi$ alone. Water at a uniform suction still drains downward under gravity.
- **Using the Green–Ampt equation from $t = 0$ under rainfall.** Before ponding, infiltration equals rainfall; the ponded-surface equation applies only after $t_p$, from the cumulative infiltration $F_p$.
- **Mixing units in Green–Ampt.** $K_s t$, $F$ and $\psi_f \Delta\theta$ must all be in the same length unit (mm here).
- **Taking $f_c$ or $K_s$ from a lab sample as catchment-scale values.** Macropores, crusts and spatial variability can change effective values by an order of magnitude.
- **Treating specific yield and storativity as interchangeable.** They differ by a factor of a thousand or more.

## 5.9 Exercises

```{code-cell} python
:tags: [remove-cell]
glue("p1_S", 0.31 * 600, display=False); glue("p1_def", (0.36 - 0.31) * 600, display=False)
glue("p2_q", 4.0 * (1.2 / 0.3), display=False)
FH5 = lambda t: 5 + (50 - 5) * np.exp(-1.8 * t)
glue("p3_f", float(FH5(1.0)), display=False); glue("p3_F", float(5 * 2 + 45 / 1.8 * (1 - np.exp(-1.8 * 2))), display=False)
Fp4 = 4.0 * 90 * 0.25 / (20 - 4.0); glue("p4_Fp", Fp4, display=False); glue("p4_tp", Fp4 / 20 * 60, display=False)
glue("p4_F2", float(green_ampt_F(2.0, 4.0, 90.0, 0.25)), display=False)
glue("p5_h", float(np.sqrt(5 ** 2 + (0.3 / 365) / 2.5 * 400 * 400)), display=False)
glue("p5_q", 0.3 / 365 * 400 * 1000, display=False)
```

1. A 0.6 m soil layer has a porosity of 0.46, a field capacity of 0.36 and a current water content of 0.31. Find the water stored (mm) and the rain needed to bring it to field capacity.
2. Two tensiometers in a uniform soil read $\psi$ = −1.5 m at 0.3 m depth and $\psi$ = −2.4 m at 0.6 m depth. The conductivity at this wetness is 4 mm h⁻¹. Find the direction and magnitude of the flux.
3. For a soil with $f_0$ = 50 mm h⁻¹, $f_c$ = 5 mm h⁻¹ and $k_H$ = 1.8 h⁻¹, find the infiltration capacity after 1 hour and the cumulative infiltration after 2 hours of ponding.
4. A soil has $K_s$ = 4 mm h⁻¹, $\psi_f$ = 90 mm and $\Delta\theta$ = 0.25. Rain falls at 20 mm h⁻¹. Find the cumulative infiltration and time at ponding. Separately, find the cumulative infiltration after 2 hours of continuous ponding from $t = 0$.
5. Two streams 800 m apart flow at 5 m above the base of an unconfined aquifer with $K_s$ = 2.5 m d⁻¹ and recharge of 300 mm per year. Find the water table height midway and the baseflow that one stream receives from this strip of aquifer, per kilometer of channel.

```{admonition} Answers
:class: dropdown
1. {glue:text}`p1_S:.0f` mm stored; {glue:text}`p1_def:.0f` mm needed.
2. Head at 0.3 m depth = −1.5 − 0.3 = −1.8 m; at 0.6 m = −2.4 − 0.6 = −3.0 m. Flow is downward at {glue:text}`p2_q:.0f` mm h⁻¹.
3. $f_p$ = {glue:text}`p3_f:.1f` mm h⁻¹; $F$ = {glue:text}`p3_F:.1f` mm.
4. $F_p$ = {glue:text}`p4_Fp:.1f` mm at $t_p$ = {glue:text}`p4_tp:.0f` min; $F$(2 h) = {glue:text}`p4_F2:.1f` mm.
5. Midway water table {glue:text}`p5_h:.2f` m; about {glue:text}`p5_q:.0f` m³ d⁻¹ per km.
```

## 5.10 Further reading

- Dingman, S. L. (2015). *Physical Hydrology* (3rd ed.). Waveland Press. The chapters on soil water, infiltration and groundwater.
- Chow, V. T., Maidment, D. R., and Mays, L. W. (1988). *Applied Hydrology*. McGraw-Hill. The chapter on subsurface water.
- 윤용남 (2007). *수문학: 기초와 응용* (in Korean). 청문각. The chapters on infiltration and groundwater.
- Green, W. H., and Ampt, G. A. (1911). Studies on soil physics, 1. The flow of air and water through soils. *Journal of Agricultural Science*, 4, 1–24.
- Philip, J. R. (1957). The theory of infiltration: 1. The infiltration equation and its solution. *Soil Science*, 83, 345–358.
- Rawls, W. J., Brakensiek, D. L., and Miller, N. (1983). Green–Ampt infiltration parameters from soils data. *Journal of Hydraulic Engineering*, 109, 62–70.
- van Genuchten, M. Th. (1980). A closed-form equation for predicting the hydraulic conductivity of unsaturated soils. *Soil Science Society of America Journal*, 44, 892–898.

---

*Figures in this chapter are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
