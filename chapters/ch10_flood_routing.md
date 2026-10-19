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

# 10. Flood routing

```{admonition} Core ideas
:class: tip
1. **Flood routing** computes how a flood hydrograph changes as it passes through a reservoir or along a river. Storage delays the peak (**translation**) and lowers it (**attenuation**).
2. All **hydrologic routing** methods solve the same continuity equation, $I - O = dS/dt$, with a different assumption about storage $S$.
3. In a reservoir with a free spillway, storage and outflow both depend only on the water level (**level-pool routing**). The **storage-indication method** solves the routing step by step, and the peak outflow occurs where the outflow hydrograph crosses the falling inflow.
4. In a river reach, storage depends on both inflow and outflow. The **Muskingum** method assumes $S = K[xI + (1 - x)O]$ and gives an explicit routing formula whose three coefficients add up to 1.
5. A cascade of **linear reservoirs** is a flexible alternative with the same physical meaning: $K$ or $nk$ is the travel time of the reach.
6. **Hydraulic routing** solves the **Saint-Venant equations**. Their simplified forms, the **kinematic** and **diffusion waves**, explain what the hydrologic methods represent: the kinematic wave only translates the flood, and the diffusion term attenuates it.
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

## 10.1 What routing does

Chapter 9 produced a flood hydrograph at the outlet of a catchment. Engineers usually need the flood somewhere else: below a dam, at a town further downstream, or at the confluence with another river. As a flood wave moves through a reservoir or a river reach, water is temporarily stored. Storage has two effects (Figure 10.2 shows both):

- **attenuation**: the peak outflow is lower than the peak inflow, because part of the inflow volume is held back while the peak passes;
- **translation** (lag): the peak arrives later, because water takes time to travel and to fill and empty the storage.

The volume of the flood is not changed, only redistributed in time (ignoring losses such as evaporation or seepage).

**Hydrologic routing** treats the reservoir or reach as a lumped control volume and uses only the continuity equation plus an assumed storage relation. **Hydraulic routing** solves the equations of unsteady open-channel flow along the river. Both start from conservation of mass for the control volume:

\begin{equation}
I(t) - O(t) = \frac{dS}{dt},
\tag{10.1}
\end{equation}

where $I$ is the inflow, $O$ the outflow and $S$ the storage. Over a time step $\Delta t$, with the flows varying linearly within the step,

\begin{equation}
\frac{I_1 + I_2}{2} - \frac{O_1 + O_2}{2} = \frac{S_2 - S_1}{\Delta t},
\tag{10.2}
\end{equation}

where subscripts 1 and 2 denote the beginning and end of the step. At each step $I_1$, $I_2$, $O_1$ and $S_1$ are known, and $O_2$ and $S_2$ are unknown. One equation is not enough; the second comes from a **storage relation** linking $S$ to $O$ (and possibly $I$). The methods in this chapter differ only in that relation.

## 10.2 Reservoir routing

### Storage and outflow both depend on the water level

In a reservoir whose water surface is nearly horizontal (a **level pool**), the storage depends only on the water level $z$, through the **storage–elevation** curve, obtained by integrating the surface area of the reservoir over elevation. If the outflow is uncontrolled, for example over a free spillway crest, it also depends only on $z$. A spillway of crest length $L_w$ with crest elevation $z_c$ discharges

$$
O = C_w\, L_w\,(z - z_c)^{3/2}, \qquad z > z_c,
$$

where $C_w$ is a discharge coefficient (about 2.0–2.2 m¹ᐟ² s⁻¹ for an ogee crest; see the section on [weirs](https://daehakim-waterlab.github.io/hydraulics/chapters/ch06_flow_in_open_channels.html#flow-measurement-with-weirs) in the Hydraulics book). Eliminating $z$ between the two curves gives a single **storage–outflow** relation $S(O)$.

### The storage-indication method

Rearrange Equation (10.2) so that all unknowns are on the left:

\begin{equation}
\frac{2 S_2}{\Delta t} + O_2 = (I_1 + I_2) + \left(\frac{2 S_1}{\Delta t} - O_1\right).
\tag{10.3}
\end{equation}

Everything on the right is known. From the storage–outflow relation, tabulate the function $2S/\Delta t + O$ against $O$ once. At each step, compute the right side, read $O_2$ from the table by interpolation, compute $2S_2/\Delta t - O_2 = (2S_2/\Delta t + O_2) - 2O_2$ for the next step, and move on. This is the **storage-indication** (or modified Puls) method. The time step should be short enough to describe the rising limb well, typically one fifth of the time to peak of the inflow or less.

```{code-cell} python
# Reservoir geometry: surface area grows with elevation above the spillway crest (z = 0 at the crest)
A0, dA = 2.6e6, 0.35e6                   # m² at the crest, increase per meter
S_of_z = lambda z: A0 * z + 0.5 * dA * z ** 2               # storage above the crest, m³
Cw, Lw = 2.1, 45.0                        # spillway coefficient (m^0.5/s) and length (m)
O_of_z = lambda z: Cw * Lw * np.clip(z, 0, None) ** 1.5      # m³/s

def inflow_hydrograph(t, Qb=20.0, Qp=420.0, tp=12.0, m=4.0):
    """Gamma-shaped inflow (m³/s) with base flow Qb, peak Qp at tp hours."""
    return Qb + (Qp - Qb) * (t / tp) ** m * np.exp(m * (1 - t / tp))

def level_pool(I, dt_h, z0=0.0):
    dt = dt_h * 3600
    zz = np.linspace(0, 8, 4001)
    SI = 2 * S_of_z(zz) / dt + O_of_z(zz)                     # storage-indication table
    z = np.empty(len(I)); O = np.empty(len(I)); z[0] = z0; O[0] = O_of_z(z0)
    for n in range(len(I) - 1):
        rhs = I[n] + I[n + 1] + (2 * S_of_z(z[n]) / dt - O[n])
        z[n + 1] = np.interp(rhs, SI, zz)
        O[n + 1] = O_of_z(z[n + 1])
    return z, O

dt_h = 0.5
t_res = np.arange(0, 72 + dt_h, dt_h)
z_start = (20.0 / (Cw * Lw)) ** (2 / 3)                      # level at which base flow passes the spillway
I_res = inflow_hydrograph(t_res)
z_res, O_res = level_pool(I_res, dt_h, z_start)
```

```{code-cell} python
:tags: [hide-input]
from scipy.integrate import solve_ivp
# Independent check: integrate dS/dt = I - O(z) with an adaptive ODE solver
dzdt = lambda tt, zz: (inflow_hydrograph(tt / 3600) - O_of_z(zz[0])) / (A0 + dA * zz[0])
sol = solve_ivp(dzdt, (0, t_res[-1] * 3600), [z_start], t_eval=t_res * 3600, rtol=1e-8, atol=1e-10)
O_ode = O_of_z(sol.y[0])
t, I, O, z = t_res, I_res, O_res, z_res                      # short names for plotting
ip, iq = I.argmax(), O.argmax()
fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 6.0), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
a1.plot(t, I, color=c1, lw=2.2, label="Inflow")
a1.plot(t, O, color=c2, lw=2.2, label="Outflow (storage indication)")
a1.plot(t, O_ode, color=ink, lw=1, ls=":", label="Outflow (ODE solver check)")
a1.fill_between(t, O, I, where=I > O, color=c1, alpha=0.12)
a1.fill_between(t, O, I, where=I < O, color=c2, alpha=0.12)
a1.annotate("peak outflow where\noutflow crosses inflow", (t[iq], O[iq]), xytext=(t[iq] + 12, O[iq] - 90),
            fontsize=9.5, arrowprops=dict(arrowstyle="-", color=ink2, lw=0.8))
a1.set_ylabel("Q (m³ s⁻¹)"); a1.legend(loc="upper right", fontsize=9.5); a1.set_ylim(0, 470)
a2.plot(t, z, color=c3, lw=2.2); a2.set_ylabel("Head on crest (m)")
a2.set_xlabel("Time (h)"); a2.set_xlim(0, 72)
plt.show()
V_in = np.trapezoid(I, t * 3600); V_out = np.trapezoid(O, t * 3600) + S_of_z(z[-1]) - S_of_z(z[0])
assert abs(V_in - V_out) / V_in < 1e-3, (V_in, V_out)        # mass balance closes
for k_, v in dict(r_Ip=I.max(), r_Op=O.max(), r_lag=t[iq] - t[ip], r_zmax=z.max(),
                  r_att=100 * (1 - O.max() / I.max()), r_ode=O_ode.max()).items():
    glue(k_, float(v), display=False)
```

*Figure 10.1. Level-pool routing of an inflow flood through a reservoir with a 45 m free spillway. The peak inflow of {glue:text}`r_Ip:.0f` m³ s⁻¹ is attenuated by {glue:text}`r_att:.0f`% to {glue:text}`r_Op:.0f` m³ s⁻¹ and delayed by {glue:text}`r_lag:.1f` h; the water rises {glue:text}`r_zmax:.2f` m above the crest. While inflow exceeds outflow (blue shading) the reservoir fills; afterwards (orange) it empties. An adaptive ODE solver gives the same peak ({glue:text}`r_ode:.0f` m³ s⁻¹).*

Because outflow depends only on the level, and the level is highest when storage is largest, the peak outflow of an uncontrolled reservoir occurs exactly when $dS/dt = 0$, that is when $O = I$: **the outflow peak lies on the falling limb of the inflow**. This is a useful check on any level-pool result. Gated reservoirs are routed in the same way, with the outflow given by the operating rule instead of a fixed curve.

## 10.3 The Muskingum method

### Storage in a river reach

In a river reach the water surface is not level. On the rising limb of a flood, the upstream end is higher than it would be in steady flow and the reach holds extra water in a **wedge**; on the falling limb, the wedge is negative. Storage therefore depends on both inflow and outflow. G. T. McCarthy, working on flood control in the Muskingum River basin in Ohio in the 1930s, proposed a linear combination:

\begin{equation}
S = K\,\bigl[x\,I + (1 - x)\,O\bigr],
\tag{10.4}
\end{equation}

where $K$ (h) is a storage constant, close to the travel time of the flood wave through the reach, and $x$ (dimensionless, 0 to 0.5) is a weighting factor. With $x = 0$ storage depends on outflow only, and the reach acts as a linear reservoir; with $x = 0.5$, inflow and outflow weigh equally and the flood is translated without attenuation. Natural rivers usually have $x$ between 0 and 0.3. (The method was developed in the 1930s; its history is part of the story of flood methods told in [*A Brief History*](https://daehakim-waterlab.github.io/water-history/history-of-hydrology/#floods-from-formulas-to-the-unit-hydrograph).)

### The routing equation

Substitute Equation (10.4) at the start and end of the step into Equation (10.2) and solve for $O_2$:

\begin{equation}
O_2 = C_0\, I_2 + C_1\, I_1 + C_2\, O_1 ,
\tag{10.5}
\end{equation}

with

\begin{equation}
C_0 = \frac{\Delta t - 2Kx}{2K(1 - x) + \Delta t}, \qquad
C_1 = \frac{\Delta t + 2Kx}{2K(1 - x) + \Delta t}, \qquad
C_2 = \frac{2K(1 - x) - \Delta t}{2K(1 - x) + \Delta t}.
\tag{10.6}
\end{equation}

Adding the three numerators gives the denominator, so

$$
C_0 + C_1 + C_2 = 1 ,
$$

which guarantees that a steady inflow produces the same steady outflow. The coefficients should all be non-negative, which requires $2Kx \le \Delta t \le 2K(1 - x)$. A negative $C_0$ makes the outflow dip at the start of the flood, a sign that the time step is too short for the chosen $K$ and $x$.

```{code-cell} python
def muskingum_coefficients(K, x, dt):
    den = 2 * K * (1 - x) + dt
    return (dt - 2 * K * x) / den, (dt + 2 * K * x) / den, (2 * K * (1 - x) - dt) / den

def muskingum(I, K, x, dt, O0=None):
    C0, C1, C2 = muskingum_coefficients(K, x, dt)
    O = np.empty_like(I); O[0] = I[0] if O0 is None else O0
    for n in range(len(I) - 1):
        O[n + 1] = C0 * I[n + 1] + C1 * I[n] + C2 * O[n]
    return O
```

### Estimating $K$ and $x$

Where inflow and outflow hydrographs have been measured for the same flood, storage is found by accumulating Equation (10.1): $S(t) = \int_0^t (I - O)\,dt$. For trial values of $x$, plot $S$ against the weighted flow $xI + (1 - x)O$. The rising and falling limbs form a loop; the value of $x$ that collapses the loop into the narrowest band is chosen, and $K$ is the slope of the line through it. Example 10.3 automates this by choosing the $x$ that maximizes the linear correlation.

### Muskingum–Cunge

Cunge (1969) showed that the Muskingum equation, with a particular choice of parameters, is a numerical solution of the diffusion wave equation (Section 10.5). For a reach of length $\Delta x$ with flood-wave celerity $c$, bed slope $S_0$, width $B$ and reference discharge $Q_0$,

\begin{equation}
K = \frac{\Delta x}{c}, \qquad x = \frac{1}{2}\left(1 - \frac{Q_0}{B\,S_0\,c\,\Delta x}\right).
\tag{10.7}
\end{equation}

The parameters now come from channel properties instead of from a past flood, which allows routing on ungauged rivers and through reaches whose geometry is changed by engineering works.

## 10.4 Linear reservoir cascades

A river reach can also be represented by a series of $n$ linear reservoirs, each with storage $S = kQ$, exactly as the Nash cascade represents a catchment (Chapter 9). For a time step in which the inflow $\bar{I}$ is taken as constant, the exact solution of Equation (9.2) (Chapter 9) for one reservoir is

$$
O_2 = O_1\, e^{-\Delta t/k} + \bar{I}\,\bigl(1 - e^{-\Delta t/k}\bigr),
$$

and the outflow of each reservoir is the inflow to the next. The total lag of the cascade is $nk$; one reservoir gives the strongest attenuation and an immediate rise, and more reservoirs give a delayed, more symmetric response (Figure 10.2). Muskingum routing with $x = 0$ is a numerical approximation of one linear reservoir, and a cascade with several reservoirs mimics the effect of a positive $x$.

```{code-cell} python
:tags: [hide-input]
def reservoir_cascade(I, n, k, dt):
    O = I.copy()
    a = np.exp(-dt / k)
    for _ in range(n):
        inflow, out = O, np.empty_like(O); out[0] = inflow[0]
        for m in range(len(inflow) - 1):
            out[m + 1] = out[m] * a + 0.5 * (inflow[m] + inflow[m + 1]) * (1 - a)
        O = out
    return O

dt = 0.5
t = np.arange(0, 96 + dt, dt)
I = inflow_hydrograph(t, Qb=30.0, Qp=600.0, tp=14.0, m=5.0)
lag = 8.0
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.plot(t, I, color=ink, lw=2.2, label="Inflow")
for n, col in [(1, c2), (3, c1), (8, c3)]:
    O = reservoir_cascade(I, n, lag / n, dt)
    ax.plot(t, O, color=col, lw=2, label=f"{n} reservoir{'s' if n > 1 else ''}, k = {lag / n:.2f} h")
ax.plot(t, muskingum(I, 8.0, 0.2, dt), color=ink2, lw=1.6, ls="--", label="Muskingum, K = 8 h, x = 0.2")
ax.set_xlabel("Time (h)"); ax.set_ylabel("Q (m³ s⁻¹)"); ax.set_xlim(0, 72); ax.set_ylim(0, 650)
ax.legend(loc="upper right", fontsize=9.5)
plt.show()
C = muskingum_coefficients(8.0, 0.2, dt)
glue("mk_C0", float(C[0]), display=False)
```

*Figure 10.2. Routing the same inflow through linear reservoir cascades with a total lag of 8 h, and with the Muskingum method ($K$ = 8 h, $x$ = 0.2, $\Delta t$ = 0.5 h). All outflows have the same volume as the inflow. A single reservoir attenuates most; more reservoirs translate more and attenuate less. The Muskingum curve dips slightly below the initial flow at the start, because with this short time step $C_0$ = {glue:text}`mk_C0:.3f` is negative.*

## 10.5 Hydraulic routing: the Saint-Venant equations

### The equations

The physics of a flood wave in a river is described by the one-dimensional equations of unsteady open-channel flow, published by Adhémar Barré de Saint-Venant in 1871 (see [*A Brief History*](https://daehakim-waterlab.github.io/water-history/industrial-age-engineering-hydraulics/#pipes-channels-and-groundwater-empirical-hydraulics-matures)). For a channel with cross-sectional area $A$, discharge $Q$, depth $y$, bed slope $S_0$ and friction slope $S_f$:

\begin{equation}
\frac{\partial A}{\partial t} + \frac{\partial Q}{\partial x} = q_L ,
\tag{10.8}
\end{equation}

\begin{equation}
\underbrace{\frac{1}{A}\frac{\partial Q}{\partial t}}_{\text{local acceleration}}
+ \underbrace{\frac{1}{A}\frac{\partial}{\partial x}\!\left(\frac{Q^2}{A}\right)}_{\text{convective acceleration}}
+ \underbrace{g\,\frac{\partial y}{\partial x}}_{\text{pressure}}
- \underbrace{g\,(S_0 - S_f)}_{\text{gravity and friction}} = 0 ,
\tag{10.9}
\end{equation}

where $q_L$ is the lateral inflow per unit length. The first is conservation of mass; the second, conservation of momentum. Steady, gradually varied flow (Chapter 6 of the [Hydraulics](https://daehakim-waterlab.github.io/hydraulics/chapters/ch06_flow_in_open_channels.html#gradually-varied-flow) book) is the special case with no time derivatives. Solving the full equations, the **dynamic wave**, needs cross-section data along the river, boundary conditions and a numerical scheme, and is done with programs such as HEC-RAS.

### Simplified forms

In most natural floods the acceleration terms are much smaller than the gravity and friction terms. Dropping terms one by one gives a hierarchy of approximations:

- **Kinematic wave**: keep only $S_f = S_0$. The flow is uniform at every instant, and discharge is a single-valued function of area, $Q = \alpha A^{m}$ (from the Manning equation, $m = 5/3$ for a wide channel). Substituting into Equation (10.8) gives

\begin{equation}
\frac{\partial Q}{\partial t} + c\,\frac{\partial Q}{\partial x} = c\,q_L, \qquad c = \frac{dQ}{dA} = m\,V ,
\tag{10.10}
\end{equation}

  a wave that travels at the **kinematic celerity** $c$ (5/3 of the mean velocity in a wide channel) **without changing shape**: pure translation, no attenuation.

- **Diffusion wave**: also keep the pressure term. The result is an advection–diffusion equation,

\begin{equation}
\frac{\partial Q}{\partial t} + c\,\frac{\partial Q}{\partial x} = D_h\,\frac{\partial^2 Q}{\partial x^2}, \qquad D_h = \frac{Q}{2\,B\,S_0},
\tag{10.11}
\end{equation}

  in which the **hydraulic diffusivity** $D_h$ spreads and flattens the wave as it travels. Attenuation is larger for wide, flat rivers carrying large flows.

- **Dynamic wave**: all terms, needed when backwater, tides or rapid gate operations matter, or on very flat rivers.

For constant $c$ and $D_h$ the diffusion wave has an exact solution (due to Hayami): the response at distance $L$ to an instantaneous unit input is

\begin{equation}
h(t) = \frac{L}{\sqrt{4\pi D_h t^3}}\exp\!\left[-\frac{(L - c\,t)^2}{4 D_h t}\right],
\tag{10.12}
\end{equation}

which can be convolved with an inflow hydrograph exactly like a unit hydrograph. Figure 10.3 compares the kinematic wave, the diffusion wave and Muskingum–Cunge for a long reach. Muskingum–Cunge, whose parameters come from Equation (10.7), follows the diffusion wave closely; the kinematic wave only shifts the inflow. This explains what the hydrologic methods represent: $K$ is the travel time of the kinematic wave, and $x$ controls the attenuation that the diffusion term produces.

```{code-cell} python
:tags: [hide-input]
# A 40 km reach of a wide river; properties at the reference (peak-ish) discharge
L_reach, B, S0, n_man, Q0 = 60e3, 150.0, 0.0002, 0.035, 600.0
y0 = (Q0 * n_man / (B * S0 ** 0.5)) ** 0.6                     # normal depth, wide channel (R ≈ y)
V0 = Q0 / (B * y0)
c = 5 / 3 * V0                                                  # kinematic celerity, m/s
Dh = Q0 / (2 * B * S0)                                          # hydraulic diffusivity, m²/s

dt = 0.25
t = np.arange(0, 120 + dt, dt)
I = inflow_hydrograph(t, Qb=30.0, Qp=600.0, tp=14.0, m=5.0)
# Kinematic: translation by L/c
O_kin = np.interp(t - L_reach / c / 3600, t, I, left=I[0])
# Diffusion (Hayami kernel) convolved with the inflow above base flow
ts = t[1:] * 3600
kern = L_reach / np.sqrt(4 * np.pi * Dh * ts ** 3) * np.exp(-(L_reach - c * ts) ** 2 / (4 * Dh * ts))
kern = np.r_[0.0, kern] * dt * 3600
O_dif = I[0] + np.convolve(I - I[0], kern)[: len(t)]
# Muskingum–Cunge for the whole reach, split into sub-reaches of 15 km, routed with Δt = 1 h
nsub = 4; dx = L_reach / nsub; dt_mc = 1.0
K_mc = dx / c / 3600
x_mc = 0.5 * (1 - Q0 / (B * S0 * c * dx))
t_mc = np.arange(0, t[-1] + dt_mc, dt_mc)
O_mc = np.interp(t_mc, t, I)
for _ in range(nsub):
    O_mc = muskingum(O_mc, K_mc, x_mc, dt_mc)

fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.plot(t, I, color=ink, lw=2, label="Inflow at the upstream end")
ax.plot(t, O_kin, color=c2, lw=2, ls="--", label="Kinematic wave (translation only)")
ax.plot(t, O_dif, color=c1, lw=2.2, label="Diffusion wave (Hayami solution)")
ax.plot(t_mc, O_mc, color=c3, lw=1.8, ls="-.", label=f"Muskingum–Cunge, {nsub} sub-reaches")
ax.set_xlabel("Time (h)"); ax.set_ylabel("Q (m³ s⁻¹)"); ax.set_xlim(0, 72); ax.set_ylim(0, 650)
ax.legend(loc="upper right", fontsize=9.5)
plt.show()
for k_, v in dict(sv_y=y0, sv_V=V0, sv_c=c, sv_D=Dh, sv_T=L_reach / c / 3600, sv_Kmc=K_mc, sv_xmc=x_mc,
                  sv_dif=O_dif.max(), sv_mc=O_mc.max()).items():
    glue(k_, float(v), display=False)
```

*Figure 10.3. A flood routed along a 60 km reach of a wide, flat river ($B$ = 150 m, $S_0$ = 0.0002, $n$ = 0.035). At the reference discharge of 600 m³ s⁻¹ the depth is {glue:text}`sv_y:.2f` m, the velocity {glue:text}`sv_V:.2f` m s⁻¹, the kinematic celerity {glue:text}`sv_c:.2f` m s⁻¹ (travel time {glue:text}`sv_T:.1f` h) and the hydraulic diffusivity {glue:text}`sv_D:,.0f` m² s⁻¹. The diffusion wave attenuates the peak to {glue:text}`sv_dif:.0f` m³ s⁻¹ and Muskingum–Cunge ($K$ = {glue:text}`sv_Kmc:.2f` h, $x$ = {glue:text}`sv_xmc:.2f` per 15 km sub-reach, $\Delta t$ = 1 h) gives {glue:text}`sv_mc:.0f` m³ s⁻¹.*

## 10.6 Worked examples

**Example 10.1 — Reservoir routing.** For the reservoir of Figure 10.1, print the first steps of the storage-indication calculation and check that the peak outflow occurs when the outflow equals the inflow.

```{code-cell} python
dt_s = 0.5 * 3600
print("  t(h)   I(m³/s)  2S/Δt-O   z(m)    O(m³/s)")
for n in range(0, 41, 8):
    print(f"{t_res[n]:6.1f} {I_res[n]:9.1f} {2 * S_of_z(z_res[n]) / dt_s - O_res[n]:9.1f} "
          f"{z_res[n]:7.3f} {O_res[n]:9.1f}")
iq = O_res.argmax()
print(f"Peak outflow {O_res[iq]:.1f} m³/s at t = {t_res[iq]:.1f} h; inflow then = {I_res[iq]:.1f} m³/s "
      f"(inflow half a step later: {I_res[iq + 1]:.1f})")
```

**Example 10.2 — Muskingum routing.** A reach has $K$ = 6 h and $x$ = 0.25. Route the inflow below with $\Delta t$ = 3 h, starting from a steady flow of 40 m³ s⁻¹. Check the coefficients and the volume balance.

```{code-cell} python
K, x, dt3 = 6.0, 0.25, 3.0
C0, C1, C2 = muskingum_coefficients(K, x, dt3)
print(f"C0 = {C0:.4f}, C1 = {C1:.4f}, C2 = {C2:.4f}, sum = {C0 + C1 + C2:.4f}")
print(f"Non-negative coefficients need {2 * K * x:.1f} h <= Δt <= {2 * K * (1 - x):.1f} h")
I_r = np.array([40, 70, 160, 260, 290, 250, 190, 140, 100, 75, 60, 50, 45, 42, 40, 40, 40, 40], float)
O_r = muskingum(I_r, K, x, dt3, O0=40.0)
print("O:", ", ".join(f"{o:.0f}" for o in O_r))
print(f"Peak: inflow {I_r.max():.0f} at {I_r.argmax() * 3:.0f} h, outflow {O_r.max():.0f} at {O_r.argmax() * 3:.0f} h")
print(f"Volumes: in {I_r.sum() * 3 * 3600 / 1e6:.2f}, out {O_r.sum() * 3 * 3600 / 1e6:.2f} million m³ "
      "(the remaining difference is still stored at the end)")
```

**Example 10.3 — Estimating $K$ and $x$ from a measured flood.** Synthetic "measured" inflow and outflow for a reach were produced by Muskingum routing with $K$ = 7 h, $x$ = 0.22 and 2% measurement noise. Recover the parameters.

```{code-cell} python
rng = np.random.default_rng(3)
dt1 = 1.0
tt = np.arange(0, 80, dt1)
I_m = inflow_hydrograph(tt, Qb=25.0, Qp=310.0, tp=12.0, m=4.0)
O_m = muskingum(I_m, 7.0, 0.22, dt1) * (1 + 0.02 * rng.standard_normal(len(tt)))
S_m = np.r_[0, np.cumsum(0.5 * (I_m[1:] + I_m[:-1]) - 0.5 * (O_m[1:] + O_m[:-1]))] * dt1   # m³/s × h
best = max(np.arange(0, 0.5, 0.01),
           key=lambda xx: np.corrcoef(xx * I_m + (1 - xx) * O_m, S_m)[0, 1])
K_fit = np.polyfit(best * I_m + (1 - best) * O_m, S_m, 1)[0]
print(f"Estimated x = {best:.2f}, K = {K_fit:.2f} h  (true: 0.22, 7 h)")
```

```{code-cell} python
:tags: [hide-input]
fig, ax = plt.subplots(figsize=(6.0, 4.0))
for xx, col in [(0.0, c2), (best, c1), (0.45, c3)]:
    ax.plot(xx * I_m + (1 - xx) * O_m, S_m, color=col, lw=1.8, label=f"x = {xx:.2f}")
ax.set_xlabel("Weighted flow xI + (1 − x)O (m³ s⁻¹)"); ax.set_ylabel("Storage S (m³ s⁻¹ · h)")
ax.legend(loc="upper left", fontsize=9.5)
plt.show()
```

*Figure 10.4. Storage plotted against the weighted flow for three values of $x$. The loop is widest for $x$ = 0 (outflow only) and closes into a nearly straight line at the best estimate; the slope of that line is $K$.*

**Example 10.4 — Muskingum–Cunge parameters.** For the river of Figure 10.3, compute $K$ and $x$ for the whole 60 km reach and for sub-reaches of 15 km and 5 km, with the limits on the time step.

```{code-cell} python
print(f"x = 0 when Δx = Q0 / (B S0 c) = {Q0 / (B * S0 * c) / 1e3:.1f} km")
for dxx in (60e3, 15e3, 5e3):
    Kc = dxx / c / 3600
    xc = 0.5 * (1 - Q0 / (B * S0 * c * dxx))
    print(f"Δx = {dxx / 1e3:4.0f} km: K = {Kc:5.2f} h, x = {xc:6.3f}, "
          f"Δt limits {max(2 * Kc * xc, 0):.2f}–{2 * Kc * (1 - xc):.2f} h")
```

The weighting factor falls as the sub-reach gets shorter and becomes negative below about 12 km, which is outside the meaningful range. A single 60 km reach gives a valid $x$ but needs a time step of about 8 hours, too coarse to describe a flood that rises in 10 hours. Sub-reaches of 15 km with a 1-hour step satisfy both conditions, which is the combination used in Figure 10.3: in Muskingum–Cunge the sub-reach length and the time step are chosen together.

## 10.7 Common mistakes

- **A time step outside $2Kx \le \Delta t \le 2K(1 - x)$.** Negative coefficients produce dips or spurious oscillations in the outflow.
- **Mixing units in the storage-indication function.** $2S/\Delta t$ must be in m³ s⁻¹: $S$ in m³ and $\Delta t$ in seconds.
- **A reservoir peak outflow that does not lie on the falling limb of the inflow.** For an uncontrolled spillway, the outflow peak must equal the inflow at that moment. If not, the calculation is wrong.
- **Forgetting the initial condition.** Routing must start from a consistent state (for example a steady flow with $O = I$), or the first steps carry a spurious transient.
- **Using Muskingum parameters from a small flood for an extreme one.** Celerity and attenuation change with discharge, and a wide floodplain can change them completely.
- **Expecting the kinematic wave to attenuate a flood.** In its exact form it only translates; any attenuation in a kinematic model comes from numerical diffusion.

## 10.8 Exercises

```{code-cell} python
:tags: [remove-cell]
cc = muskingum_coefficients(10.0, 0.2, 6.0)
glue("v1_C0", float(cc[0]), display=False); glue("v1_C1", float(cc[1]), display=False); glue("v1_C2", float(cc[2]), display=False)
glue("v2_lo", 2 * 10.0 * 0.2, display=False); glue("v2_hi", 2 * 10.0 * 0.8, display=False)
Ov = muskingum(np.array([50, 120, 240, 200, 140, 90, 60, 50], float), 10.0, 0.2, 6.0, O0=50.0)
glue("v3_O", ", ".join(f"{o:.0f}" for o in Ov), display=False)
y5 = (250 * 0.03 / (60 * 0.001 ** 0.5)) ** 0.6; V5 = 250 / (60 * y5); c5 = 5 / 3 * V5
glue("v5_y", y5, display=False); glue("v5_c", c5, display=False); glue("v5_T", 25e3 / c5 / 3600, display=False)
glue("v5_D", 250 / (2 * 60 * 0.001), display=False)
glue("v4_O", 2.0 * 30 * 1.2 ** 1.5, display=False)
```

1. Compute the Muskingum coefficients for $K$ = 10 h, $x$ = 0.2 and $\Delta t$ = 6 h, and check their sum.
2. For $K$ = 10 h and $x$ = 0.2, what range of time steps keeps all coefficients non-negative?
3. Route the inflow 50, 120, 240, 200, 140, 90, 60, 50 m³ s⁻¹ (at 6-hour intervals) with the parameters of Exercise 1, starting from 50 m³ s⁻¹.
4. A spillway has $C_w$ = 2.0 m¹ᐟ² s⁻¹ and a crest 30 m long. What is the outflow when the water is 1.2 m above the crest? If at that moment the inflow is larger than this, is the reservoir level rising or falling?
5. A wide channel 60 m wide with $S_0$ = 0.001 and $n$ = 0.03 carries 250 m³ s⁻¹. Find the normal depth, the kinematic celerity, the travel time of the flood wave through a 25 km reach, and the hydraulic diffusivity.

```{admonition} Answers
:class: dropdown
1. $C_0$ = {glue:text}`v1_C0:.4f`, $C_1$ = {glue:text}`v1_C1:.4f`, $C_2$ = {glue:text}`v1_C2:.4f`; sum = 1.
2. {glue:text}`v2_lo:.0f` h ≤ Δt ≤ {glue:text}`v2_hi:.0f` h.
3. {glue:text}`v3_O` m³ s⁻¹.
4. {glue:text}`v4_O:.0f` m³ s⁻¹; rising, since $dS/dt = I - O > 0$.
5. $y$ = {glue:text}`v5_y:.2f` m, $c$ = {glue:text}`v5_c:.2f` m s⁻¹, travel time {glue:text}`v5_T:.1f` h, $D_h$ = {glue:text}`v5_D:,.0f` m² s⁻¹.
```

## 10.9 Further reading

- Dingman, S. L. (2015). *Physical Hydrology* (3rd ed.). Waveland Press. The sections on streamflow routing.
- Chow, V. T., Maidment, D. R., and Mays, L. W. (1988). *Applied Hydrology*. McGraw-Hill. The chapters on lumped flow routing and distributed flow routing.
- 윤용남. *수문학* (in Korean). The chapter on flood routing.
- Cunge, J. A. (1969). On the subject of a flood propagation computation method (Muskingum method). *Journal of Hydraulic Research*, 7, 205–230.
- McCarthy, G. T. (1938). *The unit hydrograph and flood routing*. Conference of the North Atlantic Division, US Army Corps of Engineers.

---

*Figures in this chapter are generated by the Python code shown. Text and figures: © Daeha Kim, CC BY-NC 4.0.*
