# Parker instability diagnostics

Magnetic buoyancy (Parker) diagnostics in `kplot.volume.disk`, modelled on
Jiang, Ng, Chabanov & Rezzolla (2025), arXiv:2502.14962, Sec. V C (Figs. 7, 8).
Units are AthenaK code units, $G=c=M_\odot=1$, with magnetic pressure $b^2/2$
($b^2$ is the comoving field strength, $b^2 \equiv B^2/4\pi$ in Gaussian units).

## 1. What the paper does

The paper uses Parker's (1966) criterion

$$
\mathcal{P} := \frac{d\log p}{d\log\rho} - 1 - \frac{\beta^{-1}(1+2\beta^{-1})}{2+3\beta^{-1}},
\qquad \beta = \frac{p}{b^2/2},
$$

and calls the plasma unstable where $\mathcal{P}<0$. The time at which $\mathcal{P}$ changes sign is
called the *breakout*.
- **Fig. 7:** spacetime diagrams of $\mathcal{P}$ and $\beta^{-1}$ on the $(x,z)$ plane, averaged over
  polar angle in bins of $r=\sqrt{x^2+z^2}$, split into a "funnel" ($\theta<\pi/6$) and a "disk"
  (the rest). Worldlines of iso-density contours are overlaid.
- **Fig. 8:** $(x,z)$ maps of $\mathcal{P}$, $\beta^{-1}$, specific entropy $s$ and Lorentz factor
  $W$, with field lines and fluid lines.

The paper does not specify how $d\log p/d\log\rho$ is evaluated.

### Where the formula comes from
Parker considered an **isothermal** atmosphere with **uniform** $\alpha=\beta^{-1}$, supported against
gravity by gas plus a horizontal field, and **2D** undular perturbations ($k_\perp = 0$) with adiabatic
index $\gamma$. They are unstable for

$$
\gamma < \frac{(1+\alpha)^2}{1+\tfrac32\alpha}
\iff
\gamma - 1 - \frac{\alpha(1+2\alpha)}{2+3\alpha} < 0 .
$$

So $\mathcal{P}$ is Parker's result with the *perturbation* index $\gamma$ replaced by the
*background* slope $d\log p/d\log\rho$. The two coincide only for an isothermal, uniform-$\beta$
background. In a post-merger disk the temperature rises with height, which pushes the background
slope below 1. $\mathcal{P}$ is then negative **even for $b\to 0$**. At $t\approx 54$ ms in
`MHD_M1_120`, with $\beta^{-1}\sim10^{-4}$, about 55% of the volume has $\mathcal{P}<0$, and that comes
from thermal stratification, not magnetism.

## 2. The undular Newcomb criterion

Newcomb (1961, Phys. Fluids 4, 391) treats the general case: a plane-parallel atmosphere with
gravity $-g\hat z$, a horizontal field of fixed direction, arbitrary profiles $\rho(z)$, $p(z)$, $b(z)$,
magnetohydrostatic equilibrium

$$
-\frac{d}{dz}\Big(p+\frac{b^2}{2}\Big) = \rho g ,
$$

and adiabatic perturbations with $\Gamma_1=(\partial\ln p/\partial\ln\rho)_s$. The undular (Parker)
modes, with $k_\parallel\neq0$ and $k_\perp\to\infty$ (the most unstable, 3D, case), are unstable at
height $z$ if

$$
-\frac{d\ln\rho}{dz} < \frac{\rho g}{\Gamma_1 p} .
$$

How to read it:
- The criterion compares the actual density drop with height against the drop an adiabatically
  displaced parcel would undergo.
- For $b=0$, $\rho g=-dp/dz$ and the criterion reduces to Schwarzschild convection ($N^2<0$).
- A field carries part of the weight, so $\rho g > -dp/dz$. That raises the right-hand side and
  allows instability in a convectively stable gas, which is magnetic buoyancy.
- For Parker's isothermal, uniform-$\alpha$ atmosphere it gives $\gamma<1+\alpha$. That is less
  restrictive than the 2D Parker bound because $k_\perp\to\infty$ modes are more unstable.

(Interchange modes, $k_\parallel=0$, use $\Gamma_1p+b^2$ in the denominator. They are always harder to
excite and are not computed.)

## 3. What KPlot computes

Both criteria are evaluated per cell on the native mesh along a direction $\hat n$
(`--parker-dir`):
- `z`: $\hat n=\mathrm{sign}(z)\,\hat z$, away from the disk midplane (the default; buoyancy in a
  disk acts vertically);
- `r`: $\hat n=\hat r$ from the BH, as in the paper.

Every variable is centred on the BH tracker, so the BH drift is followed.

| Quantity | Definition | Unstable |
|---|---|---|
| $\mathcal{P}$ (`P`) | Eq. above, with $d\ln p/d\ln\rho = \partial_n\ln p/\partial_n\ln\rho$ | $<0$ |
| $\mathcal{N}_u$ (`N`) | $\dfrac{A-G}{\lvert A\rvert+\lvert G\rvert}$, $\;A=-\partial_n\ln\rho$, $\;G=\dfrac{-\partial_n(p+b^2/2)}{\Gamma_1 p}$ | $<0$ |

Implementation details:
- **Weight from the pressure gradient.** $\rho g$ is replaced by $-\partial_n(p+b^2/2)$, which
  assumes magnetohydrostatic balance. That avoids a GR effective gravity, but it is not valid in
  strongly dynamic flow.
- **Bounded, no division blow-up.** $\mathcal{N}_u\in[-1,1]$; its sign is Newcomb's criterion and its
  magnitude is the relative margin. Unlike $\mathcal{P}$ it does not divide by $\partial_n\ln\rho$,
  so it does not blow up at the midplane.
- **$\Gamma_1$ from the EOS table.** $\Gamma_1 = c_s^2\,(e+p)/p = c_s^2\,h\,m_b/Q_1$, using `cs2`,
  `Q1` and `Q7` of the CompOSE table at $(\rho, Y_e, T)$.
- **Smoothing.** $\ln\rho$, $\ln p$ and $p+b^2/2$ are box-smoothed over `--parker-smooth`$^3$ cells
  inside each meshblock (NaN-aware) before `np.gradient`. Differences are one-sided at meshblock
  edges.
- **Masks.**
  - Only cells with $\rho>$ `rho_cut` outside `r_exclude` are used. The bound criterion is **not**
    applied, so the polar outflow is kept.
  - $\mathcal{P}$ is masked where $\lvert\partial_n\ln\rho\rvert\,\Delta z<$ `--parker-eps`.
  - $\mathcal{N}_u$ is masked where $(\lvert A\rvert+\lvert G\rvert)\,\Delta z<$ `--parker-eps`.

### Outputs
- `slices/disk_slices_<snap>.npz`: nearest-cell xz plane through the BH. It holds `P_xz`, `N_xz`,
  `betainv_xz`, `s_xz`, `W_xz`, `rho_xz`, the Eulerian field `Bx_xz`, `Bz_xz` and the coordinate
  velocity `vx_xz`, `vz_xz`. These are plotted as `frames/parker_xz/` (Fig. 8 analogue).
- `parker/disk_parker_<snap>.csv`: one region (all polar angles, no funnel/disk split), in
  spherical-$r$ bins from `max(r_exclude, 1)` to `--parker-rmax`. Columns are proper-volume-weighted
  ($\sqrt\gamma\,dV$) statistics:
  - mean and median of $\mathcal{P}$ and $\mathcal{N}_u$ (`*_mean`, `*_p50`);
  - the unstable volume fraction (`*_f_unstable`) and the valid (unmasked) fraction (`*_f_valid`);
  - $\langle\beta^{-1}\rangle$, $\langle\Gamma_1\rangle$ and $\langle\rho\rangle$.

  These are plotted as `frames/scalars/disk_parker_spacetime.png` (Fig. 7 analogue, medians, with
  white zero contours and cyan $10^{9\ldots12}$ g/cm$^3$ worldlines). Volume weighting is used
  because mass weighting would be dominated by the midplane.
- The settings used are stored under `"parker"` in `scalars/disk_scalars_<snap>.json`.

## 4. Caveats
- **Assumptions.** Both criteria are local (WKB), Newtonian and plane-parallel, and assume a
  *horizontal* field. In the disk that is $b_\phi$; in the field-dominated funnel the assumption is
  not strictly valid. Rotation and shear are ignored; they can stabilise or destabilise
  long-wavelength modes (Foglizzo & Tagger 1994, 1995).
- **Field-free limit.** As $\beta^{-1}\to0$, $\mathcal{N}_u$ becomes the Schwarzschild criterion.
  Patches with $\mathcal{N}_u<0$ at negligible $\beta^{-1}$ are therefore convectively unstable or
  out of hydrostatic balance (turbulence, outflows), *not* Parker-unstable. Read $\mathcal{N}_u<0$
  as magnetic buoyancy only where $\beta^{-1}$ is non-negligible, i.e. compare with the
  $\beta^{-1}$ panel.
- **Time resolution.** 3D dumps every 250 M (about 1.2 ms) resolve the breakout time only coarsely.
- **Resolution.** An under-resolved MRI (low quality factor $Q$) keeps $\beta^{-1}$ small, so no
  magnetically driven instability is expected in that case. The setup still runs and produces the
  diagnostics.
