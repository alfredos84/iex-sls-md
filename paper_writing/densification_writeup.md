# Local densification analysis of the nanoindented glasses

This note collects, in paper-ready form, the theory and the results of the
local densification analysis performed on the four indented glasses
(CaO–AM, CaO–IOX, Ca-free–AM, Ca-free–IOX). It is split in two parts: a
**Theory** block, meant to go in the Methods section near the beginning of
the paper, and a **Results** block, meant to go in the Nanoindentation
subsection together with the summary figure and the table.

---

## Part A — Theory (Methods section)

### A.1 Local deformation gradient (Falk–Langer)

To quantify the residual, irreversible volumetric change experienced by
the glass network after a full load–hold–unload indentation cycle, we
adopt the non-affine local deformation gradient formalism of Falk and
Langer, in the same spirit as the atomic-scale densification descriptor
introduced by Pedone et al. [Pedone 2026, *Acta Materialia* 316, 122463].

For every atom *i*, we identify its neighbors *j* within a cutoff
$r_c = 4.0$ Å in a reference configuration (the equilibrated, pre-loading
state of the indenter, $t_0$) and track the same neighbor set in a
deformed configuration (post-unload, $t_1$). Let
$\mathbf{r}_{ij}^{0} = \mathbf{r}_j^{0}-\mathbf{r}_i^{0}$ and
$\mathbf{r}_{ij} = \mathbf{r}_j-\mathbf{r}_i$ be the interatomic vectors
in the reference and deformed configurations, respectively. The local
deformation gradient tensor $\mathbf{F}_i$ that best maps
$\mathbf{r}_{ij}^{0}\to\mathbf{r}_{ij}$ in a least-squares sense is

$$
\mathbf{X}_i=\sum_j \mathbf{r}_{ij}\otimes\mathbf{r}_{ij}^{0},
\qquad
\mathbf{Y}_i=\sum_j \mathbf{r}_{ij}^{0}\otimes\mathbf{r}_{ij}^{0},
\qquad
\mathbf{F}_i=\mathbf{X}_i\,\mathbf{Y}_i^{-1}.
$$

Neighbor sets with fewer than 6 neighbors, or with a poorly conditioned
$\mathbf{Y}_i$ (eigenvalue ratio $>100$, i.e. a nearly coplanar local
environment), are excluded from the analysis as unreliable fits.

The local, atomic-scale residual densification is then defined from the
determinant of $\mathbf{F}_i$ (mass conservation implies
$\rho_i' = \rho_i/\det\mathbf{F}_i$):

$$
\left(\frac{\Delta\rho}{\rho}\right)_i \;=\; \frac{1}{\det \mathbf{F}_i}-1 .
$$

Positive values indicate local densification (volume contraction),
negative values indicate local dilation.

### A.2 Plastically active atoms and the densification descriptor *D*

Following the same rationale as Pedone et al., we restrict all
quantitative descriptors to the subset of atoms classified as
*plastically active*, i.e. atoms that underwent a local volumetric
strain in excess of a fixed threshold $\varepsilon_{th}$:

$$
\text{atom } i \text{ is active if } \;\left|\left(\Delta\rho/\rho\right)_i\right| > \varepsilon_{th}.
$$

The threshold was not adopted a priori: it was fixed empirically from the
noise floor of the method itself. Evaluating $(\Delta\rho/\rho)_i$ for a
bulk region unaffected by the indenter (far from the tip axis and from
the free surface, $>5\times10^{5}$ atoms pooled over all 11 replicas)
gives a thermal-noise standard deviation $\sigma_{\rm bulk}=0.046$, with
the 1st/99th percentiles of this noise distribution falling almost
exactly at $\mp 0.10$–$0.12$. A threshold of

$$
\varepsilon_{th}=0.10
$$

(coincidentally identical to the value adopted by Pedone et al. for their
combined volumetric/deviatoric criterion) therefore corresponds to
$\approx 2.2\,\sigma_{\rm bulk}$: it admits only $3.7\%$ of the
undeformed bulk population as false positives while retaining $51\%$ of
the population in the tip-affected region, and was adopted on this basis
rather than by direct analogy with the reference study.

The scalar densification descriptor reported for each glass is then the
average residual volumetric compaction over the active-atom population,

$$
D \;=\; \big\langle \Delta\rho/\rho \big\rangle_{\rm active},
$$

evaluated between the pre-loading and post-unload configurations. This is
the direct, volumetric-only analogue of the *D* coordinate defined by
Pedone et al. (their Eq. 23); we do not report the companion deviatoric
descriptor *S* (based on $D^2_{\min}$) or the composite
Densification–Shear-Fraction (DSF) index, since the latter requires
reference densification- and shear-dominated end-member glasses
(vitreous SiO$_2$ and Cu$_{64}$Zr$_{36}$) that were not simulated in this
work.

In addition to the scalar descriptor *D*, a spatially resolved,
2-dimensional map of the local densification field is obtained by
replicating the chunk/atom bin-density protocol of Pedone et al.
(Fig. 7 therein): atoms are binned in real space into a regular
$2\times2\times2$ Å$^3$ grid, the local mass density of each bin is
compared against the bulk density $\rho_0$ of the same glass prior to
indentation (computed over an undisturbed sub-region, $z\in[10,80]$ Å,
of the pre-loading configuration), and the relative change is expressed
as a percentage:

$$
\text{local densification}(\%) = \frac{\rho_{\rm local}(\text{bin})-\rho_0}{\rho_0}\times 100 .
$$

The map is averaged over a 12 Å-thick slice centered on the indenter
axis, replica-averaged, and lightly smoothed (Gaussian, $\sigma=1.5$
bins) for visualization.

---

## Part B — Results (Nanoindentation section)

### B.1 Figure: densification maps through the indentation cycle

**Figure X.** *Local densification maps at the three stages of the
indentation cycle — Start (pre-loading, equilibrated configuration),
Max load (end of the loading branch) and Unload (after full tip
retraction) — for the four indented glasses (columns: CaO–AM, CaO–IOX,
Ca-free–AM, Ca-free–IOX). Each panel shows the chunk/atom bin-density
descriptor (Section A.2) on a $y$–$z$ slice through the indenter axis
($2\times2\times2$ Å$^3$ bins, 12 Å-thick slice, replica-averaged over
3 replicas — 2 for CaO–AM, whose third replica is a non-comparable pilot
sample — and lightly Gaussian-smoothed). The color scale spans
$\pm20\%$, set symmetrically about the maximum positive densification
value actually observed in the replica-averaged maps (Unload stage), in
the same spirit as the color-range convention of Pedone et al.; the dark
blue background denotes bins outside the glass slab (free surface, or
the cavity carved by the indenter).*

In the figure it can be observed that, already at *Start*, all four
panels show only weak, spatially uncorrelated red/blue speckle of small
amplitude — the thermal noise floor of the descriptor, with no
indentation-related structure, as expected since the tip has not yet
engaged the surface. At *Max load*, a clear, approximately V-shaped
densified region (red) appears directly beneath the indenter apex in
all four glasses, coinciding with the region of maximum hydrostatic
compression under the conical tip. At *Unload*, part of this densified
region persists — indicating that the compaction is, at least in part,
irreversible/plastic rather than purely elastic — while its intensity
and spatial extent are visibly reduced relative to Max load, consistent
with partial elastic recovery upon tip retraction. Comparing columns at
the Unload stage, the densified region appears visibly less intense and
less extended for the IOX glasses (columns 2 and 4) than for the
corresponding AM glasses (columns 1 and 3), for both CaO and Ca-free
compositions — a first, qualitative indication that ion exchange reduces
the glass's capacity for further densification under indentation.

### B.2 Quantifying the difference: the *D* descriptor

To quantify this observation beyond visual inspection, we computed the
scalar densification descriptor $D=\langle\Delta\rho/\rho\rangle_{\rm
active}$ (Section A.2) between the pre-loading and post-unload
configurations, over the *entire* simulation cell (not restricted to a
region under the tip), for every replica, and averaged over replicas.

**Table X.** *Densification descriptor $D=\langle\Delta\rho/\rho\rangle_{\rm active}$
(dimensionless, $\varepsilon_{th}=0.10$) after unloading, averaged over
replicas ($n$ = number of replicas; mean $\pm$ sample standard
deviation).*

| Glass | Condition | $n$ | $D$ |
|---|---|---|---|
| CaO | AM | 2 | $+0.0300 \pm 0.0013$ |
| CaO | IOX | 3 | $+0.0137 \pm 0.0031$ |
| Ca-free | AM | 3 | $+0.0237 \pm 0.0026$ |
| Ca-free | IOX | 3 | $+0.0021 \pm 0.0011$ |

For both glass compositions, $D$ is unambiguously and reproducibly
smaller for the ion-exchanged surface than for the as-melted one: in CaO,
IOX densifies $\approx 2.2\times$ less than AM ($D=0.014$ vs. $0.030$);
in Ca-free glass, the reduction is even more pronounced, roughly one
order of magnitude ($D=0.002$ vs. $0.024$). In both cases the
replica-to-replica standard deviation is small compared with the AM–IOX
difference, indicating that the effect is well resolved and not an
artifact of replica variability.

**Why does the ion-exchanged glass densify less?** Ion exchange replaces
the smaller Na$^+$ modifier ions with the larger K$^+$ ion within a
network that was not allowed to relax its volume during the (isochoric,
sub-$T_g$) exchange process. This "ion stuffing" mechanically dilates
the local network around every exchanged site and leaves the
ion-exchanged glass in a state of residual compressive stress and
already-reduced local free volume relative to the pristine (AM) network.
Because the densification measured here is precisely the network's
capacity to contract locally and accommodate the indenter — i.e. to
consume available free volume — a glass that has already had part of
that free volume consumed by the stuffing process has intrinsically less
capacity left to densify further under load. This is the same mechanism
identified by Pedone et al. when comparing pristine sodium silicate (NS)
to a pre-densified sodium silicate (NS 1.5) under the identical
indentation protocol: the pre-compacted glass shows a "more localized
compaction due to its reduced capacity for further densification"
relative to the pristine one. Our ion-exchanged glasses are, in this
specific sense, mechanically analogous to their pre-densified end
member: the exchange process itself acts as a pre-densification step,
so that the indentation-induced *additional* densification measured by
$D$ is correspondingly smaller.
