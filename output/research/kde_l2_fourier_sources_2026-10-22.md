# Sources for exact, spatial-quadrature and Fourier evaluation of KDE distances

Research date: 22 October 2026. No files under `Abschlussarbeit_2021/` were changed.

## Recommendation

Use **Silverman (1986), Section 3.5, pp. 61–66**, especially Eqs. (3.51), (3.56), and (3.61)–(3.62), as the methodological precedent. The application is least-squares cross-validation. [Book DOI](https://doi.org/10.1201/9781315140919); [original book text inspected, pp. 62–65](https://www.scribd.com/document/997694117/Silverman-1986).

For a related numerical comparison, use **Gramacki and Gramacki (2017), Sections 4–6**. Section 4, Eqs. (10)–(17), develops FFT evaluation of quadratic kernel sums; Section 5 includes density derivative functionals; Section 6.1 studies accuracy and Section 6.3 compares execution times. Their method uses linear binning and discrete convolution. Their comparisons do not validate this project's particular continuous-frequency quadrature. [Published article](https://doi.org/10.1016/j.csda.2016.09.001); [author preprint, with the section and equation numbering checked here](https://arxiv.org/pdf/1511.07482).

**Search outcome:** I found a close precedent for the computational idea and related speed/accuracy studies. I did not verify a publication comparing all three of the project's methods for this particular bandwidth-distance matrix. The distinction matters: the citations support the strategy, but do not establish an implementation-specific speedup or error tolerance.

## Exact connection to the thesis: our algebraic identification

The thesis defines the Gaussian KDE inner products and the corresponding half squared distances in [the bandwidth chapter, lines 307–333](/Users/iljasukhanov/MASTERARBEIT/Abschlussarbeit_2021/LaTeX/chapters/5_bandwidth_selection.tex:307).

Using its Fourier convention, define

\[
 I_n(a)=\frac{1}{2\pi}\int_{\mathbb R}|P_n(\omega)|^2e^{-a\omega^2}\,d\omega,
 \qquad P_n(\omega)=\frac1n\sum_j e^{-iX_j\omega},\quad a>0.
\]

Then direct substitution into the chapter's formulas gives

\[
 G_{hs}=I_n\!\left(\frac{h^2+s^2}{2}\right),\qquad
 D_1(\hat f_h,\hat f_s)
 =\frac12I_n(h^2)+\frac12I_n(s^2)
 -I_n\!\left(\frac{h^2+s^2}{2}\right).
\]

In particular, with \(t=\sqrt{(h^2+s^2)/2}\), we have \(G_{hs}=\|\hat f_t\|_2^2\). These identities assume both KDEs use the same observations with equal weights, as in the chapter.

Silverman's unitary convention has \(u(\omega)=P_n(-\omega)/\sqrt{2\pi}\). Consequently his Eq. (3.56), for a Gaussian kernel, is precisely

\[
 \psi(0)=I_n(h^2)-2I_n(h^2/2).
\]

Thus both calculations reduce to the **same family of Gaussian-weighted integrals of one empirical power spectrum**. This identification is our derivation; the book does not itself state the project's cross-bandwidth matrix formula. [Silverman, Section 3.5, Eqs. (3.51)–(3.56)](https://www.scribd.com/document/997694117/Silverman-1986).

A second fully accessible mathematical reference is **Chiu (2000), Section 2, pp. 1348–1349**: Eq. (2.1) gives the KDE characteristic function as the empirical characteristic function times the kernel transform; Eq. (2.2) applies Parseval to integrated squared error. This establishes the mathematical approach, without a computational comparison. [Journal PDF](https://www3.stat.sinica.edu.tw/statistica/oldpdf/A10n419.pdf).

## An analytical comparison, derived for this project

The following operation counts are our analysis, not timings quoted from the papers. Let \(n\) be the number of observations, \(N\) the number of bandwidths, \(M\) the number of spatial nodes, and \(K+1\) the number of frequency nodes. Assume shared grids, direct KDE/characteristic-function evaluation, and ordinary matrix multiplication.

| Method | Work to assemble all Gram entries | Approximation to assess |
|---|---|---|
| Exact Gaussian pair sums | \(O(N^2n^2)\) | Floating-point error; no integral discretization |
| Spatial quadrature | \(O(NnM+N^2M)\) | Spatial-domain truncation and quadrature resolution |
| Frequency quadrature as written in the chapter | \(O(nK+N^2K)\) | Frequency cutoff and quadrature resolution |

For the frequency method, compute \(|P_n(\omega_k)|^2\) once in \(O(nK)\). Form the \(N\) Gaussian multiplier vectors in \(O(NK)\), then their weighted Gram matrix in \(O(N^2K)\). For spatial quadrature, directly evaluating the KDE family costs \(O(NnM)\), followed by a weighted Gram product in \(O(N^2M)\). These counts explain the potential saving; they do not determine which implementation is fastest for fixed data and a prescribed accuracy. [Exact and spatial implementations](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/kde_tools/density.py:11), [spatial Gram assembly](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/kde_tools/quadrature.py:100).

For an elementary cutoff bound, our derivation uses \(|P_n|\le1\). Writing \(a=(h^2+s^2)/2\), the omitted part of the Gram integral satisfies

\[
 0\le G_{hs}-G_{hs}^{[0,\Omega]}
 \le\frac1\pi\int_\Omega^\infty e^{-a\omega^2}\,d\omega
 =\frac{\operatorname{erfc}(\sqrt a\,\Omega)}{2\sqrt{\pi a}}.
\]

The last expression uses the definition \(\operatorname{erfc}(z)=(2/\sqrt\pi)\int_z^\infty e^{-t^2}\,dt\) from [NIST DLMF, Eq. (7.2.2)](https://dlmf.nist.gov/7.2#E2); the bound follows by substitution and is original reasoning here. It controls cutoff error only. Quadrature error still needs a separate bound or convergence check.

## Code-versus-chapter distinction

The current [`kde_family_fft`](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/kde_tools/fourier/reconstruction.py:8) directly evaluates the empirical characteristic function at periodic-grid frequencies, applies Gaussian smoothing, reconstructs densities with `irfft`, clips negative values, and renormalizes. [`beta2_divergence_matrix_fft`](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/kde_tools/fourier/similarities.py:11) then forms a spatial Gram matrix. The main [`spectral_scale_selection_kde.ipynb`](/Users/iljasukhanov/MASTERARBEIT/Code/Code_kde/spectral_scale_selection_kde.ipynb:31) and the fast L2 notebook use this helper. It takes a different sequence of steps from the written direct frequency integral, but that observation alone does not establish a mathematical disagreement.

### Qualification checked after the follow-up discussion

Before clipping and renormalization, inverse-FFT reconstruction followed by the spatial Gram product is algebraically equal to a weighted frequency Gram product. Here is our derivation from [NumPy's DFT and inverse-DFT definitions and normalization](https://numpy.org/doc/stable/reference/routines.fft.html#implementation-details). Let \(L=2H\), \(M\) be even, \(\Delta x=L/M\), \(\omega_k=2\pi k/L\), and

\[
 A_{hk}=\frac1n\sum_j e^{-i(X_j+H)\omega_k}e^{-h^2\omega_k^2/2}.
\]

For \(k=0,M/2\), replace \(A_{hk}\) by its real part, as required by the even-length [`irfft` endpoint convention](https://numpy.org/doc/stable/reference/generated/numpy.fft.irfft.html#notes). Write the resulting coefficients as \(\widetilde A_{hk}\). Discrete Fourier orthogonality gives

\[
 \Delta x\sum_{m=0}^{M-1}f^{\rm raw}_h(x_m)f^{\rm raw}_s(x_m)
 =\frac1L\sum_{k=0}^{M/2}c_k\operatorname{Re}
 \left(\widetilde A_{hk}\overline{\widetilde A_{sk}}\right),
 \qquad c_0=c_{M/2}=1,\quad c_k=2\text{ otherwise}.
\]

With \(\Delta\omega=2\pi/L\), this is the chapter's trapezoidal spectral sum, apart from treatment of the imaginary part at the highest frequency. The clipping and renormalization in the actual helper can break exact equivalence. Periodic-grid interpretation, frequency endpoints, clipping and renormalization should therefore be documented when describing the implementation.

A fresh numerical check on one Gaussian sample (n=100, 20 bandwidths from 0.01 to 2, H=12, M=8192) found maximum absolute Gram differences of 2.55e-15 between raw spatial and discrete-frequency calculations, and 1.83e-15 between raw spatial calculation and the chapter's continuous-integrand trapezoidal sum. The implemented corrected half-squared distances differed from the exact Gaussian pair formula by at most 2.23e-14. This is one illustrative check, not validation of every experiment or selected bandwidth. [Settings and results](/Users/iljasukhanov/MASTERARBEIT/output/analysis/kde_fourier_parseval_audit_2026-10-22.json).

## Optional background comparisons

**Raykar, Duraiswami and Zhao (2010), Table 1 and Sections 6.2–6.4, especially Figure 2**, compare direct KDE evaluation, binning, binned FFT and their fast approximation in runtime and numerical error. These are density-value comparisons, not a comparison of the three Gram-matrix methods above. [Published article](https://doi.org/10.1198/jcgs.2010.09046); [published PDF](https://www.umiacs.umd.edu/labs/cvl/pirl/vikas/publications/raykar_JCGS_2010_FastKDE.pdf); [author manuscript used to inspect section/figure locators](https://users.umiacs.umd.edu/~ramanid/pubs/Raykar_Duraiswami_Zhao_JCGS_2010.pdf).

## Prepared text

The adjacent `.tex` file is proposed replacement text for the TODO only; it has not been inserted into the protected thesis. It cites the methodological precedent, gives the operation count as our analysis, and explicitly describes the related benchmark's different scope. The adjacent `.bib` file contains the references without modifying the thesis bibliography.
