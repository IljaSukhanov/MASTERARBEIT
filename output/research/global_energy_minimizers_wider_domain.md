# Why increasing the upper bandwidth limit moves the endpoint spike

This is an original algebraic derivation for the Gaussian sample energy used
in the experiment. It is not a quoted theorem.

The implementation in
[qhat.py, `_energy`](../../Code/Code_kde/kde_tools/qhat.py#L54), together with
[continuous_minimizers.py, `qr_log_energy`](../../Code/Code_kde/kde_tools/continuous_minimizers.py#L49), gives

\[
\widehat Q_r(h)=\frac{h^{-1-2r}}{4\sqrt\pi n^2}
\left\{\frac34n+2\sum_{i<j}e^{-u_{ij}}
\left(u_{ij}^2-3u_{ij}+\frac34\right)\right\},
\qquad u_{ij}=\frac{(X_i-X_j)^2}{4h^2}.
\]

For a fixed finite sample, every $u_{ij}$ tends to zero as $h$ tends to
infinity. The bracket therefore tends to

\[
\frac34n+2\binom n2\frac34=\frac34n^2.
\]

Consequently,

\[
\boxed{\widehat Q_r(h)\sim\frac{3}{16\sqrt\pi}\,h^{-1-2r}}
\qquad (h\longrightarrow\infty).
\]

In particular, the energy tends to zero for all three parameters used here:
$r=0$, $13/14$, and $1$. The energy is positive at finite positive bandwidth:
it is a positive multiple of the squared $L^2$ norm of the bandwidth
derivative of a finite Gaussian KDE, and that derivative is not identically
zero. Thus the infimum on $(0,\infty)$ is zero and is not attained at a finite
bandwidth. Interior local valleys remain possible.

This explains why increasing the finite upper search limit cannot in general
remove the preference for its endpoint. An endpoint selected at $h=2$ can be
replaced by a still lower energy at a larger bandwidth. The fixed-sample
control in the [energy notebook](../../Code/Code_kde/notebooks/continuous_ise_qhat_distributions_vs_n.ipynb)
compares detected interior minima with both endpoints; change its `H_MAX`
to inspect a wider interval. A finite scan is not a rigorous global-optimality
certificate.

The separate wider-domain Monte Carlo study was removed during the thesis-only
cleanup. The derivation above is retained, together with the reproducible
[sampling inputs](../../Code/Code_kde/inputs/README.md).
