"""Reproduce the numerical examples in report.md; no thesis sources are edited."""

from pathlib import Path
import json
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qhat-matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import eval_hermitenorm


OUT = Path(__file__).resolve().parent


def energy_log_derivatives(samples, h, r):
    """Return Q, dQ/d(log h), d²Q/d(log h) using all ordered pairs."""
    n = len(samples)
    u = ((samples[:, None] - samples[None, :]) / (2*h))**2
    p = u*u - 3*u + 0.75
    g = 2*u**3 - 11*u*u + 10.5*u - 0.75
    k = 4*u**4 - 36*u**3 + 76*u*u - 33*u + 0.75
    w = np.exp(-u)/(4*np.sqrt(np.pi)*n*n*h**(1+2*r))
    return np.array([np.sum(w*p), np.sum(w*(g-2*r*p)),
                     np.sum(w*(k-4*r*g+4*r*r*p))])


def spatial_roughness(samples, h, j):
    """Independent integration of the squared spatial Gaussian KDE derivative."""
    def integrand(x):
        z = (x-samples)/h
        derivative = ((-1)**j * np.mean(eval_hermitenorm(j, z)*np.exp(-z*z/2))
                      / (np.sqrt(2*np.pi)*h**(j+1)))
        return derivative*derivative
    edges = np.unique(np.r_[samples.min()-11*h, samples, samples.max()+11*h])
    return sum(quad(integrand, a, b, epsabs=1e-12, epsrel=2e-11)[0]
               for a, b in zip(edges[:-1], edges[1:]))


u, r = sp.symbols("u r", real=True)
p = u*u-3*u+sp.Rational(3, 4)
P = sp.expand((2*u-1-2*r)*p-2*u*sp.diff(p, u))
P_reference = (2*u**3-(11+2*r)*u*u
               +(sp.Rational(21, 2)+6*r)*u-sp.Rational(3, 4)*(1+2*r))
assert sp.simplify(P-P_reference) == 0

small_sample = np.array([-0.9, -0.2, 0.15, 1.2])
checks = []
for h, rv in [(0.08, 0.0), (0.3, 13/14), (0.9, 1.0)]:
    q, qz, qzz = energy_log_derivatives(small_sample, h, rv)
    R2, R3 = [spatial_roughness(small_sample, h, j) for j in (2, 3)]
    q_spatial = 0.5*h**(4-2*rv)*R2
    derivative_spatial = h**(3-2*rv)*((2-rv)*R2-h*h*R3)
    checks.append(dict(h=h, r=rv, q=float(q),
                       q_spatial=float(q_spatial),
                       derivative=float(qz/h),
                       derivative_spatial=float(derivative_spatial)))
    assert np.isclose(q, q_spatial, rtol=1e-10, atol=1e-12)
    assert np.isclose(qz/h, derivative_spatial, rtol=1e-10, atol=1e-12)

a = np.array([-3, -1, 0, 1, 3], dtype=float)
samples = (a[:, None]+a[None, :]/100).ravel()
rv = 13/14
z_grid = np.linspace(np.log(0.003), np.log(20), 8193)
grid_values = np.array([energy_log_derivatives(samples, np.exp(z), rv) for z in z_grid])
roots = []
for i in np.flatnonzero(grid_values[:-1, 1]*grid_values[1:, 1] < 0):
    z = brentq(lambda z: energy_log_derivatives(samples, np.exp(z), rv)[1],
               z_grid[i], z_grid[i+1], xtol=1e-14)
    h = np.exp(z)
    q, qz, qzz = energy_log_derivatives(samples, h, rv)
    R2, R3, R4 = [spatial_roughness(samples, h, j) for j in (2, 3, 4)]
    B = h*h*R3/R2
    curvature_spatial = 2*h**(2-2*rv)*(h**4*R4-(2-rv)*(3-rv)*R2)
    assert np.isclose(B, 2-rv, rtol=2e-10)
    assert np.isclose(qzz/h**2, curvature_spatial, rtol=1e-8)
    roots.append(dict(h=float(h), q=float(q),
                      kind="minimum" if qzz > 0 else "maximum",
                      B_spatial=float(B), second_derivative=float(qzz/h**2)))
assert [x["kind"] for x in roots] == ["minimum", "maximum", "minimum", "maximum"]

sign_checks = []
for h in [0.01, 0.018, 0.03, 1.0, 1.8, 3.0]:
    n = len(samples)
    qz = energy_log_derivatives(samples, h, rv)[1]
    sign_sum = qz*4*np.sqrt(np.pi)*n*n*h**(1+2*rv)
    sign_checks.append(dict(h=h, derivative_sum=float(sign_sum)))

# Exact Gaussian-sampling expectation, with sigma=1.
aa, bb = 1+2*rv, 4-2*rv
threshold = 1+aa/(2*(bb/7)**3.5)
s0 = bb/7
expected_roots = []
for lo, hi in [(1e-12, s0), (s0, bb/5)]:
    s = brentq(lambda s: 199*s**2.5*(bb-5*s)-aa, lo, hi, xtol=1e-14)
    expected_roots.append(float(np.sqrt(s/(1-s))))

# Independently compare direct off-diagonal roughness to the BCV identity.
h = 0.3
n = len(small_sample)
q0 = energy_log_derivatives(small_sample, h, 0)[0]
R2 = spatial_roughness(small_sample, h, 2)
bcv = 1/(2*np.sqrt(np.pi)*n*h) + h**4/4*(R2-3/(8*np.sqrt(np.pi)*n*h**5))
assert np.isclose(bcv, q0/2+13/(32*np.sqrt(np.pi)*n*h), rtol=1e-12)

result = dict(symbolic_derivative=str(P), spatial_checks=checks,
              counterexample=dict(r=rv, sample=samples.tolist(), extrema=roots,
                                  derivative_sign_checks=sign_checks,
                                  exhaustive_root_isolation=False),
              gaussian_expectation=dict(threshold_n=float(threshold),
                                        n=200, sigma=1, stationary_h=expected_roots),
              bcv_identity_checked=True)
(OUT/"checks.json").write_text(json.dumps(result, indent=2)+"\n")

# A diagnostic that exposes the sign changes even for shallow energy minima.
hs = np.exp(z_grid)
Bs = 2-rv-grid_values[:, 1]/(2*grid_values[:, 0])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(10, 4.4), layout="constrained")
ax.plot(hs, Bs, color="#234e70", lw=2,
        label=r"$B(h)=h^2\|\widehat f_h^{(3)}\|_2^2/\|\widehat f_h^{(2)}\|_2^2$")
ax.axhline(2-rv, color="#b45309", linestyle="--", lw=1.5,
           label=r"Stationary point: $B(h)=2-r=15/14$")
ax.fill_between(hs, Bs, 2-rv, where=Bs < 2-rv,
                color="#f59e0b", alpha=0.25,
                label=r"$\widehat Q_r$ increases")
for root in roots:
    is_minimum = root["kind"] == "minimum"
    ax.scatter(root["h"], 2-rv, marker="v" if is_minimum else "^", s=65,
               facecolor="#b45309" if is_minimum else "white",
               edgecolor="#b45309", zorder=5)
    ax.annotate(("local min\n" if is_minimum else "local max\n")+f'h={root["h"]:.4g}',
                (root["h"], 2-rv), xytext=(-20 if is_minimum else 24,
                                                         -44 if is_minimum else 24),
                textcoords="offset points", ha="center", fontsize=9,
                arrowprops=dict(arrowstyle="-", color="#6b7280", lw=0.7))
ax.set(xscale="log", xlim=(0.003, 20), ylim=(0.45, 3.5),
       xlabel="Bandwidth h (log scale)", ylabel=r"$B(h)$",
       title="One fixed sample can have two local energy minima\n"
             + r"$r=13/14$; dashed line: $B=2-r$; orange intervals: energy increases")
ax.grid(axis="y", alpha=0.15)
fig.savefig(OUT/"local_extrema.png", dpi=180, bbox_inches="tight", pad_inches=0.15)
fig.savefig(OUT/"local_extrema.pdf", bbox_inches="tight", pad_inches=0.15)
plt.close(fig)
print(json.dumps({"verified_extrema": roots, "gaussian_threshold_n": threshold,
                  "gaussian_n200_stationary_h": expected_roots,
                  "files": ["checks.json", "local_extrema.png", "local_extrema.pdf"]}, indent=2))
