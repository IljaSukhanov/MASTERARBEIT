"""Choose a fixed AR(3) start from the uniform causal-coefficient prior.

This is integration over coefficient space, not a new time-series experiment.
Run with the project's existing Python scientific stack; outputs stay here.
"""
from pathlib import Path
import os

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ar-prior-start-matplotlib')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import check_grad, minimize, minimize_scalar
from scipy.special import betaincinv, roots_legendre
from scipy.stats import qmc
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent


def directions(a, b, c):
    """The p=3 recursion used by ar_tools.functions.jones_ar_coefficients."""
    phi = np.column_stack([a*(1-b)-c*b, b-c*a*(1-b), c])
    normal = np.column_stack([np.ones(len(phi)), -phi])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    return phi, normal


def integrate_prior(n):
    """Tensor Gauss-Legendre rule including the three PACF densities."""
    x, w = roots_legendre(n)
    a, b, c = np.meshgrid(x, x, x, indexing='ij')
    weights = np.einsum('i,j,k->ijk', w/2, w*(1-x)/2, w*3*(1-x*x)/4).ravel()
    phi, normal = directions(a.ravel(), b.ravel(), c.ravel())
    np.testing.assert_allclose(weights.sum(), 1, atol=2e-13, rtol=0)
    np.testing.assert_allclose(weights@phi, [0, -1/3, 0], atol=2e-13, rtol=0)
    return weights, normal


def risk_and_gradient(z, weights, normal):
    length = np.linalg.norm(z)
    u = z/length
    dots = normal@u
    absolute = np.clip(np.abs(dots), 0, 1-1e-15)
    risk = float(weights@np.arccos(absolute))
    gradient = -normal.T@(weights*np.sign(dots)/np.sqrt(1-absolute*absolute))
    return risk, (gradient-u*(u@gradient))/length


def plane_direction(theta):
    return np.array([np.cos(theta), 0, np.sin(theta), 0])


with threadpool_limits(limits=1):
    weights, normal = integrate_prior(20)
    error = check_grad(lambda x: risk_and_gradient(x, weights, normal)[0],
                       lambda x: risk_and_gradient(x, weights, normal)[1],
                       np.array([.98, .04, .18, -.03]))
    assert error < 2e-6, error
    starts = [*np.eye(4), np.array([.9664, 0, .257, 0]),
              *np.random.default_rng(20261021).normal(size=(43, 4))]
    searches = []
    for start in starts:
        fit = minimize(risk_and_gradient, start/np.linalg.norm(start),
                       args=(weights, normal), jac=True, method='BFGS',
                       options={'gtol': 1e-9, 'maxiter': 250})
        direction = fit.x/np.linalg.norm(fit.x)
        if direction[0] < 0:
            direction *= -1
        searches.append((float(fit.fun), direction, bool(fit.success)))
    searches.sort(key=lambda item: item[0])
    assert np.linalg.norm(searches[0][1][[1, 3]]) < 2e-5
    theta = np.arctan2(searches[0][1][2], searches[0][1][0])

    convergence = []
    e1 = np.array([1., 0., 0., 0.])
    for n in [20, 40, 64, 96]:
        weights, normal = integrate_prior(n)
        optimum = minimize_scalar(
            lambda t: risk_and_gradient(plane_direction(t), weights, normal)[0],
            bounds=(theta-.08, theta+.08), method='bounded',
            options={'xatol': 1e-12},
        )
        theta = float(optimum.x)
        candidate = plane_direction(theta)
        convergence.append((n, candidate.copy(), np.degrees(optimum.fun),
                            np.degrees(risk_and_gradient(e1, weights, normal)[0])))
    assert abs(convergence[-1][2]-convergence[-2][2]) < 5e-5
    assert np.linalg.norm(convergence[-1][1]-convergence[-2][1]) < 1e-5
    assert convergence[-1][2] < convergence[-1][3]

    # Full-sphere refinement at higher quadrature order; no planar restriction.
    full_refinements = []
    rw, rq = integrate_prior(40)
    for start in [candidate, np.array([1., .1, .25, .1]), np.array([1., -.1, .25, -.1])]:
        fit = minimize(risk_and_gradient, start/np.linalg.norm(start), args=(rw, rq),
                       jac=True, method='BFGS', options={'gtol': 1e-9, 'maxiter': 250})
        full_direction = fit.x/np.linalg.norm(fit.x)
        if full_direction[0] < 0:
            full_direction *= -1
        assert np.linalg.norm(full_direction-convergence[1][1]) < 2e-5
        full_refinements.append((np.degrees(fit.fun), full_direction))

    mean_direction = weights@normal
    mean_direction /= np.linalg.norm(mean_direction)
    matrix = (normal.T*weights)@normal
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    projector_mean = eigenvectors[:, -1]
    projector_mean *= np.sign(projector_mean[0])
    coefficient_mean_start = np.array([1., 0, 1/3, 0])/np.sqrt(1+1/9)

    # Independent randomized Sobol validation of the prior integrals.
    validation = []
    for replication in range(8):
        points = qmc.Sobol(3, scramble=True, seed=20261021+replication).random_base2(17)
        a = 2*points[:, 0]-1
        b = 1-2*np.sqrt(1-points[:, 1])
        c = 2*betaincinv(2, 2, points[:, 2])-1
        _, q = directions(a, b, c)
        candidate_loss = np.degrees(np.arccos(np.clip(np.abs(q@candidate), 0, 1)).mean())
        e1_loss = np.degrees(np.arccos(q[:, 0]).mean())
        validation.append((candidate_loss, e1_loss))
    validation = np.array(validation)
    np.testing.assert_allclose(validation.mean(axis=0), [convergence[-1][2], convergence[-1][3]], atol=.002, rtol=0)

    # One-dimensional section through the best candidate, for visualization.
    ts = np.radians(np.linspace(-20, 40, 181))
    curve = np.array([np.degrees(risk_and_gradient(plane_direction(t), rw, rq)[0]) for t in ts])
    plt.rcParams.update({'font.size': 11, 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(8.2, 4.9), layout='constrained')
    ax.plot(np.degrees(ts), curve, color='#2b6589', linewidth=2)
    ax.scatter([0], [convergence[-1][3]], color='#b7703b', s=55, label=f'e1: {convergence[-1][3]:.3f}°')
    ax.scatter([np.degrees(theta)], [convergence[-1][2]], color='#2e7f5b', s=55,
               label=f'Numerical candidate: {convergence[-1][2]:.3f}°')
    ax.set(xlabel=r'Tilt t in u(t) = (cos(t), 0, sin(t), 0), degrees',
           ylabel='Expected acute angle to population direction, degrees',
           title='A fixed starting direction from the uniform AR(3) coefficient prior')
    ax.grid(alpha=.18)
    ax.legend(loc='upper center', framealpha=.95)
    fig.savefig(HERE/'expected_angle.png', dpi=170, bbox_inches='tight', pad_inches=.12)
    fig.savefig(HERE/'expected_angle.pdf', bbox_inches='tight', pad_inches=.12)
    plt.close(fig)

    lines = [
        '# A fixed starting direction chosen from the uniform causal AR(3) prior',
        '',
        'This is a new numerical calculation over the coefficient prior, independent of the 100 models used in Experiments 3 and 4. No time-series fitting or thesis editing is performed.',
        '',
        '## Objective',
        '',
        'Let Phi be uniform with respect to coefficient-space Lebesgue volume on the causal region C3, and let v(Phi)=(1,-Phi)/sqrt(1+||Phi||^2). Minimize R(u)=E[arccos(|u dot v(Phi)|)] over ||u||=1. Opposite directions are identified. This defines the best fixed direction for expected acute angle, not necessarily the best optimizer initialization for final coefficient error.',
        '',
        'The expected-distance minimizer is a Frechet median on the space of unoriented directions; squared distance defines a different objective. See [Yang, introduction and section 2](https://arxiv.org/pdf/1110.3899). The population direction is the one stated in [the thesis proposition](../../../Abschlussarbeit_2021/LaTeX/chapters/3_application_to_time_series.tex).',
        '',
        '## Uniform-prior integration and coefficient asymmetry',
        '',
        'The Jones sampler uses independent a=2 Beta(1,1)-1, b=2 Beta(1,2)-1, c=2 Beta(2,2)-1 for AR(3). Its recursion gives phi1=a(1-b)-cb, phi2=b-ca(1-b), phi3=c. The joint density of (a,b,c) is 3/16 (1-b)(1-c^2). The Jacobian determinant is (1-b)(1-c^2), so the induced coefficient density is constant, 3/16, on C3. The transformation onto the causal region is the Jones parametrization.',
        '',
        'My direct calculation from independence gives E[Phi]=(0,-1/3,0). Therefore the coefficient prior is not centered at zero. The third direction coordinate corresponds to the negative of the second AR coefficient. Sources: [Jones (1987), pp. 134-138](https://doi.org/10.2307/2347544), [implemented sampler](../../../Code/Code_AR_processes/ar_tools/functions.py).',
        '',
        '## Numerical candidate',
        '',
        f'Best direction found: {candidate.tolist()}. It corresponds to initial coefficients {(-candidate[1:]/candidate[0]).tolist()}. Its tilt from e1 is {np.degrees(theta):.8f} degrees.',
        '',
        f'Expected acute angle: candidate {convergence[-1][2]:.8f} degrees; e1 {convergence[-1][3]:.8f} degrees. Improvement: {convergence[-1][3]-convergence[-1][2]:.8f} degrees.',
        '',
        '| Quadrature nodes per PACF | Direction first component | Direction third component | Candidate expected angle (degrees) | e1 expected angle (degrees) |',
        '| --- | --- | --- | --- | --- |',
    ]
    for n, u, loss, baseline in convergence:
        lines.append(f'| {n} | {u[0]:.10f} | {u[2]:.10f} | {loss:.10f} | {baseline:.10f} |')
    lines += [
        '',
        f'The initial full-sphere search used {len(searches)} starts at 20^3 quadrature nodes, including all coordinate axes and 43 Gaussian starts. The best value matched the (first, third)-coordinate-plane solution. Three full-sphere refinements at 40^3 nodes agreed with that candidate. Some starts reached other stationary or poorly converged points. This is numerical evidence; no proof of global optimality or uniqueness is claimed.',
        '',
        f'Independent validation used 8 scrambled Sobol replicates of 2^17 prior draws each. Mean expected-angle estimates: candidate {validation[:,0].mean():.8f}, e1 {validation[:,1].mean():.8f} degrees. Replicate ranges: candidate [{validation[:,0].min():.8f}, {validation[:,0].max():.8f}], e1 [{validation[:,1].min():.8f}, {validation[:,1].max():.8f}].',
        '',
        '![Expected angle versus tilt](expected_angle.png)',
        '',
        '## Different distances give different centers',
        '',
        f'For squared Euclidean distance to the canonically oriented v(Phi), expansion gives E||u-v||^2=2-2 u dot E[v], hence u=E[v]/||E[v]||. Numerical direction: {mean_direction.tolist()}. This objective distinguishes v and -v and is not the acute-angle objective.',
        '',
        f'For squared projector distance, ||uu^T-vv^T||_F^2=2(1-(u dot v)^2), so the optimum is a leading eigenvector of E[vv^T]. Numerical direction: {projector_mean.tolist()}. See [Mardia and Patrangenaru, section 4, equations (4.3)-(4.4), reprint pp. 18-19](https://arxiv.org/pdf/math/0508280#page=18). This is a different, sign-invariant squared-sine criterion.',
        '',
        f'For squared Euclidean coefficient loss, the best fixed coefficient vector is E[Phi]=(0,-1/3,0), whose induced direction is {coefficient_mean_start.tolist()}. Averaging coefficients and averaging normalized directions do not commute.',
        '',
        '## What this does and does not establish',
        '',
        'This construction is a principled fixed initialization for the specified order, prior and distance. It is independent of the time-series observations and the earlier 100 coefficient draws. It does not guarantee a smaller empirical width, faster convergence or lower final coefficient error. Before selecting it as an algorithm default, freeze the prior-derived vector and compare it with e1 and random starts on independent time-series experiments with matched step schedules and stopping rules.',
        '',
        f'Checks: quadrature weights sum to one, analytic coefficient mean reproduced, objective-gradient finite-difference error {error:.5g}, quadrature convergence checked, independent randomized-Sobol agreement checked. The global optimum is not certified.',
        '',
        'Reproduce with `python3 compute_prior_start.py`. This script writes only this report and the PNG/PDF plot in its own directory.',
    ]
    (HERE/'findings.md').write_text('\n'.join(lines)+'\n')
    print('Candidate:', candidate)
    print('Initial coefficients:', -candidate[1:]/candidate[0])
    print('Expected angle candidate/e1:', convergence[-1][2], convergence[-1][3])
    print('Independent Sobol means:', validation.mean(axis=0))
    print('Saved:', HERE/'findings.md')
