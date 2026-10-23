import numpy as np


def descent_chain(widths):
    """Definitions 1.2.8–1.2.10 and Eq. (1.6) of the supplied preview, pp. 5–6.

    The rule excludes k=1,K, uses positive/0=infinity and 0/0=0, and
    chooses the smallest order on ties. Only monotone input is accepted.
    """
    m = np.asarray(widths, dtype=float)
    if m.ndim != 1 or len(m) < 4 or not np.isfinite(m).all():
        raise ValueError("Expected at least four finite widths.")
    tolerance = 32 * np.finfo(float).eps * max(1.0, float(np.abs(m).max()))
    if np.any(np.diff(m) > tolerance):
        raise ValueError("The descent chain requires nonincreasing widths.")
    # Re-evaluating a padded dot product can change the width by a few ulps.
    # Only machine-precision-sized drops become zero; a substantive upward
    # step is rejected above. This prevents an ulp-sized final drop from
    # generating infinite odds. No saved fitted curve is smoothed.
    drops = np.r_[0.0, np.maximum(m[:-1] - m[1:], 0.0)]
    drops[drops <= tolerance] = 0.0
    remaining = np.r_[np.cumsum(drops[:0:-1])[::-1], 0.0]
    alpha = np.zeros(len(m))
    alpha[0] = 1.0
    np.divide(drops[1:], remaining[:-1], out=alpha[1:], where=remaining[:-1] > 0)
    stationary = alpha / alpha.sum()
    odds = np.zeros(len(m))
    np.divide(drops, remaining, out=odds, where=remaining > 0)
    odds[(remaining == 0) & (drops > 0)] = np.inf
    selected_order = int(np.argmax(odds[1:-1]) + 1)

    transition = np.zeros((len(m), len(m)))
    for i in range(len(m)):
        if remaining[i] == 0:
            transition[i, 0] = 1.0
        else:
            transition[i, i + 1:] = drops[i + 1:] / remaining[i]
    assert np.allclose(transition.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    residual = float(np.max(np.abs(stationary @ transition - stationary)))
    assert residual < 1e-12
    assert selected_order == int(np.argmax(stationary[1:-1]) + 1)
    return selected_order, drops, remaining, alpha, stationary, odds, residual


def width_selectors(widths):
    m = np.asarray(widths, dtype=float)
    max_order = len(m) - 2
    return {
        "First difference": int(np.argmax((m[:-1] - m[1:])[:max_order]) + 1),
        "Second difference": int(np.argmax(np.diff(m, n=2)) + 1),
    }


def earlier_bend_certificate(prefix_widths, terminal_upper):
    """Original deterministic diagnostic, not a changed selection rule.

    Prefix entries must be global minima through the true break dimension.
    terminal_upper is any feasible width in the last dimension. If an earlier
    candidate wins here, it also wins for every smaller terminal minimum:
    A/(a-b)>B/(c-b) iff A(c-b)-B(a-b)>0. Such a win implies A>B,
    so decreasing b can only increase this positive margin.
    """
    m = np.asarray(prefix_widths, dtype=float)
    b = float(terminal_upper)
    if len(m) < 3 or not np.all(np.diff(m) <= 0) or b >= m[-1]:
        raise ValueError("Need a monotone prefix and terminal upper bound below it.")
    drops = -np.diff(m)
    earlier_scores = drops[:-1] / (m[1:-1] - b)
    index = int(np.argmax(earlier_scores))
    a, A, c, B = m[index + 1], drops[index], m[-1], drops[-1]
    margin = A * (c - b) - B * (a - b)
    return {"earlier_order": index + 1,
            "earlier_score_at_terminal_upper": float(earlier_scores[index]),
            "true_score_at_terminal_upper": float(B / (c - b)),
            "cross_product_margin": float(margin),
            "earlier_drop": float(A), "true_drop": float(B),
            "earlier_bend_wins_for_all_smaller_terminal_widths": bool(margin > 1e-10)}


def all_population_selectors(widths):
    m = np.asarray(widths, dtype=float)
    results = width_selectors(m)
    results["Descent chain"] = descent_chain(m)[0]
    return results
