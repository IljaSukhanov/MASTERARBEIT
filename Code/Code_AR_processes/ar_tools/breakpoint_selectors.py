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


def select_order_nearest_width(widths_by_k, reference_width, *, candidate_orders):
    candidate_widths = np.asarray(widths_by_k, dtype=float)[candidate_orders]
    if not np.isfinite(reference_width) or not np.any(np.isfinite(candidate_widths)):
        raise ValueError("A finite reference and at least one finite candidate width are required.")
    # candidate_orders are ascending, so argmin resolves exact ties toward lower order.
    return int(candidate_orders[np.argmin(np.abs(candidate_widths - reference_width))])


def tail_excess_ratios(width_curve, *, max_dimension, candidate_orders):
    """Compute the proposed signed T_d for d=1,...,max_dimension-2."""
    widths = np.asarray(width_curve, dtype=float)
    if widths.shape != (max_dimension,):
        raise ValueError(f"Expected implied widths for d=1,...,{max_dimension}.")
    terminal_width = widths[max_dimension - 1]
    ratios = np.full(len(candidate_orders), np.nan)
    denominators = np.full(len(candidate_orders), np.nan)
    if not np.isfinite(terminal_width):
        return ratios, denominators
    for index, d in enumerate(candidate_orders):
        numerator = widths[d - 1] - terminal_width
        tail_excesses = widths[d:max_dimension - 1] - terminal_width
        if not np.isfinite(numerator) or not np.all(np.isfinite(tail_excesses)):
            continue
        denominator = float(np.mean(tail_excesses))
        denominators[index] = denominator
        if denominator != 0.0:
            ratios[index] = numerator / denominator
    return ratios, denominators


def select_tail_ratio(widths, *, max_dimension, candidate_orders):
    """Signed terminal-excess ratio, with no regularizer or tolerance."""
    scores = np.empty(len(candidate_orders))
    for index, d in enumerate(candidate_orders):
        numerator = widths[d - 1] - widths[max_dimension - 1]
        denominator = np.mean(widths[d:max_dimension - 1] - widths[max_dimension - 1])
        if denominator > 0:
            scores[index] = numerator / denominator
        elif denominator == 0 and numerator > 0:
            scores[index] = np.inf
        else:
            scores[index] = np.nan
    defined = np.isfinite(scores) | np.isposinf(scores)
    return (int(candidate_orders[np.nanargmax(scores)]) if defined.any() else None), scores


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


def _sse(values):
    # A repeated analytical plateau has zero variance exactly; avoid roundoff
    # from taking the mean of repeated identical floating-point values.
    if np.ptp(values) == 0:
        return 0.0
    residual = values - values.mean()
    return float(residual @ residual)


def _ratio(numerator, denominator):
    return numerator / denominator if denominator > 0 else (np.inf if numerator > 0 else 0.0)


def all_population_selectors(widths):
    m = np.asarray(widths, dtype=float)
    results = width_selectors(m)
    results["Descent chain"] = descent_chain(m)[0]
    drops = m[:-1] - m[1:]
    denominator = drops[:-1] + drops[1:]
    drift = np.full(len(m) - 2, -np.inf)
    np.divide(drops[:-1] - drops[1:], denominator, out=drift, where=denominator > 0)
    results["Local drift"] = int(np.argmax(drift) + 1)
    for method in ("Drop-to-tail energy", "Tail SSE ratio", "Tail variance ratio"):
        scores = []
        for q in range(1, len(m) - 1):
            if method == "Drop-to-tail energy":
                numerator, denominator = drops[q - 1] ** 2, float((drops[q:] ** 2).sum())
            else:
                previous, tail = m[q - 1:], m[q:]
                numerator, denominator = _sse(previous), _sse(tail)
                if method == "Tail variance ratio":
                    numerator /= len(previous)
                    denominator /= len(tail)
            scores.append(_ratio(numerator, denominator))
        results[method] = int(np.argmax(scores) + 1)
    results["Minimum width — first tied minimum"] = int(np.argmin(m[1:]) + 1)
    return results
