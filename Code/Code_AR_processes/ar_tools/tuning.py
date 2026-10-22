"""Paired geometric step-schedule comparisons on fixed clouds and AR paths."""

from time import perf_counter

import numpy as np
from scipy.spatial.distance import pdist
from threadpoolctl import threadpool_limits

from . import descent
from .coefficient_errors import coefficient_mae
from .functions import is_stationary, jones_ar_coefficients, simulate_uniform_ar, sliding_window_point_cloud

__all__ = [
    "tune_geometric_width_schedule", "tune_geometric_half_life_schedules",
    "summarize_geometric_replications", "tune_fixed_ar_realizations",
    "tune_ar_sample_sizes", "compare_ar_schedules", "summarize_configuration_wins",
    "tune_jones_ar_realizations",
]


def tune_geometric_width_schedule(
    points,
    initial_directions,
    gamma0_grid,
    beta_grid,
    *,
    iterations,
    patience,
    improvement_tolerance,
    optimizer_seed,
    progress=False,
    stationarity_check=False,
    max_restarts=100,
    delta=1e-8,
):
    """Select gamma_m = gamma0 * beta**m by attained empirical width.

    Every candidate uses the supplied starting directions and the same
    fallback seed for each supplied start. Ties are broken
    by total evaluations, then gamma0 and beta. True coefficients are not
    inputs to this search. Histories retain the best width across starts at
    each iteration, holding each stopped run at its last recorded best width.
    Optional stationarity checkpoints/restarts use the shared optimizer.
    delta controls the meaningful best-width decrease required to reset
    patience, independently of improvement_tolerance used for record updates.
    Evaluation counts include discarded attempts. Starts that exhaust the
    restart limit are excluded from width selection. If every start fails,
    the candidate has NaN width/direction and explicit failure metadata.
    """
    points = np.asarray(points, dtype=float)
    directions = np.array(initial_directions, dtype=float, copy=True)
    if (
        points.ndim != 2 or len(points) < 2 or points.shape[1] < 1
        or not np.all(np.isfinite(points))
    ):
        raise ValueError("points must be a finite (n, k) cloud with n >= 2 and k >= 1.")
    if (
        directions.ndim != 2 or len(directions) < 1
        or directions.shape[1] != points.shape[1]
        or not np.all(np.isfinite(directions))
        or np.any(np.linalg.norm(directions, axis=1) <= np.finfo(float).eps)
    ):
        raise ValueError("initial_directions must contain finite nonzero vectors matching k.")
    grids = []
    for values, name in ((gamma0_grid, "gamma0_grid"), (beta_grid, "beta_grid")):
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be a nonempty finite one-dimensional grid.")
        if np.any(values <= 0) or (name == "beta_grid" and np.any(values >= 1)):
            raise ValueError("Require gamma0 > 0 and 0 < beta < 1.")
        grids.append(np.unique(values))
    for value, name in ((iterations, "iterations"), (patience, "patience")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if not np.isfinite(improvement_tolerance) or improvement_tolerance < 0:
        raise ValueError("improvement_tolerance must be finite and nonnegative.")
    if (isinstance(delta, (bool, np.bool_))
            or not isinstance(delta, (int, float, np.integer, np.floating))
            or not np.isfinite(delta) or delta < 0):
        raise ValueError("delta must be a finite nonnegative number.")
    if (
        isinstance(optimizer_seed, (bool, np.bool_))
        or not isinstance(optimizer_seed, (int, np.integer)) or optimizer_seed < 0
    ):
        raise ValueError("optimizer_seed must be a nonnegative integer.")

    rows, per_start, histories = [], [], {}
    winning_key, winning_row, winning_results = None, None, None
    dimension = points.shape[1]
    total_candidates = len(grids[0]) * len(grids[1])
    with threadpool_limits(limits=1):
        for gamma0 in grids[0]:
            for beta in grids[1]:
                candidate_id = f"grid_{len(rows):03d}"
                started = perf_counter()
                runs = [
                    descent.projected_subgradient_empirical_width(
                        points,
                        iterations=iterations,
                        initial_step=float(gamma0),
                        geometric_decay=float(beta),
                        patience=patience,
                        improvement_tolerance=improvement_tolerance, delta=delta,
                        initial_direction=direction,
                        rng=np.random.SeedSequence([
                            optimizer_seed, 0, dimension, 21, start_index,
                        ]),
                        step_schedule="geometric",
                        stationarity_check=stationarity_check,
                        max_restarts=max_restarts,
                    )
                    for start_index, direction in enumerate(directions)
                ]
                eligible = [index for index, run in enumerate(runs)
                            if run.get("stationarity_check_passed", True)]
                selected_index = min(eligible, key=lambda index: runs[index]["minimum_width"]) if eligible else 0
                selected = runs[selected_index]
                direction = selected["direction"].copy() if eligible else np.full(dimension, np.nan)
                if direction[0] < 0:
                    direction *= -1
                width = float(selected["minimum_width"]) if eligible else np.nan
                if eligible and (
                    not np.isfinite(width) or not np.all(np.isfinite(direction))
                    or not np.isclose(np.linalg.norm(direction), 1.0)
                    or not np.isclose(np.ptp(points @ direction), width)
                ):
                    raise RuntimeError(f"Invalid attained width for {candidate_id}.")
                row = {
                    "candidate_id": candidate_id,
                    "gamma0": float(gamma0), "beta": float(beta),
                    "minimum_width": width, "selected_start": selected_index + 1,
                    "delta": float(delta),
                    "total_evaluations": sum(
                        run.get("total_evaluations", len(run["width_history"])) for run in runs
                    ),
                    "patience_stops": sum(run["termination_reason"] == "patience" for run in runs),
                    "runtime_seconds": perf_counter() - started,
                }
                if stationarity_check:
                    row.update(
                        stationarity_check=True,
                        stationarity_check_passed=bool(eligible),
                        restart_limit_reached=any(run["restart_limit_reached"] for run in runs),
                        restart_count=sum(run["restart_count"] for run in runs),
                        attempt_count=sum(run["attempt_count"] for run in runs),
                        final_evaluations=len(selected["width_history"]),
                        checkpoint_spectral_radius=selected["checkpoint"]["spectral_radius"],
                    )
                row.update({f"u_{j}": float(value) for j, value in enumerate(direction)})
                rows.append(row)
                valid_runs = [runs[index] for index in eligible]
                longest = max(len(run["best_width_history"]) for run in (valid_runs or runs))
                histories[candidate_id] = np.minimum.reduce([
                    np.pad(
                        run["best_width_history"],
                        (0, longest - len(run["best_width_history"])), mode="edge",
                    )
                    for run in valid_runs
                ]) if valid_runs else np.full(longest, np.nan)
                for start_index, run in enumerate(runs):
                    per_start.append({
                        "candidate_id": candidate_id,
                        "gamma0": float(gamma0), "beta": float(beta),
                        "start": start_index + 1,
                        "minimum_width": (float(run["minimum_width"])
                                          if run.get("stationarity_check_passed", True) else np.nan),
                        "evaluations": run.get("total_evaluations", len(run["width_history"])),
                        "termination_reason": run["termination_reason"],
                        "selected": bool(eligible) and start_index == selected_index,
                    })
                key = (width if np.isfinite(width) else np.inf,
                       row["total_evaluations"], float(gamma0), float(beta))
                if winning_key is None or key < winning_key:
                    winning_key, winning_row, winning_results = key, row, runs
                if progress:
                    print(
                        f"Pair {len(rows)}/{total_candidates}: gamma0={gamma0:.6g}, "
                        f"beta={beta:g}, best width={width:.10f}", flush=True,
                    )
    return {
        "candidates": rows, "per_start": per_start, "histories": histories,
        "selected": dict(winning_row), "best_results": winning_results,
        "direction": np.array([winning_row[f"u_{j}"] for j in range(dimension)]),
    }


def tune_geometric_half_life_schedules(
    points,
    initial_direction,
    scale_grid,
    half_life_grid,
    *,
    iterations,
    patience,
    improvement_tolerance,
    optimizer_seed,
    progress=False,
    stationarity_check=False,
    max_restarts=100,
    step_rule="absolute",
    delta=1e-8,
):
    """Tune geometric steps from one fixed starting direction.

    beta = 2**(-1/H). The default absolute rule uses each scale as gamma0.
    Opt-in diameter_normalized uses gamma0 = scale/L, where L is the exact
    Euclidean diameter of this empirical cloud, computed once per search.
    The existing fixed-cloud tuner performs the optimization in both cases.
    """
    points = np.asarray(points, dtype=float)
    direction = np.asarray(initial_direction, dtype=float)
    if (points.ndim != 2 or len(points) < 2 or points.shape[1] < 1
            or not np.all(np.isfinite(points))):
        raise ValueError("points must be a finite (n, k) cloud with n >= 2.")
    if (direction.shape != (points.shape[1],)
            or not np.all(np.isfinite(direction))
            or np.linalg.norm(direction) <= np.finfo(float).eps):
        raise ValueError("initial_direction must be a finite nonzero k-vector.")
    grids = []
    for values, name in ((scale_grid, "scale_grid"), (half_life_grid, "half_life_grid")):
        values = np.asarray(values, dtype=float)
        if (values.ndim != 1 or not len(values)
                or not np.all(np.isfinite(values)) or np.any(values <= 0)
                or len(np.unique(values)) != len(values)):
            raise ValueError(f"{name} must contain distinct positive finite values.")
        grids.append(np.sort(values))
    scales, half_lives = grids
    betas = np.exp2(-1.0 / half_lives)
    if np.any((betas <= 0) | (betas >= 1)):
        raise ValueError("Half-lives must produce representable beta values in (0, 1).")
    if step_rule not in {"absolute", "diameter_normalized"}:
        raise ValueError("step_rule must be absolute or diameter_normalized.")
    diameter = float(pdist(points, metric="euclidean").max()) if step_rule == "diameter_normalized" else None
    if diameter is not None and (not np.isfinite(diameter) or diameter <= 0):
        raise ValueError("Diameter-normalized steps require a positive finite cloud diameter.")
    initial_steps = scales if diameter is None else scales / diameter
    search = tune_geometric_width_schedule(
        points, direction[None, :], initial_steps, betas,
        iterations=iterations, patience=patience,
        improvement_tolerance=improvement_tolerance, delta=delta,
        optimizer_seed=optimizer_seed, progress=progress,
        stationarity_check=stationarity_check, max_restarts=max_restarts,
    )
    effective = {(row["gamma0"], row["beta"]): row for row in search["candidates"]}
    start_details = {row["candidate_id"]: row for row in search["per_start"]}
    candidates, per_start, histories = [], [], {}
    for scale_index, scale in enumerate(scales):
        for h_index, (half_life, beta) in enumerate(zip(half_lives, betas)):
            original = effective[(float(initial_steps[scale_index]), float(beta))]
            effective_id = original["candidate_id"]
            candidate_id = f"{step_rule}_s{scale_index:02d}_h{h_index:02d}"
            row = dict(original, candidate_id=candidate_id, step_rule=step_rule,
                       step_value=float(scale), half_life=float(half_life))
            if diameter is not None:
                row["cloud_diameter"] = diameter
            candidates.append(row)
            histories[candidate_id] = search["histories"][effective_id]
            per_start.append(dict(start_details[effective_id],
                                  candidate_id=candidate_id, step_rule=step_rule,
                                  step_value=float(scale), half_life=float(half_life)))
    ordered = sorted(candidates, key=lambda row: (
        row["minimum_width"] if np.isfinite(row["minimum_width"]) else np.inf,
        row["total_evaluations"], row["gamma0"],
        row["beta"], row["step_rule"],
    ))
    selected = ordered[0]
    best_by_rule = {step_rule: selected}
    return {
        "candidates": candidates, "per_start": per_start, "histories": histories,
        "selected": dict(selected), "best_by_rule": best_by_rule,
        "direction": np.array([selected[f"u_{j}"] for j in range(points.shape[1])]),
        "initial_direction": direction / np.linalg.norm(direction),
        "effective_schedule_count": len(effective),
    }


def summarize_geometric_replications(results):
    """Rank shared settings by mean achieved minimum width across paired paths.

    Rows must contain the complete paired candidate-by-replication design.
    Every realized path contributes equally. Ties in mean width are broken by
    mean evaluations and parameter labels. No external optimum or accuracy
    target enters the score.
    """
    if not results:
        raise ValueError("Supply at least one replication result.")
    grouped = {}
    for row in results:
        grouped.setdefault(row["candidate_id"], []).append(row)
    expected_cases = {(row["replication"], row.get("T")) for row in results}
    summaries = []
    for candidate_id, rows in grouped.items():
        if (len(rows) != len(expected_cases)
                or {(row["replication"], row.get("T")) for row in rows} != expected_cases):
            raise ValueError("Each candidate must occur once on every replication/sample-size case.")
        first = rows[0]
        labels = ("step_rule", "step_value", "half_life", "beta")
        if any(any(row[key] != first[key] for key in labels) for row in rows):
            raise ValueError("Candidate labels must be identical across replications.")
        widths = np.array([row["minimum_width"] for row in rows], dtype=float)
        if np.any(~np.isfinite(widths)) or np.any(widths < 0):
            raise ValueError("Achieved minimum widths must be finite and nonnegative.")
        summary = {"candidate_id": candidate_id, **{key: first[key] for key in labels},
                   "replications": len({row["replication"] for row in rows}),
                   "cases": len(rows),
                   "mean_minimum_width": float(widths.mean()),
                   "median_minimum_width": float(np.median(widths)),
                   "worst_minimum_width": float(widths.max()),
                   "mean_evaluations": float(np.mean([row["total_evaluations"] for row in rows])),
                   "gamma0_min": float(min(row["gamma0"] for row in rows)),
                   "gamma0_max": float(max(row["gamma0"] for row in rows))}
        summaries.append(summary)
    summaries.sort(key=lambda row: (
        row["mean_minimum_width"], row["mean_evaluations"],
        row["step_rule"], row["step_value"], row["half_life"],
    ))
    return [dict(row, rank=index + 1) for index, row in enumerate(summaries)]


def summarize_configuration_wins(
    results, *, score_key="minimum_width", epsilon=1e-8, allow_nonfinite=False,
):
    """Give every qualifying configuration a full point on each paired case.

    Each (replication, T) case must contain every candidate exactly once.
    Award +1 to every score <= the smallest score + epsilon, where epsilon
    is a finite nonnegative absolute tolerance. The reference is the best
    achieved score among supplied candidates on that case.
    Points are never split between tied configurations or allocated by speed.
    Each configuration can win at most once per case, while total points can
    exceed the number of cases. Equal final counts use parameter labels for
    display order only; no average objective value enters the summary.
    With allow_nonfinite=True, undefined scores receive no credit and the
    reference uses only finite scores. An entirely undefined case awards no
    points but remains in the denominator. Nothing is exported.
    """
    if not results:
        raise ValueError("Supply at least one replication result.")
    if (isinstance(epsilon, (bool, np.bool_))
            or not isinstance(epsilon, (int, float, np.integer, np.floating))
            or not np.isfinite(epsilon) or epsilon < 0):
        raise ValueError("epsilon must be a finite nonnegative number.")
    epsilon = float(epsilon)
    candidates, cases = {}, {}
    labels = ("step_rule", "step_value", "half_life", "beta")
    for row in results:
        if not allow_nonfinite and not np.isfinite(row[score_key]):
            raise ValueError("All comparison scores must be finite.")
        candidate_id = row["candidate_id"]
        candidates.setdefault(candidate_id, []).append(row)
        cases.setdefault((row["replication"], row.get("T")), []).append(row)
    for rows in candidates.values():
        if any(any(row[key] != rows[0][key] for key in labels) for row in rows):
            raise ValueError("Candidate labels must be identical across replications.")
    expected_ids = set(candidates)
    counts = dict.fromkeys(candidates, 0)
    winners, case_summaries = [], []
    for case, rows in sorted(cases.items()):
        if len(rows) != len(candidates) or {row["candidate_id"] for row in rows} != expected_ids:
            raise ValueError("Every candidate must occur once in every paired case.")
        finite_rows = [row for row in rows if np.isfinite(row[score_key])]
        minimum_score = float(min(row[score_key] for row in finite_rows)) if finite_rows else np.nan
        winning_threshold = minimum_score + epsilon
        co_winners = [row for row in finite_rows if float(row[score_key]) <= winning_threshold]
        for row in sorted(co_winners, key=lambda row: row["candidate_id"]):
            counts[row["candidate_id"]] += 1
            winners.append(dict(row, winning_score=float(row[score_key]),
                                minimum_score=minimum_score, winning_threshold=winning_threshold,
                                absolute_gap=float(row[score_key]) - minimum_score,
                                co_winners=len(co_winners)))
        case_summaries.append({"replication": case[0], "T": case[1],
                               "minimum_score": minimum_score, "winning_threshold": winning_threshold,
                               "co_winners": len(co_winners),
                               "undefined_scores": len(rows) - len(finite_rows)})
    ranking = []
    for candidate_id, rows in candidates.items():
        ranking.append({
            "candidate_id": candidate_id, **{key: rows[0][key] for key in labels},
            "wins": counts[candidate_id], "win_rate": counts[candidate_id] / len(cases),
            "undefined_scores": sum(not np.isfinite(row[score_key]) for row in rows),
            "mean_evaluations": float(np.mean([row["total_evaluations"] for row in rows])),
        })
    ranking.sort(key=lambda row: (
        -row["wins"], row["step_rule"], row["step_value"],
        row["half_life"], row["candidate_id"],
    ))
    ranking = [dict(row, rank=index + 1) for index, row in enumerate(ranking)]
    total_points = sum(row["wins"] for row in ranking)
    assert total_points == len(winners) == sum(case["co_winners"] for case in case_summaries)
    assert all(0 <= row["wins"] <= len(cases) for row in ranking)
    return {"ranking": ranking, "selected": dict(ranking[0]), "winners": winners,
            "case_summaries": case_summaries, "cases": len(cases),
            "score_key": score_key, "total_points": total_points,
            "epsilon": epsilon,
            "undefined_cases": sum(not np.isfinite(case["minimum_score"]) for case in case_summaries),
            "tied_cases": sum(case["co_winners"] > 1 for case in case_summaries),
            "maximum_co_winners": max(case["co_winners"] for case in case_summaries)}


def tune_fixed_ar_realizations(
    phi,
    sample_size,
    initial_direction,
    scale_grid,
    half_life_grid,
    *,
    replications,
    process_seed,
    burnin,
    iterations,
    patience,
    improvement_tolerance,
    optimizer_seed,
    progress=False,
    summarize=True,
    stationarity_check=False,
    max_restarts=100,
    n_jobs=1,
    delta=1e-8,
    step_rule="absolute",
    coefficient_diagnostics=True,
):
    """Run paired grids on fixed-coefficient AR(3) paths at one size.

    Set summarize=False to return raw runs without aggregate width selection.
    Optional checkpoints/restarts use the same optimizer as Jones tuning.
    n_jobs controls independent realization workers; seeded paths and starts
    do not depend on the number of workers. Exhausted fits have NaN widths.
    step_rule supports absolute or diameter_normalized steps. Supply a
    (replications, 4) array to use one starting direction per realization.
    """
    return tune_ar_sample_sizes(
        phi, [sample_size], initial_direction, scale_grid, half_life_grid,
        replications=replications, process_seed=process_seed, burnin=burnin,
        iterations=iterations, patience=patience,
        improvement_tolerance=improvement_tolerance, delta=delta, optimizer_seed=optimizer_seed,
        progress=progress, summarize=summarize,
        stationarity_check=stationarity_check, max_restarts=max_restarts, n_jobs=n_jobs,
        step_rule=step_rule, coefficient_diagnostics=coefficient_diagnostics,
    )


def _ar_realization_cases(
    phi, sample_sizes, initial_direction, *, replications, process_seed, burnin,
    coefficient_seed=None,
):
    """Yield paired clouds with fixed phi, or Jones draws when phi is None."""
    random_coefficients = phi is None
    if not random_coefficients:
        phi = np.asarray(phi, dtype=float)
    directions = np.array(initial_direction, dtype=float, copy=True)
    if not random_coefficients and (
            phi.shape != (3,) or not np.all(np.isfinite(phi)) or phi[-1] == 0
            or not is_stationary(phi)[0]):
        raise ValueError("Use finite causal AR(3) coefficients with nonzero phi_3.")
    sizes = np.asarray(sample_sizes)
    if (sizes.ndim != 1 or not len(sizes) or sizes.dtype.kind not in "iu"
            or np.any(sizes < 5) or len(np.unique(sizes)) != len(sizes)):
        raise ValueError("sample_sizes must contain distinct integers >= p+2.")
    sizes = np.sort(sizes)
    for value, name, lower in ((replications, "replications", 1),
                               (process_seed, "process_seed", 0),
                               (burnin, "burnin", 0)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or value < lower):
            raise ValueError(f"{name} must be an integer >= {lower}.")
    if directions.shape not in {(4,), (replications, 4)}:
        raise ValueError("initial_direction must be a four-vector or a (replications, 4) array.")
    norms = np.linalg.norm(directions, axis=-1, keepdims=True)
    if (not np.all(np.isfinite(directions)) or not np.all(np.isfinite(norms))
            or np.any(norms <= np.finfo(float).eps)):
        raise ValueError("Every initial_direction must be finite and nonzero.")
    directions /= norms
    if random_coefficients:
        if (isinstance(coefficient_seed, (bool, np.bool_))
                or not isinstance(coefficient_seed, (int, np.integer)) or coefficient_seed < 0):
            raise ValueError("coefficient_seed must be a nonnegative integer.")
        coefficient_seeds = np.random.SeedSequence(coefficient_seed).spawn(replications)
    seeds = np.random.SeedSequence(process_seed).spawn(replications)
    for replication, seed in enumerate(seeds, start=1):
        direction = directions if directions.ndim == 1 else directions[replication - 1]
        realized_seed = int(seed.generate_state(1, dtype=np.uint64)[0])
        metadata = {}
        if random_coefficients:
            realized_coefficient_seed = int(coefficient_seeds[replication - 1].generate_state(
                1, dtype=np.uint64,
            )[0])
            realized_phi, pacf = jones_ar_coefficients(
                3, rng=realized_coefficient_seed, return_pacf=True,
            )
            stationary, eigenvalues = is_stationary(realized_phi)
            if not stationary or realized_phi[-1] == 0:
                raise RuntimeError("Jones draw is not a finite causal AR(3) model; no redraw was made.")
            metadata = {
                "coefficient_seed": realized_coefficient_seed,
                "spectral_radius": float(np.max(np.abs(eigenvalues))),
                **{f"phi_{j+1}": float(value) for j, value in enumerate(realized_phi)},
                **{f"pacf_{j+1}": float(value) for j, value in enumerate(pacf)},
            }
        else:
            realized_phi = phi
            metadata = {f"phi_{j+1}": float(value) for j, value in enumerate(realized_phi)}
        series = simulate_uniform_ar(realized_phi, sample_size=int(sizes[-1]), burnin=burnin,
                                     half_width=1.0, rng=realized_seed)
        true_direction = np.r_[1.0, -realized_phi]
        coefficient_normalizer = float(np.linalg.norm(true_direction))
        true_direction /= coefficient_normalizer
        for sample_size in sizes:
            points = sliding_window_point_cloud(series[:sample_size], 4)
            yield {"replication": replication, "T": int(sample_size),
                   "process_seed": realized_seed,
                   "points": points, "initial_direction": direction, **metadata,
                   **{f"initial_u_{j}": float(value) for j, value in enumerate(direction)},
                   "population_minimum_width": 2.0 / coefficient_normalizer,
                   "empirical_true_direction_width": float(np.ptp(points @ true_direction))}


def tune_ar_sample_sizes(
    phi, sample_sizes, initial_direction, scale_grid, half_life_grid, *,
    replications, process_seed, burnin, iterations, patience,
    improvement_tolerance, optimizer_seed, progress=False, summarize=True,
    stationarity_check=False, max_restarts=100, n_jobs=1,
    delta=1e-8,
    step_rule="absolute", coefficient_diagnostics=True,
):
    """Select by mean achieved minimum width across paired paths and sizes.

    One longest AR(3) path is simulated per replication; smaller sizes use its
    prefixes. All grids, starting directions and optimizer fallback seeds are
    paired within every cloud. Each initial step is used directly on every path.
    The theoretical population minimum and the empirical width at the true
    coefficient direction are reported only as diagnostics; they do not
    change optimization or ranking. No numerical global optimum is computed.
    With summarize=False, aggregate width ranking and selection are skipped.
    step_rule selects absolute or diameter-normalized steps on each cloud.
    """
    clouds = _ar_realization_cases(
        phi, sample_sizes, initial_direction, replications=replications,
        process_seed=process_seed, burnin=burnin,
    )
    return _tune_ar_cases(
        clouds, initial_direction, scale_grid, half_life_grid, replications=replications,
        iterations=iterations, patience=patience, improvement_tolerance=improvement_tolerance, delta=delta,
        optimizer_seed=optimizer_seed, progress=progress, summarize=summarize,
        stationarity_check=stationarity_check, max_restarts=max_restarts, n_jobs=n_jobs,
        step_rule=step_rule, coefficient_diagnostics=coefficient_diagnostics,
    )


def tune_jones_ar_realizations(
    sample_size, initial_direction, scale_grid, half_life_grid, *, replications,
    coefficient_seed, process_seed, burnin, iterations, patience,
    improvement_tolerance, optimizer_seed, progress=False, summarize=True,
    stationarity_check=False, max_restarts=100, step_rule="absolute", n_jobs=1,
    coefficient_diagnostics=True,
    delta=1e-8,
):
    """Tune paired grids on distinct Jones-sampled causal AR(3) models.

    Each model receives an independent coefficient seed and innovation seed.
    All candidates share its one realized cloud and starting direction.
    initial_direction accepts one common four-vector or a (replications, 4)
    array with one direction per model, indexed by replication - 1. Rows are
    normalized separately and returned as initial_u_0, ..., initial_u_3 in
    every result and case. Pass the same array to pair starts across step rules.
    The existing Jones sampler is used without truncation or rejection by
    spectral radius. Coefficients, partial autocorrelations and seeds are
    returned with every result and case. Nothing is exported. Set summarize=False
    to return raw runs without aggregate width ranking and selection.
    stationarity_check=True enables a checkpoint after patience evaluations
    of each attempt, with a fresh full budget on every restart. The initial
    direction, coefficient draws and simulated paths are unchanged.
    Exhausted candidates have NaN scores and explicit failure flags; use
    summarize=False and summarize_configuration_wins(allow_nonfinite=True)
    to retain them in the design without awarding them win credit.
    step_rule="diameter_normalized" interprets scale_grid as c in gamma0=c/L,
    with L the empirical cloud diameter. n_jobs parallelizes independent models.
    Set coefficient_diagnostics=False to skip fitted-coefficient recovery and
    MAE reporting; model-generation metadata and width references are retained.
    """
    if not isinstance(coefficient_diagnostics, (bool, np.bool_)):
        raise ValueError("coefficient_diagnostics must be a boolean.")
    clouds = _ar_realization_cases(
        None, [sample_size], initial_direction, replications=replications,
        process_seed=process_seed, burnin=burnin, coefficient_seed=coefficient_seed,
    )
    return _tune_ar_cases(
        clouds, initial_direction, scale_grid, half_life_grid, replications=replications,
        iterations=iterations, patience=patience, improvement_tolerance=improvement_tolerance, delta=delta,
        optimizer_seed=optimizer_seed, progress=progress, summarize=summarize,
        stationarity_check=stationarity_check, max_restarts=max_restarts,
        step_rule=step_rule, n_jobs=n_jobs, coefficient_diagnostics=coefficient_diagnostics,
    )


def _tune_ar_cases(
    clouds, initial_direction, scale_grid, half_life_grid, *, replications,
    iterations, patience, improvement_tolerance, optimizer_seed, progress, summarize,
    stationarity_check=False, max_restarts=100, n_jobs=1, step_rule="absolute",
    coefficient_diagnostics=True,
    delta=1e-8,
):
    """Run the same grid and aggregation for either AR coefficient design."""
    if (isinstance(n_jobs, (bool, np.bool_))
            or not isinstance(n_jobs, (int, np.integer)) or n_jobs < 1):
        raise ValueError("n_jobs must be a positive integer.")
    if n_jobs != 1:
        from joblib import Parallel, delayed

        completed = Parallel(n_jobs=int(n_jobs), return_as="generator")(
            delayed(_tune_ar_cases)(
                [case], initial_direction, scale_grid, half_life_grid,
                replications=replications, iterations=iterations, patience=patience,
                improvement_tolerance=improvement_tolerance, delta=delta, optimizer_seed=optimizer_seed,
                progress=False, summarize=False, stationarity_check=stationarity_check,
                max_restarts=max_restarts, n_jobs=1, step_rule=step_rule,
                coefficient_diagnostics=coefficient_diagnostics,
            ) for case in clouds
        )
        rows, cases = [], []
        for batch in completed:
            rows.extend(batch["results"])
            cases.extend(batch["cases"])
            if progress:
                case = batch["cases"][0]
                details = (f"{case['restart_count']} restarts; {case['failed_runs']} failed runs; "
                           if stationarity_check else "")
                print(f"Path {case['replication']}/{replications}, T={case['T']}: "
                      f"best achieved width={case['grid_best_width']:.10f}; "
                      f"{len(batch['results'])} {step_rule} schedules; {details}"
                      f"{case['runtime_seconds']:.1f}s", flush=True)
        return _finish_ar_tuning(rows, cases, initial_direction, summarize=summarize)
    rows, cases = [], []
    for case in clouds:
        started = perf_counter()
        metadata = {key: value for key, value in case.items()
                    if key not in {"points", "initial_direction"}}
        search = tune_geometric_half_life_schedules(
            case["points"], case["initial_direction"], scale_grid, half_life_grid, iterations=iterations,
            patience=patience, improvement_tolerance=improvement_tolerance, delta=delta,
            optimizer_seed=optimizer_seed,
            stationarity_check=stationarity_check, max_restarts=max_restarts,
            step_rule=step_rule,
        )
        for candidate in search["candidates"]:
            rows.append(dict(
                candidate, **metadata,
                population_width_gap=candidate["minimum_width"] - metadata["population_minimum_width"],
                empirical_true_direction_gap=(
                    candidate["minimum_width"] - metadata["empirical_true_direction_width"]
                ),
            ))
        winner = search["selected"]
        cases.append({**metadata,
                      "grid_best_width": winner["minimum_width"],
                      "grid_population_width_gap": winner["minimum_width"] - metadata["population_minimum_width"],
                      "grid_empirical_true_direction_gap": (
                          winner["minimum_width"] - metadata["empirical_true_direction_width"]
                      ),
                      "grid_winner_candidate_id": winner["candidate_id"],
                      "grid_winner_step_rule": winner["step_rule"],
                      "grid_winner_step_value": winner["step_value"],
                      "grid_winner_half_life": winner["half_life"],
                      "runtime_seconds": perf_counter() - started})
        if coefficient_diagnostics:
            winning_direction = np.array([winner[f"u_{j}"] for j in range(4)])
            estimated_phi = np.full(3, np.nan)
            if np.all(np.isfinite(winning_direction)) and abs(winning_direction[0]) > 1e-10:
                estimated_phi = -winning_direction[1:] / winning_direction[0]
            true_phi = np.array([metadata[f"phi_{j+1}"] for j in range(3)])
            coefficient_error = coefficient_mae(estimated_phi, true_phi)
            cases[-1].update(
                **{f"estimated_phi_{j+1}": float(value) for j, value in enumerate(estimated_phi)},
                coefficient_mae=coefficient_error,
            )
        if stationarity_check:
            cases[-1].update(
                restart_count=sum(row["restart_count"] for row in search["candidates"]),
                total_evaluations=sum(row["total_evaluations"] for row in search["candidates"]),
                failed_runs=sum(not row["stationarity_check_passed"] for row in search["candidates"]),
            )
        if progress:
            if "coefficient_seed" in metadata:
                diagnostics = (
                    f"population min={metadata['population_minimum_width']:.10f}, "
                    f"empirical-population gap={cases[-1]['grid_population_width_gap']:+.10f}, "
                )
            elif coefficient_diagnostics:
                estimates_text = np.array2string(estimated_phi, precision=6, separator=", ",
                                                suppress_small=False, max_line_width=200)
                error_text = f"{coefficient_error:.8g}" if np.isfinite(coefficient_error) else "undefined"
                diagnostics = f"estimated phi={estimates_text}, coefficient MAE={error_text}, "
            else:
                diagnostics = ""
            restart_text = (f"{cases[-1]['restart_count']} restarts; {cases[-1]['failed_runs']} failed runs; "
                            if stationarity_check else "")
            print(f"Path {case['replication']}/{replications}, T={case['T']}: "
                  f"best achieved width={winner['minimum_width']:.10f}, "
                  f"{diagnostics}"
                  f"{len(search['candidates'])} {step_rule} schedules; "
                  f"{restart_text}"
                  f"{cases[-1]['runtime_seconds']:.1f}s", flush=True)
    return _finish_ar_tuning(rows, cases, initial_direction, summarize=summarize)


def _finish_ar_tuning(rows, cases, initial_direction, *, summarize):
    """Assemble raw case results, optionally applying the existing summary."""
    directions = np.asarray(initial_direction, dtype=float)
    result = {"results": rows, "cases": cases,
              "initial_direction": directions / np.linalg.norm(directions, axis=-1, keepdims=True)}
    if summarize:
        ranking = summarize_geometric_replications(rows)
        result.update(ranking=ranking, selected=dict(ranking[0]),
                      best_by_rule={rule: next(row for row in ranking if row["step_rule"] == rule)
                                    for rule in {row["step_rule"] for row in ranking}})
    return result


def compare_ar_schedules(
    phi, sample_sizes, initial_direction, schedules, *, replications, process_seed,
    burnin, iterations, patience, improvement_tolerance, optimizer_seed, progress=False,
    record_histories=False,
    delta=1e-8,
):
    """Evaluate frozen schedules on fresh paired AR(3) paths, without selection.

    Each schedule supplies candidate_id, step_rule, step_value and half_life.
    Only absolute steps are supported, with step_value used directly as gamma0.
    Repeated schedules share one numerical run on each cloud.
    With record_histories=True, optimizer results are retained under
    (replication, T, candidate_id), ending at their actual stopping iterations.
    All results are returned in memory; nothing is saved or retuned.
    """
    if not schedules:
        raise ValueError("Supply at least one frozen schedule.")
    if len({row["candidate_id"] for row in schedules}) != len(schedules):
        raise ValueError("Frozen schedule identifiers must be distinct.")
    for row in schedules:
        if (row["step_rule"] != "absolute"
                or not np.isfinite(row["step_value"]) or row["step_value"] <= 0
                or not np.isfinite(row["half_life"]) or row["half_life"] <= 0):
            raise ValueError("Require absolute steps and positive finite initial step and half-life.")
    rows, cases, histories = [], [], {}
    for case in _ar_realization_cases(
        phi, sample_sizes, initial_direction, replications=replications,
        process_seed=process_seed, burnin=burnin,
    ):
        metadata = {key: case[key] for key in ("replication", "T", "process_seed")}
        evaluated = {}
        recorded = {}
        for schedule in schedules:
            beta = float(np.exp2(-1.0 / schedule["half_life"]))
            gamma0 = float(schedule["step_value"])
            key = (gamma0, beta)
            if key not in evaluated:
                search = tune_geometric_width_schedule(
                    case["points"], case["initial_direction"][None, :], [gamma0], [beta],
                    iterations=iterations, patience=patience,
                    improvement_tolerance=improvement_tolerance, delta=delta, optimizer_seed=optimizer_seed,
                )
                evaluated[key] = search["selected"]
                if record_histories:
                    recorded[key] = search["best_results"][0]
            rows.append(dict(evaluated[key], **metadata,
                             candidate_id=schedule["candidate_id"], step_rule=schedule["step_rule"],
                             step_value=float(schedule["step_value"]),
                             half_life=float(schedule["half_life"])))
            if record_histories:
                histories[(case["replication"], case["T"], schedule["candidate_id"])] = recorded[key]
        cases.append(dict(metadata))
        if progress:
            print(f"Fresh path {case['replication']}/{replications}, T={case['T']}: "
                  f"evaluated {len(schedules)} frozen schedules", flush=True)
    return {"results": rows, "cases": cases, "histories": histories}
