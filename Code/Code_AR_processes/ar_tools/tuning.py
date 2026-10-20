"""Paired geometric step-schedule comparisons on fixed clouds and AR paths."""

from time import perf_counter

import numpy as np
from scipy.spatial.distance import pdist
from threadpoolctl import threadpool_limits

from . import descent
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
):
    """Select gamma_m = gamma0 * beta**m by attained empirical width.

    Every candidate uses the supplied starting directions and the same
    fallback seed for each supplied start. Ties are broken
    by total evaluations, then gamma0 and beta. True coefficients are not
    inputs to this search. Histories retain the best width across starts at
    each iteration, holding each stopped run at its last recorded best width.
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
                        improvement_tolerance=improvement_tolerance,
                        initial_direction=direction,
                        rng=np.random.SeedSequence([
                            optimizer_seed, 0, dimension, 21, start_index,
                        ]),
                        step_schedule="geometric",
                    )
                    for start_index, direction in enumerate(directions)
                ]
                selected_index = int(np.argmin([run["minimum_width"] for run in runs]))
                selected = runs[selected_index]
                direction = selected["direction"].copy()
                if direction[0] < 0:
                    direction *= -1
                width = float(selected["minimum_width"])
                if (
                    not np.isfinite(width) or not np.all(np.isfinite(direction))
                    or not np.isclose(np.linalg.norm(direction), 1.0)
                    or not np.isclose(np.ptp(points @ direction), width)
                ):
                    raise RuntimeError(f"Invalid attained width for {candidate_id}.")
                row = {
                    "candidate_id": candidate_id,
                    "gamma0": float(gamma0), "beta": float(beta),
                    "minimum_width": width, "selected_start": selected_index + 1,
                    "total_evaluations": sum(len(run["width_history"]) for run in runs),
                    "patience_stops": sum(run["termination_reason"] == "patience" for run in runs),
                    "runtime_seconds": perf_counter() - started,
                }
                row.update({f"u_{j}": float(value) for j, value in enumerate(direction)})
                rows.append(row)
                longest = max(len(run["best_width_history"]) for run in runs)
                histories[candidate_id] = np.minimum.reduce([
                    np.pad(
                        run["best_width_history"],
                        (0, longest - len(run["best_width_history"])), mode="edge",
                    )
                    for run in runs
                ])
                for start_index, run in enumerate(runs):
                    per_start.append({
                        "candidate_id": candidate_id,
                        "gamma0": float(gamma0), "beta": float(beta),
                        "start": start_index + 1,
                        "minimum_width": float(run["minimum_width"]),
                        "evaluations": len(run["width_history"]),
                        "termination_reason": run["termination_reason"],
                        "selected": start_index == selected_index,
                    })
                key = (width, row["total_evaluations"], float(gamma0), float(beta))
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
):
    """Compare absolute and diameter-normalized steps from one paired start.

    The thesis parameterization is beta = 2**(-1/H). For each scale s, the
    absolute rule uses gamma0=s and the normalized rule uses gamma0=s/L,
    with L the actual Euclidean diameter of the supplied cloud. The existing
    fixed-cloud tuner performs the optimization; shared effective schedules
    are evaluated only once and then receive both rule labels if needed.
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
    diameter = float(pdist(points).max())
    if not np.isfinite(diameter) or diameter <= 0:
        raise ValueError("The cloud must have positive finite Euclidean diameter.")
    actual_steps = np.unique(np.r_[scales, scales / diameter])
    search = tune_geometric_width_schedule(
        points, direction[None, :], actual_steps, betas,
        iterations=iterations, patience=patience,
        improvement_tolerance=improvement_tolerance,
        optimizer_seed=optimizer_seed, progress=progress,
    )
    effective = {(row["gamma0"], row["beta"]): row for row in search["candidates"]}
    start_details = {row["candidate_id"]: row for row in search["per_start"]}
    candidates, per_start, histories = [], [], {}
    for rule in ("absolute", "diameter_normalized"):
        for scale_index, scale in enumerate(scales):
            gamma0 = float(scale if rule == "absolute" else scale / diameter)
            for h_index, (half_life, beta) in enumerate(zip(half_lives, betas)):
                original = effective[(gamma0, float(beta))]
                effective_id = original["candidate_id"]
                candidate_id = f"{rule}_s{scale_index:02d}_h{h_index:02d}"
                row = dict(original, candidate_id=candidate_id, step_rule=rule,
                           step_value=float(scale), half_life=float(half_life),
                           lipschitz_constant=diameter)
                candidates.append(row)
                histories[candidate_id] = search["histories"][effective_id]
                per_start.append(dict(start_details[effective_id],
                                      candidate_id=candidate_id, step_rule=rule,
                                      step_value=float(scale), half_life=float(half_life)))
    # Use the same numerical ordering as the underlying tuner. Rule labels
    # resolve the otherwise identical candidates when L=1.
    ordered = sorted(candidates, key=lambda row: (
        row["minimum_width"], row["total_evaluations"], row["gamma0"],
        row["beta"], row["step_rule"],
    ))
    selected = ordered[0]
    best_by_rule = {
        rule: next(row for row in ordered if row["step_rule"] == rule)
        for rule in ("absolute", "diameter_normalized")
    }
    return {
        "candidates": candidates, "per_start": per_start, "histories": histories,
        "selected": dict(selected), "best_by_rule": best_by_rule,
        "direction": np.array([selected[f"u_{j}"] for j in range(points.shape[1])]),
        "initial_direction": direction / np.linalg.norm(direction),
        "diameter": diameter, "effective_schedule_count": len(effective),
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


def summarize_configuration_wins(results, *, score_key="minimum_width", epsilon=1e-8):
    """Give every qualifying configuration a full point on each paired case.

    Each (replication, T) case must contain every candidate exactly once.
    Award +1 to every score <= the smallest score + epsilon, where epsilon
    is a finite nonnegative absolute tolerance. The reference is the best
    achieved score among supplied candidates on that case.
    Points are never split between tied configurations or allocated by speed.
    Each configuration can win at most once per case, while total points can
    exceed the number of cases. Equal final counts use the full-precision
    mean score, then mean evaluations and labels for display order only.
    Nothing is exported.
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
        if not np.isfinite(row[score_key]):
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
        minimum_score = float(min(row[score_key] for row in rows))
        winning_threshold = minimum_score + epsilon
        co_winners = [row for row in rows if float(row[score_key]) <= winning_threshold]
        for row in sorted(co_winners, key=lambda row: row["candidate_id"]):
            counts[row["candidate_id"]] += 1
            winners.append(dict(row, winning_score=float(row[score_key]),
                                minimum_score=minimum_score, winning_threshold=winning_threshold,
                                absolute_gap=float(row[score_key]) - minimum_score,
                                co_winners=len(co_winners)))
        case_summaries.append({"replication": case[0], "T": case[1],
                               "minimum_score": minimum_score, "winning_threshold": winning_threshold,
                               "co_winners": len(co_winners)})
    ranking = []
    for candidate_id, rows in candidates.items():
        ranking.append({
            "candidate_id": candidate_id, **{key: rows[0][key] for key in labels},
            "wins": counts[candidate_id], "win_rate": counts[candidate_id] / len(cases),
            "mean_score": float(np.mean([row[score_key] for row in rows])),
            "mean_evaluations": float(np.mean([row["total_evaluations"] for row in rows])),
        })
    ranking.sort(key=lambda row: (
        -row["wins"], row["mean_score"], row["mean_evaluations"],
        row["step_rule"], row["step_value"], row["half_life"], row["candidate_id"],
    ))
    ranking = [dict(row, rank=index + 1) for index, row in enumerate(ranking)]
    total_points = sum(row["wins"] for row in ranking)
    assert total_points == len(winners) == sum(case["co_winners"] for case in case_summaries)
    assert all(0 <= row["wins"] <= len(cases) for row in ranking)
    return {"ranking": ranking, "selected": dict(ranking[0]), "winners": winners,
            "case_summaries": case_summaries, "cases": len(cases),
            "score_key": score_key, "total_points": total_points,
            "epsilon": epsilon,
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
):
    """Tune one shared schedule on fixed-coefficient AR(3) paths at one size."""
    return tune_ar_sample_sizes(
        phi, [sample_size], initial_direction, scale_grid, half_life_grid,
        replications=replications, process_seed=process_seed, burnin=burnin,
        iterations=iterations, patience=patience,
        improvement_tolerance=improvement_tolerance, optimizer_seed=optimizer_seed,
        progress=progress,
    )


def _ar_realization_cases(
    phi, sample_sizes, initial_direction, *, replications, process_seed, burnin,
    coefficient_seed=None,
):
    """Yield paired clouds with fixed phi, or Jones draws when phi is None."""
    random_coefficients = phi is None
    if not random_coefficients:
        phi = np.asarray(phi, dtype=float)
    direction = np.array(initial_direction, dtype=float, copy=True)
    if not random_coefficients and (
            phi.shape != (3,) or not np.all(np.isfinite(phi)) or phi[-1] == 0
            or not is_stationary(phi)[0]):
        raise ValueError("Use finite causal AR(3) coefficients with nonzero phi_3.")
    if (direction.shape != (4,) or not np.all(np.isfinite(direction))
            or np.linalg.norm(direction) <= np.finfo(float).eps):
        raise ValueError("initial_direction must be a finite nonzero four-vector.")
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
    if random_coefficients:
        if (isinstance(coefficient_seed, (bool, np.bool_))
                or not isinstance(coefficient_seed, (int, np.integer)) or coefficient_seed < 0):
            raise ValueError("coefficient_seed must be a nonnegative integer.")
        coefficient_seeds = np.random.SeedSequence(coefficient_seed).spawn(replications)
    direction /= np.linalg.norm(direction)
    seeds = np.random.SeedSequence(process_seed).spawn(replications)
    for replication, seed in enumerate(seeds, start=1):
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
                   "population_minimum_width": 2.0 / coefficient_normalizer,
                   "empirical_true_direction_width": float(np.ptp(points @ true_direction))}


def tune_ar_sample_sizes(
    phi, sample_sizes, initial_direction, scale_grid, half_life_grid, *,
    replications, process_seed, burnin, iterations, patience,
    improvement_tolerance, optimizer_seed, progress=False,
):
    """Select by mean achieved minimum width across paired paths and sizes.

    One longest AR(3) path is simulated per replication; smaller sizes use its
    prefixes. All grids, starting directions and optimizer fallback seeds are
    paired within every cloud. Normalization uses each cloud's actual diameter.
    The theoretical population minimum and the empirical width at the true
    coefficient direction are reported only as diagnostics; they do not
    change optimization or ranking. No numerical global optimum is computed.
    """
    clouds = _ar_realization_cases(
        phi, sample_sizes, initial_direction, replications=replications,
        process_seed=process_seed, burnin=burnin,
    )
    return _tune_ar_cases(
        clouds, initial_direction, scale_grid, half_life_grid, replications=replications,
        iterations=iterations, patience=patience, improvement_tolerance=improvement_tolerance,
        optimizer_seed=optimizer_seed, progress=progress,
    )


def tune_jones_ar_realizations(
    sample_size, initial_direction, scale_grid, half_life_grid, *, replications,
    coefficient_seed, process_seed, burnin, iterations, patience,
    improvement_tolerance, optimizer_seed, progress=False,
):
    """Tune paired grids on distinct Jones-sampled causal AR(3) models.

    Each model receives an independent coefficient seed and innovation seed.
    All candidates share its one realized cloud and the supplied fixed start.
    The existing Jones sampler is used without truncation or rejection by
    spectral radius. Coefficients, partial autocorrelations and seeds are
    returned with every result and case. Nothing is exported.
    """
    clouds = _ar_realization_cases(
        None, [sample_size], initial_direction, replications=replications,
        process_seed=process_seed, burnin=burnin, coefficient_seed=coefficient_seed,
    )
    return _tune_ar_cases(
        clouds, initial_direction, scale_grid, half_life_grid, replications=replications,
        iterations=iterations, patience=patience, improvement_tolerance=improvement_tolerance,
        optimizer_seed=optimizer_seed, progress=progress,
    )


def _tune_ar_cases(
    clouds, initial_direction, scale_grid, half_life_grid, *, replications,
    iterations, patience, improvement_tolerance, optimizer_seed, progress,
):
    """Run the same grid and aggregation for either AR coefficient design."""
    rows, cases = [], []
    for case in clouds:
        started = perf_counter()
        metadata = {key: value for key, value in case.items()
                    if key not in {"points", "initial_direction"}}
        search = tune_geometric_half_life_schedules(
            case["points"], case["initial_direction"], scale_grid, half_life_grid, iterations=iterations,
            patience=patience, improvement_tolerance=improvement_tolerance,
            optimizer_seed=optimizer_seed,
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
                      "diameter": search["diameter"],
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
        if progress:
            print(f"Path {case['replication']}/{replications}, T={case['T']}: "
                  f"best achieved width={winner['minimum_width']:.10f}, "
                  f"population min={metadata['population_minimum_width']:.10f}, "
                  f"empirical-population gap={cases[-1]['grid_population_width_gap']:+.10f}, "
                  f"L={search['diameter']:.6g}, {len(search['candidates'])} schedules; "
                  f"{cases[-1]['runtime_seconds']:.1f}s", flush=True)
    ranking = summarize_geometric_replications(rows)
    return {"results": rows, "cases": cases, "ranking": ranking,
            "selected": dict(ranking[0]),
            "best_by_rule": {rule: next(row for row in ranking if row["step_rule"] == rule)
                             for rule in ("absolute", "diameter_normalized")},
            "initial_direction": np.asarray(initial_direction) / np.linalg.norm(initial_direction)}


def compare_ar_schedules(
    phi, sample_sizes, initial_direction, schedules, *, replications, process_seed,
    burnin, iterations, patience, improvement_tolerance, optimizer_seed, progress=False,
    record_histories=False,
):
    """Evaluate frozen schedules on fresh paired AR(3) paths, without selection.

    Each schedule supplies candidate_id, step_rule, step_value and half_life.
    The two allowed rules and stopping settings match the tuning helpers.
    Repeated effective schedules share one numerical run on each cloud.
    With record_histories=True, optimizer results are retained under
    (replication, T, candidate_id), ending at their actual stopping iterations.
    All results are returned in memory; nothing is saved or retuned.
    """
    if not schedules:
        raise ValueError("Supply at least one frozen schedule.")
    if len({row["candidate_id"] for row in schedules}) != len(schedules):
        raise ValueError("Frozen schedule identifiers must be distinct.")
    for row in schedules:
        if (row["step_rule"] not in {"absolute", "diameter_normalized"}
                or not np.isfinite(row["step_value"]) or row["step_value"] <= 0
                or not np.isfinite(row["half_life"]) or row["half_life"] <= 0):
            raise ValueError("Require a known step rule and positive finite scale and half-life.")
    rows, cases, histories = [], [], {}
    for case in _ar_realization_cases(
        phi, sample_sizes, initial_direction, replications=replications,
        process_seed=process_seed, burnin=burnin,
    ):
        metadata = {key: case[key] for key in ("replication", "T", "process_seed")}
        diameter = float(pdist(case["points"]).max())
        if not np.isfinite(diameter) or diameter <= 0:
            raise ValueError("The cloud must have positive finite diameter.")
        evaluated = {}
        recorded = {}
        for schedule in schedules:
            beta = float(np.exp2(-1.0 / schedule["half_life"]))
            gamma0 = float(schedule["step_value"])
            if schedule["step_rule"] == "diameter_normalized":
                gamma0 /= diameter
            key = (gamma0, beta)
            if key not in evaluated:
                search = tune_geometric_width_schedule(
                    case["points"], case["initial_direction"][None, :], [gamma0], [beta],
                    iterations=iterations, patience=patience,
                    improvement_tolerance=improvement_tolerance, optimizer_seed=optimizer_seed,
                )
                evaluated[key] = search["selected"]
                if record_histories:
                    recorded[key] = search["best_results"][0]
            rows.append(dict(evaluated[key], **metadata,
                             candidate_id=schedule["candidate_id"], step_rule=schedule["step_rule"],
                             step_value=float(schedule["step_value"]),
                             half_life=float(schedule["half_life"]), lipschitz_constant=diameter))
            if record_histories:
                histories[(case["replication"], case["T"], schedule["candidate_id"])] = recorded[key]
        cases.append(dict(metadata, diameter=diameter))
        if progress:
            print(f"Fresh path {case['replication']}/{replications}, T={case['T']}: "
                  f"evaluated {len(schedules)} frozen schedules", flush=True)
    return {"results": rows, "cases": cases, "histories": histories}
