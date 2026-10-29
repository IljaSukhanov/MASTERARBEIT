"""Reusable functions for spectral_scale_selection_kde. Experiment settings stay in the notebook."""

from __future__ import annotations
from kde_tools.bandwidth_histograms import grouped_grid_counts
from kde_tools.fourier.reconstruction import kde_family_fft
from kde_tools.spectral_sweeps import run_l2_point
import json
import matplotlib.pyplot as plt
import numpy as np


def run_beta2_n_sweep_point(
    current_n,
    *,
    beta2_n_sweep_M,
    beta2_n_sweep_N,
    beta2_n_sweep_cache_root,
    beta2_n_sweep_density_half_width,
    beta2_n_sweep_density_spatial_points,
    beta2_n_sweep_frequencies,
    beta2_n_sweep_full_samples,
    beta2_n_sweep_h_grid,
    beta2_n_sweep_import_directories,
    beta2_n_sweep_lambda_values,
    beta2_n_sweep_oracle_bandwidths,
    beta2_n_sweep_oracle_indices,
    beta2_n_sweep_output_directory,
    beta2_n_sweep_output_root,
    beta2_n_sweep_parseval_weights,
    beta2_n_sweep_reuse_checkpoints,
    beta2_n_sweep_settings,
    beta2_n_sweep_true_density,
):
    current_n = int(current_n)
    spectral_result = run_l2_point(
        current_n,
        beta2_n_sweep_full_samples,
        beta2_n_sweep_h_grid,
        beta2_n_sweep_settings,
        beta2_n_sweep_lambda_values,
        beta2_n_sweep_output_directory,
        reuse=beta2_n_sweep_reuse_checkpoints,
        cache_root=beta2_n_sweep_cache_root,
        import_directories=beta2_n_sweep_import_directories,
        diagnostics=True,
    )
    selected_indices = spectral_result["selected_indices"][:, 0]
    localized_selected_indices = spectral_result["selected_indices"][:, 1]
    ise_selected_indices = np.full(beta2_n_sweep_M, -1, dtype=int)
    qhat_selected_indices = np.full(beta2_n_sweep_M, -1, dtype=int)
    controls_path = spectral_result["point_directory"] / "grid_controls.npz"
    controls_identity = json.dumps(
        {
            "samples_sha256": spectral_result["samples_sha256"],
            "grid": beta2_n_sweep_h_grid.tolist(),
            "half_width": beta2_n_sweep_density_half_width,
            "spatial_points": beta2_n_sweep_density_spatial_points,
            "method": "periodic_reconstructed_ise_log_derivative_q2_v1",
        },
        sort_keys=True,
    )
    controls_completed = 0
    controls_source = controls_path
    if beta2_n_sweep_reuse_checkpoints and (not controls_path.is_file()):
        for candidate in sorted(
            beta2_n_sweep_output_root.glob(f"*/n{current_n}/grid_controls.npz")
        ):
            with np.load(candidate, allow_pickle=False) as saved:
                if (
                    str(saved["identity_json"].item()) == controls_identity
                    and int(saved["completed"].item()) == beta2_n_sweep_M
                ):
                    controls_source = candidate
                    break
    if beta2_n_sweep_reuse_checkpoints and controls_source.is_file():
        with np.load(controls_source, allow_pickle=False) as saved:
            if str(saved["identity_json"].item()) != controls_identity:
                raise ValueError("Saved grid controls differ from the requested inputs.")
            ise_selected_indices = saved["ise_selected_indices"].copy()
            qhat_selected_indices = saved["qhat_selected_indices"].copy()
            controls_completed = int(saved["completed"].item())
        if (
            ise_selected_indices.shape != (beta2_n_sweep_M,)
            or qhat_selected_indices.shape != (beta2_n_sweep_M,)
            or (not 0 <= controls_completed <= beta2_n_sweep_M)
        ):
            raise ValueError("Invalid control checkpoint dimensions or completion count.")
        for values in (ise_selected_indices, qhat_selected_indices):
            if (
                not np.issubdtype(values.dtype, np.integer)
                or np.any(values[:controls_completed] < 0)
                or np.any(values[:controls_completed] >= beta2_n_sweep_N)
            ):
                raise ValueError("Invalid saved control selection indices.")
        print(
            f"n={current_n}: reused {controls_completed} ISE/Q-hat control pairs from {controls_source}",
            flush=True,
        )
    controls_reused_count = controls_completed

    def save_controls():
        temporary = controls_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                identity_json=controls_identity,
                ise_selected_indices=ise_selected_indices,
                qhat_selected_indices=qhat_selected_indices,
                completed=controls_completed,
            )
        temporary.replace(controls_path)

    try:
        for repetition in range(controls_completed, beta2_n_sweep_M):
            current_samples = beta2_n_sweep_full_samples[repetition, :current_n]
            current_densities, current_dx = kde_family_fft(
                samples=current_samples,
                bandwidths=beta2_n_sweep_h_grid,
                half_width=beta2_n_sweep_density_half_width,
                spatial_points=beta2_n_sweep_density_spatial_points,
            )
            current_ise_values = current_dx * np.sum(
                (current_densities - beta2_n_sweep_true_density[None, :]) ** 2, axis=1
            )
            ise_selected_indices[repetition] = int(np.argmin(current_ise_values))
            current_density_coefficients = np.fft.rfft(current_densities, axis=1)
            current_log_derivative_coefficients = (
                -beta2_n_sweep_h_grid[:, None] ** 2
                * beta2_n_sweep_frequencies[None, :] ** 2
                * current_density_coefficients
            )
            current_qhat_values = (
                0.5
                * current_dx
                / beta2_n_sweep_density_spatial_points
                * np.sum(
                    beta2_n_sweep_parseval_weights[None, :]
                    * np.abs(current_log_derivative_coefficients) ** 2,
                    axis=1,
                )
            )
            qhat_selected_indices[repetition] = int(np.argmin(current_qhat_values))
            controls_completed = repetition + 1
            if controls_completed % 25 == 0 or controls_completed == beta2_n_sweep_M:
                save_controls()
            if controls_completed % 100 == 0:
                print(
                    f"n={current_n}: grid ISE/Q-hat controls {controls_completed}/{beta2_n_sweep_M}",
                    flush=True,
                )
    finally:
        save_controls()
    result = {
        "selected_indices": selected_indices.copy(),
        "counts": np.bincount(selected_indices, minlength=beta2_n_sweep_N),
        "localized_selected_indices": localized_selected_indices.copy(),
        "localized_counts": np.bincount(localized_selected_indices, minlength=beta2_n_sweep_N),
        "controls_reused_from": str(controls_source) if controls_reused_count else None,
        "ise_selected_indices": ise_selected_indices.copy(),
        "ise_counts": np.bincount(ise_selected_indices, minlength=beta2_n_sweep_N),
        "qhat_selected_indices": qhat_selected_indices.copy(),
        "qhat_counts": np.bincount(qhat_selected_indices, minlength=beta2_n_sweep_N),
        "oracle_bandwidth": beta2_n_sweep_oracle_bandwidths[current_n],
        "oracle_index": beta2_n_sweep_oracle_indices[current_n],
    }
    result["relative_frequencies"] = result["counts"] / beta2_n_sweep_M
    assert all(
        (
            result[key].sum() == beta2_n_sweep_M
            for key in ("counts", "ise_counts", "qhat_counts", "localized_counts")
        )
    )
    print(f"completed n={current_n}", flush=True)
    return (current_n, spectral_result, result)


def plot_spectral_sample_size_overlay(
    *,
    count_key,
    plot_title,
    lambda_text,
    output_filename,
    csv_filename,
    tick_step,
    beta2_n_sweep_M,
    beta2_n_sweep_N,
    beta2_n_sweep_binning,
    beta2_n_sweep_h_grid,
    beta2_n_sweep_output_directory,
    beta2_n_sweep_results,
    project_root,
):
    sample_overlay_sizes = [300, 400, 500, 1000]
    if all((n in beta2_n_sweep_results for n in sample_overlay_sizes)):
        sample_overlay_counts = np.stack(
            [beta2_n_sweep_results[n][count_key] for n in sample_overlay_sizes]
        )
        overlay_starts, overlay_stops, overlay_grouped = grouped_grid_counts(
            sample_overlay_counts, beta2_n_sweep_N, beta2_n_sweep_M, beta2_n_sweep_binning
        )
        sample_overlay_figure, sample_overlay_axis = plt.subplots(figsize=(11.4, 6.6))
        sample_overlay_figure.subplots_adjust(left=0.085, right=0.98, top=0.84, bottom=0.21)
        overlay_centers = (overlay_starts + overlay_stops - 1) / 2
        overlay_colors = ("#74BFE5", "#318FCA", "#1556A3", "#061C3F")
        for n, color, values in zip(sample_overlay_sizes, overlay_colors, overlay_grouped):
            occupied = np.flatnonzero(values)
            sample_overlay_axis.bar(
                overlay_centers[occupied],
                values[occupied] / beta2_n_sweep_M,
                width=0.94 * (overlay_stops - overlay_starts)[occupied],
                color=color,
                alpha=0.3,
                edgecolor="white",
                linewidth=0.3,
                label="_nolegend_",
            )
        from matplotlib import patheffects as overlay_path_effects

        for n, color, values in zip(sample_overlay_sizes, overlay_colors, overlay_grouped):
            occupied = np.flatnonzero(values)
            first = max(0, occupied[0] - 1)
            last = min(len(values), occupied[-1] + 2)
            sample_overlay_axis.plot(
                overlay_centers[first:last],
                values[first:last] / beta2_n_sweep_M,
                color=color,
                linestyle="-",
                linewidth=1.9,
                label=f"$n={n}$",
                zorder=4,
                path_effects=[
                    overlay_path_effects.Stroke(linewidth=3.1, foreground="white"),
                    overlay_path_effects.Normal(),
                ],
            )
        occupied_bins = np.flatnonzero(overlay_grouped.sum(axis=0))
        overlay_left = max(-0.5, overlay_starts[occupied_bins[0]] - 2 * beta2_n_sweep_binning - 0.5)
        overlay_right = min(
            beta2_n_sweep_N - 0.5,
            overlay_stops[occupied_bins[-1]] + 2 * beta2_n_sweep_binning - 0.5,
        )
        overlay_h_ticks = np.array([0.05, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6])
        overlay_tick_positions = np.interp(
            np.log(overlay_h_ticks), np.log(beta2_n_sweep_h_grid), np.arange(beta2_n_sweep_N)
        )
        visible = (overlay_tick_positions >= overlay_left) & (
            overlay_tick_positions <= overlay_right
        )
        sample_overlay_axis.set_xticks(
            overlay_tick_positions[visible], [f"{h:g}" for h in overlay_h_ticks[visible]]
        )
        sample_overlay_axis.set_xlim(overlay_left, overlay_right)
        sample_overlay_axis.set_ylim(0, 1.1 * overlay_grouped.max() / beta2_n_sweep_M)
        from matplotlib.ticker import MultipleLocator, PercentFormatter

        sample_overlay_axis.yaxis.set_major_locator(MultipleLocator(tick_step))
        sample_overlay_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        sample_overlay_axis.set_xlabel(
            "selected bandwidth h", fontsize=12, labelpad=12, color="#304967"
        )
        sample_overlay_axis.set_ylabel(
            "relative frequency", fontsize=12, labelpad=12, color="#304967"
        )
        sample_overlay_axis.tick_params(labelsize=10, colors="#304967", length=0)
        sample_overlay_axis.grid(axis="y", color="#E5ECF6", linewidth=0.6)
        sample_overlay_axis.set_axisbelow(True)
        for side in ("top", "right", "left"):
            sample_overlay_axis.spines[side].set_visible(False)
        sample_overlay_axis.spines["bottom"].set_color("#E5ECF6")
        sample_overlay_axis.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=4,
            frameon=False,
            fontsize=11,
            labelcolor="#304967",
        )
        sample_overlay_figure.suptitle(
            plot_title
            + "\n"
            + f"$N={beta2_n_sweep_N}$, $M={beta2_n_sweep_M}$, ${lambda_text}$; "
            + f"{beta2_n_sweep_binning} grid positions per bin",
            y=0.965,
            fontsize=15,
            color="#304967",
        )
        sample_overlay_pdf = project_root / "output/pdf" / output_filename
        sample_overlay_pdf.parent.mkdir(parents=True, exist_ok=True)
        sample_overlay_figure.savefig(sample_overlay_pdf, format="pdf", bbox_inches="tight")
        np.savetxt(
            beta2_n_sweep_output_directory / csv_filename,
            np.column_stack((overlay_starts, overlay_stops - 1, overlay_grouped.T)),
            delimiter=",",
            fmt="%d",
            header="first_grid_index,last_grid_index,n300,n400,n500,n1000",
            comments="",
        )
        plt.show()
        plt.close(sample_overlay_figure)
        return {
            "figure": sample_overlay_figure,
            "axis": sample_overlay_axis,
            "counts": overlay_grouped,
            "pdf": sample_overlay_pdf,
        }
