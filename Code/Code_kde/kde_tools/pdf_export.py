"""Automatic PDF export for figures displayed by the KDE notebooks."""

from __future__ import annotations

import itertools
import re
import weakref
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure as MatplotlibFigure

try:
    from plotly.basedatatypes import BaseFigure as PlotlyBaseFigure
    from plotly.io import kaleido as plotly_kaleido
except ImportError:  # pragma: no cover - Plotly is optional to the helper itself.
    PlotlyBaseFigure = None
    plotly_kaleido = None


_ORIGINAL_MATPLOTLIB_SHOW = plt.show
_ORIGINAL_PLOTLY_SHOW = (
    PlotlyBaseFigure.show if PlotlyBaseFigure is not None else None
)
_ACTIVE_EXPORTER: _AutomaticPdfExporter | None = None
_IPYTHON_EVENTS = None


def _safe_name(value: str, fallback: str = "figure") -> str:
    """Return a short filesystem-safe name."""
    without_markup = re.sub(r"<[^>]+>", " ", str(value))
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", without_markup)
    normalized = normalized.strip("_").lower()
    return normalized[:80] or fallback


class _AutomaticPdfExporter:
    def __init__(self, output_directory: Path) -> None:
        self.output_directory = output_directory
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self._counter = itertools.count(1)
        self._saved_matplotlib_figures: weakref.WeakSet[MatplotlibFigure] = (
            weakref.WeakSet()
        )

    def _next_path(self, title: str, fallback: str) -> Path:
        number = next(self._counter)
        return self.output_directory / (
            f"{number:03d}_{_safe_name(title, fallback)}.pdf"
        )

    def save_matplotlib_figure(self, figure: MatplotlibFigure) -> Path:
        title = ""
        if figure._suptitle is not None:
            title = figure._suptitle.get_text()
        if not title:
            title = next(
                (
                    axis.get_title()
                    for axis in figure.axes
                    if axis.get_title()
                ),
                "",
            )
        path = self._next_path(title, "matplotlib_figure")
        figure.savefig(path, format="pdf", bbox_inches="tight")
        self._saved_matplotlib_figures.add(figure)
        print(f"Saved PDF: {path}")
        return path

    def save_new_matplotlib_figures(self) -> list[Path]:
        paths = []
        for number in plt.get_fignums():
            figure = plt.figure(number)
            if figure not in self._saved_matplotlib_figures:
                paths.append(self.save_matplotlib_figure(figure))
        return paths

    def save_plotly_figure(self, figure: Any) -> Path:
        title = ""
        layout_title = getattr(figure.layout, "title", None)
        if layout_title is not None:
            title = layout_title.text or ""
        path = self._next_path(title, "plotly_figure")
        if plotly_kaleido is not None:
            plotly_kaleido.scope.mathjax = None
        figure.write_image(str(path), format="pdf")
        print(f"Saved PDF: {path}")
        return path


def _show_matplotlib_and_save(*args: Any, **kwargs: Any) -> Any:
    if _ACTIVE_EXPORTER is not None:
        _ACTIVE_EXPORTER.save_new_matplotlib_figures()
    return _ORIGINAL_MATPLOTLIB_SHOW(*args, **kwargs)


def _save_after_notebook_cell(*args: Any, **kwargs: Any) -> None:
    """Include figures displayed implicitly by the notebook's inline backend."""
    if _ACTIVE_EXPORTER is not None:
        _ACTIVE_EXPORTER.save_new_matplotlib_figures()


def _show_plotly_and_save(
    figure: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    if _ACTIVE_EXPORTER is not None:
        _ACTIVE_EXPORTER.save_plotly_figure(figure)
    if _ORIGINAL_PLOTLY_SHOW is None:  # pragma: no cover
        raise RuntimeError("Plotly is not available.")
    return _ORIGINAL_PLOTLY_SHOW(figure, *args, **kwargs)


def enable_automatic_pdf_saving(
    notebook_name: str,
    *,
    output_root: str | Path | None = None,
) -> Path:
    """Save every subsequently displayed Matplotlib or Plotly figure as PDF."""
    global _ACTIVE_EXPORTER, _IPYTHON_EVENTS

    if output_root is None:
        output_root = Path(__file__).resolve().parents[1] / "outputs" / "pdfs"
    output_directory = Path(output_root) / _safe_name(
        notebook_name,
        "kde_notebook",
    )
    _ACTIVE_EXPORTER = _AutomaticPdfExporter(output_directory)

    plt.show = _show_matplotlib_and_save
    try:
        from IPython import get_ipython
        shell = get_ipython()
    except ImportError:
        shell = None
    if shell is not None and _IPYTHON_EVENTS is None:
        _IPYTHON_EVENTS = shell.events
        # Run before the inline backend flushes and closes its figures.
        _IPYTHON_EVENTS.register("post_execute", _save_after_notebook_cell)
        callbacks = _IPYTHON_EVENTS.callbacks["post_execute"]
        callbacks.remove(_save_after_notebook_cell)
        callbacks.insert(0, _save_after_notebook_cell)
    if PlotlyBaseFigure is not None:
        PlotlyBaseFigure.show = _show_plotly_and_save

    return output_directory


def disable_automatic_pdf_saving() -> None:
    """Restore the original display functions."""
    global _ACTIVE_EXPORTER, _IPYTHON_EVENTS

    plt.show = _ORIGINAL_MATPLOTLIB_SHOW
    if PlotlyBaseFigure is not None and _ORIGINAL_PLOTLY_SHOW is not None:
        PlotlyBaseFigure.show = _ORIGINAL_PLOTLY_SHOW
    _ACTIVE_EXPORTER = None
    if _IPYTHON_EVENTS is not None:
        _IPYTHON_EVENTS.unregister("post_execute", _save_after_notebook_cell)
        _IPYTHON_EVENTS = None
