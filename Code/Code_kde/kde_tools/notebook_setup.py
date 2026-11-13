"""Shared notebook paths and automatic PDF export."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NotebookPaths:
    code: Path
    project: Path
    data: Path
    images: Path
    pdfs: Path


def setup_notebook(name: str) -> NotebookPaths:
    """Enable PDF saving once and return the experiment's output folders."""
    from .pdf_export import enable_automatic_pdf_saving

    code = Path(__file__).resolve().parents[1]
    data = code / "outputs" / "data" / name
    images = code / "outputs" / "images" / name
    return NotebookPaths(
        code=code,
        project=code.parents[1],
        data=data,
        images=images,
        pdfs=enable_automatic_pdf_saving(name),
    )
