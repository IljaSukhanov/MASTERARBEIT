"""Delivery copies for the exact figure paths already used by the thesis."""

from pathlib import Path
import shutil


KDE_DIRECTORY = Path(__file__).resolve().parents[1]


def copy_thesis_pdf(pdf_directory, experiment_name, filename):
    """Keep the thesis's existing asset path after combining notebook sections."""
    source = Path(pdf_directory)/filename
    if not source.is_file():
        raise FileNotFoundError(f'The automatic PDF export is missing: {source}')
    destination = KDE_DIRECTORY/'outputs/pdfs'/experiment_name/filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copyfile(source, destination)
    return destination


def save_minimizer_panel(figure, n, output_directory=None):
    """Save one selected-size panel at the chapter's existing includegraphics path."""
    directory = (Path(output_directory) if output_directory is not None else
                 KDE_DIRECTORY.parent/'output/pdf/continuous_minimizers_selected_n')
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/f'continuous_minimizers_n{int(n)}.pdf'
    figure.savefig(path, format='pdf', bbox_inches='tight')
    return path
