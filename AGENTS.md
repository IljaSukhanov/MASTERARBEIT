## Thesis protection

- Treat every file under `Abschlussarbeit_2021/` as protected.
- A request to create, regenerate, replace, or import a figure authorizes only:
  1. changing the figure file; and
  2. changing the corresponding `\includegraphics{...}` path.
- Never infer permission to edit thesis prose, captions, labels, references, TODOs, equations, structure, or formatting from a figure or code request.
- Any other thesis-source modification requires a separate, explicit request describing that exact text change.
- If uncertain, stop and ask before modifying the thesis.

## KDE figure exports

- For active code and notebooks under `Code/Code_kde/`, every figure that is plotted or displayed must also be saved automatically as a PDF.
- New KDE notebooks must enable the shared automatic PDF exporter from `kde_tools.pdf_export` in their setup cell.
- Save these automatic exports under `Code/Code_kde/outputs/pdfs/<notebook-name>/`.
