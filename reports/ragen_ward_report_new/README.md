# RAGEN-Ward LaTeX Report

This folder contains an English ACL-style LaTeX version of the RAGEN-Ward report.

## Main File

Compile:

```text
latex/acl_latex.tex
```

The project uses the ACL template files:

- `latex/acl.sty`
- `latex/acl_natbib.bst`
- `latex/custom.bib`

The main body is split into `latex/sections/*.tex` for easier editing.

## Recommended Overleaf Settings

- Compiler: `pdfLaTeX`
- Main document: `latex/acl_latex.tex`
- Bibliography: BibTeX, using `custom.bib`

The file `latex/acl_lualatex.tex` is only an alternative wrapper that inputs the same report. For this English version, `pdfLaTeX` is the intended path.

## Notes

- The full ACL Anthology bibliography is intentionally not included to keep upload size and compile time low.
- All cited references used by this report are in `latex/custom.bib`.
- Figures are copied into `latex/figures/`.
- Local compilation was not performed because the machine does not have LaTeX installed.
