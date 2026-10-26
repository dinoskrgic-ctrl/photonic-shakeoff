# Photonic shake-off: leakage decay can destroy the qubit–photon bound state in passive leakage filters

Dino Škrgić and Dejan Stojkovic, Department of Physics, University at Buffalo, SUNY

Placeholder for arXiv and journal, once accepted (fingers crossed)! 

This repository holds the LaTeX source of the paper and the code and data that reproduce every figure and number in it.

## Layout

```
paper/                      LaTeX source (revtex4-2, prresearch) and the compiled PDF
  shakeoff_prr.tex          manuscript
  shakeoff_refs.bib         references
  shakeoff_prrNotes.bib     revtex control file (revtex rewrites it on every run)
  shakeoff_prr.bbl          formatted bibliography (needed for arXiv)
  fig_*.pdf, kc_*.pdf       the six figures, as included by the .tex
  shakeoff_prr.pdf          compiled paper
code/                       scripts and saved data (run from inside this folder)
  reproduce_all.sh          redraws every figure and table
  aws_cat_data/             data digitized from Putterman et al. (see below)
  aws_cat_extract/          scripts that digitized them (provenance only)
```

## Building the paper

```
cd paper
pdflatex shakeoff_prr
bibtex shakeoff_prr
pdflatex shakeoff_prr
pdflatex shakeoff_prr
```

or `latexmk -pdf shakeoff_prr`. BibTeX prints a number of "is a string literal, `journal' is a missing field" messages from `apsrev4-2.bst` and warnings for arXiv-only entries; both are harmless.

## Reproducing the figures

Python 3 with `numpy`, `scipy`, `matplotlib` and `SciencePlots` (`pip install -r code/requirements.txt`). The figure scripts use the SciencePlots `science` style, which typesets labels with LaTeX, so a LaTeX installation is needed for them.

```
cd code
bash reproduce_all.sh         # all figures and tables from the saved data (about a minute)
bash reproduce_all.sh full    # recompute all data first (about 1.5 h on two cores)
```

The scripts write their figures into `code/`; copy them into `paper/` to rebuild the paper with them.

| Figure / table | Script | Input data | Recomputed (`full`) by |
|---|---|---|---|
| Fig. 1 `fig_schematic.pdf` | `fig_schematic.py` | none | |
| Fig. 2 `fig_nl2000_repro.pdf` | `fig_nl2000_repro.py` | `nl2000_reproduce.npz` | `nl2000_reproduce.py` |
| Fig. 3 `fig_markov.pdf` | `nl2000_markov_fig.py` | `nl2000_threshold.npy`, `nl2000_markov_limit.npy` | `nl2000_threshold.py`, `nl2000_markov_limit.py` |
| Fig. 4 `kc_selfion.pdf` | `kc_selfion_fig.py` | `selfion_KC.npz`, `selfion_KP.npz` | `kc_selfion_run.py KC`, `kc_selfion_run.py KP` |
| Fig. 5 `kc_selfion_scaling.pdf` | `kc_selfion_scaling_fig.py` | `kc_selfion_scaling.npy` | `kc_selfion_scaling.py` |
| Fig. 6 `fig_aws_cat.pdf` | `aws_filter_channel_fig.py` | `aws_cat_data/` | (no recomputation needed) |
| Table I (App. F) | `ghz_mapping.py` | none | |
| Sec. IV, App. G (Thorbeck filter) | `thorbeck_filter_shakeoff.py` | none (writes `thorbeck_filter_shakeoff.npz`) | |
| Sec. IV, App. H (cat qubit numbers) | `aws_filter_channel.py` | `aws_cat_data/` | |

Shared modules: `kc_bath.py` (loaded-line bath, exact Green's function, ring modes; running it prints self-tests), `kc_selfion.py` (two-excitation dynamics and the branching formula), `kc_transmon_freq.py` and `kc_transmon_part2.py` (dispersion and two-quantum Hamiltonian), `nl2000_reproduce.py` (the Nikolopoulos–Lambropoulos model). `audit_checks.py` runs the extra checks quoted in the text (non-adjacent band pairs, ring return times, dropped runs); its output is saved in `audit_checks.out`.

## Data from Putterman et al.

`code/aws_cat_data/` contains data digitized from the figures of

> H. Putterman *et al.*, "Preserving phase coherence and linearity in cat qubits with exponential bit-flip suppression", Phys. Rev. X **15**, 011070 (2025), [arXiv:2409.17556](https://arxiv.org/abs/2409.17556).

The traces were read from the vector figure PDFs in the arXiv source, not from images; `aws_cat_data/README.md` describes each file and its units. The extraction scripts are in `code/aws_cat_extract/` (their README refers to them as `../code/`). They need the arXiv source tarball, which is not included, and they are not run by `reproduce_all.sh`. All credit for these measurements belongs to the original authors; please cite their paper if you use the data.

## Citation

```bibtex
@article{Skrgic_Stojkovic_shakeoff,
  author  = {\v{S}krgi\'c, Dino and Stojkovic, Dejan},
  title   = {Photonic shake-off: leakage decay can destroy the qubit--photon bound state in passive leakage filters},
  journal = {Phys. Rev. Res.},
  volume  = {VOL},
  pages   = {ARTICLE},
  year    = {YEAR},
  doi     = {10.1103/XXXXXXX},
  eprint  = {XXXX.XXXXX},
  archivePrefix = {arXiv}
}
```
