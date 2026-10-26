#!/usr/bin/env bash
# Reproduce every figure and number in "Photonic shake-off" (paper/shakeoff_prr.tex).
#   bash reproduce_all.sh          -> figures + tables from the saved data            (~1 min)
#   bash reproduce_all.sh full     -> recompute all data first, then figures           (~1.5 h on 2 cores)
# Requires: python3, numpy, scipy, matplotlib, scienceplots.
set -e
cd "$(dirname "$0")"
if [ "${1:-}" = "full" ]; then
  python3 kc_bath.py                    # self-tests of the bath: Green's function, C_g defect ring, exact pole   (~10 s)
  python3 kc_selfion_run.py KC &        # Fig. 4: capacitive line, gaps 1-4: theory + exact two-excitation runs   (~50 min)
  python3 kc_selfion_run.py KP; wait    # Fig. 4: inductive line, gaps 1-4
  python3 kc_selfion_scaling.py         # Fig. 5: kappa scaling of the loss                                       (~10 min)
  python3 nl2000_reproduce.py           # Fig. 2: N&L 2000 Figs. 2, 4a, 4b, 5a, 6 + convergence of 4a              (~1 min)
  python3 nl2000_threshold.py           # Fig. 3: threshold scan in the N&L model, 12 points                       (~3 min)
  python3 nl2000_markov_limit.py        # Fig. 3: three points deep in the band (finer grid)                       (~2 min)
fi
python3 fig_schematic.py                # Fig. 1  -> fig_schematic.pdf
python3 fig_nl2000_repro.py             # Fig. 2  -> fig_nl2000_repro.pdf
python3 nl2000_markov_fig.py            # Fig. 3  -> fig_markov.pdf   (+ fig_markov_data.txt)
python3 kc_selfion_fig.py               # Fig. 4  -> kc_selfion.pdf
python3 kc_selfion_scaling_fig.py       # Fig. 5  -> kc_selfion_scaling.pdf
python3 ghz_mapping.py                  # Table I (Appendix F): device parameters in GHz
python3 thorbeck_filter_shakeoff.py     # Sec. IV + Appendix G: shake-off in the filter model of Thorbeck et al.
python3 aws_filter_channel.py           # Sec. IV + Appendix H: filter-band heating channel, Putterman et al. cat qubit
python3 aws_filter_channel_fig.py       # Fig. 6  -> fig_aws_cat.pdf (+ .png)
# python3 audit_checks.py > audit_checks.out   # non-adjacent pairs, ring return times, dropped runs (~10 min)
echo "done"
