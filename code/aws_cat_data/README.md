# AWS cat-qubit raw data (extracted 16 Oct 2026)

**Source:** Putterman et al., "Preserving phase coherence and linearity in cat qubits with exponential bit-flip suppression", PRX 15, 011070 (2025), arXiv:2409.17556v2. The data come from the arXiv source tarball (figure PDFs), not from screenshots.
- Vector traces were read from the PDF path coordinates and calibrated against the figure's own tick marks.
- Colour maps were inverted with the figure's own colour bar (nearest colour; the residual colour error is ~1e-4, so the mapping is exact up to the colour-bar resolution of ~0.03 in parity or ~0.4 dB).

Scripts are in `../code/`:
- `pdfstreams2.py` decompresses the PDF streams;
- `paths2.py`, `s21.py`, `rects.py`, `parity.py` and `panelab.py` do the extraction;
- `test.py` runs the analysis.

| File | What it is | Columns / units |
|---|---|---|
| `filter_S21.txt` | Main-text Fig. 2(b): S21 of the 4-pole buffer filter, 2–6 GHz | GHz, dB. Uncalibrated (convolved with a 2.5–4 GHz isolator, per the caption); clipped below −80 dB in the plot. Passband ≈ 2.59–3.55 GHz (−10 dB from the in-band median). Edge peaks at 2.640 and 3.464 GHz; buffer notch at ~3.01 GHz. |
| `parity_vs_pump_delay.txt` | App. Fig. "pump-induced parity changes", panel (c): storage parity vs pump frequency and delay. Fock 0/1 superposition prepared, pump at ε_p ≈ 0.08–0.1 | pump GHz (cell centre, 20.5 MHz step, 7.10–7.90), delay µs, parity |
| `parity_late_vs_pump.txt` | Same data, mean over 200–300 µs (≈ steady-state parity) | pump GHz, parity |
| `buffer_S21_vs_pump.npz` | Same figure, panel (b): S21 map of the buffer resonance vs pump frequency | arrays `pump_GHz` (169), `probe_GHz` (81), `S21_dB` (81×169) |
| `buffer_freq_measured_vs_pump.txt` | Measured buffer frequency (S21 minimum) per pump column of panel (b) | pump GHz, probe GHz, min dB. The scatter of ±10–15 MHz is real (pump power delivery varies with frequency, per the paper). |
| `model_buffer_freq_vs_pump.txt` | Panel (a): the authors' Floquet-model Stark-shifted buffer frequency, 5 curves for ε_p = 0.02–0.1 | curve index (0 = ε_p 0.1), pump GHz, buffer GHz. The kinks at 7.115 GHz (ε 0.1) and 7.156 GHz (ε 0.08) are their a†b†³ resonance. |

**Parameters quoted in the text** [read]:
- storage ω_s/2π = 5.35 GHz;
- buffer ω_b/2π = 3.01 GHz (unpumped), κ_b/2π = 10.7 MHz;
- g₂/2π ≈ 578 kHz;
- buffer at saddle point (φ_Σ, φ_Δ) = (π/2, π/2);
- desired 3WM at ω_p = 2ω̄_s − ω̄_b (≈ 7.7 GHz in their model);
- undesired a†b†³ at 2ω_p = ω̄_s + 3ω̄_b (≈ 7.2 GHz);
- unexplained parity dip near 7.35 GHz (their App. G.3, last paragraph);
- storage T1 = 76 µs (79 µs with pure two-photon dissipation);
- bit-flip saturation at 0.3–0.6 s, unexplained (Sec. VI; App. N excludes ancilla/coupler heating).
