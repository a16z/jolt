# Akita five-workload CPU/Metal matrix — 2026-09-05

Completed 90 cells on one M4 Max: BTreeMap, Fibonacci, SHA2-chain, SHA3-chain,
and Collatz range; every integer scale from 2^20 through 2^28; optimized Akita
CPU versus the production hybrid Metal backend. All 96 scored proofs verified.

At T28, mean Metal throughput is **8.243 MHz measured**, or **9.315 MHz projected
on M5 Max** using the campaign's fixed 1.13x factor. This five-program quick
sweep is below the 10 MHz target. The original three programs within this
sweep project to 9.862 MHz; the earlier 10.498 MHz acceptance result is a
separate, longer-cooling three-program experiment, not a five-program claim.

## Five-program means

| log2 T | CPU mean s | Metal mean s | CPU mean MHz | Metal mean MHz | Geometric speedup | Projected M5 Metal MHz |
|---|---:|---:|---:|---:|---:|---:|
| 20 | 1.606 | 1.494 | 0.661 | 0.714 | 1.08x | 0.806 |
| 21 | 2.607 | 2.418 | 0.828 | 0.897 | 1.08x | 1.014 |
| 22 | 4.865 | 4.419 | 0.895 | 0.988 | 1.10x | 1.117 |
| 23 | 9.512 | 8.678 | 0.924 | 1.023 | 1.10x | 1.156 |
| 24 | 19.453 | 9.228 | 0.900 | 1.834 | 2.07x | 2.072 |
| 25 | 21.440 | 6.071 | 1.592 | 5.563 | 3.51x | 6.286 |
| 26 | 42.756 | 9.254 | 1.599 | 7.324 | 4.60x | 8.276 |
| 27 | 80.824 | 19.029 | 1.695 | 7.228 | 4.26x | 8.167 |
| 28 | 157.482 | 33.125 | 1.741 | 8.243 | 4.75x | 9.315 |

| T28 program | CPU s | Metal s |
|---|---:|---:|
| BTreeMap | 129.293 | 27.934 |
| Fibonacci | 177.087 | 35.153 |
| SHA2-chain | 154.511 | 30.040 |
| SHA3-chain | 189.485 | 40.686 |
| Collatz range | 137.035 | 31.813 |

## Method and limits

Guests were compiled ahead of time. Each observation used a fresh process;
the reported wall covers proof generation, excluding guest compilation,
tracing, preprocessing, and verification. Verification must still succeed.
CPU and Metal use matching inputs and proof parameters within each scale.
MHz counts padded trace rows, not actual executed instructions.

There is one sample per cell except six flagged Metal cells: all five T25
cells and Fibonacci T28 received one 120-second cooled repeat. Both samples
remain in each repeated cell's arithmetic mean. Ordinary gaps were 5 seconds
at T20–24, 15 seconds at T25–27, and 30 seconds at T28. Per-scale seconds and
MHz are arithmetic means across five programs; speedup is the geometric mean
of CPU/Metal ratios. Mean MHz is not the reciprocal of mean wall time.
These sparse observations are descriptive, with no confidence intervals.

The T25 timing drop persisted in repeats. Production one-hot chunk geometry
changes at T25, alongside several Metal dispatch thresholds. The sweep crosses
configuration boundaries; it is not a fixed-kernel linear scaling study.

The original study stopped at SHA3 Metal T25: its active register domain
exceeded the Metal read/write family's 64-register capacity. The user approved
a repair. Exact active masks now select the existing optimized CPU family
when capacity is exceeded, including the fused Stage-1 handoff. Other eligible
families remain on Metal. Shaders, relations, transcript, parameters, and
verifier checks were not changed. SHA3 T25–28 is explicitly a hybrid result.

The original 57 completed cells were retained; the repaired binary filled
33 cells after a supported-route BTreeMap T28 drift check passed at 27.934 s
against the predefined 26.85 s anchor and 10% band. This reuse check does not
prove zero regression. The two source revisions are recorded per observation.
BTreeMap T20 uses the documented input override so it actually pads to T20.

Peak process RSS was 86.877 GiB, below the 88 GiB guard; all scored runs had
zero swaps. No watchdog abort was treated as noise. An independent completion
audit checked all cells, paired trace sizes, raw/source/binary/guest hashes,
CSV values, and aggregate calculations. No runs or lock remain.

## Revision and validation

Original measured harness: d9b4da730. Repaired measured revision: 107f28a8e.
Integrated locally on feat/akita-metal as 1bff57f9a and eebadf651; source trees
match. Unrelated user changes were preserved. Not pushed as part of this task.

Repair validation: 329/329 serial Metal kernel tests; reference parity at
64/65/72/128 active registers and the Stage-1 handoff; host and host+ZK clippy;
Metal/test-utils clippy; formatting; release build; all requested verified
end-to-end cells. Could not verify the optional Metal/test-utils/allocative
clippy combination, which fails outside the matrix's build configuration.

Full local evidence is under
`benchmark-runs/akita-five-workload-matrix-2026-09-05/`:

- `resume/REPORT.md`: per-program/per-scale tables and all five-program means.
- `resume/results.csv`, `resume/averages.json`, `resume/observations.json`.
- `resume/METHOD.md`, `resume/AUDIT.md`, and `REPAIR-VALIDATION.md`.
- Both manifests, append-only event logs, and raw run outputs.

The benchmark-runs directory is local evidence, not included in this commit.
