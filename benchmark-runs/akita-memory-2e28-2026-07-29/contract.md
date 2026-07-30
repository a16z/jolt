# Akita memory experiments for 2^28

Date: 2026-07-29 EDT

## Objective

Reduce Akita's K256 prover working-set slope without slowing the prover. The
eventual 2^28 run should peak below 95 GiB, report zero process swaps, and
produce no system swapouts during the run.

K256 and the proof protocol are frozen for this campaign. Each engineering
change is measured and committed independently.

## R1CS retained-row removal

Question: what exact time and phase-RSS trade results from never retaining the
208-byte `R1CSCycleInputs` row cache?

The user has selected the memory-minimal policy regardless of trace size. The
measurement records its cost; it does not decide whether the cache remains.

- Screen: traced 2^22 K256 proof.
- Full validation: traced 2^26 K256 proof.
- Metrics: round-zero materialization, claimed-input evaluation, stage 1,
  `prove_packed`, phase RSS, maximum RSS, and swaps.
- Correctness: proof verification plus the host and host+zk muldiv gates.

## Compact trace-index storage

Question: can one compact K256 row-major source replace the retained `u16`
commit/opening lanes and fixed-width `RaIndices` without losing the existing
trace-cache speedup?

Outcome note: this campaign validated the storage pieces separately rather
than attempting the full replacement at once. Native byte lanes landed, and
`RaIndices` now dies before opening, but Stage 1–7 still read the fixed-width
rows. The proposed 82-byte/cycle unification therefore remains a follow-up;
the measured results are recorded in `results.md`.

The first screen changes storage only. A K256 lane uses one byte. Instruction,
bytecode, and increment lanes are always present; RAM uses a separate validity
bitmap so `None` remains distinguishable from logical lane zero.

- Expected storage reduction: at least 82 bytes/cycle, approximately 5.1 GiB
  at 2^26 and 20.5 GiB at 2^28.
- Focused metrics: cache construction, commitment accumulation,
  Booleanity initialization, `evaluate_and_fold`, `decompose_fold`, and Stage 7.
- Promotion gate: proof parity; at least 4.5 GiB lower 2^26 peak or an exactly
  attributed phase-window reduction; no focused aggregate regression above 2%;
  no full-prover regression larger than the established 0.48-second noise band.

## Validation discipline

All full measurements use `PERF_LOG_K_CHUNK=8`,
`PERF_LOOKUPS_RA_VIRTUAL_LOG_K_CHUNK=32`, `PERF_TRACE=1`, a quiet-machine
gate, and an RSS sampler. Parent and candidate are interleaved before a result
is called revalidated. Traces use the short `mem-*` names in
`benchmark-runs/perfetto_traces/`.
