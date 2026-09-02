# Lane W1D — 2^27 pressure-tier degradation (st4, st6b) root-cause + structural fix

## Mission

At 2^25 the Metal prover is healthy (st4 41% GPU, st6b 54%, prove 19.82 s). At
2^27 (~90 GiB footprint on a 128 GiB box) the same stages collapse: st4 18%
GPU / 10.44 s canonical, st6b 25% / 13.87 s, and the st5→st6b seam contains a
9.16 s zero-GPU hole partly made of pressure-inflated prepares. Root-cause the
degradation mechanism, then fix it structurally. Proof bytes must stay
identical.

## What is already known (M5 campaign, W4 U1 — read `.metal-m5-box-journal.md` §Wave 4)

- Stage 5 retires two arena slabs totaling ~30 GiB at 2^27. If they stay
  physical into st6b, page pressure inflates `IncClaimReduction::prepare`
  (2.6→7.3 s), `BytecodeReadRafCycle::prepare` (→3.0 s), st6b `prove_batch`
  (→9.1 s), `EqPolynomial::evals_parallel` (→8.2 s).
- W3-era T6 tree "happened to" reclaim them before st6b (94.47→100.6 s
  variance across merges); T2's timing left them physical → st6b 19.8-21.8 s.
- **Failed attempt (do not repeat):** `MADV_FREE_REUSABLE` on parked arena free
  ranges + `MADV_FREE_REUSE` on carve (`69e7d75d4`, reverted `276396aed`) —
  neutral at 2^26, ineffective at 2^27. Close-out verdict: "a real fix must
  structurally end or decommit stage-5 ownership before the stage-6b
  adoptions."
- st4 has its own non-pressure problem visible even in the instrumented trace:
  `RegistersReadWriteChecking::prepare` = 4.70 s zero-GPU hole + first round
  1.6 s zero-GPU + rounds at 18% util. Canonical st4 grew 2.468→10.44 s from
  2^25→2^27 (4.2× for 4× work is fine, but util halved — separate the
  pressure component from the algorithmic component with evidence, not
  assumption).

## Deliverables, in order

1. **Root-cause artifact BEFORE any fix** (`.journals/lane-reports/w1d-rootcause.md`):
   - Which allocations own the ~30 GiB at the st5→st6b boundary (arena slab
     inventory: who allocated, who parks, who adopts). Instrument or log the
     arena state at stage boundaries (footprint per slab, `vm_stat` deltas,
     page-fault counters around the inflated prepares).
   - Why `MADV_FREE_REUSABLE` failed (compressor interaction? wrong range?
     wired by Metal residency?).
   - st4: what fraction of the 2^27 cost is pressure vs shape (e.g., compare a
     2^26 run where footprint ≈51 GiB — the pressure tier starts between 2^26
     and 2^27).
2. **Structural fix**: end/decommit stage-5 arena ownership before st6b
   adoptions. Candidate shapes (evaluate, don't assume): scoped drop of the
   stage-5 slabs (actual `munmap`/dealloc, not madvise) at the st5 boundary;
   restructure slab reuse so st6b carves from the same physical pages instead
   of faulting fresh ones; move the two st6b prepares' inputs off the retired
   slabs entirely.
3. **st4 prepare/round hole** if the evidence says it's pressure-coupled; if
   it's shape-coupled (CPU table build like the other prepares), document and
   hand to a wave-2 port lane — do NOT start a registers-RWC kernel rewrite
   here (W4 U3 already rejected a bounded prototype; CSR rewrite is out of
   lane scope).

## Where things live

- Arena/slab code: search `crates/` for the parked-arena implementation
  (M5 keywords: "arena", "carve", "retire", "MADV_FREE_REUSABLE",
  "park"; the U1 attempt/revert commits `69e7d75d4`/`276396aed` show the exact
  files — `git show` them).
- Stage-5 driver `crates/jolt-prover/src/stages/stage5.rs`, st6b
  `stage6b.rs`, slots in `crates/jolt-kernels/src/metal/slots/`.
- ProofSession state (cross-stage carriers): `crates/jolt-kernels/src/backend.rs`.

## Ground rules

- Worktree `~/dev/jolt/.worktrees/gpuutil-w1d`, branch `gpu/util-w1d`. Never
  push. Commit per unit; root-cause artifact committed before fix commits.
- **Hard gate: proof bytes identical** + full gate matrix (see
  `.journals/gpu-util.md` Protocol).
- You are the ONLY lane allowed 2^27 runs, and only: one at a time, under
  `/tmp/jolt-gpu.lock.d`, after checking no sibling build storm (`uptime` load
  < 6) and ≥95 GiB free (`memory_pressure` / footprint of running procs), swap
  checked before/after (`sysctl vm.swapusage`). Prefer 2^26 (51 GiB) for
  iteration — the pressure tier boundary is between 2^26 and 2^27, so 2^26
  healthy + 2^27 sick is itself evidence.
- Cheap diagnostics first: dtrace/log counters, `vm_stat 1` sidecars, footprint
  sampling (`footprint --pid`), NOT repeated full 2^27 sweeps. Budget ≈6
  2^27 runs total for the whole lane.
- Coordination: W1B owns IncClaimReduction/BytecodeReadRafCycle slot code. You
  own allocator/arena/lifetime + stage-driver ownership changes. If the fix
  needs slot-code edits, route through the orchestrator.

## Kill / accept gates

- Root-cause artifact is mandatory output even if no fix lands.
- Fix gate at 2^27: st4+st6b combined −4 s vs canonical 24.32 s (st4 10.442 +
  st6b 13.874), with 2^25 neutral (±1% total). Partial wins with clean
  attribution are acceptable if honestly quantified.

## Reporting

message_parent at: (i) root-cause artifact done (with the mechanism named),
(ii) fix candidate chosen + first 2^26/2^27 evidence, (iii) final. Lane report
`.journals/lane-reports/w1d.md` committed.
