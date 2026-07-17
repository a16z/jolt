# Akita prover perf plan

## Goal

Optimize the Akita (packed) prover on `feat/akita-protocol` until, at 2^20
sha2-chain on this machine, **akita prove time is ≤ 0.5× dory prove time**,
measured back-to-back in the same session.

Baseline (2026-07-12, nothing concurrent): akita prove 81.33s / verify 1.94s;
dory prove 13.20s / verify 0.57s → target ≈ **≤6.6s prove** (≈12× from here).
Prove time is the objective. Don't grossly regress akita verify or peak memory
(16GB machine). The ~9s one-time transparent setup is EXCLUDED from the target.

## State

The full Akita prover port is committed on `feat/akita-protocol` as
`380dad2f3 feat: prover working` — that commit is the perf baseline, tree
clean, muldiv/advice/committed e2es all green over Akita. Build on it: land
each accepted iteration as its own commit (`perf(akita): <what>`) once gates
pass, so wins stay bisectable; discard failed attempts with `git restore`.
Do NOT push. Context ledgers:
`~/.claude/projects/-Users-mgeorghiades-a16z-jolt/memory/akita_perf_goal.md`
(baselines, leads), plus `akita_surgery_port.md` and
`akita_integration_state.md`. CLAUDE.md rules apply throughout (nextest only,
`#[expect]` not `#[allow]`, comment policy).

## Harness

Iterate at 2^18 (fast); confirm milestones at 2^20.

- headline (NO tracing):

  ```bash
  PERF_LOG_T=18 cargo nextest run --release -p jolt-prover-legacy \
    --features host,akita --run-ignored all -E 'test(sha2_chain_akita_perf)' \
    --no-capture
  ```

- profile: same command + `PERF_TRACE=1` → writes
  `benchmark-runs/perfetto_traces/sha2-2exp{N}-akita.json` (keep every trace —
  the user views them in Perfetto)
- dory denominator (ratio checks only): `-E 'test(sha2_chain_dory_perf)'` with
  `--features host`
- Nothing else runs during a measurement. Laptop thermals swing ±10–15%:
  compare cool-machine runs or span RATIOS, and re-measure the dory
  denominator in the same session as any milestone claim.

## Loop

Repeat without pausing between iterations until target or hard-blocked:

1. **TRACE FIRST.** Take a fresh 2^18 trace and rank top spans by inclusive
   time. Hypotheses come from the trace, never from static code reading. If
   the top span is opaque, add finer tracing spans (tracing is already a dep
   of jolt-openings; instrument jolt-akita/adapter layers as needed),
   re-trace, then hypothesize. (No 2^18 akita trace exists yet — that's
   iteration 1's first action.)
2. **DESIGN BEFORE CODING:** name the span, its measured cost, the algorithmic
   change, and an argued expected gain. While >2× off target, only large
   argued wins qualify — no micro-optimizations (inlining, devirt, allocation
   shaving). ONE bottleneck per iteration. If the biggest cost needs a
   data-format change, change the format — don't work around it.
3. **IMPLEMENT.** Larger refactors are explicitly sanctioned, INCLUDING the
   format of the packed witness we commit to (lane layout, chunk width,
   one-hot grouping, object structure). Any witness-format or protocol change
   must move prover (jolt-prover-legacy), verifier (jolt-verifier),
   jolt-claims, and the packed clear-claims builder coherently in the same
   iteration. No serialized fixtures pin akita transcript bytes yet — format
   changes are still cheap.
4. **GATES** before accepting any measurement (all must pass):
   - `cargo nextest run -p jolt-prover-legacy --features host,akita -E
     'test(e2e_akita)' --cargo-quiet` (muldiv_e2e_akita + advice_e2e_akita +
     muldiv_e2e_akita_committed_program; these include tamper-rejection and
     are the soundness net for format changes)
   - `cargo nextest run -p jolt-openings --cargo-quiet` (includes the
     sparse-vs-dense oracle test)
   - if any shared (non-cfg(akita)) code was touched: muldiv e2e with
     `--features host` AND `--features host,zk` must pass unchanged
   - clippy zero warnings: `--all` with `host`; `--all` with `host,zk`;
     `-p jolt-prover-legacy` with `host,akita` (if you touched akita-gated
     code in jolt-verifier/jolt-akita/jolt-openings, clippy those with their
     akita features too); `cargo fmt -q`
5. **MEASURE** clean (no tracing) vs the iteration's before-number. If gates
   pass and the win is real, commit it (`perf(akita): <what>`); otherwise
   `git restore` the attempt and attack the next-ranked span. A failed
   hypothesis is never a reason to stop.
6. **LEDGER:** append one line per iteration to the akita_perf_goal.md memory
   ledger (hypothesis → change → before/after @2^18 → gates → commit sha).
   At each milestone (≈ every cumulative 2×), run 2^20 akita + dory
   back-to-back and record the ratio.

## Leads

> Status 2026-07-16 (post-#1683 distillation, slices A–E of
> `specs/akita-1683-distillation.md`): the fused-inc pipeline + native
> W_jolt opening + K16 toggle + jolt-owned schedule catalogs are landed.
> Measured at 2^20 sha2-chain, same machine back-to-back:
> **akita prove 8.19s vs dory 13.64s = 0.60×** (verify 36.8ms vs 610ms;
> the dory harness lacks the prepared-pairing-cache init, so the verify
> ratio is conservative toward dory). Goal target 0.5× — remaining gap
> ~0.1×; next levers are quangvdao's list (witness ownership/copy
> removal, column-derivation reuse, Hachi-style layout experiment,
> prepared-setup cache). CAUTION: akita upstream #302 (quantum SIS
> cutover) made one-hot setup construction ~400× slower (83s at the
> 2^20 W_jolt shape vs 0.2s pre-#302) — per-shape and cacheable, but the
> setup cache (lever 5) is now urgent, and prove-side #302 impact vs
> machine variance is unattributed (quang's pre-#302 prove was
> 5.5–6.1s on his machine). Older analysis below and in
> `specs/akita-optimal-committed-data.md` is kept as context; the union
> vs grouped accounting there is superseded by the native common-point
> opening.


From the original branch's trace analysis. Treat as hypotheses to RE-VALIDATE
against your own fresh traces, not as a work plan. Ranked by expected size:

1. **STRUCTURAL/FORMAT (the big one** — kernel wins alone were measured
   insufficient): commit W_jolt as GROUPED strict one-hot columns through the
   akita backend's one-hot flavor + grouped commitment protocol instead of
   today's sparse-unit UNION object. The backend's own flavor bench measured
   one-hot commit+open ≈5.4× cheaper than sparse-unit at equal domain/ones.
   The pinned backend rev (LayerZero-Labs/akita @ c8290922 — external git
   dep, consume via API, don't edit; flag to the user only if a capability is
   genuinely missing) already ships `backend/onehot/` and the grouped
   protocol. Open design questions (K=256 preset vs log_k_chunk=4 lanes; MSB
   as K=2 one-hot; grouped-root gate rejects non-strict-one-hot) are at the
   tail of akita_integration_state.md.
2. Per-object native-opening pipeline duplication (mat_vec_mul_ntt, relation
   quotient, ring-switch repeated per commitment object) → batch/collapse
   across objects.
3. CPU utilization was ~170% of 800% during akita prove on the original
   branch — hunt serial fractions in backend/adapter layers AFTER the
   structural items land.
4. The stage-8 packed reduction was just rewritten sparse
   (jolt-openings/src/packing.rs) — confirm from the trace it's no longer
   dominant before touching it again.

Reference: the original branch's best was 55.4s at 2^20 (2^18 composition:
PIOP ~1.3s + commit ~2.3s + reduction ~1.8s + backend opening ~4.7s, vs dory
5.6s total). The 2× goal requires beating that branch's own endpoint — hence
the structural lead.

## Report

When done or blocked: final table (akita vs dory, prove/verify, 2^18 and
2^20), landed-changes list, the per-iteration ledger, and trace files for the
final state. Hard-blocked means a decision only the user can make (upstream
rev bump, soundness-relevant protocol semantics) — not a failed hypothesis,
not a long build.
