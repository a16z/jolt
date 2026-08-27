# Lane S14 (wave 14): st1 Spartan outer — attribution + lazy-form kernels

**Verdict: RETAIN — modeled −1.66 s st1 @2^27 (bar ≥1.0), byte parity
proven in both arms, zero resident-data change.** Commit `83a866fd8` on
`lane/metal-w14-st1` (off e6a7e3225). Kill switch
`JOLT_METAL_OUTER_LAZY=0` restores the eager kernels. KernelId::ALL
85 → **88**.

## Attribution @2^27 (the one instrumented profile)

FrBind 257.8 µs (healthy). Traced run 42.95 s wall — record-class
window (best 42.43 / median 42.67). st1 = 4.995 s, 99.5% attributed;
all four slices are solo synchronous device dispatches (host glue
< 1%, no waits of note, `TraceRecord` shared from st0):

| slice | s | share | what it is |
|---|---:|---:|---|
| `outer_t1` (`jk_outer_t1`) | 2.059 | 41% | uniskip message: 9 extended-node sums over 134 M rows |
| `outer_azbz` (`jk_outer_azbz`) | 1.670 | 33% | Az/Bz table materialization (2×2^28 Fr) + first endpoints |
| `outer_claims` (`jk_outer_claims`) | 0.925 | 19% | 35 final opening sums (device since Aug 4, `9d9362aa6`) |
| `outer_round` ×13 (device CBs) | 0.315 | 6% | fused bind+endpoints round loop; CPU tail < 2^16 trivial |
| host glue (alloc 3 ms, uniskip poly 3 ms, prepare, tail binds) | ~0.03 | <1% | |

No single item ≥ half the stage; the dominant coherent slice is the
shared per-row pattern across the three big kernels: **every integer
row value paid a Montgomery conversion multiply before its weight
multiply** (t1 74 mont-muls/row, azbz 81, claims 52), while the host
twin (`extended_products`) had long been integer-lazy.

## The cut: lazy-form kernel twins (default on)

Three new kernels, values identical mod p at every store/partial
boundary (CIOS `fr_mont_mul` is exact and canonical for one raw
operand < 2^256 with the other < p):

- **`jk_outer_t1_lazy`** — Az·Bz per node in the integer domain
  (i64 × i192 → i256; host bounds `|az| < 2^22, |bz| < 2^152,
  product < 2^174`); ONE weight multiply on the raw residue; partials
  leave in standard form, host multiplies each node sum by semantic
  R once (`mont_form_fix`). Coefficient ladder uses `jk_i192_mul_i32`
  (max |c| = 140140 = 2^17.1, pinned by new test
  `extension_coefficients_fit_i32`) — 6 partial products vs 12.
  74 → 20 mont-muls/row.
- **`jk_outer_azbz_lazy`** — guard integers are {−1,0,1} in
  production: Az folds Lagrange weights as masked adds (Montgomery is
  linear; generic |a|>1 arm kept for exactness). Bz accumulates
  raw-weight products in standard form, one R2 multiply per stream
  before the store — **stored tables byte-identical**, round loop and
  everything downstream untouched. 81 → 26 mont-muls/row.
- **`jk_outer_claims_lazy`** — 18 boolean columns fold the eq weight
  directly (masked add, Montgomery partial; bool tiles skip the wide
  lanes entirely); 17 wide columns meet the weight raw (standard
  partial, host R fix per column). 52 → 17 mont-muls/row.

## Numbers

Same-window pairs, fixture attribution rig + criterion
(oracle-asserted in both arms before timing; synthetic full-range
i128/u128 lanes stress the sign/raw and generic-guard paths *harder*
than production):

| kernel | 2^24 eager | 2^24 lazy | Δ | 2^22 Δ |
|---|---:|---:|---:|---:|
| t1 | 225.3 ms | 154.9 ms | **−31.2%** | −32.0% |
| azbz | 191.0 ms | 125.9 ms | **−34.4%** | −30.2% |
| claims (criterion, p=0.00) | 107.3 ms | 55.6 ms | **−48.2%** | −45.9% |
| round loop (untouched) | 43.7 ms | 46.1 ms | noise | noise |

**Modeled @2^27** (ratios × traced spans): t1 −0.643 + azbz −0.575 +
claims −0.446 = **−1.66 s**; st1 4.995 → ≈3.34 s (stage vector 5.07 →
≈3.4). Post-cut st1 shape: t1 ~1.42 · azbz ~1.10 · claims ~0.48 ·
rounds ~0.32.

**Scale-transfer:** no resident data grows with rows — the lazy
kernels read/write the SAME buffers byte-for-byte (no new tables, no
new allocations; RSS shape unchanged). The Miller-table failure mode
does not apply; the standard gate kill-switch ABBA @2^27
(`JOLT_METAL_OUTER_LAZY=0` as the B arm) is still the wall evidence.

## Gates (all green)

- metal suites 412/412 (`jolt-kernels -p jolt-dory -p jolt-eval`,
  metal features) — includes the outer lockstep parity tests (every
  round poly + all 35 output claims vs optimized host) and the claims
  device-vs-host oracle.
- byte-diff `prover-fixtures` 20/20 first pass; **metal-armed byte-diff
  (`prover-fixtures,metal`) 20/20 in BOTH arms** (default lazy, and
  `JOLT_METAL_OUTER_LAZY=0`) — both match legacy proof bytes ⇒ arms
  byte-identical e2e.
- clippy `--all --features host -D warnings` clean; release example
  build OK; e2e smoke @2^22 metal 2.41 s.

## Doors closed / notes

- **gpu-util parked door "st1 claimed_inputs device port (~0.6 s)" is
  STALE — remove:** it landed 2026-08-04 (`9d9362aa6`, metal-sat w2);
  the 5.07 vector already contained it. This lane cut the device
  claims kernel another −48%.
- t1+azbz fusion (single record walk): DEAD by protocol order — azbz
  needs the uniskip challenge drawn from the transcript *after* t1's
  message is absorbed; storing raw groups instead would be 608 B/row.
- Remaining st1 floor is record loads (~120 B/row ×2 walks), integer
  group construction, and 9 threadgroup reductions per t1 thread —
  mont-mul mass is now ≤ 27% of original. Parked lever with mechanism:
  multi-row (grid-stride) accumulation to amortize t1's 9 tg-sums and
  the row loads — byte-exact regroup (field addition exact), unpriced.
- Attribution-only spans added (`outer_t1/azbz/round/claims`,
  `MetalOuterRemainder::prepare`, `outer_pair_tables_alloc`,
  `SpartanOuter::output_claims`) — no renames.

## Discipline

One 2^27 profile only (FrBind-gated); iteration at 2^22/2^24 on the
fixture rig; timed decisions: lazy A/B one pair per scale (+criterion
for claims), i32 tweak one confirm run; 20-30 s cooldowns between
arms; all cargo under `lockf` cargo lock, all timed GPU under the GPU
lock; sibling worktrees and `commitment.rs` untouched; no push.
