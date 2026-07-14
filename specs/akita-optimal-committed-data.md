# Optimal committed-data form for the packed (Akita) path

Working note for co-designing the W_jolt commitment format with the akita
backend (local checkout at `../akita`, baseline upstream `33169b8d`). The
question: what is the cheapest sound way to commit and open Jolt's packed
one-hot witness data? This is the canonical record of the investigation —
it supersedes the leads section of
`specs/akita-perf-plan.md`.

## How we got here (measured trajectory, sha2-chain 2^20)

| date | format / change | prove | verify | commit |
|---|---|---|---|---|
| 2026-07-12 | sparse-unit UNION, `log_k_chunk = 4` (~60 slots, 2^30 domain), backend `c8290922` | **81.3 s** | 1.94 s | baseline (`380dad2f3`) |
| 2026-07-13 | batched one-hot members, `log_k_chunk = 8` (29 × 28v polys, one group, one batched open) | **13.8 s** | 0.23 s | `52decbb76` |
| 2026-07-13 | backend bump to upstream tip `33169b8d` (runtime-ring API; audited SIS resizing `n_a` 5→6 at 28v) | **~17 s** | 0.16 s | `254c33b3a` |

Dory reference, same machine/session: 13.4 s / 0.67 s. Target: ≤ 6.6 s
(0.5× dory).

Tried and rejected, with reasons (all measured, none speculative):

- **Object-parallel reduction rounds** — neutral; the round sweeps are
  memory-bound and concurrent streams contend.
- **Paired-round sweep-halving** — neutral; halves traversals but doubles
  split-eq lookups per visit, and lookups are the bound (`f6e303292`).
- **`D64OneHotMultiChunkW2R2` preset** — 21 s at 2^20, worse than plain.
- **`D128OneHot` preset** — not K=256-compatible (chunk-size default 1).
- **Union revival on the de-tiered backend** — 72 s commit / 112 s open at
  33v (see the decisive-measurements section).
- **One-hot-only setup + moved (never cloned) group hints** — landed;
  killed a ~7 GB setup spike and an OOM-class clone at stage 8.

## The data versus the committed object

One committed `Ra` member answers "which of `K = 256` addresses did cycle `t`
select" over `T = 2^20` cycles:

| | size |
|---|---|
| information content | `T` one-byte symbols ≈ 1 MB / member |
| PIOP object (one-hot multilinear extension) | `K·T = 2^28` cells, ≤ `2^20` ones |
| whole `W_jolt` (~29 members at 2^20) | ~29 MB of data → ~2^32.9 committed cells, ~3.0·10^7 ones |

The PIOP's sumchecks emit evaluation claims on the **28-variable one-hot
extension**, not on the symbol vector — so whatever we commit must support
openings of that extension. The 8 extra variables are pure encoding.

## The dimension rent (why 2^28 hurts)

The backend commitment is an Ajtai/SIS map `t = A·w`, `A ∈ R_q^{n_a × m}`:

- the 2^28 cells pack into `m ≈ 2^22` ring-element columns (D = 64
  coefficients each, times commit digits);
- committing a one-hot poly costs `n_a` ring-adds **per one** (sparse-aware);
- `n_a` is set by the audited SIS tables and **grows with `m`** — more
  columns give an attacker more freedom, so binding against *arbitrary*
  (non-one-hot) openings must be priced for the whole domain:

| member `num_vars` | 24 | 26 | **28 (ours)** | 30 | 32 | 33 | 34–36 |
|---|---|---|---|---|---|---|---|
| `n_a` (one-hot tables, 138-bit L∞) | 5 | 5 | **6** | 6 | 6 | 7 | 7 |

So commit CPU ≈ `ones × n_a × c_flavor`, and we pay security rent on a
2^28-cell domain to store 1 MB of furniture. No one-hot re-packing escapes
this: ones are layout-invariant, and `n_a(num_vars)` only worsens as objects
grow (one big 2^33 union ⇒ `n_a = 7`). Power-of-two padding is a non-issue
in the current format (the kernel skips zeros; members aren't padded; `T`
pow-2 is PIOP-forced).

## Union vs grouped batch: where the 33 variables go

The union and the grouped batch hold the same 2^33 cells of data; they
differ in where the member index lives, and the SIS accounting charges for
that placement:

| | member index | address dimension | SIS witness span | `n_a` basis | fold depth |
|---|---|---|---|---|---|
| union (one 33v object) | in the witness | in the witness | 2^33 cells, one instance | m(33v) → 7 | 33 vars |
| grouped batch (today) | protocol (ρ-RLC at open) | in the witness | 30 × 2^28, separate images | m(28v) → 6 | 28 vars |
| committed-symbols (proposed) | protocol (ρ-RLC) | protocol (sumcheck leg) | 30 × 2^23 | m(23v) ≤ 5 | 23 vars |

One SIS relation `t = A·w` must resist collisions across **all columns of
that instance**; separate images `t_i = A·w_i` confine the attacker to one
member's columns, so each instance is priced at its own `m`. The union's 5
selector bits carry no data, yet the matrix width, digit planes, block/NTT
machinery, witness residency, and fold levels all scale with them — that is
the measured 72 s commit / 112 s open. The batch re-introduces the member
index at open time as a ρ-combination of *claims*, which is linear and
cheap.

The committed-symbols mode is the same externalization one level deeper:
a member's 8 address bits are also deterministic selector structure (a
function of the symbol), currently sitting inside the witness as 255
zero-cells of domain per hot cell. Moving them into an opening-time
sumcheck relation is to the address dimension what grouping was to the
member dimension.

## Measured constants (sha2-chain, 2^20, M-series 8-core)

| path | per-one commit | commit wall | open wall | notes |
|---|---|---|---|---|
| sparse-unit union, old rev `c8290922` (k=4, 2^30) | ~270 ns | ~17 s | ~59 s | open hit the (now deleted) tiered ≥30-var path |
| sparse-unit union, new rev (k=8, 33v, 3.0·10^7 ones) | ~2.4 µs | 72.2 s | 112.2 s | de-tiered and still dead — the sparse kernel itself is superlinear in `num_vars` |
| one-hot members, old rev (29 × 28v, n_a=5-era) | ~1.5 µs | 6.5 s | 3.5 s | iteration-1 format |
| one-hot members, new rev `33169b8d` (n_a=6) | ~1.9 µs | 7.5–8.5 s | 4.0–4.7 s | current (in-pipeline vs standalone bench); commit CPU 57 s at 7.6× parallel |

Two corrections this table forced on our intuitions:

1. **The sparse-unit (Full-flavor) commit kernel is ~7–10× cheaper per one
   at small `num_vars`** (CRT-NTT pointwise digit accumulation vs `n_a × D`
   wide coefficient-adds + rotation/LUT bookkeeping) — but the advantage
   does not survive scale: at 33 vars its per-one cost is ~2.4 µs, worse
   than one-hot at 28 vars. Per-one constants are only meaningful at a
   stated `num_vars`.
2. The one-hot kernel runs ~6× above its arithmetic floor (~0.3 µs/one) —
   vectorization headroom exists independent of any format change.

## Candidate forms

| form | commit est. @2^20 | mechanism | cost / risk |
|---|---|---|---|
| today: 29 one-hot members, 28v | 7.5 s | — | — |
| packing-only optimum (cycle-split to 26v) | ~6.3 s | `n_a` 6→5 | ×4 members/claims; weak |
| K=2^16 fusion (`log_k_chunk = 16`) | ~4.5 s | ones ×0.52, `n_a`→7 | chunk-65536 preset + regenerated schedules; u16 indices exist upstream |
| ~~sparse-unit union on de-tiered backend~~ | ~~1–2 s?~~ | small-`num_vars` kernel constant | **measured DEAD below: 72 s commit / 112 s open at 33v** |
| **committed-symbols mode (the optimal path)** | **~2.8 s stock, ~1 s with a bit-aware backend preset/kernel** | commit the bit-planes (2^23 cells, m ≈ 2^17) via the dense path: per symbol `n_a·D` mults/8 ≈ 144 add-eq vs one-hot's 384 (2.7× stock); unit-norm sizing + narrow-CRT accumulation close the rest | opening gains a degree-9 one-hot-evaluation sumcheck leg + bit-booleanity |
| multi-group merge of advice/W_prog (orthogonal) | −0.5–1 s of opens | upstream #275 | protocol plumbing only |

### The committed-symbols co-design, in one paragraph

Commit the *data* (per member: 8 bit-columns `b_j` of the symbol vector,
2^23 cells), and move the one-hot *semantics* into the opening: the claim
"the one-hot extension evaluates to `v` at `(r_addr, r_cycle)`" is

```
v = Σ_t eq(r_cycle, t) · Π_{j<8} ( r_j·b_j(t) + (1−r_j)·(1−b_j(t)) )
```

— a degree-9 sumcheck over `T` whose final claims are evaluations of the
committed bit-columns, plus a `b² = b` booleanity leg (which replaces part
of what the PIOP's hamming/booleanity machinery pays for today). Binding is
clean: the SIS commitment pins the bit-columns; the extension is a
deterministic function of them. The SIS instance shrinks 32× in columns,
dropping `n_a` below every sizing step that hurts us, and the commit becomes
bounded by reading the data. This subsumes fusion (which only halves ones)
and all packing games.

## Decisive measurements (this doc's data section)

Bench: `jolt-akita` `flavor_bench` (`BENCH_LOG_T`, `BENCH_SLOTS`,
`BENCH_SKIP_UNION`, `BENCH_SKIP_ONEHOT`), production shape `log_t = 20`,
30 slots, against the local de-tiered backend.

Measured 2026-07-13 against the local backend at `33169b8d`:

| case | setup | commit | open | verify |
|---|---|---|---|---|
| one-hot batch (28v, 30 polys) | 2.2 s (one-hot-only) | **8.5 s** | **4.0 s** | 0.16 s |
| sparse-unit union (33v, 3.0·10^7 ones) | 35.6 s | **72.2 s** | **112.2 s** | 5.7 s |

**Verdict: the union path is dead, de-tiering notwithstanding.** The sparse
kernel's cheap per-one constant does not survive 33 variables — its per-one
cost grew to ~2.4 µs (10× its small-object constant; the block/NTT machinery
is itself superlinear in `num_vars`) and the open remains pathological.
The one-hot flavor stays the vehicle, and the **committed-symbols mode is
the path** to the ~1–2 s commit budget the 6.6 s end-to-end target implies.
A useful corollary: per-one kernel constants are only meaningful at a stated
`num_vars` — both flavors degrade with domain size, which is one more
argument for committing the 2^23-cell data instead of the 2^28-cell
encoding.

## The no-protocol-change floor

If the format and protocol stay exactly as landed (29 one-hot members,
batched commit, current reduction), the remaining knobs and their stacked
floor against the measured breakdown (commit 7.5 + open 4.7 + reduction 1.9
+ PIOP 2.3 + glue 0.5 ≈ 17 s):

| knob | owner | expected | confidence |
|---|---|---|---|
| one-hot accumulate vectorization (NEON, accumulator batching, narrow-CRT) | akita, kernel-only | commit → 2–3.5 s | medium-high (6× over arithmetic floor is measured) |
| `n_a` sizing at 28v (6 → 5?) under the 138-bit tables | akita, config | ×0.83 on commit, partial on open | unknown — estimator question |
| open-phase parallelization (~1.5× utilization today) | akita, same proof bytes | open → 1.5–2.5 s | medium |
| multi-group merge of advice/W_prog (#275) | jolt plumbing | −0.5–1 s | high |
| ~~reduction sweep-halving (paired rounds)~~ | jolt | **measured neutral** — the sweep is bound by split-eq table accesses (4/position paired vs 2×2 single), not arithmetic; object-parallel rounds also measured neutral earlier. The reduction window (~1.9 s) is access-bound from every angle | — |
| PIOP stage polish | jolt | −0.3–0.5 s | medium |

Stacked: **best case ≈ 6.5–7 s (0.5× dory), realistic ≈ 8–9 s (0.6×)**.
The two dominant levers are akita-side; without protocol changes this is a
backend-engineering exercise with a sizing dependency. The committed-symbols
path exists to buy margin and move the decisive lever jolt-side; the
Phase-0 gate below decides which world we operate in. The three akita asks
(kernel, sizing, open parallelism) are needed in both worlds and should be
filed regardless.

## Refactor plan for the committed-symbols path

Phased, each with gates; the critical risk is prototyped first.

**Phase 0 — kill-switch prototype (the eq-product sumcheck).** The opening
must prove, per member, `v = Σ_t eq(r_c,t)·Π_j(r_j·b_j(t)+(1−r_j)(1−b_j(t)))`
— degree-9 rounds over 20 variables. This is where the saved commit cost
could reappear (naively ~10^9 mults for 29 members). Two structural weapons:
the factors are 2-valued (`b_j ∈ {0,1}` ⇒ factor `∈ {r_j, 1−r_j}`), so
unbound regions collapse to a 256-entry symbol table `E[sym]`; and Jolt's
`ra_virtual`/`PrefixSuffixDecomposition` machinery already proves this shape
of `eq × Π` sumcheck efficiently. Deliverable: a standalone bench at
production shape (29 members, T = 2^20, one batched instance).
**Gate: ≤ ~3 s wall, else the path dies** → fallback is K=2^16 fusion plus
the kernel asks.

**Phase 1 — commitment format (jolt side; commit needs no backend change).**
Each member commits its bit-plane polynomial: 8 symbol bits packed as
`(bit_plane ‖ cycle)` = one 23-var poly per member (member count stays 29;
statements stay per-member). Committed through the existing dense
small-coefficient path — single-digit decomposition, ~2^17 ring columns per
member. Honest arithmetic: dense pays `n_a·D` ring-*mults* per column with
64 bits packed per column ⇒ ~144 add-equiv per symbol vs one-hot's 384 —
**~2.7× stock (commit ≈ 2.5–3 s)**. The rest of the win (→ ~1 s) is the
backend co-design item (Phase 1b): a unit-norm/bit-aware preset (stock
full-flavor tables price generic digit norms, `n_a = 6` at 23v where the
tiny bit norm should admit ~4–5) plus narrow-CRT accumulation for the bit
path. Assembly is a copy of the
symbols we already hold. Setup shape `(23, 29)`.

**Phase 2 — opening reduction (the surgery: jolt-openings + verifier +
claims).** The packed reduction gains a degree-9 product integrand for
bit-plane members, mixed with degree-2 singleton objects in one batched
sumcheck: round polynomials become variable-degree (proof format change);
the reduction binds 20 vars instead of 28 (the address dimension leaves the
domain — the same externalization as union→group); per-member final claims
become bit-plane evaluations, opened by the backend as small dense objects
(23-var fold depth: open ≈ 2.5–3.5 s, down from 4.7); booleanity legs
`b² = b` RLC-fold into the same sumcheck (with boolean bits, exactly-one-hot
per row is structural). Full FS mirror + e2e re-pins; no akita fixtures pin
bytes yet.

**Phase 3 — PIOP pruning (separately gated).** With one-hotness structural,
the lattice booleanity / hamming-weight-1 machinery becomes partially
redundant (~1–2 s candidate), but it reshapes the claim flow feeding the
leaves — its own iteration.

**Phase 4 — multi-group merge (orthogonal).** Advice/W_prog into the same
commitment via upstream #275 (~0.5–1 s of singleton opens).

**Backend co-dev track (parallel, `../akita`):** audited schedule entries
for the `(23, ~29)` dense keys; NEON on the one-hot accumulate (fallback
path + anything staying one-hot); if Phases 1–2 prove out, an upstreamable
first-class symbols/bit-plane input mode.

Budget forecast at 2^20 (s):

| phase | commit | open | reduction (+new leg) | PIOP | total |
|---|---|---|---|---|---|
| today | 7.5 | 4.7 | 1.9 | 2.3 | ~17 |
| after 1+2 (stock backend) | ~2.8 | ~3 | ~2.5–4 (gated) | 2.3 | ~11–12.5 |
| + 1b (bit-aware preset/kernel, akita side) | ~1 | ~3 | ~2.5–4 | 2.3 | ~9–10.5 |
| after 3+4 | ~1 | ~2.5 | ~2.5 | ~1.5 | ~7.5–8.5 |
| + kernel/serial-fraction asks | | | | | → 6.6 target |

Ownership: Phases 0/2/3 are entirely jolt-side; Phase 1 is jolt-side with
at most a regenerated schedule entry; **Phase 1b is the akita co-design
item** (unit-norm preset + narrow-CRT bit kernel + schedule regen); Phase 4
is jolt plumbing over existing upstream APIs. NEON and the fusion preset
remain akita-side fallback tracks.

## Current state and code map (for the cleanup phase)

The pipeline today: `W_jolt` = one commitment object of ~29 row-major
K=256 `OneHotPolynomial` members (one per committed column), committed by
`AkitaScheme::commit_one_hot_group` and opened by one native batched proof;
`W_prog` and the advice byte columns remain singleton objects. The packed
reduction settles every leaf claim (one per member, plus the singletons')
in a single sumcheck, after which groups open natively.

| piece | where |
|---|---|
| member assembly (`assemble_one_hot_members`) | `jolt-prover-legacy/src/zkvm/lattice.rs` |
| prove pipeline (`prove_packed`, member statements, groups) | `jolt-prover-legacy/src/zkvm/akita.rs` |
| leaf-point mapping (`one_hot_member_point`, symbol‖cycle → cycle‖lane; msb as lanes {0,1}) | `jolt-claims/src/protocols/jolt/lattice/packing.rs` |
| reduction sumcheck + grouped native tail (`prove/verify_packed_openings`, `PackedObjectGroup`, `open_batch`) | `jolt-openings/src/packing.rs` |
| backend adapter (`commit_one_hot_group`, `open_batch`/`verify_batch`, one-hot-only setups) | `jolt-akita/src/{scheme,native_batching,adapters}.rs` |
| verifier mirror | `jolt-verifier/src/stages/stage8/packed.rs` |
| flavor/measurement bench (`flavor_bench`, `BENCH_*` env knobs) | `jolt-akita/src/scheme.rs` |
| packed config (`log_k_chunk = 8` forced under `cfg(feature = "akita")`) | `jolt-prover-legacy/src/zkvm/config.rs`, `preprocessing.rs` |

The `akita-*` deps build from the local `../akita` clone (committed on this
branch for co-design; pin a fork rev before the branch travels). (This spec
lives in `specs/` — the repo `.gitignore` swallows `docs/` directories.)

## Status (2026-07-14): paused at the decision gate

Optimization is paused with the state at ~17 s (1.27× dory). The cheap
jolt-side knobs are exhausted (see the floor table); what remains is:

1. **Phase 0 of the refactor plan** — the eq-product sumcheck prototype,
   whose ≤3 s gate decides between the committed-symbols path and the
   backend-knob path;
2. **the three akita asks** (kernel vectorization, open parallelization,
   `n_a` sizing), needed in both worlds;
3. an engineering follow-up independent of the above: a clean
   **commitment-strategy seam** in jolt-prover/jolt-verifier so the packed
   path can swap between the single-object (union/full packed polynomial)
   and grouped-members strategies without surgery — both formats have now
   existed, and the next format (bit-planes) is on the table, so the
   abstraction should make strategy changes cheap.

## Engineering debt tracked for future PRs

Identified during the 2026-07-14 simplification pass; each is a
self-contained PR candidate.

1. **Fold the byte-column reconstruction twins.** `advice_bytes.rs`
   (untrusted + trusted) and `program_image_reconstruction.rs` are three
   instances of one protocol object (γ-batched byte-column reconstruction
   over `(byte ‖ place)` with a pre-bound word point), and the verifier's
   `stage8/reconstruction.rs` mirrors all three. Parameterizing one instance
   over `(kind, claim source)` is ~−350 lines prover-side and ~−300–400
   verifier-side, and the bit-plane strategy will want the parameterized
   form anyway.
2. **Narrow the legacy prover's PCS bounds.** `JoltCpuProver` and its main
   impl demand `StreamingCommitmentScheme + ZkEvalCommitment` for
   everything, so the packed path must stub both (the `AkitaPackedScheme`
   panic shims). The packed path only uses `gen_from_elf`,
   `prove_batched_sumcheck`, and the stage-1–5 drivers — splitting the impl
   so those sit under a plain `CommitmentScheme` bound deletes the streaming
   and zk-eval shims and decouples future packed provers from the base
   commit pipeline. Deferred: the method-closure move is ~1,000 lines of
   shared-file surgery, wrong to bundle with other changes.
3. Smaller items: `indexed_family` hoist in base `clear_claims` (~−40);
   stage-7 address-phase scheduling triplication (needs a shared trait or
   local macro — judged not clearly better today); `Arc`'d commitment bytes
   (clone cost is ~ms; churn not yet justified).

## Open questions for the akita side

1. `n_a = 6` at 28 vars under the 138-bit L∞ tables — is 5 admissible under
   a refined estimate for this shape?
2. A chunk-65536 one-hot preset (u16 indices) + regenerated schedules, if
   fusion stays interesting after the committed-symbols spike.
3. NEON vectorization of the one-hot accumulate path (~6× above arithmetic
   floor today).
4. The committed-symbols opening leg: best placement inside the staged
   opening sumchecks (stage-1 fuse vs a dedicated pre-stage).
