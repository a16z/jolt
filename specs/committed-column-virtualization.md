# Q3 design note: committed-column virtualization at 2^26 — measured analysis and recommendation

Status: AWAITING APPROVAL (Q3 protocol gate, specs/akita-prover-codesign-2e26.md).
Recommendation up front: **do not implement any virtualization; close Q3.**
The item's premise fails on measured per-column costs, and the largest
recoverable component turns out to be an engineering artifact (kernel load
imbalance, −3.3 s, no protocol surface) now queued as Q1b. Details and
per-candidate analysis below; every claim is file:line-grounded.

## 1. Measured inventory (2^26 sha2-chain, committed mode, K=256)

One native akita commitment group, NOT eight: `OneHotTrace` = 29 uniform
columns of arity 34 (`lattice/packing.rs:49-57`, `lattice/strategy.rs:35-55`),
committed by a single `commit_one_hot_group_owned` call (packed.rs:1441,
jolt-akita scheme.rs:116; mixed arity/K rejected at scheme.rs:123-140).
Catalog: `jolt_fp128_d64_onehot_k256.rs:451` — ppb 2^21, 128 live blocks per
column, 3712 flat block-rows, n_a=6, num_digits_inner=1.

| columns | count | hot entries | marginal commit cost (traced) |
|---|---|---|---|
| InstructionRa(0..16) | 16 | T each | 1.62 s each |
| UnsignedIncChunk(0..8) | 8 | T each | 1.62 s each |
| UnsignedIncMsb | 1 | **T** (boolean value, but committed as a full K=256 one-hot; only lanes {0,1} used — packed.rs:710, no dense/K=2 backend path exists) | 1.34-1.62 s |
| BytecodeRa(0..2) | 2 | T each | 1.62 s each (2.67 s total) |
| RamRa(0..2) | 2 | **~7-8 % of T** (None on non-memory cycles emits no entry — akita blocks.rs:231-237; density fit from per-thread CPU: 6.75 vs 43.38 CPU-s, corroborated by build_blocks 0.03-0.045 s vs 0.27-0.38 s) | **~0.12 s each** |

Total 44.0 s commit ≈ 27.15 dense-column-equivalents × 1.62 s. The campaign
spec's "1.55 s per committed column" uniformity assumption (45/29) is wrong
in composition: RamRa is nearly free already.

**Kernel imbalance discovered during this audit (not a protocol item):** the
merge sweep statically partitions 3712 block-rows into 16 contiguous ranges
(`column_sweep.rs:404`); one thread receives the nearly-empty RamRa range and
finishes at 25.6 s while the rest run to 62.3 s — 15 busy threads for 74.7 %
of the merge window, 52.2 idle core-seconds = **3.26 s of wall**. An
entry-count-weighted partition recovers this with byte-identical output.
This is also the resolution of Q1's 61-vs-50 ns/accum e2e-vs-bench gap (the
bench uses uniform blocks; e2e has the RamRa hole). Queued as **Q1b [ENG]**.

## 2. Per-candidate analysis

Protocol context (agent-mapped, file:line in the reports): committed RA
chunks are opened by exactly three sumchecks each — the family
virtualization (`BytecodeReadRaf` / `RamRaVirtualization`), `Booleanity`,
and `HammingWeightClaimReduction` — then one stage-8 PCS opening. All
upstream Twist/Shout machinery works on the *virtual* full-width ra.

### 2a. BytecodeRa(0..2) — REJECT (soundness-circular)

Savings if dropped: 2.67 s. The binding chain is:
R1CS/Spartan pins PC → read-raf's RAF legs pin
`Σ_{j,k} ra(k,j)·eq(j)·Int(k) = PC@SpartanOuter` and `@SpartanShift`
(read_raf_checking.rs:2055-2058, 1843-1852; Int(k)=k, identity_poly.rs:66-77)
→ booleanity (ra²=ra, booleanity.rs:594-613) + hamming weight
(Σ_k ra_i(k,·)=1, hamming_weight.rs:241) make ra a genuine unit indicator, so
`Σ_k k·ra(k,j)` IS the hot index and the RAF claim forces hot_index(j)=PC(j).

The committed one-hot columns are therefore not derivable data — they ARE
the range/decomposition argument for PC. Any replacement derivation
sumcheck must independently prove "PC(j) decomposes into d in-range 8-bit
digits" over T cycles, which is exactly what one-hot commitment + booleanity
+ hamming provides; every standard alternative (Lasso-style lookup, digit
range check) reintroduces an equivalent committed object. Re-encoding as one
K=2^13 column halves entries but multiplies the column's ring-element
footprint ×32 (live blocks 128 → 4096; ppb is capped by the expanded-A
budget, 12.9 GB at 2^21), inflating fold cost and proof size — the same
geometry wall that deferred K=2^16 to Q7. Dead.

### 2b. RamRa(0..2) — REJECT (payoff evaporated)

Savings if dropped: **~0.24 s**, not the ~3.1 s the uniform model implied —
the columns are ~7-8 % dense because `remap_address` yields None on
non-memory cycles and None emits no committed entry. Same circularity as
bytecode besides (RafEvaluation pins `Σ_k ra(k)·unmap(k) = RamAddress`,
raf_evaluation.rs:44-51, with booleanity + the RamHammingWeight-targeted
hamming leg supplying one-hot-ness for exactly the 0-or-1-access semantics,
hamming_weight.rs:228-243). Not worth any protocol surface at 0.24 s.

### 2c. UnsignedIncMsb — VIABLE BUT NET ~−1 s; NOT RECOMMENDED

The only structurally clean candidate. The msb column m is opened by
exactly two sumchecks (agent-verified): Booleanity (m²−m leg,
lattice/relations/booleanity.rs:64-65) and the stage-7 hamming decode leg,
whose identity `Σ_j 2^{8j}·G_chunk_j(ρ) + 2^64·G_msb(ρ) = FusedInc(r_cycle)
+ 2^64` (lattice/relations/hamming_weight.rs:158-189; verifier point-sharing
hinge stage7/verify.rs:180-188) is the msb's only tie to the fused stream.

Derivation design: define m virtually as
`m(t) := (FusedInc(t) + 2^64 − Σ_j 2^{8j}·val_j(t)) / 2^64`, where
`val_j(t) = Σ_k k·chunk_j(k,t)` is linear in the committed chunk columns.
Replace the two legs with:
1. decode leg drops its G_msb term; the verifier solves the identity for
   the msb contribution instead of receiving it (the input side already
   carries FusedInc + 2^64);
2. a NEW degree-2 cycle sumcheck `0 = Σ_t eq(r,t)·(m(t)² − m(t))` whose
   output claim reduces to (a) chunk-column value-form openings at the bound
   point and (b) a FusedInc opening at a fresh point — the latter cascades
   into the read-raf fused stages (LATTICE_N_STAGES 9 → 10, one more staged
   val + gamma slot, geometry/bytecode.rs:216-255) or a second claim
   reduction.

Cost bound: −1.34 to −1.62 s commit, +0.3-0.5 s for the new T-scale
degree-2 sumcheck and the extra fused stage ⇒ **net ≈ −1 s**. Surfaces
touched: packed witness layout + digest tags (strategy.rs:87-91, serde tags
proof_parts.rs:281-285), lattice booleanity params/instances, hamming
decode weights and input terms, read-raf fused-stage count, verifier
stage6b/7 wiring, catalogs (29 → 28 columns ⇒ gate 3 + fresh
drift/coverage), and the BlindFold caveat below. High surface-to-payoff
ratio; the accept bar (≥1.5 s) is not met by the expected net.

BlindFold note (pre-existing, found during the audit): the lattice columns
are asserted-out of zk at read-raf (`debug_assert!(num_val_stages ==
BASE_N_STAGES)`, read_raf_checking.rs:1282) but
`HammingWeightClaimReduction`'s zk constraints (hamming_weight.rs:367-478)
silently omit the inc-chunk/msb/decode terms with no guard — if zk+akita
are ever combined the constraints are wrong before any Q3 work. Worth an
assert regardless of this note's outcome.

### 2d. Per-family K reshapes (inc 5×13-bit, bytecode K=2^13) — REJECT

Entry-count arithmetic is attractive (inc 9T→5T, bytecode 2T→1T) but the
ring-element footprint of a one-hot column scales with K (T·K/D ring
positions), so live blocks per column scale ×32 at K=2^13 under the
ppb-capped-by-A-memory constraint (12.9 GB at 2^21). Fold/opening work and
proof size scale with block count; booleanity/hamming/stage-8 single-point
batching fragments per K. This is the Q7 geometry analysis at smaller K —
already deferred to ≥2^28 with seed-streamed A.

## 3. Recommendation

1. **Close Q3 with no implementation.** Corrected candidate payoffs
   (BytecodeRa: circular; RamRa: 0.24 s; msb: net ~−1 s below the accept
   bar; K reshapes: Q7-deferred geometry) do not justify protocol surface.
2. **Take Q1b [ENG] instead**: entry-weighted merge-sweep partition,
   expected ≈ −3.3 s at 2^26, byte-identical output, akita-side only —
   implementable immediately without this note's approval.
3. Optional hygiene (no perf): add the missing zk+lattice guard in
   `HammingWeightClaimReduction` constraints (§2c).

If the goal (≤65 s) is still unmet after Q1b/Q4/Q5/Q6, the msb elimination
(§2c) is the only Q3 candidate worth revisiting, at expected net −1 s.
