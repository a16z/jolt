
## 2. The matrix-MLE problem: Ã(rx,ry), B̃, C̃ of the verifier-algebra R1CS

R1CS after §3 moves the 36.3k Straus digit booleans out: m ≈ 9.9k algebra (plan-relation §2, M) + ≈600 Dory Fr scalars + ≈100 heads/IO ≈ **10.6k → 2^14
rows**; n = **2^18** (witness index space = T1 row space, §3 L1/L2); nnz(A)+nnz(B)+nnz(C) ≈ 4–5 per row (Horner 4, eq step 6, product 3) ≈ **45–55k →
2^16** (E). Native `linear_form_bilinear_eval` (`jolt-r1cs/src/constraint.rs:265`) is O(nnz) ≈ 50k mults ≈ **1.0 M gas** on-chain (E) — the item to kill.
Spartan placement in the stream is fixed by data dependence: outer (needs τ) = stage-A member (14 rounds head-aligned, 0 B); inner (needs rx) = stage-B
member (18 rounds, 0 B); anything needing (rx, ry) can only run in stage C or later.

| option | mechanism | proof bytes | on-chain | prover | VK | verdict |
|---|---|---:|---:|---:|---|---|
| **(a) SPARK, LogUp memory checking, hosted in stage C** | key-time commit `row(k), col(k), valA/B/C(k), m_x(i), m_y(i)` (packed: 1–2 G1). Per proof: `E_x(k)=eq(rx,row(k))`, `E_y(k)=eq(ry,col(k))`, helpers `h_Lx = 1/(α−row−βE_x)`, `h_Ly`, `h_Rx(i)=1/(α−i−β·eq(rx,i))`, `h_Ry` (6 columns, ≤2^18, one packed group). Stage-C members: Σ_k (ρ_a valA+ρ_b valB+ρ_c valC)(k)·E_x·E_y (deg 3, 16 rounds), 4 zero-checks `h·(α−…)=1` (deg 3 with eq), 2 LogUp sums Σ_k h_L = Σ_i m·h_R (linear); +3 slot-tensor rounds bind the group's slot bits so each group yields ONE claim | stage C deg 2→3: +23×32 = **704** · claims 2 × 32 = 64 · commitment 32 → **≈800 B** | eq(rx,r′), eq(ry,r″), identity MLE, LogUp assembly ≈500 ops (10k) + 2 ecMul in the RLC (15k) → **≈25k gas** | 6 columns ≤2^18 (≈0.15 s commits) + 3 small members ≈ **+0.3 s** | +2 G1 | **pick** |
| (b) uniform / structured R1CS (Jolt `R1csKey::evaluate_matrix_mles` pattern, `jolt-r1cs/src/key.rs:255`: eq(cycle)·M̃_local) | block catalog (plan-relation §2, M counts): Horner steps 1,229 (1 constraint, shift-1 acc), eq steps ≈700 (2), LT/EqPlusOne steps ≈100 (3–5), γ-power chains ≈300 (1), Lagrange kernels ≈10×30, heads/products ≈600 (irregular), **54 table-MLE gadgets ≈6,000 (54 distinct formulas, irregular)**. Uniform share ≈ 35–40%; broadcast challenges need per-round mask polys (irregular round lengths d ∈ {2,3,5,10}) → O(#rounds·log) ≈ 3k ops | 0 B for the uniform part; the irregular 6.6k rows still need (a) or O(nnz≈25k) on-chain | ≈ 60k gas uniform part + 500k irregular (or (a) anyway) | padding ≈1.5× rows (2^15), negligible | — | reject: does not remove (a); W4-R emitter is a declarative walk, not block-structured (≈3–5 agent-days to restructure) |
| (c) index-committed product form: `E_x(k)=Π_j (rx_j b_j(k)+(1−rx_j)(1−b_j(k)))` over 14+18 key-time bit columns | no LogUp; but zero-checks are degree ≥4 with eq → cannot sit in stage C (deg 3): own 16-round deg-5 stage = 2,560 B, or 30 degree-2 chain columns (2 groups, 30 claims) in stage C | ≥ 1.7 KB | ≈ same as (a) | +30 columns ≈ 0.5 s | +2 G1 | reject on bytes |
| (a′) second tiny sumcheck without memory checking | `Σ_k A_k eq(rx,row(k)) eq(ry,col(k))` with committed indices is (c) or degree-32 round polys (14 KB) | — | — | — | — | not sound/cheap as stated; (a) is its sound form |

Why 800 B and not ≈3 KB: the SPARK sumchecks are head-aligned members of the existing 23-round stage C (0 round bytes beyond the degree bump), the
witness columns are slots of one packed group whose slot bits the member binds itself (1 claim), key polys are VK commitments entering the single
eq-weighted RLC (0 B), and the whole R1CS check already shares the one HyperKZG opening (W is a slot). Moving SPARK to a fourth stage costs 1,536 B
+ a second point reduction; running Spartan as a prefix stage costs 2,240 B. Both worse.

## 3. Public MLEs the verifier must evaluate — make every one O(log) or ≤256-entry sparse

| public object (today) | size | today's on-chain cost | design | on-chain after | bytes | prover |
|---|---:|---:|---|---:|---:|---:|
| T2 Straus digit selectors δ(k,step,j) (N1: 12,224 selectors, 61,120 bits, 3,728 fixed operand offsets, 11 shift relations — M) | 12k×17 | Σ_{k,step} eq(r,(k,step))·eq(r_j,digit) ≈ 20k ops ≈ 400k gas (plan-v3, E) | **committed one-hot columns** δ_j, j<17 (signed w=5), over the online-op rows: booleanity + Σ_j δ_j = 1 as stage-A row constraints (deg 2); selected operand = Σ_j δ_j(u)·T2[table(k(u), j, c(u))] | selector kernel Σ_j δ_j(r_A)·eq(r_j, j)·eq(r_k,u_k)·eq(r_c,u_c) ≈ 17×(3·log) ≈ 300 ops; the δ_j(r_A) come out of column batching | 17 columns → 254 total (fits 256, no extra rounds); 0 claims | 17 bit columns × 2^19 ≈ 0.04 s |
| digit ↔ GLV mini-scalar binding | 143 scalars | in R1CS as 36.3k booleans (2^16 R1CS) | scalar_k = Σ_s Σ_j j·2^{5s}·δ_j(k,s): stage-A linking member Σ_u pow(u)·digit(u) = Σ_k ρ^k W[base+k]; pow = Π_i(1−x_i+x_i·2^{5·2^i}) closed form; scalar wires contiguous in W | ≈100 ops | 0 B (head-aligned, 1 stage claim recomputed) | negligible; R1CS shrinks 47k → 10.6k rows |
| fixed operand offsets (table row of digit j for base k) | 3,728 offsets → nnz ≈ 2.5M (u,v) pairs | O(nnz) or SPARK at 2^22 — impossible | **strided T2 row layout**: index = (region ∥ k ∥ window/j ∥ coeff) with power-of-two extents per region (GT/G1/G2 table, GT/G1/G2 online, Miller, FE); non-power-of-two counts only at the top (k < 153/40/40 truncates) → wiring = eq on shared bits × selector kernel | included above | 0 | rows: coeff 12→16 (+33%), windows 13→16 / 26→32 / 14→16 (+15%), table j 17→32 (+27k rows) → **≈1.4–1.7× → 2^19** (E; N1 compact = 261,550 → ≈400k) |
| 11 shift relations (accumulator/table/point-state) | 11 | O(rows) declared edges (0.3 s native) | shift by a constant stride in the strided layout → `EqPlusOne`-class kernels | 11 × ≈60 ops | 0 | 0 |
| T1 wiring: 66 wired inputs, 8–12 row shapes (same-G, round permutation, message schedule, chaining) | 116 rows/compression, irregular | O(rows) per shape | rows per compression 124 → **128** (`table.rs`: 112+8+4 = 124 today, M): index = (compression ∥ position); shape = eq(r_comp,u_comp)·P̃(r_pos,u_pos) with P̃ a 128-entry map (one source per position), chaining = EqPlusOne(comp) | ≈12 shapes × (128 + 2×128 table) ≈ 5k ops (E) → 100k gas; lever: closed forms from Blake3's G-index/rotation regularity ≈1.5k | 0 | rows 1,980×128 = 253k (97% of 2^18); with L3+L4 (C_tot ≈1,150) 147k |
| L1 absorbed-Fr link (1,253 Fr → 10k word links), L2 challenge link (366), L4 chunk↔bytes, L5 scalars | 10k+ | per-segment affine maps ≈372×20 = 7.4k ops, or SPARK +400 B | **W index space = T1 row space** (n = 2^18): the R1CS variable of the Fr absorbed at (comp, byte 0/32) sits at W index (comp, 0/8); challenge variables at (comp, 120); with L4 (no labels) every Fr starts at byte 0 or 32 of a block → links are eq-on-low-bits × 8 shift kernels; inner sumcheck 14 → 18 rounds (head-aligned in B, 0 B) | ≈300 ops | 0 | W commit unchanged (sparse zeros free); SPARK col index 18 bits (h_Ry 2^18: 0.14 s) |
| T1/T2 relation coefficients at r_A (γ powers, α, q limbs, sel = row parity, IV constants) | ≈300 | — | already O(1) each; IV/flag constants become wiring constants of position rows (structured) | ≈600 ops | 0 | 0 |
| Dory VK GT constants (χ[k], Δ1R, Δ2R, HT) as T2 public operand rows | ≈ 200–1,000 Fr | O(size) | evaluate at r_A on-chain (≤1k ops) or 1 VK commitment + 1 claim | ≤ 20k gas | 0 / 32 | 0 |
| Spartan public columns x (≈12) | ≈60 nnz | O(nnz) | unchanged (tiny) | ≈1k ops | 0 | 0 |

Result: no O(rows) or O(nnz) public evaluation remains; total public-MLE work ≈ 13k Fr ops (§1 rows 5–12) ≈ 260k gas (E), of which T1 wiring 100k is the
lever. Soundness unchanged from plan-v3 §7 (one-hot δ ⇒ one table entry per selector; kernels are exact 0/1 matrices; LogUp α,β random).

## 4. Best single-layer design — bytes / gas / prover

Shape: rows 2^19 domain (T2 strided; T1 2^18 head-aligned), 254 columns (T1 163 bits + T2 54 u16 + 18 helpers (s=3) + 1 multiplicity + 17 selectors + W),
k = 16 → 16 groups, ℓ = 23; SPARK group + 2 VK groups; outer transcript Keccak256 chained digest; HyperKZG 4-pair form.

| line | count | now (19872523d formats) | free levers (this plan) | unbuilt levers | tag |
|---|---|---:|---:|---:|---|
| packed commitments | 16 (k=16) | 512 | 512 | 256 (k=32; opening 2^24, +10 s) | M format |
| stage A rounds | 19 × deg 5 compressed | 3,040 | 3,040 | 1,824 + ≈96 (KZG-committed rounds: 32 + 2×32 per round, in-flight `CommittedStageProof`; +19 ecMul ≈ +150k gas) | M format / E |
| stage B rounds | D=4 × log 256 = 32 × deg 2 | 2,048 | 2,048 | — (in-flight per-factor `ColumnReduction`: 8 rounds = 512 B but one claim per factor column ≈ 254 × 32 = 8 KB — keep the tensor for the real relation) | M format |
| stage C rounds | 23 × deg 3 (hosts SPARK) | 1,472 (deg 2, no SPARK) | 2,208 | — | E |
| stage member outputs (`stage_claims`) | ≈15 | 480 | **0** — verifier recomputes every member output (`checked_stage_claims` already does) | | M format |
| reduced / residual claims | 4 stage-B + az,bz,cz,W(ry) + 2 SPARK | 256 | 320 | | E |
| HyperKZG opening | 22 fold G1 + 1 witness + 3ℓ Fr | 2,944 | **2,208** — drop `v[2]`: the verifier derives P_{i+1}(r²) from the fold identity it already checks (`scheme.rs` consistency loop), one inversion | ≈700 Mercury/Samaritan-class (O(1) G1, unbuilt 1–2 d) | M format / E |
| public IO | K+12 = 28 challenges of 128 bits | 896 | **448** (two per word) | | M count |
| **wire total** | | **11,648** | **10,784 B** | **9,276** (Mercury) · **8,156** (+committed A) | E |
| EVM calldata | + 32 B per G1 (39 / 21 / 40) | | 12,032 B | ≈9.9 KB / ≈9.4 KB | E |

Gas (Cancun, E; §1 detail): calldata 192.5k · base 21k · transcript 15k · rounds 18k · eq/tensor tables 40k · T1 relation 26k · T1 wiring 100k · T2 relation
80k · T2 kernels 30k · links 20k · Spartan 20k · SPARK 10k · inversions 5k · RLC 19 terms 146k · HyperKZG MSM 26 terms 200k · divisor 23k · 4-pair
pairing 183k → **≈1.13 M**. Levers: T1 closed-form kernels −70k · 3-witness HyperKZG (2-pair) −85k, +64 B, +2 full-size MSMs prover (+5 s @2^23) ·
Mercury-class PCS −200k (MSM + calldata) · k=32 −62k. All four → ≈0.71 M; without the 3-witness trade → ≈0.80 M.

Prover (E, 10 threads, from measured unit costs): G-shape gate 4.95 s @2^17×k=8 = opening 1.35 (M law) + 3.6 non-opening (M). Scaling: non-opening
×4 rows (T2 at 2^19; T1 stays 2^18) ≈ 12–14 s · commits: T2 chunks 54 × 25 ms 1.35 s, helpers 18 × 288 ms 5.2 s, T1 bits 0.14 s, selectors/SPARK/W
≈0.2 s · opening k·2^19 = 2^23 ≈ 10.8 s (law ×2/doubling from 2^20 = 1.35 s M) · witness replay 0.3 s → **≈25 s (k=16)**; k=8: opening 5.4 s → ≈20 s
(+512 B, +123k gas); if T2 fits 2^18 (compact layout with a cheaper stride scheme): ≈13 s (k=16). Native verifier ≈ 5 ms (13k ops + 48 ecMul + pairing).

Verifier contract constants: 4 G2 + 3 G1 (SRS) · 2 SPARK VK G1 · Q̃_T1/Q̃_T2 term tables ≈ 6 KB · 12 T1 wiring maps ≈ 1.5 KB · region/stride
descriptors · Blake3 IV/flags · Dory GT constants (≈ 200–1,000 Fr or 1 G1) · profile digest → 20–30 KB → data contract (EIP-170).

Transcript decision: **Keccak256 chained digest** (`DigestTranscript<sha3::Keccak256, Fr>` — one type alias + `sha3` dep; `transcript-keccak`'s
spongefish duplex is not it). Per event 30 + 6·⌈(64+payload)/32⌉ gas + ≈60 glue → ≈100; 140 events ≈ 15k. Blake2b-F (EIP-152) measured ≈ 700 per
compression incl. Solidity glue (N4 5,443/7 + reduction) → ≈100k for the same schedule. `challenge()` decode (125-bit · 2^{−128}) = 1 mulmod on-chain;
prefer `challenge_scalar` (128-bit BE) for the wrapper's outer transcript to avoid it.

HyperKZG decision: keep the single cubic-divisor witness; on-chain move the divisor scalars to G1: e(B−R−z₀W,[1]₂)·e(−z₁W,[β]₂)·e(−z₂W,[β²]₂)·e(−W,[β³]₂)=1
(4 pairs 183.4k M, 3 ecMul, 0 B, 0 prover). Alternative 3 witnesses (2 pairs 114.7k, +64 B, +2 MSMs ≈ +5 s at 2^23) — take it only if gas outranks
prover in a later decision. Never G2 scalar mults on-chain.

## 5. Build lanes (each ≤ 1 agent-day; dependency order)

| lane | scope | depends on | status |
|---|---|---|---|
| W4-S (stream A/B/C + one opening + Spartan) | production stream 19872523d | — | **done** (4.95 s / 10,304 B @2^17 G-shape M) |
| W4-T1 hash_table | T1 rows/columns/wiring/witness | — | **running** |
| W4-T2 limb_table | T2 chunks/helpers/LogUp | — | **running** |
| W4-R relation | verifier-algebra R1CS emitter + replay witness | — | **running** |
| W4-A byte audit (+ in-flight `StageAEncoding::KzgCommitted`, `ColumnReduction`) | exact per-field bytes | W4-S | running (uncommitted on disk) |
| **P1 free byte levers** | drop `stage_claims` (recompute), drop `v[2]`, 128-bit IO packing, EVM calldata codec (uncompressed G1) | W4-S | 0.5 d |
| **P2 Keccak outer transcript** | `DigestTranscript<Keccak256>` alias, `challenge_scalar` everywhere in the stream, byte-layout spec for Solidity | W4-S | 0.5 d |
| **P3 HyperKZG EVM form** | 4-pair rearrangement in a Solidity `verifyOpening` + Foundry gas (N4 project `/Volumes/Dev/scratch/wrapper-verifier-gas`); Rust verifier unchanged | P2 | 1 d |
| **P4 T1 strided layout + kernels** | 128 rows/compression, shape maps P̃, chaining EqPlusOne, L1/L2 via W-index embedding, L4 label drop | W4-T1 | 1 d |
| **P5 T2 strided layout + selectors** | region/stride descriptors, one-hot δ columns + constraints, 11 shift kernels, digit→scalar link member, row-count measurement (2^18 vs 2^19) | W4-T2, N1 | 1 d |
| **P6 SPARK-in-stage-C** | key-time row/col/val/m commitments, E/h columns, stage-C degree-3 members + slot-tensor rounds, RLC with VK commitments, tamper tests | W4-R, W4-S | 1 d |
| **P7a Solidity core** | transcript, 74 round checks, eq tables, RLC MSM, pairing; measured on a synthetic proof | P1–P3 | 1 d |
| **P7b Solidity tables** | Q̃_T1/Q̃_T2 evaluators, T1/T2 kernels, links, SPARK closed forms, data contract for constants | P4–P6, P7a | 1–2 d |
| **P8 e2e + gas table** | real fibonacci @2^18/2^20 proof → wrapped → Foundry verify; bytes / gas / prover table (all M) | P7b | 0.5 d |
| L-MERC (lever) | O(1)-proof multilinear KZG PCS (Mercury/Samaritan class): −1.5 KB, −200k gas, +0.5–1 s | P3 | 1–2 d, unmeasured |
| L-KZGR (lever) | committed stage-A rounds (in-flight): −1.2 KB, +150k gas | W4-A | 0.5 d after audit |

## 6. Risks (top 2) and notes

1. **T2 stride padding → 2^19** (E 1.4–1.7×; N1 compact 261,550 rows already 99.8% of 2^18 — M). Consequence: prover ≈25 s at k=16 (opening 2^23 10.8 s,
   helpers 5.2 s) or k=8 (+512 B, +123k gas). If a compact layout is kept instead, the fixed-offset operand wiring is an O(nnz ≈ 2.5M) public matrix —
   ≈ 50M gas — so the strided layout is not optional; P5 must measure the padded row count before anything else is sized. Mitigations: cheaper
   coefficient-row counts (12 → 16 is the main padder; a c-major layout with #ops padded to 2^15 gives 12 × 32k = 393k, same bucket), σ=11 (L=18)
   −6%, L3 GT codec + L4 (T1 only, does not help T2).
2. **Byte floor ≈ 9–11 KB is the round polynomials** (74 rounds = 7.3 KB, 68%): stage A 19×5, stage B 32×2, stage C 23×3. No packing/opening lever touches
   it; the only in-architecture reductions are committed stage-A rounds (−1.2 KB, +150k gas — bytes rank first, so this lever is *in* if gas ≤ 1.3 M is
   acceptable) and fewer sequential stages, which the rx→ry→SPARK data dependence forbids. "A couple of KB" needs a different reduction (e.g. all
   sumchecks over one shared point set via the Dory-assist staging invariant, which conflicts with column batching — plan-v3 §3), not this stream.
3. Lesser: (i) column budget 254/256 — one more column costs +16 stage-B rounds (+1 KB); (ii) in-flight `ColumnReduction` (one claim per factor column)
   regresses bytes by ≈8 KB for the real 254-column relation — the tensor form must stay for T1/T2; (iii) contract size → data contract; (iv) stage C at
   degree 3 makes the two-point-opening lever (−1.4 KB) unavailable while SPARK lives there; (v) SPARK soundness relies on LogUp with the table
   `{(i, eq(rx,i))}` — entries distinct (eq(rx,·) is injective for random rx with overwhelming probability) — state it in the spec.
