# Bytecode Commitment (Planning / Progress Notes)

This file is a **living design doc** for implementing **bytecode commitment** to remove verifier work linear in bytecode size \(K\), especially in recursion contexts (e.g. `examples/recursion/`).

## Problem statement (what is slow today?)

### Where the verifier is doing \(O(K)\) work

- **Stage 6 verifier constructs `BytecodeReadRafSumcheckVerifier` by calling `BytecodeReadRafSumcheckParams::gen`**, passing the full `BytecodePreprocessing`.
  - This happens in:
    - `jolt-core/src/zkvm/verifier.rs` **L409–L417**

- `BytecodeReadRafSumcheckParams::gen` currently **materializes 5 full `val_polys` of length `K`** by iterating the entire bytecode.
  - `compute_val_polys(...)` call site:
    - `jolt-core/src/zkvm/bytecode/read_raf_checking.rs` **L773–L784**
  - The fused per-instruction loop is here:
    - `jolt-core/src/zkvm/bytecode/read_raf_checking.rs` **L874–L1009**

- In `expected_output_claim`, the verifier then **evaluates each `val_poly` at `r_address`**, which is also \(O(K)\).
  - `val.evaluate(&r_address_prime.r)`:
    - `jolt-core/src/zkvm/bytecode/read_raf_checking.rs` **L648–L666**
  - `MultilinearPolynomial::evaluate` builds EQ tables and does a split-eq evaluation (still linear in coeff count):
    - `jolt-core/src/poly/multilinear_polynomial.rs` **L682–L772**

Net: for large bytecode (e.g. \(K \approx 2^{20}\)), the verifier is doing millions of field ops per verification, which explodes in recursion.

## Relevant existing patterns we can mirror

### 1) Two-phase claim reduction spanning Stage 6 → Stage 7 (Advice)

- Stage 6 includes Advice claim reduction Phase 1:
  - `jolt-core/src/zkvm/verifier.rs` **L446–L486**
- Stage 7 conditionally includes Advice claim reduction Phase 2:
  - `jolt-core/src/zkvm/verifier.rs` **L508–L529**
- Advice reduction module:
  - `jolt-core/src/zkvm/claim_reductions/advice.rs` (full file)

### 2) “Trusted commitment in preprocessing-only context” (Advice)

- Untrusted advice: prover commits during proving (`DoryContext::UntrustedAdvice`) and includes commitment in proof.
  - `jolt-core/src/zkvm/prover.rs` **L636–L667**
- Trusted advice: commitment/hint computed in preprocessing-only context (`DoryContext::TrustedAdvice`), verifier has commitment; prover just appends it to transcript.
  - `jolt-core/src/zkvm/prover.rs` **L669–L688**
- Dory contexts currently supported:
  - `jolt-core/src/poly/commitment/dory/dory_globals.rs` **L160–L166**

### 3) Single Stage 8 joint opening (Dory batch opening)

Stage 8 collects polynomial claims, samples gamma, combines commitments, and verifies a single opening.

- Stage 8 verifier:
  - `jolt-core/src/zkvm/verifier.rs` **L542–L691**

Advice polynomials get a **Lagrange embedding factor** so a smaller context polynomial can be batched with main polynomials:

- `compute_advice_lagrange_factor`:
  - `jolt-core/src/poly/opening_proof.rs` **L635–L672**

## Key batching detail (important for scheduling reductions)

Batched sumcheck instances are “front-loaded” via a **global round offset**:

- Default `round_offset` shifts shorter instances to the **end**:
  - `jolt-core/src/subprotocols/sumcheck_prover.rs` **L30–L37**
  - `jolt-core/src/subprotocols/sumcheck_verifier.rs` **L24–L30**
- `BatchedSumcheck` uses that offset to decide whether an instance is active in a global round:
  - `jolt-core/src/subprotocols/sumcheck.rs` **L79–L93**

This matters because it explains why Stage 6 “cycle rounds” can align across many instances even if they have different `num_rounds()`.

## Bytecode commitment: what we likely need to commit to

### Bytecode-side “fields” referenced in `compute_val_polys`

From `jolt-core/src/zkvm/bytecode/read_raf_checking.rs` **L874–L1009**, Val polynomials depend on:

- **Instruction scalar fields**
  - `instr.address` (a.k.a. unexpanded PC)
  - `instr.operands.imm`
- **Circuit flags**: `NUM_CIRCUIT_FLAGS = 13`
  - `jolt-core/src/zkvm/instruction/mod.rs` **L59–L86**, **L121**
- **Instruction flags**: `NUM_INSTRUCTION_FLAGS = 7`
  - `jolt-core/src/zkvm/instruction/mod.rs` **L104–L119**, **L122**
- **Register operands**: `rd`, `rs1`, `rs2` (used via `eq_r_register[...]` lookup)
  - This strongly suggests committing to **one-hot indicators** `1_{rd=r}`, `1_{rs1=r}`, `1_{rs2=r}` for all `r` (linear combination with EQ table).
- **Lookup table selector**
  - `NUM_LOOKUP_TABLES = LookupTables::<32>::COUNT` (currently 41)
  - `jolt-core/src/zkvm/lookup_table/mod.rs` **L118–L166**
- **RAF / interleaving flag**
  - `!circuit_flags.is_interleaved_operands()` (non-linear in circuit flags, so likely needs its own committed boolean field if we want linear combination only).
  - `jolt-core/src/zkvm/instruction/mod.rs` **L124–L135**

## Decisions so far (from discussion)

### Commitment granularity + packing (key)

We will **commit to the “atomic” bytecode fields**, but **pack/chunk them so each committed polynomial’s “lane” dimension fits `k_chunk = 2^{log_k_chunk}`**.

- `log_k_chunk` is **either 4 or 8** (so `k_chunk` is **16 or 256**), chosen from trace length:
  - `jolt-core/src/zkvm/config.rs` **L133–L151**

#### Canonical lane ordering (authoritative)

We fix a canonical total ordering of “lanes” (fields) so packing/chunking is purely mechanical and future-proof:

1. **`rs1` one-hot lanes**: 128 lanes (registers 0..127)
2. **`rs2` one-hot lanes**: 128 lanes
3. **`rd` one-hot lanes**: 128 lanes
4. **`unexpanded_pc` lane** (scalar)
5. **`imm` lane** (scalar)
6. **circuit flags** lanes: 13 boolean lanes (`NUM_CIRCUIT_FLAGS`)
7. **instruction flags** lanes: 7 boolean lanes (`NUM_INSTRUCTION_FLAGS`)
8. **lookup-table selector** lanes: 41 boolean lanes (`NUM_LOOKUP_TABLES`)
9. **RAF/interleave flag** lane: 1 boolean lane (`raf_flag := !circuit_flags.is_interleaved_operands()`)

Lane counts:
- registers: `3 * REGISTER_COUNT = 3 * 128 = 384`
  - `REGISTER_COUNT` definition: `common/src/constants.rs` **L1–L5**
- “dense-ish” bytecode fields: `2 + 13 + 7 + 41 + 1 = 64`
  - flags definitions: `jolt-core/src/zkvm/instruction/mod.rs` **L59–L86** (circuit), **L104–L119** (instruction)
  - lookup tables count: `jolt-core/src/zkvm/lookup_table/mod.rs` **L118–L166**

Total lanes = **384 + 64 = 448**.

Packing policy:
- We chunk the lane list into consecutive blocks of size `k_chunk`.
- Each block becomes one committed “bytecode commitment polynomial”.
- **`k_chunk=16`**: 448 lanes ⇒ **28 commitments** (exactly `3*(128/16)=24` for registers + `64/16=4` for the rest).
- **`k_chunk=256`**: 448 lanes ⇒ **2 commitments**:
  - chunk0: `rs1[0..127] || rs2[0..127]` (256 lanes)
  - chunk1: `rd[0..127] || (all remaining 64 lanes) || (64 lanes padding)`

Notes:
- Even though the first 384 lanes are “one-hot structured”, the packing is defined by lanes, so rs1/rs2/rd can be packed together when `k_chunk=256`.
- We will likely encode all lanes as field elements in the packed polynomial (booleans as 0/1), but **the representation choice (dense vs specialized one-hot)** is still an implementation detail (see Remaining plan questions below).

### Embedding policy

We will **not** require the main Dory matrix to grow to fit bytecode commitments. Instead we:

- keep each bytecode-commit polynomial within the main `k_chunk` address-dimension, and
- use a claim reduction (Stage 6→7) so these commitments can be batched into the single Stage 8 opening, similar to advice.

### Domain / padding

Bytecode commitments use the same **padding-to-power-of-two** policy as other committed polynomials:

- the “instruction index” dimension is padded to a power of 2 (like other `T`-style dimensions).
- the “lane/index” dimension is `k_chunk` (16 or 256), with unused lanes zero-padded.

### Ownership / preprocessing storage

Bytecode commitments should behave like **trusted preprocessing**:

- verifier has them in shared preprocessing (like trusted advice commitment is “known” to verifier),
- we define an enum where shared preprocessing stores **either**:
  - raw bytecode (`BytecodePreprocessing`), **or**
  - commitments (+ minimal metadata).

## Remaining plan questions (to settle before coding)

1. **Representation / PCS support for packed bytecode polynomials**:
   - Packing into `k_chunk` lanes means each packed polynomial has `k_chunk * bytecode_len` coefficients (very large).
   - We likely need a **streaming / implicit** polynomial representation (similar in spirit to `RLCPolynomial`) so Stage 8 can include bytecode commitments in the joint opening without materializing all coefficients.
2. **“rs1+rs2 as one-hot” wording (important clarity)**:
   - A single `OneHotPolynomial` can only select **one** lane index per column.
   - Packing `rs1` and `rs2` into the same 256-lane chunk means two 1s per instruction; this may need to be represented as a packed dense-bool polynomial (still sparse), or via a different encoding.
3. **Reduction batching**: we want **one** `BytecodeClaimReduction` sumcheck that batches all bytecode commitments and normalizes to the unified point (like `AdviceClaimReduction` + `HammingWeightClaimReduction` patterns).
4. **Stage 6 refactor** (required for mid-stage emission):
   - Stage 6 must split into **Stage 6a (log_K)** and **Stage 6b (log_T)** so bytecode-field claims emitted after the address rounds can be consumed immediately.
   - This also requires splitting `Booleanity` into address/cycle sumchecks (it is internally two-phase today):
     - `jolt-core/src/subprotocols/booleanity.rs` **L399–L453** (phase switch), **L455–L478** (cache_openings)
5. **Exact API surface**:
   - what concrete type should live in `JoltSharedPreprocessing` for the commitment-only variant (commitments-only vs commitments+opening hints)?
   - which `SumcheckId` values should be used for the new reduction’s intermediate/final cached openings?

---

## BytecodeReadRaf Stage 6a: what claims should be emitted?

The “emission point” is already explicit in the prover today: it happens right when we transition from the first `log_K` (address) rounds into the remaining `log_T` (cycle) rounds.

In `BytecodeReadRafSumcheckProver::init_log_t_rounds`:

- The prover computes the 5 stage-specific scalars:
  - `poly.final_sumcheck_claim()` for each stage Val polynomial, plus the RAF-injected identity contribution for stages 1 and 3:
    - `jolt-core/src/zkvm/bytecode/read_raf_checking.rs` **L307–L335**
- It also finalizes the address point by reversing the collected low-to-high challenges:
  - `jolt-core/src/zkvm/bytecode/read_raf_checking.rs` **L337–L340**

Those 5 scalars are stored in:

- `self.bound_val_evals: Option<[F; 5]>`
  - `jolt-core/src/zkvm/bytecode/read_raf_checking.rs` **L320–L335**

**Stage 6a should emit exactly these 5 scalars as “bytecode field claims”**, keyed by a new `SumcheckId` / `OpeningId`, with opening point = the address point `r_address` produced at the end of the address rounds.

Implementation detail we’ll likely choose:

- Emit **Val-only** claims `Val_s(r_address)` (no RAF Int injected), and let `BytecodeReadRaf` add the constant RAF terms itself (since `Int(r_address)=1`).
  - Today RAF is injected in `bound_val_evals` at **L324–L331**; we can split this for cleaner “bytecode-only” claim reduction.

Why this is the “right” interface:

- Stage 6b (the cycle-phase continuation of BytecodeReadRaf) needs these 5 scalars as weights for the remaining `log_T` rounds (today they’re read from `bound_val_evals` during the `round >= log_K` branch).

## BytecodeClaimReduction: what it should prove (high level)

We mirror the structure of `AdviceClaimReduction` (`jolt-core/src/zkvm/claim_reductions/advice.rs`), but with different “payload polynomials” and a simpler address schedule thanks to `k_chunk`.

### Inputs (from Stage 6a)

- The 5 “Val stage” claims:
  - `c_s := Val_s(r_bc)` for `s ∈ {1..5}`, where `r_bc` is the Stage 6a address point (bytecode-index point).
- The point `r_bc` itself (implicitly stored as the opening point associated with `c_s`).

### Witness (committed) polynomials

Let `B_i` be the committed bytecode chunk polynomials induced by the canonical lane ordering.

- `i ∈ [0, n_chunks)` where `n_chunks = ceil(448 / k_chunk)`:
  - `k_chunk=16` ⇒ `n_chunks=28`
  - `k_chunk=256` ⇒ `n_chunks=2`
  - See lane spec above.

Each `B_i` is a polynomial over:
- **lane/address vars**: `log_k_chunk`
- **bytecode-index vars**: `log_K_bytecode` (padded / embedded as needed; see “bytecode_len vs trace_len” note below)

### The identity to prove (batched)

Define a per-stage lane weight table `w_s[lane]` derived from:
- stage gammas sampled in `BytecodeReadRafSumcheckParams::gen`:
  - `jolt-core/src/zkvm/bytecode/read_raf_checking.rs` **L738–L742**
- register EQ tables (`eq_r_register_4`, `eq_r_register_5`) and the stage formulas in `compute_val_polys`:
  - `jolt-core/src/zkvm/bytecode/read_raf_checking.rs` **L752–L783**, **L874–L1009**

Then for each stage:

- \(c_s = \sum_{lane,k} B[lane,k] \cdot w_s[lane] \cdot \mathrm{eq}(r_{bc}, k)\)

We will batch the 5 stages with a transcript challenge \( \eta \) (powers), so the sumcheck instance has a **single scalar input claim**:

- \(C_{\text{in}} = \sum_s \eta^s \cdot c_s\)

and proves:

- \(C_{\text{in}} = \sum_{lane,k} B[lane,k] \cdot W_{\eta}(lane) \cdot \mathrm{eq}(r_{bc}, k)\)
  - where \(W_{\eta}(lane) := \sum_s \eta^s \cdot w_s[lane]\)

This keeps verifier complexity small: evaluating \(W_{\eta}\) at a point costs `O(k_chunk)` and computing \(\mathrm{eq}(r_{bc}, \cdot)\) uses `EqPolynomial`.

### Reduction target (Stage 8 compatibility)

BytecodeClaimReduction will run in two phases like advice:

- **Phase 1 (Stage 6b)**: bind the bytecode-index variables (cycle-phase rounds).
  - Cache an intermediate claim (like `AdviceClaimReductionCyclePhase`).
- **Phase 2 (Stage 7)**: bind the lane variables (`log_k_chunk` rounds).
  - When each `B_i` is fully bound (len==1), cache its final opening `B_i(final_point)` for batching into Stage 8.

Verifier then reconstructs the stage-6a claim(s) from:
- the final `B_i(final_point)` openings,
- the scalar `EqPolynomial::mle(r_bc, final_point_k)`,
- the scalar `W_eta(final_point_lane)`,
exactly analogous to `AdviceClaimReductionVerifier::expected_output_claim`.

### bytecode_len vs trace_len (defensive padding)

If `bytecode_len > padded_trace_len` (rare but possible for “mostly dead code”), we need to ensure:
- the main Dory URS / generators are large enough, and
- any “bytecode index variable count” that is driven by Stage 6 cycle rounds has enough randomness.

Pragmatic policy:
- set `padded_trace_len = max(padded_trace_len, bytecode_len.next_power_of_two())` *when bytecode commitments are enabled*,
  similar in spirit to `adjust_trace_length_for_advice` in `jolt-core/src/zkvm/prover.rs`.

### Preliminary “field count” if committed separately (worst-case baseline)

If we commit one polynomial per “atomic linear field”:

- `pc` + `imm`: **2**
- circuit flags: **13**
- instruction flags: **7**
- register one-hots: **3 * REGISTER_COUNT**
  - Note: `REGISTER_COUNT = 32 (RISC-V) + 96 (virtual) = 128` in this repo
    - `common/src/constants.rs` **L1–L5**
- lookup table one-hots: **41**
- raf/interleave flag: **1**

Total baseline (with `REGISTER_COUNT=128`): **2 + 13 + 7 + 384 + 41 + 1 = 448 polynomials**.

This is too many to *open individually*, but may be fine if we **derive only a few linear-combo commitments** (see open design options below).

## Proposed direction (high-level)

Goal: make verifier’s `BytecodeReadRaf` expected-claim computation **not materialize or evaluate length-K `val_polys`**, and instead consume **opening claims** that are later checked against a **trusted bytecode commitment** via Stage 8.

Key idea: mirror advice:

- **(A) Commit to bytecode (trusted preprocessing)**
  - Add a dedicated Dory context (e.g. `DoryContext::Bytecode`) whose matrix is a top-left block of main, like advice.
  - Verifier has these commitments “for free” (hard-coded / preprocessing).

- **(B) Emit bytecode-related evaluation claims during Stage 6**
  - Similar to how advice emits `RamValEvaluation` openings that later get reduced, `BytecodeReadRaf` should stop evaluating `val_polys` itself and instead *read* an opening claim (or small number of claims) from the opening accumulator.

- **(C) New two-phase “BytecodeClaimReduction” sumcheck**
  - Stage 6 phase: bind cycle-derived coordinates (last `log_T` rounds)
  - Stage 7 phase: bind address-derived coordinates (`log_k_chunk` rounds)
  - Cache final opening(s) so Stage 8 can batch them.

- **(D) Stage 8 batches bytecode commitments**
  - Include bytecode commitment(s) and reduced claim(s) in `polynomial_claims` with an embedding/Lagrange factor (same pattern as advice).

## Open design questions (need alignment before coding)

1. **Embedding feasibility**
   - Bytecode commitment context must fit in main Dory matrix: need `(sigma_bytecode <= sigma_main)` and `(nu_bytecode <= nu_main)`.
   - If program has **small trace length but huge bytecode**, do we:
     - pad `T` upward (like `adjust_trace_length_for_advice`), or
     - allow a second opening / separate Stage 8, or
     - impose a constraint “recursion requires T big enough”?

2. **Granularity**
   - Commit per field (many polynomials), or
   - commit a smaller set + derive per-stage Val polynomials by linear combinations of commitments, or
   - pack fields into one polynomial `p(k, idx)` (but then Val is *not* a simple linear combo of `p` at one point; needs more thought).

3. **How many bytecode “claims” should Stage 6 consume?**
   - 5 claims (one per stage Val polynomial), or
   - 1 claim (random linear combo of stage Vals, or another fixed fold) to minimize downstream reduction/opening cost.

4. **Where should the “initial” bytecode openings live?**
   - As `OpeningId::Committed(CommittedPolynomial::..., SumcheckId::BytecodeReadRaf)` entries, analogous to other committed openings, or
   - a new `OpeningId` variant (like `TrustedAdvice(...)`) if we need special casing.

5. **Commitment ownership**
   - Should bytecode commitments be stored inside `JoltSharedPreprocessing` / `JoltVerifierPreprocessing`, or passed separately like `trusted_advice_commitment`?

6. **Transcript binding**
   - We likely need to append trusted bytecode commitment(s) to the transcript in `JoltVerifier::verify` (similar to trusted advice):
     - `jolt-core/src/zkvm/verifier.rs` **L190–L203**

---

## Next steps (for plan agreement)

1. Decide **commit granularity** (per-field vs derived vs packed) with a target of minimizing **recursive verifier cycles**.
2. Decide **embedding policy** when bytecode is larger than main Dory dims.
3. Define the **exact claims** `BytecodeReadRaf` will consume (count + meaning).
4. Define the new **BytecodeClaimReduction** parameters (analogous to `AdviceClaimReductionParams`) and which Stage 6/7 rounds it occupies.

---

## Detailed implementation plan (agreed direction)

This section is an implementation checklist in dependency order.

### Step 1 — Refactor Stage 6 into two substages (6a + 6b)

**Goal**: make “end of BytecodeReadRaf address rounds” a real stage boundary so we can:
- emit `Val_s(r_bc)` claims **immediately** after binding `r_bc`,
- start `BytecodeClaimReduction` during the subsequent **cycle** randomness (what will become Stage 6b),
- avoid verifier doing any \(O(K_{\text{bytecode}})\) work.

#### 1.1 Proof object / serialization changes

- Split `stage6_sumcheck_proof` into:
  - `stage6a_sumcheck_proof` (address rounds)
  - `stage6b_sumcheck_proof` (cycle rounds)
- Transcript ordering: **run Stage 6a sumcheck → append Stage 6a claims → run Stage 6b sumcheck → append Stage 6b claims** (breaking change OK).
- Files:
  - `jolt-core/src/zkvm/proof_serialization.rs` (`JoltProof` struct)
  - any serialize/deserialize helpers that assume a single Stage 6 proof.

#### 1.2 Prover plumbing

- In `jolt-core/src/zkvm/prover.rs`:
  - Replace `prove_stage6()` with `prove_stage6a()` + `prove_stage6b()`.
  - Update the main `prove()` flow to call both and store both proofs.
  - Stage 6 instances currently assembled at `prover.rs` **L1206–L1214** must be split across 6a/6b.

Target contents:
- **Stage 6a (max rounds = `max(log_K_bytecode, log_k_chunk)`)**:
  - `BytecodeReadRafAddr` (new; `log_K_bytecode` rounds)
  - `BooleanityAddr` (new; `log_k_chunk` rounds; will be active only in last `log_k_chunk` rounds via front-loaded batching)
- **Stage 6b (max rounds = `log_T`)**:
  - `BytecodeReadRafCycle` (new; `log_T` rounds)
  - `BooleanityCycle` (new; `log_T` rounds)
  - existing Stage-6 cycle-only instances (unchanged logic, just move them here):
    - `RamHammingBooleanity` (`log_T`)
    - `RamRaVirtualization` (`log_T`)
    - `InstructionRaVirtualization` (`log_T`)
    - `IncClaimReduction` (`log_T`)
    - AdviceClaimReduction Phase 1 (if present) **needs a `round_offset` update** because Stage 6b `max_num_rounds` will now be `log_T` (see Step 2.3).
  - `BytecodeClaimReduction` phase 1 (new; `log_T` rounds; see Step 4)

#### 1.3 Verifier plumbing

- In `jolt-core/src/zkvm/verifier.rs`:
  - Replace `verify_stage6()` with `verify_stage6a()` + `verify_stage6b()`.
  - Update the main `verify()` call chain to include both.

### Step 2 — Split Booleanity into two sumchecks (address + cycle)

Reason: `Booleanity` is currently a *single* sumcheck with an internal phase transition at `log_k_chunk`:
- `jolt-core/src/subprotocols/booleanity.rs` **L399–L446**

But Stage 6 is becoming two proofs, so Booleanity must be representable as two separate sumcheck instances.

#### 2.1 New sumcheck instances

Create:
- `BooleanityAddressSumcheck` (`num_rounds = log_k_chunk`)
- `BooleanityCycleSumcheck` (`num_rounds = log_T`)

We will reuse most of the existing prover state splitting exactly at the current transition:
- address phase ends where today `eq_r_r` is computed and `H` is initialized (**L415–L445**)
- cycle phase reuses `D` and `H` binding (**L446–L452**)

#### 2.2 Chaining between 6a and 6b (important)

To make `BooleanityCycle` a standalone sumcheck, it needs an **input claim**:
- the output of `BooleanityAddress`, i.e. the partially summed claim after binding `r_address`.

We will follow the **AdviceClaimReduction** pattern:
- Stage 6a prover computes this intermediate claim and stores it in the opening accumulator under a new `SumcheckId` (see Step 5).
- Stage 6a verifier treats that stored claim as the expected output of `BooleanityAddress`.
- Stage 6b `BooleanityCycle` uses that stored claim as its `input_claim`.

This avoids needing BatchedSumcheck to “return per-instance output claims”.

#### 2.3 Update advice reduction round alignment (PINNED)

`AdviceClaimReductionProver::round_offset` currently assumes Stage 6 max rounds includes `log_k_chunk + log_T` (it aligns to the start of Booleanity’s cycle segment).
With Stage 6b max rounds = `log_T`, this must be updated to avoid underflow and to align to Stage 6b round 0.

File:
- `jolt-core/src/zkvm/claim_reductions/advice.rs` (`round_offset` in both prover+verifier impls)

### Step 3 — Split BytecodeReadRaf into two sumchecks (address + cycle)

Reason: we need a real stage boundary right after binding `r_bc` (bytecode-index address point), because:
- `Val_s(r_bc)` is computed exactly at the transition today in `init_log_t_rounds`
  - `jolt-core/src/zkvm/bytecode/read_raf_checking.rs` **L307–L340**

#### 3.1 New sumcheck instances

Create:
- `BytecodeReadRafAddressSumcheck` (`num_rounds = log_K_bytecode`)
- `BytecodeReadRafCycleSumcheck` (`num_rounds = log_T`)

#### 3.2 Stage 6a emissions (the key interface)

At the end of address rounds (today’s `init_log_t_rounds` boundary):
- emit **Val-only** claims:
  - `c_s := Val_s(r_bc)` for `s=1..5`
  - RAF terms are *not* included; verifier can add them succinctly because `Int(r_bc)=1`.
- batch these 5 claims with a random \(\eta\) in later reduction (Step 4), but still store the 5 scalars in the opening map.

Also emit the **cycle-phase input claim** for `BytecodeReadRafCycle`:
- this is the output claim of the address-only sumcheck (the partially summed value over cycle variables).

Both kinds of values must land in `opening_claims` so the verifier has them without recomputation.

### Step 4 — Implement `BytecodeClaimReduction` (two-phase, single instance)

This is the new sumcheck that replaces verifier’s \(O(K_{\text{bytecode}})\) evaluation of `val_polys`.

#### 4.1 High-level role

Input: the 5 `Val_s(r_bc)` scalars from Stage 6a.

Output: a set of committed-polynomial openings for the **bytecode commitment chunk polynomials** at the unified Dory opening point, so Stage 8 can batch them.

#### 4.2 Batching the 5 stage claims

We will batch the 5 `Val_s(r_bc)` using a transcript challenge \(\eta\):

- \(C_{\text{in}} = \sum_s \eta^s \cdot Val_s(r_{bc})\)

and prove this equals a single linear functional of the committed bytecode polynomials:

- \(C_{\text{in}} = \sum_{lane,k} B[lane,k] \cdot W_{\eta}(lane) \cdot \mathrm{eq}(r_{bc}, k)\)

No per-lane openings are needed; correctness follows from linearity.

#### 4.3 Two phases aligned to new stages

- **Phase 1 (Stage 6b)**: bind the bytecode-index variables using Stage 6b cycle challenges.
  - cache an intermediate claim (like `AdviceClaimReductionCyclePhase`) to start Phase 2.
- **Phase 2 (Stage 7)**: bind the lane variables (`log_k_chunk` rounds).
  - when each chunk polynomial is fully bound, cache its final opening for Stage 8.

The address phase should be simpler than advice because lane vars = exactly `log_k_chunk` (no partial consumption).

### Step 5 — `SumcheckId` / opening bookkeeping (naming + flow)

#### 5.1 How `SumcheckId` actually enters the proving / verifying flow

`SumcheckId` is part of the **key** used to store scalar claims in the opening accumulator maps.
Concretely, the key type is `OpeningId`, and it embeds `SumcheckId`:

- `OpeningId::Committed(CommittedPolynomial, SumcheckId)`
- `OpeningId::Virtual(VirtualPolynomial, SumcheckId)`
- `OpeningId::TrustedAdvice(SumcheckId)` / `OpeningId::UntrustedAdvice(SumcheckId)`
  - `jolt-core/src/poly/opening_proof.rs` **L136–L175**

**Prover side**: each sumcheck instance labels the claims it emits in `cache_openings(...)` by calling `ProverOpeningAccumulator::append_*` with a `SumcheckId`.
Those become entries in `opening_claims` (serialized into the proof).

**Verifier side**: the verifier is initialized with these claim scalars already present (from `opening_claims`), and each instance’s `cache_openings(...)` uses the same `SumcheckId` to populate the **opening point** for the existing claim (and to keep the transcript in sync).

#### 5.2 Why advice has two `SumcheckId`s (`...CyclePhase` and final)

Advice claim reduction spans Stage 6 → Stage 7, so it must store:

- an **intermediate** scalar after Phase 1 (cycle binding), and
- the **final** advice evaluation after Phase 2 (address binding).

This is why `SumcheckId` has both:

- `AdviceClaimReductionCyclePhase` (intermediate)
- `AdviceClaimReduction` (final)
  - `jolt-core/src/poly/opening_proof.rs` **L157–L160**

Where it’s used:

- Phase 2 starts from the Phase 1 intermediate:
  - `AdviceClaimReductionParams::input_claim` (AddressVariables case):
    - `jolt-core/src/zkvm/claim_reductions/advice.rs` **L190–L216**
- Phase 1 and Phase 2 both cache openings under their respective IDs:
  - `AdviceClaimReductionProver::cache_openings`:
    - `jolt-core/src/zkvm/claim_reductions/advice.rs` **L466–L518**

So neither is unused; they identify *two different stored claims*.

#### 5.3 Naming rule of thumb (must match variable order)

Two-phase protocols in this repo come in **both** variable orders:

- **cycle → address**: advice claim reduction, bytecode claim reduction
- **address → cycle**: booleanity, bytecode read+raf

So the naming should reflect **what phase 1 binds**:

- `XCyclePhase`: output claim after Phase 1 binds the **cycle-derived** variables
- `XAddressPhase`: output claim after Phase 1 binds the **address-derived** variables
- `X` (or `XFinal`): final output after all variables are bound

For protocols we split into two physical sumchecks (Stage 6a + 6b) but want downstream stability:

- keep the existing “final” `SumcheckId` if other modules already key off it (e.g. `HammingWeightClaimReduction` expects `SumcheckId::BytecodeReadRaf` today),
- add a new `...AddressPhase` id for the Stage 6a pre-phase when the protocol binds address first.

#### 5.4 Concrete `SumcheckId` changes for this rollout

File to update:
- `jolt-core/src/poly/opening_proof.rs` (`SumcheckId` enum)

We will add:

- **Address → cycle protocols (Stage 6 split)**:
  - `BytecodeReadRafAddressPhase` (new; Stage 6a sumcheck; binds **address** first)
  - `BooleanityAddressPhase` (new; Stage 6a sumcheck; binds **address** first)
  - keep `BytecodeReadRaf` and `Booleanity` as the “final” IDs (Stage 6b sumchecks + cached openings) so downstream modules that key off them (e.g. HW reduction) remain stable.

- **Cycle → address protocols (two-phase reductions)**:
  - `BytecodeClaimReductionCyclePhase` (new; phase 1 output after binding **cycle** vars in Stage 6b)
  - `BytecodeClaimReduction` (new; final output after binding **lane/address** vars in Stage 7)
  - (existing) `AdviceClaimReductionCyclePhase` / `AdviceClaimReduction` already follow this pattern.

We will also add **new `VirtualPolynomial` variants** for scalar claims that are *not* openings of committed polynomials:

- **Stage 6a (BytecodeReadRafAddressPhase)**:
  - `VirtualPolynomial::BytecodeValStage(usize)` for the 5 Val-only claims.
  - `VirtualPolynomial::BytecodeReadRafAddrClaim` for the address-phase output claim that seeds the cycle-phase sumcheck.
- **Stage 6a (BooleanityAddressPhase)**:
  - `VirtualPolynomial::BooleanityAddrClaim` for the address-phase output claim that seeds the cycle-phase sumcheck.
- **Stage 6b → Stage 7 (BytecodeClaimReduction)**:
  - `VirtualPolynomial::BytecodeClaimReductionIntermediate` for the cycle-phase intermediate claim (analogous to advice’s `...CyclePhase`), used as Stage 7 input.

#### 5.5 Quick “protocol → variable order → IDs” table (sanity)

- **BytecodeReadRaf**: address → cycle
  - Stage 6a: `SumcheckId::BytecodeReadRafAddressPhase`
  - Stage 6b: `SumcheckId::BytecodeReadRaf` (final)
- **Booleanity**: address → cycle
  - Stage 6a: `SumcheckId::BooleanityAddressPhase`
  - Stage 6b: `SumcheckId::Booleanity` (final)
- **BytecodeClaimReduction**: cycle → lane/address
  - Stage 6b: `SumcheckId::BytecodeClaimReductionCyclePhase` (intermediate stored)
  - Stage 7: `SumcheckId::BytecodeClaimReduction` (final)
- **AdviceClaimReduction** (existing): cycle → address (two-phase)
  - Stage 6: `SumcheckId::AdviceClaimReductionCyclePhase`
  - Stage 7: `SumcheckId::AdviceClaimReduction`

### Step 6 — Bytecode commitments in preprocessing + transcript

#### 6.1 New Dory context + storage

Add a new `DoryContext::Bytecode` (like Trusted/UntrustedAdvice) so we can commit to bytecode chunk polynomials in preprocessing and hand the commitments to the verifier.

Update shared preprocessing to store either:
- raw `BytecodePreprocessing`, or
- `{ bytecode_len, k_chunk, commitments: Vec<Commitment>, (optional) layout metadata }`

#### 6.2 Canonical lane ordering implementation

Implement an enum (or equivalent) encoding the authoritative lane ordering:
- rs1 lanes (0..127), rs2 lanes (0..127), rd lanes (0..127), then dense fields.
Then chunk into blocks of size `k_chunk` to get commitment indices.

This ordering must be used consistently by:
- commitment generation
- `BytecodeClaimReduction` weight construction
- Stage 8 batching / VMV contribution

### Step 7 — Stage 8 batching integration (bytecode polynomials)

Stage 8 currently builds a streaming `RLCPolynomial` from:
- dense trace polys
- onehot RA polys
- advice polys (passed directly)

We need to extend this to include “bytecode commitment chunk polynomials”:
- they are **not** streamed from trace
- they are too large to materialize when bytecode is big

Implementation direction:
- extend the streaming RLC machinery to support an additional source (“stream from bytecode”),
  analogous to how it already streams onehot polys from trace.

Files involved:
- `jolt-core/src/poly/rlc_polynomial.rs` (extend streaming context + VMP to include bytecode chunk polys)
- `jolt-core/src/zkvm/prover.rs` / `verifier.rs` Stage 8 claim collection (include bytecode chunk claims with appropriate embedding factor, like advice)

### Step 8 — Defensive padding: bytecode_len vs trace_len

When bytecode commitments are enabled, ensure we have enough cycle randomness to bind bytecode-index vars:

- `padded_trace_len = max(padded_trace_len, bytecode_len.next_power_of_two())`

This is analogous to `adjust_trace_length_for_advice` in `jolt-core/src/zkvm/prover.rs`.

### Step 9 — Tests / validation

- Unit tests:
  - lane ordering + chunking (k_chunk=16 ⇒ 28 chunks, k_chunk=256 ⇒ 2 chunks)
  - bytecode_len > trace_len padding path
- E2E:
  - prove+verify with bytecode commitment enabled, both layouts (CycleMajor/AddressMajor)
- Recursion benchmark:
  - confirm verifier cycle count no longer scales with bytecode length.
