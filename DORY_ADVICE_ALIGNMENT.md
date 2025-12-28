# Single Dory Opening for Jolt (with preprocessing-only advice commitments)
This note documents the **alignment** needed to fold **trusted + untrusted advice** into the **single Stage 8 Dory opening proof**, even though advice commitments may have a different “native” dimension than the main witness polynomials.

It’s written as an end-to-end walkthrough: **commitment → Stage 6 advice claim reduction → Stage 7 wiring → Stage 8 streaming RLC + Dory opening**.

---

### TL;DR (what finally made it work)
- **Commit advice in separate Dory contexts** (`TrustedAdvice`, `UntrustedAdvice`) as a **balanced matrix** \(2^{\nu_a}\times 2^{\sigma_a}\) (preprocessing-only; dims derived deterministically from advice size).
- **Embed** that advice matrix into the main Dory matrix as the **top-left block** (rows \([0..2^{\nu_a})\), cols \([0..2^{\sigma_a})\)) and account for this embedding with **row/col selector factors** in Stage 7 and in the streaming RLC evaluation (Stage 8).
- For \(\nu_a>0\), the advice evaluation point implied by Dory’s point ordering generally pulls coordinates from **both** Stage 6 (**cycle**) and Stage 7 (**address**). We implement **two-phase advice claim reduction**:
  - **Stage 6 (Phase 1)** binds the cycle-derived advice coordinates and outputs an **intermediate claim**.
  - **Stage 7 (Phase 2)** consumes that intermediate claim, binds the remaining address-derived advice coordinates, and caches the **final advice opening** for Stage 8 batching.
- We keep the generic `BatchedSumcheck` logic unchanged: the advice reducers themselves handle any
  **trailing dummy variables** (\(2^{\text{dummy\_after}}\) scaling) as well as internal “gap” dummy
  rounds needed to reach the correct cycle coordinates.

---

### Key conventions (Dory matrix + endianness)

#### Dory dimensions
For a Dory matrix built from `K * T = 2^total_vars`:
- `sigma = ceil(total_vars/2)` column variables → `num_columns = 2^sigma`
- `nu    = floor(total_vars/2)` row variables    → `num_rows    = 2^nu`

In Jolt we call `DoryGlobals::initialize(K, T)` where:
- Main witness uses `K = k_chunk = 2^{log_k_chunk}`, `T = padded_trace_len = 2^{log_t}` so `total_vars = log_k_chunk + log_t`.

#### Opening-point endianness cheat-sheet
- Jolt stores `OpeningPoint<BIG_ENDIAN>` (MSB-first).
- Dory’s internal MLE routines use **little-endian** variable order (LSB-first).
- The Dory wrapper code frequently does:
  - `let mut r_le = opening_point_be.clone(); r_le.reverse();`
  - then splits as `(r_cols, r_rows) = r_le.split_at(sigma)`.

When `opening_point_be = [addr_be || cycle_be]`, reversing yields:
`r_le = [cycle_le || addr_le]`.

For example with `log_k_chunk=4`, `log_t=16`, `total_vars=20`, `sigma=10`:
- `r_cols = r_le[0..10] = cycle_le[0..10]` (**low** cycle bits)
- `r_rows = r_le[10..20]` (mix of high cycle bits + address bits)

This is why “advice variables” need to line up with the **prefix of `cycle_le`**, not the suffix.

---

### Ground truth: exact opening-point ordering (Stage 6 → Stage 7 → Stage 8 → Dory)

This section is the **precise current ordering** in the codebase. It’s meant to be the “single
source of truth” you can use when designing a rigorous **permutation / schedule** mechanism (e.g.,
for multi-row advice commitments).

#### Notation
- **Stage 6 batched-sumcheck challenges (global)**: `r6[0..R6-1]`
  - `r6[j]` is the challenge sampled in **global round** `j` (in increasing time order).
  - Each instance gets an **instance-local slice** (still in increasing order) via:
    - `r_slice = r6[offset .. offset + num_rounds]`
    - where `offset = instance.round_offset(R6)`.
- **Stage 7 batched-sumcheck challenges (global)**: `r7[0..log_k_chunk-1]`
  - Stage 7 is its own batched sumcheck (separate transcript state).

We’ll name address/cycle challenges as:
- `addr6_le = [a0, a1, …, a_{log_k_chunk-1}]` (LE / low-to-high, from Stage 6)
- `cycle6_le = [c0, c1, …, c_{log_t-1}]` (LE / low-to-high, from Stage 6)
- `addr7_le = [b0, b1, …, b_{log_k_chunk-1}]` (LE / low-to-high, from Stage 7)

#### Stage 6 Booleanity → `r_cycle_stage6`
Booleanity’s instance-local slice is (LE, address then cycle):

```text
r_bool_le = r6[offset_bool .. offset_bool + log_k_chunk + log_t]
          = [addr6_le || cycle6_le]
          = [a0..a_{log_k_chunk-1} || c0..c_{log_t-1}]
```

Booleanity then normalizes to an `OpeningPoint<BIG_ENDIAN>` by reversing each segment
independently, yielding a stored point:

```text
r_bool_be = [reverse(addr6_le) || reverse(cycle6_le)]
          = [a_{log_k_chunk-1}..a0 || c_{log_t-1}..c0]
```

Define:

```text
r_cycle_stage6_be = [c_{log_t-1}..c0]   (length = log_t)
r_cycle_stage6_le = reverse(r_cycle_stage6_be) = [c0..c_{log_t-1}]
```

With default “front-loaded” batching for Booleanity, its cycle segment begins at:

```text
cycle_start = offset_bool + log_k_chunk
```

And, under default suffix placement for Booleanity, this is exactly:

```text
cycle_start = R6 - log_t
```

So:

```text
r_cycle_stage6_le = r6[cycle_start .. cycle_start + log_t]
r_cycle_stage6_be = reverse(r6[cycle_start .. cycle_start + log_t])
```

This is also why `IncReduction` (a `log_t`-round Stage 6 instance using the default suffix
placement) naturally shares the same `r_cycle_stage6`.

#### Stage 7 HammingWeightClaimReduction → `r_address_stage7`
Stage 7 produces:

```text
addr7_le = r7 = [b0..b_{log_k_chunk-1}]
addr7_be = reverse(addr7_le) = [b_{log_k_chunk-1}..b0]
```

#### Stage 8 unified opening point (what Jolt passes into PCS::prove)
The Stage 8 Dory opening point stored in `DoryOpeningState` is:

```text
opening_point_be = [addr7_be || r_cycle_stage6_be]
                 = [b_{log_k_chunk-1}..b0 || c_{log_t-1}..c0]
```

This is the “ground truth” constraint you called out: **Stage 8 is address-from-Stage7 +
cycle-from-Stage6**.

#### What Dory actually receives (inside `DoryCommitmentScheme::prove`)
`DoryCommitmentScheme::prove` reverses the entire opening point before passing it to `dory::prove`,
so Dory receives:

```text
point_dory = reverse(opening_point_be)
          = [reverse(r_cycle_stage6_be) || reverse(addr7_be)]
          = [r_cycle_stage6_le         || addr7_le]
          = [c0..c_{log_t-1}           || b0..b_{log_k_chunk-1}]
```

#### Dory’s row/column split (critical for scheduling)
Let:
- `total_vars = log_t + log_k_chunk`
- `sigma = ceil(total_vars / 2)`  (columns exponent)
- `nu    = total_vars - sigma`    (rows exponent)

Dory interprets:

```text
col_coords = point_dory[0 .. sigma]           (length sigma, LSB-first)
row_coords = point_dory[sigma .. sigma + nu]  (length nu,    LSB-first)
```

Concretely, depending on whether `sigma <= log_t`:

- **Case A: `sigma <= log_t` (common)**

```text
col_coords = [c0 .. c_{sigma-1}]
row_coords = [c_sigma .. c_{log_t-1} || b0 .. b_{log_k_chunk-1}]
```

- **Case B: `sigma > log_t` (possible when address bits dominate)**

```text
col_coords = [c0 .. c_{log_t-1} || b0 .. b_{sigma-log_t-1}]
row_coords = [b_{sigma-log_t} .. b_{log_k_chunk-1}]
```

This is the core “ordering weirdness”: **Dory columns come first**, and in Jolt’s current
construction those columns start with the **low cycle bits** (then possibly spill into address bits).

#### Examples (fully expanded)

**Example 1: `log_k_chunk=4`, `log_t=16` ⇒ `total_vars=20`, `sigma=10`, `nu=10`**

```text
point_dory  = [c0..c15 || b0..b3]
col_coords  = [c0..c9]
row_coords  = [c10..c15 || b0..b3]   (6 cycle bits + 4 address bits)
```

**Example 2: `log_k_chunk=8`, `log_t=16` ⇒ `total_vars=24`, `sigma=12`, `nu=12`**

```text
point_dory  = [c0..c15 || b0..b7]
col_coords  = [c0..c11]
row_coords  = [c12..c15 || b0..b7]   (4 cycle bits + 8 address bits)
```

**Example 3: `log_k_chunk=4`, `log_t=8` ⇒ `total_vars=12`, `sigma=6`, `nu=6`**

```text
point_dory  = [c0..c7 || b0..b3]
col_coords  = [c0..c5]
row_coords  = [c6,c7 || b0..b3]      (2 cycle bits + 4 address bits)
```

**Example 4 (sigma > log_t): `log_k_chunk=8`, `log_t=4` ⇒ `total_vars=12`, `sigma=6`, `nu=6`**

```text
point_dory  = [c0..c3 || b0..b7]
col_coords  = [c0..c3 || b0,b1]
row_coords  = [b2..b7]
```

#### Advice claim-reduction point (general \((\nu_a,\sigma_a)\), two-phase)
The *native* advice point (in **Dory order**, little-endian) is always:

```text
advice_point_le = [ col_coords[0..sigma_a] || row_coords[0..nu_a] ]
```

With `point_dory = [cycle6_le || addr7_le]`, this splits as:
- **Cycle-derived**:
  - `col_coords[0..sigma_a] = cycle6_le[0..sigma_a]`
  - plus the cycle portion of `row_coords[0..nu_a]`, which starts at `cycle6_le[sigma_main]`
- **Address-derived**:
  - remaining suffix of `row_coords[0..nu_a]` coming from `addr7_le[0..]`

So we implement advice claim reduction as:
- **Stage 6 Phase 1**: binds the cycle-derived coordinates (aligned to the start of Booleanity’s cycle segment)
- **Stage 7 Phase 2**: binds the remaining address-derived coordinates (a prefix of `addr7_le`)

---

### Phase 1: Commitment phase (Main vs Advice contexts)

#### Main witness polynomials
- Committed under the **Main** Dory context:
  - `DoryGlobals::initialize(1 << one_hot_params.log_k_chunk, padded_trace_len)`
  - Commitments + hints are produced in `generate_and_commit_witness_polynomials()`.

These are the “native” polynomials of the main Dory matrix.

#### Trusted advice (preprocessing-only)
Goal: commitment computable **without knowing the trace length**.

We commit trusted advice under a dedicated context with a **preprocessing-only matrix shape**:
- `DoryGlobals::initialize_trusted_advice_matrix(num_rows, num_columns)`
  - `num_rows` and `num_columns` are **powers of two**
  - `T_advice = num_rows * num_columns` (polynomial length)
  - depends only on advice size (after power-of-two padding), not on the execution trace length `T`

**Canonical shape policy (balanced):**
- `advice_len = next_power_of_two(#words)`
- `advice_vars = log2(advice_len)`
- `sigma_a = ceil(advice_vars/2)`, `nu_a = advice_vars - sigma_a`
- `num_columns = 2^{sigma_a}`, `num_rows = 2^{nu_a}`

We also keep the **opening hint** from commit so Stage 8 can batch it:
- The macro-generated `commit_trusted_advice` is updated to commit in this context and return the hint.

#### Untrusted advice
Untrusted advice is committed similarly so it can be embedded consistently:
- `DoryGlobals::initialize_untrusted_advice_matrix(num_rows, num_columns)`
- The prover commits it at runtime and stores its hint for Stage 8 batching.

#### Edge case: “main matrix smaller than advice”
Because we embed advice into the **top-left block** of the main matrix, we need:

\[
\sigma_\text{main} \ge \sigma_a \quad\text{and}\quad \nu_\text{main} \ge \nu_a
\]

In `gen_from_trace()` we add a guardrail that increases `padded_trace_len` (doubling) until:
- the main matrix has enough **column and row variables** to fit advice,
- but never exceeds `preprocessing.max_padded_trace_length` (otherwise we error/panic).

---

### Phase 2: Stage 6 – Advice claim reduction (and why alignment matters)

#### What we’re reducing
RAM subprotocols can emit multiple openings for advice polynomials (trusted/untrusted), potentially at different points.

We introduce **two-phase advice claim reduction**, and run **trusted** and **untrusted** as
**separate** instances (since they may have different \((\nu_a,\sigma_a)\)):

- **Stage 6 (Phase 1)**: `AdviceClaimReductionPhase1`
  - binds the **cycle-derived** advice coordinates
  - outputs an **intermediate claim** cached under `SumcheckId::AdviceClaimReductionPhase1`
- **Stage 7 (Phase 2)**: `AdviceClaimReductionPhase2`
  - consumes that intermediate claim as its `input_claim`
  - binds the remaining **address-derived** advice coordinates
  - caches the **final advice opening** under `SumcheckId::AdviceClaimReduction` (so Stage 7/8 can batch it)

#### The core alignment requirement (Dory variable order)
In Stage 8, we want advice to behave like a **top-left block** \(2^{\nu_a}\times 2^{\sigma_a}\)
embedded into the main matrix. In Dory order (see “Ground truth”), this means the *native* advice
evaluation point must be:

```text
advice_point_le = [ col_coords[0..sigma_a] || row_coords[0..nu_a] ]
               = [ r_cols[0..sigma_a]     || r_rows[0..nu_a]     ]
```

But the Stage 6 batched sumcheck produces a single global challenge vector `r_sumcheck`.
Each instance receives a slice:
- `r_slice = r_sumcheck[offset .. offset + num_rounds]`

So we must schedule Phase 1 and Phase 2 so the bound variables correspond to the coordinates above:
- Phase 1 binds the pieces that come from `cycle6_le`
- Phase 2 binds the pieces that come from `addr7_le`

When \(\nu_a>0\) and \(\sigma_a < \sigma_\text{main}\), Phase 1 must traverse the “gap”
`cycle6_le[sigma_a .. sigma_main)` to reach the cycle-derived row bits at `cycle6_le[sigma_main..]`.
Those gap rounds are treated as **dummy internal rounds** (constant univariates), and the phase
maintains a scaling factor so the sumcheck identities remain correct.

#### How we pick the right slice (round_offset)
Stage 6 is batched across many instances; some have `num_rounds` much larger than Booleanity.

Let:
- `max_num_rounds = max_i num_rounds(i)` (e.g. often 32 due to lookup-ish instances)
- `booleanity_rounds = log_k_chunk + log_t`
- Under default front-loaded batching, Booleanity starts at:
  - `booleanity_offset = max_num_rounds - booleanity_rounds`
- Booleanity’s **cycle segment** starts after the address chunk:
  - `cycle_offset = booleanity_offset + log_k_chunk`

We set:
- `AdviceClaimReduction::round_offset(max_num_rounds) = cycle_offset`

So `AdviceClaimReduction` consumes the first `advice_vars` challenges of the cycle segment, which matches the **low Dory column bits**.

---

### Phase 3: Handling “dummy rounds after the active window” (localized to advice reducers)

When an instance is **not** placed as a suffix (e.g. Phase 1 aligned to the start of Booleanity’s
cycle segment, or Phase 2 consuming an address **prefix**), there are “dummy” variables **after**
its active window:

\[
dummy\_after = max\_num\_rounds - (offset + num\_rounds) > 0
\]

During the active rounds, those trailing dummy variables are still summed over, contributing a
constant factor \(2^{dummy\_after}\).

Instead of teaching `BatchedSumcheck` about this case, we handle it inside the advice reducers:
- divide the instance’s `previous_claim` by \(2^{dummy\_after}\) before constructing the univariate
- multiply the univariate by \(2^{dummy\_after}\)

This keeps the generic batcher simple while supporting the non-suffix placement required by Dory’s
variable ordering.

---

### Phase 4: Stage 7 – wiring advice into the final Dory opening state

Stage 7 constructs a single `DoryOpeningState`:
- `opening_point = [r_address_be || r_cycle_be]`
- plus a list of `(polynomial_id, claim)` to batch-open in Stage 8.

For advice polynomials, we **don’t** open them in their native advice context.
Instead, we treat them as embedded in the main matrix, which introduces a selector factor.

#### The embedding selector (row + column padding)
Compute:
- `r_le = reverse(opening_point_be)`
- split into `(r_cols, r_rows)` using `sigma = log2(num_columns)` from the **main** context

For an advice polynomial with dimensions \((\nu_a,\sigma_a)\), embedded as the **top-left block**:
- rows are restricted to `[0 .. 2^{nu_a})` → enforce **high row bits are 0**
- cols are restricted to `[0 .. 2^{sigma_a})` → enforce **high col bits are 0**

This produces the selector:

\[
\text{row\_factor} = \prod_{i = \nu_a}^{\nu_\text{main}-1} (1 - r\_{rows}[i])
\]
\[
\text{col\_prefix\_factor} = \prod_{i = \sigma_a}^{\sigma_\text{main}-1} (1 - r\_{cols}[i])
\]

And the embedded evaluation identity:

\[
\text{EmbeddedAdvice}(r) =
\text{row\_factor} \cdot \text{col\_prefix\_factor} \cdot
\text{AdviceMLE}([r\_{cols}[0..\sigma_a] \;||\; r\_{rows}[0..\nu_a]])
\]

**Stage 7 action:** when we pull the “raw” advice claim from advice claim-reduction (which is
`AdviceMLE([r_cols[0..sigma_a] || r_rows[0..nu_a]])`), we multiply it by `row_factor * col_prefix_factor` before
putting it into the final Dory batch opening list (`DoryOpeningState.claims`).

This is the key that makes the “native advice opening” consistent with opening everything at the
single unified main point.

---

### Phase 5: Stage 8 – streaming RLC + single Dory opening proof

Stage 8:
1. Re-initializes the **Main** Dory context with the *actual* `(k_chunk, padded_trace_len)`.
2. Builds the **joint RLC polynomial** from the trace (streaming) using the Stage 7 `DoryOpeningState`:
   - the unified opening point
   - `gamma` powers sampled from the transcript
   - the per-polynomial claims (including the scaled advice claims)
3. Calls `PCS::prove(...)` once to produce a **single Dory opening proof** for the joint polynomial.

#### Streaming RLC wiring for advice
The streaming RLC builder is given:
- `opening_proof_hints` for every polynomial being batched (including trusted/untrusted advice),
- an `advice_polys` map holding the **actual advice polynomials**.

The RLC evaluation path (`RLCPolynomial::vector_matrix_product`) includes advice by interpreting
each advice polynomial as a **top-left block embedded into the main matrix** in the same way Stage 7
scaled the claims:
- the row restriction is the high-row-bits selector (`row_factor`)
- the column restriction to the leftmost block is `col_prefix_factor`
- the remaining dependence is the advice MLE evaluated at **(low col bits || low row bits)**
- the VMV sums across multiple advice rows using `left_vec[row_idx]` for `row_idx < 2^{nu_a}`

Because Stage 6 is aligned to those low column bits (see Phase 2/3), the “raw” advice opening
cached by `AdviceClaimReduction` and the Stage 8 streaming evaluation now match.

---

### The alignment bug we hit (and the mental model that resolves it)

#### Symptom
We repeatedly saw a Stage 8 mismatch of the form:
- “RLC evaluation at the Dory point” ≠ “Σ γᵢ · claimᵢ”

When we isolated the base (non-advice) contribution, it matched. The mismatch came entirely from
the advice contribution.

#### What was actually wrong
Before the fix, `AdviceClaimReduction` was effectively consuming the **suffix** of the Stage 6
cycle challenges (because the default batched-sumcheck placement is suffix).

Concretely:
- Booleanity’s cycle segment is length `log_t` (little-endian).
- With `advice_vars` advice variables, suffix placement gives challenges:
  - `cycle_le[log_t - advice_vars .. log_t]`

But Dory’s **column** bits (for the main matrix) are the **prefix** of `cycle_le`:
- `r_cols = cycle_le[0 .. sigma]`

So advice was evaluated at the wrong slice of `cycle_le`, and therefore could not match the
streaming RLC evaluation (which depends on the low Dory column bits).

#### Fix in one line
Force `AdviceClaimReduction` to use:
- `cycle_le[0 .. advice_vars]`

by setting a custom `round_offset` that starts at the **beginning of Booleanity’s cycle segment**.

That placement creates trailing dummy rounds, so the advice sumcheck prover must compensate by a
constant factor `2^{dummy_after}` (Phase 3).

---

### Appendix: Amir’s schedule mapping (old “Stage 7 binds addr then cycle” pipeline)
On [`amir/dory-batching-squashed`](https://github.com/LayerZero-Research/jolt/tree/amir/dory-batching-squashed),
the (old) opening-reduction Stage 7 produced a single challenge stream:

```text
r7_le = [addr7_le || cycle7_le]
```

but Dory evaluates at:

```text
point_dory = reverse(opening_point_be) = [cycle7_le || addr7_le]
```

So, for an advice matrix \((\nu_a,\sigma_a)\) embedded as the top-left block, the *native* advice
point coordinates \([col_coords[0..\sigma_a] || row_coords[0..\nu_a]]\) correspond to a
**non-contiguous subset** of indices of `r7_le`:

- **Columns**: `col_coords[0..sigma_a] = cycle7_le[0..sigma_a] = r7_le[log_K + 0 .. log_K + sigma_a]`
- **Rows**: `row_coords[0..nu_a] = point_dory[sigma_main .. sigma_main + nu_a]`, which can straddle:
  - cycle part: `r7_le[log_K + sigma_main .. log_K + min(log_T, sigma_main + nu_a)]`
  - addr part:  `r7_le[0 .. max(0, sigma_main + nu_a - log_T)]`

This is exactly why Amir needed a `binding_rounds`/schedule: advice consumes a *permutation* of the
global Stage 7 rounds, derived from the ground-truth `point_dory` ordering.

#### How we translate that to the current pipeline
In today’s pipeline, the final Dory point is already `point_dory = [cycle6_le || addr7_le]`, split
across stages. So we “slice” the old mapping:

- indices in the old schedule that come from `cycle` → **Stage 6 (Phase 1)** consumes the
  cycle-derived coordinates (including traversing the `sigma_a..sigma_main` gap as dummy internal rounds)
- indices in the old schedule that come from `addr`  → **Stage 7 (Phase 2)** consumes the
  address-derived coordinates (a prefix of `addr7_le`)

---

### Code pointers (where to look)

#### Commitment contexts
- `jolt-core/src/poly/commitment/dory/dory_globals.rs`
  - `DoryContext`
  - `DoryGlobals::initialize_trusted_advice_matrix`
  - `DoryGlobals::initialize_untrusted_advice_matrix`

#### Preprocessing-only trusted advice commit
- `jolt-sdk/macros/src/lib.rs`
  - macro-generated host function commits under `TrustedAdvice` context and returns an opening hint

#### Prover wiring (guardrail, Stage 7/8)
- `jolt-core/src/zkvm/prover.rs`
  - `gen_from_trace`: padding/guardrail so `sigma_main >= advice_vars`
  - `generate_and_commit_untrusted_advice`: commit in `UntrustedAdvice` context, store hint
  - `generate_and_commit_trusted_advice`: use externally provided trusted commitment + hint
  - Stage 7: compute `row_factor` + `col_prefix_factor` and scale advice claims
  - Stage 8: pass `advice_polys` + advice hints into `build_streaming_rlc`, then call `PCS::prove`

#### Advice claim reduction alignment
- `jolt-core/src/zkvm/claim_reductions/advice.rs`
  - `AdviceClaimReduction(Prover/Verifier)::round_offset(max_num_rounds)`

#### Batched sumcheck mid-segment scaling
- `jolt-core/src/subprotocols/sumcheck.rs`
  - `zkvm/claim_reductions/advice.rs`: active-round scaling by `2^{dummy_after}` when an advice phase is not a suffix

---

### Regression tests
- `zkvm::prover::tests::advice_e2e_dory` (end-to-end trusted + untrusted advice batched into Stage 8)
- `zkvm::prover::tests::fib_e2e_dory` (sanity-check non-advice flows after batched-sumcheck change)


