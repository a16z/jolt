# Jolt Core Prover Optimization Inventory

| Field | Value |
|-------|-------|
| Author(s) | Markos Georghiades, Codex |
| Created | 2026-05-23 |
| Status | draft inventory |
| Scope | `jolt-core` prover-side performance and memory optimizations to preserve during the `jolt-prover` split |

## Purpose

This document is the optimization ledger for porting the current `jolt-core`
CPU prover into `jolt_backends::cpu` for the modular `jolt-prover` stack. It is
intentionally more mechanical than the design specs: each entry names the
current files, describes the optimization, states the porting requirement, and
identifies a test or benchmark signal.

The invariant is:

```text
If an optimization affects prover time, peak memory, proof size, or witness
materialization in jolt-core, the modular port must either preserve it or record
a measured, reviewed replacement.
```

This inventory is not a protocol spec. Protocol ownership is defined in
[`jolt-prover` Model Crate](./jolt-prover-model-crate.md) and
[`jolt-prover` CPU Backend Port](./jolt-prover-cpu-backend-port.md).

## Audit Scope

Included:

- prover orchestration in `jolt-core/src/zkvm/prover.rs`;
- witness generation and committed polynomial construction in
  `jolt-core/src/zkvm/witness.rs`;
- polynomial representations and evaluation/binding kernels in
  `jolt-core/src/poly/`;
- sumcheck prover kernels in `jolt-core/src/subprotocols/`;
- prover-side Jolt relation implementations under `jolt-core/src/zkvm/`;
- Dory/PCS prover-side helpers under `jolt-core/src/poly/commitment/dory/`;
- BlindFold prover-side witness/commitment construction under
  `jolt-core/src/subprotocols/blindfold/` and `jolt-core/src/zkvm/prover.rs`.

Excluded:

- verifier-only optimizations unless they affect prover data generation;
- guest execution optimizations in `tracer` except where the prover currently
  depends on tracer-local lazy trace behavior;
- SDK/host build optimizations.

## Preservation Rule

Each entry has a port target:

```text
protocol:
  belongs in jolt-prover / jolt-claims / jolt-verifier

witness:
  belongs in jolt-witness as a representation or provider capability

cpu-backend:
  belongs in `jolt_backends::cpu`, possibly with Jolt-specific code

pcs:
  belongs behind jolt-openings / concrete PCS implementation crates
```

For every ported frontier, reviewers should verify:

- no hot path was accidentally forced through dense `Vec<F>`;
- protocol order and transcript labels remain outside compute kernels;
- `jolt-backends` owns backend traits and request/result types;
- `jolt-prover` constructs backend requests and does not production-depend on
  concrete backend modules such as `jolt_backends::cpu`;
- optimized and reference paths agree on verifier-visible values;
- prove-time and peak-memory deltas are reported against `jolt-core`.

## Inventory

### Global Prover Shape

| ID | Optimization | Current Location | Mechanism | Port Target | Preservation / Test Signal |
|----|--------------|------------------|-----------|-------------|-----------------------------|
| OPT-GEN-001 | Minimum/power-of-two trace sizing | `zkvm/prover.rs` | Pads traces to at least 256 and then to a power of two, matching polynomial domain requirements and Dory shape assumptions. | protocol + witness | Parity test for `trace_length`, padded length, proof metadata, and verifier acceptance on tiny guests. |
| OPT-GEN-002 | Advice-aware trace padding | `zkvm/prover.rs::adjust_trace_length_for_advice` | Increases main trace domain until Dory main matrix can embed trusted/untrusted advice top-left blocks. | protocol + cpu-backend | Parity test for advice-heavy guests; assert same main/advice sigma/nu shape as `jolt-core`. |
| OPT-GEN-003 | Parallel `ram_K` computation | `zkvm/prover.rs::gen_from_trace`, `zkvm/ram/mod.rs` | Computes max remapped RAM address from trace and bytecode footprint with parallel scan. | witness + cpu-backend | `ram_K` parity against `jolt-core` on memory-heavy fixtures. |
| OPT-GEN-004 | One-time shared state derivation | `zkvm/prover.rs::gen_from_trace` | Derives `ReadWriteConfig`, `OneHotParams`, `UniformSpartanKey`, initial RAM state, and final RAM state once and reuses across stages. | protocol + witness | Stage frontier tests should assert config equality and avoid recomputation in profiles. |
| OPT-GEN-005 | Output truncation before preamble | `zkvm/prover.rs::gen_from_trace` | Truncates trailing zero output bytes so prover/verifier bind the same public IO. | protocol | Public IO parity and transcript preamble parity. |
| OPT-GEN-006 | `DoryGlobals` context guards | `zkvm/prover.rs`, `poly/commitment/dory/dory_globals.rs` | Sets layout/context-specific Dory dimensions for main, trusted advice, and untrusted advice without passing large shape structs everywhere. | pcs + cpu-backend | Preserve context selection from explicit backend/protocol config; tests for nested context correctness. |
| OPT-GEN-007 | `tracing`/pprof/allocative instrumentation | `zkvm/prover.rs`, many hot modules | Stable spans and optional heap flamegraphs identify stage and allocation regressions. | cpu-backend | Preserve comparable spans for commit, stages, Stage 8, RA, RLC, and BlindFold. |

### Witness Commitment And PCS

| ID | Optimization | Current Location | Mechanism | Port Target | Preservation / Test Signal |
|----|--------------|------------------|-----------|-------------|-----------------------------|
| OPT-COM-001 | Streaming committed witness generation | `zkvm/prover.rs::generate_and_commit_witness_polynomials`, `zkvm/witness.rs` | In CycleMajor Dory layout, streams padded trace chunks and commits row chunks without materializing all committed polynomials. | witness + cpu-backend + pcs | Commitment parity and peak-memory benchmark for CycleMajor. |
| OPT-COM-002 | Two-tier Dory chunk aggregation | `zkvm/prover.rs`, `poly/commitment/dory/commitment_scheme.rs` | Computes per-row chunk states, transposes to per-polynomial chunk state vectors, then aggregates to final commitments and opening hints. | pcs + cpu-backend | Commitment and hint parity; Stage 8 proof verification using reused hints. |
| OPT-COM-003 | AddressMajor materialized fallback | `zkvm/prover.rs::generate_and_commit_witness_polynomials` | Uses materialized parallel witness generation when streaming assumptions do not match AddressMajor layout. | cpu-backend | AddressMajor correctness and benchmark retained until a measured streaming replacement exists. |
| OPT-COM-004 | Per-polynomial one-hot `K` selection | `zkvm/witness.rs`, `OneHotParams` | `InstructionRa`, `BytecodeRa`, and `RamRa` pass the correct one-hot chunk size to PCS aggregation. | witness + cpu-backend | Commitment parity for each RA family and each chunk index. |
| OPT-COM-005 | Dedicated advice PCS contexts | `zkvm/prover.rs::generate_and_commit_untrusted_advice`, `generate_and_commit_trusted_advice` | Commits advice under separate Dory contexts with advice-specific dimensions. | protocol + pcs + cpu-backend | Trusted/untrusted advice commitment parity and verifier acceptance. |
| OPT-COM-006 | Retained opening hints | `zkvm/prover.rs`, `poly/opening_proof.rs` | Keeps PCS opening hints from commitment generation for Stage 8 joint opening. | cpu-backend + pcs | Stage 8 time and memory benchmark; proof verification. |
| OPT-COM-007 | One-hot row commitments with batch additions | `poly/one_hot_polynomial.rs::commit_rows` | Converts one-hot row support to grouped G1 additions instead of scalar MSM over dense rows. | pcs + cpu-backend | One-hot commitment parity against dense path on small fixtures; benchmark on large RA polynomials. |
| OPT-COM-008 | CycleMajor one-hot `T >> K` path | `poly/one_hot_polynomial.rs::commit_rows` | For common CycleMajor shapes, processes by cycle chunks and groups column indices by address to improve cache locality and batch additions. | pcs + cpu-backend | Benchmark on lookup-heavy traces; retain branch or measured replacement. |
| OPT-COM-009 | Dory prepared/cache state | `poly/commitment/dory/dory_globals.rs`, `jolt_dory_routines.rs` | Prepares/reuses Dory bases and layout-derived state for repeated commitments/openings. | pcs | Setup and commitment benchmarks; no repeated expensive preparation inside stages. |
| OPT-COM-010 | Blinded Dory commitments in ZK | `poly/commitment/dory/commitment_scheme.rs` | Adds hiding commitment behavior and ZK evaluation commitments for BlindFold mode. | pcs + cpu-backend | ZK commitment shape tests and BlindFold acceptance. |

### Polynomial Representations

| ID | Optimization | Current Location | Mechanism | Port Target | Preservation / Test Signal |
|----|--------------|------------------|-----------|-------------|-----------------------------|
| OPT-POLY-001 | Tagged MLE representation | `poly/multilinear_polynomial.rs` | Dispatches over dense field, compact scalar, one-hot, and RLC polynomial variants. | witness + cpu-backend | `jolt-witness` handles must preserve variant/capability information. |
| OPT-POLY-002 | Compact scalar polynomials | `poly/compact_polynomial.rs` | Stores small coefficients as `bool/u8/u16/u32/u64/u128/i64/i128/S128` until first bind. | witness + cpu-backend | Dense-vs-compact eval/bind parity and memory benchmark. |
| OPT-POLY-003 | First-bind small-scalar arithmetic | `poly/compact_polynomial.rs::bind`, `bind_parallel` | Uses scalar comparisons and difference multiplication to avoid full field conversion work. | cpu-backend | Microbench bind before/after; parity against dense bind. |
| OPT-POLY-004 | Drop compact coefficients after first bind | `poly/compact_polynomial.rs` | Frees small-scalar vector once `bound_coeffs` exists. | cpu-backend | Peak memory benchmark in sumcheck-heavy stages. |
| OPT-POLY-005 | Zero/one multiplication shortcuts | `field::OptimizedMul`, `dense_mlpoly.rs`, `compact_polynomial.rs`, `multilinear_polynomial.rs` | Avoids full field multiplication when coefficient or slope is 0/1. | cpu-backend | Microbench polynomial eval and bind; correctness parity. |
| OPT-POLY-006 | Dense split-eq evaluation | `poly/dense_mlpoly.rs::evaluate`, `split_eq_evaluate` | Splits equality table into two halves and computes nested dot products, parallel above threshold. | cpu-backend | Dense eval parity and benchmark over variable sizes. |
| OPT-POLY-007 | Compact split-eq evaluation | `poly/compact_polynomial.rs::split_eq_evaluate` | Same split-eq strategy while preserving compact coefficient arithmetic. | cpu-backend | Compact eval parity and benchmark. |
| OPT-POLY-008 | Inside-out evaluation | `poly/dense_mlpoly.rs`, `poly/compact_polynomial.rs` | Uses randomwalks-style inside-out folding, switching serial/parallel by size. | cpu-backend | Eval microbench; exact parity against dot-product eval. |
| OPT-POLY-009 | Batch dense polynomial evaluation | `poly/dense_mlpoly.rs::batch_evaluate` | Shares equality tables and cache locality across multiple dense polynomials. | cpu-backend | Batch-vs-individual parity and benchmark. |
| OPT-POLY-010 | Parallel linear combinations over mixed encodings | `poly/dense_mlpoly.rs::linear_combination`, `multilinear_polynomial.rs` | Builds dense linear combinations while using compact field multiplication paths when available. | cpu-backend | Stage 8/RLC and commitment benchmarks. |
| OPT-POLY-011 | One-hot evaluation without dense materialization | `poly/one_hot_polynomial.rs::evaluate` | Evaluates one-hot RA polynomial by summing eq-address/eq-cycle contributions for nonzero indices. | witness + cpu-backend | One-hot vs dense parity for all layout shapes. |
| OPT-POLY-012 | One-hot vector-matrix product | `poly/one_hot_polynomial.rs::vector_matrix_product` | Computes Dory VMP directly from one-hot indices, with CycleMajor fast path. | pcs + cpu-backend | Dory opening benchmark and dense parity on small shapes. |
| OPT-POLY-013 | RLC polynomial variant | `poly/rlc_polynomial.rs` | Represents joint RLC polynomial either materialized or streaming with advice side inputs. | witness + cpu-backend + pcs | Stage 8 memory benchmark and proof verification. |
| OPT-POLY-014 | Low-optimized dot product | `poly/dense_mlpoly.rs::evaluate_at_chi_low_optimized` | Uses low-level optimized dot product for chi/evaluation tables. | cpu-backend | Microbench dot product and dense eval. |

### Equality, Lagrange, And Univariate Helpers

| ID | Optimization | Current Location | Mechanism | Port Target | Preservation / Test Signal |
|----|--------------|------------------|-----------|-------------|-----------------------------|
| OPT-EQ-001 | Serial/parallel eq table threshold | `poly/eq_poly.rs` | Uses serial eq table generation for small `n` and parallel DP for large `n`. | cpu-backend | Eq table microbench; parity tests already present should move or mirror. |
| OPT-EQ-002 | Cached eq prefix tables | `poly/eq_poly.rs::evals_cached`, `evals_cached_rev` | Produces all prefix/suffix eq tables for repeated binding/evaluation. | cpu-backend | Split-eq and Spartan streaming tests. |
| OPT-EQ-003 | Aligned-block eq evaluation | `poly/eq_poly.rs::evals_for_aligned_block`, `evals_for_max_aligned_block` | Computes eq tables only for aligned suffix blocks instead of full domains. | cpu-backend | Tests for block decomposition and benchmarks in users. |
| OPT-EQ-004 | Gruen/Dao-Thaler split-eq | `poly/split_eq_poly.rs` | Factors eq into cached prefix tables and active streaming-window tables. | cpu-backend | Spartan outer/product parity and streaming-vs-linear tests. |
| OPT-EQ-005 | Streaming-window active eq tables | `poly/split_eq_poly.rs::E_out_in_for_window`, `E_active_for_window` | Computes only active window equality tables for streaming sumcheck windows. | cpu-backend | Streaming sumcheck benchmark. |
| OPT-EQ-006 | Direct degree-2/degree-3 interpolation | `poly/unipoly.rs::from_evals` | Avoids general Vandermonde interpolation for common low-degree round polynomials. | cpu-backend | Sumcheck proof construction benchmark. |
| OPT-EQ-007 | Toom/infinity interpolation | `poly/unipoly.rs::from_evals_toom` | Uses finite evaluations plus infinity coefficient for higher-degree messages. | cpu-backend | RAM val-check and relation-specific parity tests. |
| OPT-EQ-008 | Compressed univariate polynomials | `poly/unipoly.rs::compress`, `decompress`, `eval_from_hint` | Omits linear term when verifier can recover it from `H(0)+H(1)`. | protocol + cpu-backend | Proof-size parity and verifier acceptance. |
| OPT-EQ-009 | Batched Lagrange evaluation | `poly/lagrange_poly.rs` | Provides interpolation/evaluation helpers and `evaluate_many` for repeated small-domain evaluation. | cpu-backend | Uni-skip and BlindFold parity. |

### RA, One-Hot, And Lookup Pushforwards

| ID | Optimization | Current Location | Mechanism | Port Target | Preservation / Test Signal |
|----|--------------|------------------|-----------|-------------|-----------------------------|
| OPT-RA-001 | Fixed-size RA index arrays | `poly/shared_ra_polys.rs::RaIndices` | Stores instruction/bytecode/RAM chunks in stack-sized arrays, avoiding per-cycle heap allocation. | witness + cpu-backend | Allocation profile over lookup-heavy traces. |
| OPT-RA-002 | Single-pass RA index generation | `poly/shared_ra_polys.rs::compute_all_G_and_ra_indices` | Computes RA indices and `G` pushforwards in one trace pass. | witness + cpu-backend | Parity against independent dense/one-hot eval; trace-pass count benchmark. |
| OPT-RA-003 | Split-eq RA pushforward | `poly/shared_ra_polys.rs::compute_all_G_impl` | Splits cycle eq table into high/low halves for cache-friendly pushforward. | cpu-backend | RA pushforward microbench and parity. |
| OPT-RA-004 | Per-thread partial `G` accumulators | `poly/shared_ra_polys.rs::compute_all_G_impl` | Uses per-thread instruction/bytecode/RAM partial tables before reduction. | cpu-backend | Parallel scaling benchmark. |
| OPT-RA-005 | Unreduced limb accumulation | `poly/shared_ra_polys.rs`, read/write matrix modules | Accumulates products in unreduced field representations, then reduces touched entries. | cpu-backend | Microbench and field correctness tests. |
| OPT-RA-006 | Touched bitsets for sparse updates | `poly/shared_ra_polys.rs` | Tracks only modified one-hot entries for clearing and reduction. | cpu-backend | Allocation/write-count benchmark. |
| OPT-RA-007 | Delayed RA materialization | `poly/ra_poly.rs` | `RaPolynomial` stays in Round1/Round2/Round3 compact states before becoming full MLE. | witness + cpu-backend | Sumcheck memory benchmark; parity vs dense RA. |
| OPT-RA-008 | Shared RA delayed materialization | `poly/shared_ra_polys.rs::SharedRaPolynomials` | Shares eq tables and non-transposed RA indices across all RA families. | witness + cpu-backend | Hamming/RA virtualization memory benchmark. |
| OPT-RA-009 | Bounds-limited RA chunks | `poly/shared_ra_polys.rs`, `OneHotParams` | Uses hard maximum chunk counts and `u8` chunk indices because practical `K <= 256`. | witness + cpu-backend | Config tests for instruction/bytecode/RAM dimensions. |
| OPT-RA-010 | Family-order batching | `shared_ra_polys.rs`, `zkvm/witness.rs` | Processes RA families in stable order instruction, bytecode, RAM for cache and opening-order compatibility. | protocol + cpu-backend | Opening-order parity and dense eval tests. |

### Read/Write Sparse Matrices

| ID | Optimization | Current Location | Mechanism | Port Target | Preservation / Test Signal |
|----|--------------|------------------|-----------|-------------|-----------------------------|
| OPT-RW-001 | Sparse `K x T` matrix representation | `subprotocols/read_write_matrix/*` | Represents only events rather than dense RAM/register matrices. | witness + cpu-backend | Dense-reference tests on small traces; memory benchmark on large traces. |
| OPT-RW-002 | CycleMajor layout | `read_write_matrix/cycle_major.rs` | Sorts entries by `(row, col)` for binding cycle variables first. | cpu-backend | CycleMajor parity and benchmark. |
| OPT-RW-003 | AddressMajor layout | `read_write_matrix/address_major.rs` | Sorts entries by `(col, row)` for binding address variables first. | cpu-backend | AddressMajor parity and benchmark. |
| OPT-RW-004 | Parallel recursive merge binding | `cycle_major.rs`, `address_major.rs` | Splits long sorted rows/columns and merges in parallel, falling back below threshold. | cpu-backend | Bind microbench and sparse matrix parity. |
| OPT-RW-005 | Dry-run exact output sizing | `cycle_major.rs`, `address_major.rs` | Computes bound entry length before writing output. | cpu-backend | Allocation and correctness tests around sparse merges. |
| OPT-RW-006 | `MaybeUninit` / spare-capacity writes | `cycle_major.rs`, `address_major.rs` | Writes bound sparse entries into uninitialized capacity and sets length after disjoint writes. | cpu-backend | Miri-like safety review where possible; fuzz small matrices. |
| OPT-RW-007 | One-hot coefficient lookup tables | `read_write_matrix/one_hot_coeffs.rs` | Maintains bound one-hot coefficient tables for RA/WA instead of recomputing. | cpu-backend | Register/RAM read-write sumcheck benchmark. |
| OPT-RW-008 | Unreduced prover message contributions | `cycle_major.rs`, `ram.rs`, `registers.rs` | Returns unreduced products for accumulation before final reduction. | cpu-backend | Relation message parity and microbench. |
| OPT-RW-009 | Layout conversion by parallel sort/map | `address_major.rs::From<ReadWriteMatrixCycleMajor>` | Converts CycleMajor entries to AddressMajor with parallel sort and map. | cpu-backend | Conversion parity and layout benchmark. |
| OPT-RW-010 | Specialized RAM/register entries | `read_write_matrix/ram.rs`, `registers.rs` | Separate entry structs encode relation-specific prev/next values and coefficients compactly. | cpu-backend | RAM/register read-write frontier tests. |

### Sumcheck And Stage Execution

| ID | Optimization | Current Location | Mechanism | Port Target | Preservation / Test Signal |
|----|--------------|------------------|-----------|-------------|-----------------------------|
| OPT-SC-001 | Front-loaded batched sumcheck | `subprotocols/sumcheck.rs::BatchedSumcheck::prove` | Batches parallel sumchecks with transcript-derived coefficients to reduce proof/verifier work. | protocol + cpu-backend | Proof-size and challenge/order parity. |
| OPT-SC-002 | Different-round padding by powers of two | `sumcheck.rs` | Scales shorter sumcheck input claims by powers of two under batched domains. | protocol | Exact input-claim parity and verifier acceptance. |
| OPT-SC-003 | Dummy constant rounds for inactive instances | `sumcheck.rs` | Emits constant round polynomial for instances whose variables are dummy in the batched max domain. | protocol + cpu-backend | Batched sumcheck parity. |
| OPT-SC-004 | Compressed round polynomials | `sumcheck.rs`, `unipoly.rs` | Stores compressed univariate messages in clear proofs. | protocol + cpu-backend | Proof-size parity. |
| OPT-SC-005 | ZK committed round polynomials | `sumcheck.rs::prove_zk`, `univariate_skip.rs::prove_uniskip_round_zk` | Commits round coefficients with Pedersen, retaining coefficients/blindings for BlindFold. | protocol + cpu-backend + `jolt-blindfold` | ZK proof shape and BlindFold acceptance. |
| OPT-SC-006 | Committed output-claim rows | `sumcheck.rs`, `univariate_skip.rs` | Commits output claims in chunks and reuses them for BlindFold row commitments. | protocol + cpu-backend | Output-claim ordering and ZK acceptance tests. |
| OPT-SC-007 | Univariate skip first round | `subprotocols/univariate_skip.rs`, `zkvm/spartan/outer.rs`, `product.rs` | Handles high-degree first round separately to avoid generalizing every later sumcheck round. | protocol + cpu-backend | Stage 1/2 parity and proof-size checks. |
| OPT-SC-008 | Streaming sumcheck windows | `subprotocols/streaming_sumcheck.rs`, `streaming_schedule.rs` | Computes early messages from trace/windows and switches to materialized linear stage later. | cpu-backend | Streaming-vs-linear parity and memory benchmark. |
| OPT-SC-009 | Cost-aware `HalfSplitSchedule` | `streaming_schedule.rs` | Chooses window sizes by degree-dependent cost model. | cpu-backend | Schedule unit tests and stage benchmark. |
| OPT-SC-010 | `LinearOnlySchedule` fallback | `streaming_schedule.rs` | Disables streaming for stages where materialized path is preferable. | cpu-backend | Stage 1 current behavior parity. |
| OPT-SC-011 | End-of-protocol `finalize` hook | `sumcheck.rs`, `sumcheck_prover.rs` | Lets delayed-binding instances flush before caching openings. | cpu-backend | Relation tests with delayed materialization. |
| OPT-SC-012 | Background drop of stage instances | `zkvm/prover.rs`, `utils::thread::drop_in_background_thread` | Releases large prover instances asynchronously to reduce wall-clock/peak interference. | cpu-backend | Peak memory and timing comparison. |

### Spartan And R1CS-Oriented Compute

| ID | Optimization | Current Location | Mechanism | Port Target | Preservation / Test Signal |
|----|--------------|------------------|-----------|-------------|-----------------------------|
| OPT-SP-001 | Uniform Spartan key reuse | `zkvm/prover.rs`, `zkvm/r1cs/key.rs` | Builds uniform key once from trace length. | protocol + cpu-backend | Stage 1/2 parity and no repeated key construction. |
| OPT-SP-002 | Streaming Spartan outer remainder | `zkvm/spartan/outer.rs`, `streaming_sumcheck.rs` | Uses shared state and streaming windows for outer sumcheck instead of fully materializing all data. | cpu-backend | Stage 1 memory/time benchmark. |
| OPT-SP-003 | Gruen split-eq in Spartan | `poly/split_eq_poly.rs`, `spartan/outer.rs`, `spartan/product.rs` | Factors eq tables for multiquadratic sumcheck compute. | cpu-backend | Stage 1/2 parity. |
| OPT-SP-004 | Product virtualization split | `zkvm/spartan/product.rs` | Uses uniskip plus product-virtual remainder flow rather than one monolithic product sumcheck. | protocol + cpu-backend | Stage 2 parity and proof shape. |
| OPT-SP-005 | Product cycle input state | `zkvm/r1cs/inputs.rs`, `spartan/product.rs` | Specialized product inputs avoid generic R1CS row evaluation in hot loops. | cpu-backend | Stage 2 benchmark. |
| OPT-SP-006 | Shift/instruction-input specialized polys | `spartan/shift.rs`, `instruction_input.rs`, `poly/identity_poly.rs` | Uses identity/operand polynomial formulas instead of dense materialization. | witness + cpu-backend | Stage 3 parity and dense-reference tests. |

### Lookup, Bytecode, RAM, Register Relation Kernels

| ID | Optimization | Current Location | Mechanism | Port Target | Preservation / Test Signal |
|----|--------------|------------------|-----------|-------------|-----------------------------|
| OPT-REL-001 | Prefix/suffix lookup decomposition | `poly/prefix_suffix.rs`, `instruction_lookups/read_raf_checking.rs`, `bytecode/read_raf_checking.rs` | Decomposes lookup polynomials into reusable prefix and suffix parts. | cpu-backend | Lookup-heavy benchmarks; dense-reference checks. |
| OPT-REL-002 | Prefix registry reuse | `poly/prefix_suffix.rs::PrefixRegistry` | Shares prefix polynomial checkpoints and cached polys across decompositions. | cpu-backend | Instruction read-RAF benchmark. |
| OPT-REL-003 | Per-index `OnceLock` sumcheck eval cache | `poly/prefix_suffix.rs::CachedPolynomial` | Caches repeated degree-2 evals for prefix polys within a round. | cpu-backend | Benchmark with cache on/off where feasible. |
| OPT-REL-004 | Bytecode read-RAF stage gamma precomputation | `zkvm/bytecode/read_raf_checking.rs` | Precomputes stage-specific gamma powers/values for bytecode read-RAF batching. | protocol + cpu-backend | Stage 5/bytecode parity. |
| OPT-REL-005 | Simultaneous Val(k) polynomial computation | `zkvm/bytecode/read_raf_checking.rs` | Computes multiple stage-specific Val polynomials together to avoid repeated passes. | cpu-backend | Bytecode read-RAF benchmark. |
| OPT-REL-006 | RAM val-check batched relation | `zkvm/ram/val_check.rs` | Combines ValEvaluation and ValFinal at one address point with explicit gamma. | protocol + cpu-backend | Stage 4 parity and advice tests. |
| OPT-REL-007 | RAM val-check Toom message | `zkvm/ram/val_check.rs::compute_message` | Uses Toom/infinity interpolation for cubic relation message. | cpu-backend | RAM val-check unit/frontier tests. |
| OPT-REL-008 | RAM output check public IO masks | `zkvm/ram/output_check.rs` | Encodes IO output constraints using precomputed public mask evaluations. | protocol + cpu-backend | IO guest parity and tampering tests. |
| OPT-REL-009 | RAM RAF evaluation with structured polynomials | `zkvm/ram/raf_evaluation.rs` | Evaluates RAM read address/value final checks without dense RAM matrix. | cpu-backend | Stage 2 parity. |
| OPT-REL-010 | Register val-evaluation compact relation | `zkvm/registers/val_evaluation.rs` | Evaluates register read/write values using compact trace-derived values. | witness + cpu-backend | Stage 5 parity. |
| OPT-REL-011 | Increment claim reduction specialized paths | `zkvm/claim_reductions/increments.rs` | Specialized reduction for `RamInc`/`RdInc` claims and final openings. | protocol + cpu-backend | Stage 6 parity and Stage 8 opening order. |
| OPT-REL-012 | Hamming weight reduction over RA families | `zkvm/claim_reductions/hamming_weight.rs` | Batches RA hamming-weight claims across one-hot families. | protocol + cpu-backend | Stage 7 parity and RA opening tests. |
| OPT-REL-013 | Advice claim reduction two-phase state | `zkvm/claim_reductions/advice.rs`, `zkvm/prover.rs` | Advice reduction spans Stage 6/7 and caches prover state between stages. | protocol + cpu-backend | Advice e2e in standard/ZK; state reuse benchmark. |
| OPT-REL-014 | Booleanity phase splitting | `subprotocols/booleanity.rs`, `ram/hamming_booleanity.rs` | Booleanity protocols split address/cycle work and cache openings at the end. | protocol + cpu-backend | Stage 6/7 parity. |
| OPT-REL-015 | Verifier-evaluable claim cache | `subprotocols/sumcheck_claim.rs` | Caches repeated verifier-evaluable polynomial evaluations while computing expected output claims. | protocol + cpu-backend | Claim-reduction parity and microbench. |

### Stage 8 And Opening Proof

| ID | Optimization | Current Location | Mechanism | Port Target | Preservation / Test Signal |
|----|--------------|------------------|-----------|-------------|-----------------------------|
| OPT-OPEN-001 | Opening accumulator | `poly/opening_proof.rs::ProverOpeningAccumulator` | Caches committed, virtual, and advice opening claims during stages for one final proof. | protocol + cpu-backend | Missing-opening assertions and Stage 8 parity. |
| OPT-OPEN-002 | Pending claims for ZK output rows | `poly/opening_proof.rs`, `sumcheck.rs`, `univariate_skip.rs` | Tracks output claims and IDs before committing them for BlindFold. | protocol + cpu-backend | Output-claim row order tests. |
| OPT-OPEN-003 | Streaming Stage 8 RLC | `zkvm/prover.rs::prove_stage8`, `poly/opening_proof.rs`, `poly/rlc_polynomial.rs` | Builds joint polynomial directly from trace/advice/opening hints without regenerating all witnesses. | protocol + cpu-backend + pcs | Stage 8 time/memory benchmark and proof verification. |
| OPT-OPEN-004 | Dense-polynomial embedding factor | `zkvm/prover.rs::prove_stage8` | Scales `RamInc`/`RdInc` by `eq(r_address, 0)` to account for Dory matrix zero-padding. | protocol | Joint claim parity and tampering test. |
| OPT-OPEN-005 | Advice Lagrange embedding factor | `poly/opening_proof.rs::compute_advice_lagrange_factor`, `zkvm/prover.rs` | Scales advice claims from advice-domain point into main Dory opening point. | protocol + cpu-backend | Advice Stage 8 parity. |
| OPT-OPEN-006 | RLC streaming VMP | `poly/rlc_polynomial.rs::streaming_vector_matrix_product` | Computes Dory VMP from streaming trace chunks plus advice blocks. | cpu-backend + pcs | Stage 8 benchmark. |
| OPT-OPEN-007 | RLC folded one-hot tables | `poly/rlc_polynomial.rs::FoldedOneHotTables` | Premultiplies RLC coefficients and eq factors for one-hot families. | cpu-backend | RLC VMP microbench. |
| OPT-OPEN-008 | Materialized fallback for tests/layouts | `poly/rlc_polynomial.rs::materialize_from_context`, `address_major_vmp` | Allows correctness testing and AddressMajor path by materializing from streaming context. | cpu-backend | Dense-vs-streaming RLC parity. |
| OPT-OPEN-009 | Joint claim computed before PCS proof | `zkvm/prover.rs::prove_stage8` | Computes RLC joint claim once from gamma powers and scaled claims. | protocol | Joint-claim parity and verifier acceptance. |
| OPT-OPEN-010 | ZK evaluation commitment extraction | `zkvm/prover.rs::prove_stage8`, `dory/commitment_scheme.rs` | Extracts `y_com` and hidden evaluation blinding for BlindFold. | protocol + pcs | ZK Stage 8 and BlindFold tests. |

### BlindFold / ZK Prover Data

| ID | Optimization | Current Location | Mechanism | Port Target | Preservation / Test Signal |
|----|--------------|------------------|-----------|-------------|-----------------------------|
| OPT-ZK-001 | Pedersen round commitments | `subprotocols/sumcheck.rs::prove_zk`, `univariate_skip.rs` | Sends commitments instead of clear coefficients and stores private coefficients/blindings. | protocol + cpu-backend + `jolt-blindfold` | ZK proof shape and BlindFold acceptance. |
| OPT-ZK-002 | Output-claim row commitment reuse | `zkvm/prover.rs::prove_blindfold` | Reuses committed output-claim rows from sumcheck as Hyrax rows. | cpu-backend + `jolt-blindfold` | Row-count/order tests and BlindFold proof verification. |
| OPT-ZK-003 | BlindFold row commitments/openings | `zkvm/prover.rs::prove_blindfold`, `subprotocols/blindfold/protocol.rs` | Commits non-round BlindFold witness rows in parallel and opens folded witness/error rows at caller-supplied points without moving transcript ownership into the backend. | cpu-backend + `jolt-blindfold` | BlindFold row hook coverage, row opening parity, and timing benchmark. |
| OPT-ZK-004 | Baked challenge/public vectors | `zkvm/prover.rs::prove_blindfold` | Bakes transcript-derived values into BlindFold R1CS coefficients once. | protocol + `jolt-blindfold` | R1CS satisfiability and verifier acceptance. |
| OPT-ZK-005 | Witness assignment after all stages | `subprotocols/blindfold/witness.rs`, `zkvm/prover.rs` | Builds BlindFold witness only after all committed rounds, output rows, and Stage 8 data are available. | protocol + `jolt-blindfold` | Peak memory and correctness tests. |
| OPT-ZK-006 | Random satisfying instance folding | `subprotocols/blindfold/folding.rs`, `protocol.rs` | Uses Nova-style folding with random satisfying instance to hide witness; `jolt-blindfold` owns protocol scheduling while the CPU backend owns transcript-free error-row materialization and fold arithmetic hooks. | `jolt-blindfold` + cpu-backend | BlindFold hook coverage, CPU formula parity tests, standalone/e2e tests, and timing benchmark. |
| OPT-ZK-007 | Hyrax grid layout computation | `subprotocols/blindfold/layout.rs`, `relaxed_r1cs.rs` | Packs witness/error rows according to Hyrax layout and vector commitment capacity. | `jolt-blindfold` | Layout snapshot tests and proof verification. |

### Memory Management And Parallel Execution

| ID | Optimization | Current Location | Mechanism | Port Target | Preservation / Test Signal |
|----|--------------|------------------|-----------|-------------|-----------------------------|
| OPT-MEM-001 | Rayon over trace/polynomial loops | many files | Uses `par_iter`, `into_par_iter`, `par_chunks`, `rayon::join`, and `par_bridge` in hot loops. | cpu-backend | Parallel scaling benchmarks. |
| OPT-MEM-002 | `unsafe_allocate_zero_vec` | `utils::thread`, used in `rlc_polynomial.rs`, `shared_ra_polys.rs`, sparse matrices, etc. | Allocates zeroed vectors efficiently when zero bit-pattern is valid. | cpu-backend | Safety review per type and allocation benchmarks. |
| OPT-MEM-003 | `MaybeUninit` writes | `read_write_matrix/*`, `compact_polynomial.rs` | Avoids default initialization for exact-size output buffers. | cpu-backend | Fuzz small sparse matrices and bind parity. |
| OPT-MEM-004 | `mem::take` state transfer | `ra_poly.rs`, `shared_ra_polys.rs`, `address_major.rs`, `prover.rs` | Moves large vectors out of old states without cloning. | cpu-backend | Allocation profile around binds/stage transitions. |
| OPT-MEM-005 | `Arc` sharing of trace and indices | `zkvm/prover.rs`, `ra_poly.rs`, `rlc_polynomial.rs` | Shares large immutable trace/index data across polys and Stage 8 without cloning. | witness + cpu-backend | Clone-count/peak-memory audit. |
| OPT-MEM-006 | Background drops | `zkvm/prover.rs`, `utils::thread::drop_in_background_thread` | Drops large prover instances asynchronously after stages. | cpu-backend | Peak memory and wall-clock benchmark. |
| OPT-MEM-007 | Exact capacity allocation | many files | Uses known lengths, `with_capacity`, and set_len after writes in hot paths. | cpu-backend | Allocation benchmark and safety review. |
| OPT-MEM-008 | Stable chunk sizes from Rayon thread count | `rlc_polynomial.rs`, `one_hot_polynomial.rs`, `shared_ra_polys.rs` | Partitions work by thread count to reduce contention and allocation churn. | cpu-backend | Parallel scaling benchmark across thread counts. |
| OPT-MEM-009 | Layout-specific hot branches | `one_hot_polynomial.rs`, `rlc_polynomial.rs`, `prover.rs` | Switches algorithms for CycleMajor vs AddressMajor and common `T >= row_len` cases. | cpu-backend + pcs | Layout-specific benchmarks. |
| OPT-MEM-010 | Optional allocative flamegraphs | `zkvm/prover.rs`, many `Allocative` derives | Enables heap flamegraph inspection for memory regressions. | cpu-backend | Preserve or replace memory introspection. |

### Field And Arithmetic Micro-Optimizations

| ID | Optimization | Current Location | Mechanism | Port Target | Preservation / Test Signal |
|----|--------------|------------------|-----------|-------------|-----------------------------|
| OPT-FLD-001 | Small scalar trait conversions | `utils::small_scalar`, `compact_polynomial.rs` | Converts small integers to fields only when needed and supports difference multiplication. | cpu-backend | Compact polynomial benchmark. |
| OPT-FLD-002 | Unreduced multiplication products | `field`, `read_write_matrix/*`, `shared_ra_polys.rs`, `mles_product_sum.rs` | Accumulates products before reducing to field elements. | cpu-backend | Relation microbench and correctness tests. |
| OPT-FLD-003 | Specialized sum-of-products degree paths | `subprotocols/mles_product_sum.rs` | Provides specialized product-sum evaluators, including degree-4 path. | cpu-backend | Product-sum microbench. |
| OPT-FLD-004 | Low-level group addition batching | `one_hot_polynomial.rs`, `jolt_optimizations::batch_g1_additions_multi` | Uses optimized group addition routines for sparse one-hot commitments. | pcs + cpu-backend | One-hot commitment benchmark. |
| OPT-FLD-005 | BN254/Dory GLV/prepared routines | `poly/commitment/dory/jolt_dory_routines.rs`, `msm/` | Uses optimized group/MSM routines through Dory PCS. | pcs | Commitment/opening benchmarks. |

## Required Port Checklist

Before a modular CPU backend frontier replaces a `jolt-core` frontier, check:

- [ ] every optimization ID used by that frontier has an owner in the modular
      design;
- [ ] every optimization ID used by that frontier has either a parity test, a
      benchmark, or a documented reason why it is not applicable;
- [ ] no CPU backend function owns transcript labels or challenge derivation;
- [ ] all dense materialization is intentional and benchmarked;
- [ ] `jolt-core` and modular outputs agree for commitments, stage outputs,
      opening events, and verifier-visible proof fields where deterministic;
- [ ] prove-time and peak-memory deltas are reported for `muldiv` in standard
      and ZK modes;
- [ ] advice and Stage 8 deltas are reported when the frontier touches advice or
      openings.

## Suggested Test Matrix

Correctness:

```bash
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host,zk
```

Performance fixtures to keep in rotation:

```text
muldiv
fibonacci
sha2
sha2-chain
sha3
sha3-chain
memory-ops
advice-demo
advice-consumer
```

Microbench groups to add or preserve:

```text
eq table generation
dense/compact evaluation
compact binding
one-hot commitment
RA pushforward
read/write sparse bind
streaming vs linear sumcheck
Stage 8 streaming RLC
Dory commit/open
BlindFold row commitment
```

## Maintenance Rule

When a PR changes any of the files listed in this inventory, the PR should do
one of:

- update the relevant optimization entry;
- mark the optimization as intentionally removed with a benchmarked reason;
- add a new optimization entry if the PR introduces a new fast path.

The CPU backend porting spec should be considered incomplete for any frontier
whose active optimization IDs are not accounted for here.
