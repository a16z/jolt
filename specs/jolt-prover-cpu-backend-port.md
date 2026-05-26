# Spec: `jolt-prover` CPU Backend Port

| Field | Value |
|-------|-------|
| Author(s) | Markos Georghiades, Codex |
| Created | 2026-05-23 |
| Status | draft |
| PR | TBD |

## Summary

This spec describes how to port the current optimized `jolt-core` CPU prover
into the modular `jolt-prover` stack without losing prover-time or memory
optimizations. The main design constraint is strict separation between protocol
orchestration and compute: `jolt-prover` decides what must be proved and in what
order, while `jolt_backends::cpu` executes already-resolved algebraic work using
whatever specialized representations and fused kernels are needed for
performance.

This is a companion to [`jolt-prover` Model Crate](./jolt-prover-model-crate.md).
That spec defines the modular proof model; this spec defines how to preserve the
current CPU fast path while moving into that model.

Correctness, parity, and performance gates for each ported frontier are defined
in [`jolt-prover` Frontier Harness](./jolt-prover-frontier-harness.md).

The rigorous optimization ledger for the current `jolt-core` CPU prover is
[`Jolt Core Prover Optimization Inventory`](./jolt-core-prover-optimization-inventory.md).
Every ported frontier should account for the optimization IDs it touches.

## Goal

Port the existing `jolt-core` CPU prover fast path into
`jolt_backends::cpu`, a canonical modular CPU backend that is allowed to be
Jolt-specific and highly optimized internally, but is driven only by
protocol-resolved backend requests constructed by `jolt-prover`.

## Core Principle

```text
protocol chooses what must be proved
compute only computes the requested algebra
```

`jolt_backends::cpu` may use relation-specific kernels, fused passes, global
PCS contexts, unsafe preallocation, streaming trace passes, cached state, and
Jolt-specific layouts. It must not decide stage order, transcript labels,
challenge derivation, claim formulas, proof visibility, opening order, verifier
output shape, or wrapper/Dory-assist handoff semantics.

## Ownership Boundary

`jolt-prover` owns:

- stage order and stage dependencies;
- transcript absorbs, labels, challenge derivation, and RNG threading;
- clear vs BlindFold selection;
- claim formulas and metadata from `jolt-claims`;
- batching order and output-claim order;
- opening requests and opening ID order;
- verifier-owned proof and stage-output construction;
- mapping backend outputs into `jolt-verifier` proof objects;
- wrapper, recursion, and Dory-assist handoff decisions.

`jolt_backends::cpu` owns:

- concrete polynomial and witness representations used by the fast path;
- relation-specific algebraic kernels;
- streaming and materialized execution strategies;
- parallel map-reduce loops;
- sparse matrix binding and evaluation;
- Dory chunk processing, aggregation hints, and PCS compute through
  `jolt-openings`/PCS traits;
- backend caches, scratch buffers, and memory-release policy;
- fast construction of private ZK material requested by protocol requests.

`jolt-witness` owns:

- namespace-generic witness oracles/views;
- materialized, compact, derived, one-hot, sparse, and streaming witness views;
- public value and opening witness APIs;
- protocol-specific witness providers, starting with `protocols::jolt_vm`.

## Protocol-Resolved Backend Requests

The exact Rust traits can evolve, but the architectural boundary should be
request-based. `jolt-prover` builds protocol-resolved backend requests;
`jolt_backends::cpu` executes them.

The dependency direction is:

```text
jolt-prover   -> jolt-backends      // traits and request/result types
jolt-backends -/-> jolt-prover
```

`jolt-backends` owns backend traits and request/result types. `jolt-prover`
constructs requests according to protocol order and maps backend results into
verifier-owned proof objects. The SDK, CLI, host layer, or test harness selects
a concrete backend and passes it to `jolt-prover`.

Representative request families:

```text
CommitmentRequest:
  committed polynomial IDs, OracleRef slots, ViewRequirement values,
  commitment order, layout, PCS context, transcript absorb slots, advice
  inclusion policy, retention hints

StageRequest:
  stage ID, visibility, dependency outputs, relation list, batching order,
  claim slots, output slots, ZK material requested, witness view requirements

SumcheckRequest:
  relation oracle refs, view requirements, degree bounds, round count, input
  claim expression, batching coefficients, expected output ordering

OpeningRequest:
  final opening ID order, polynomial oracle refs/views, view requirements,
  points, scaling factors, RLC coefficients, clear/ZK binding mode,
  Dory-assist export slots

BlindFoldRequest:
  committed round slots, output-claim rows, challenge/public metadata,
  opening-proof binding data, vector-commitment setup
```

Requests should use the standardized witness interop vocabulary from
[`jolt-witness` crate spec](./jolt-witness-crate.md): `OracleRef`,
`OracleDescriptor`, `ViewRequirement`, `MaterializationPolicy`, and
`RetentionHint`-style concepts. This keeps protocol intent explicit while
letting the CPU backend decide whether to borrow compact data, replay a stream,
materialize in host memory, cache derived state, or keep data until Stage 8 or
BlindFold. The logical meaning of each oracle, claim, opening, public value,
and challenge comes from `jolt-claims`; the witness layer only describes how
the concrete data for that logical object can be accessed.

Backend outputs should be keyed by request slots rather than invented protocol IDs:

```text
SumcheckOutput:
  round messages/proof payload
  challenges consumed by backend state
  output claim values by request slot
  opening evaluations by request slot
  private ZK coefficients/blindings by request slot
  backend hints/caches for later requests
```

`jolt-prover` maps those slots back to `jolt-claims` IDs and
`jolt-verifier` fields. This prevents kernels from discovering protocol meaning
by inspecting IDs and deciding what stage they are in.

The first implemented CPU boundary should be a deliberately small bridge:
resolve each slot-keyed `ViewRequirement` through a `jolt-witness` provider,
verify that the provider's descriptor matches the requested encoding, and
return the resolved descriptors by request slot. Real commitment, sumcheck, and
opening payloads then extend the same result channel instead of introducing a
parallel backend-private protocol map.

For commitment requests with streaming witness requirements, the bridge should
also consume committed streams from the provider and report slot-keyed chunk
metadata, including representation kind and row count. This tests the intended
shape before real PCS work lands: the backend sees streaming `InstructionRa`,
`BytecodeRa`, `RamRa`, and increment data through the witness contract, not
through tracer internals or backend-private lookup logic.

The corresponding `jolt-prover` frontier should be equally small: build a
`CommitmentRequest` from `CommittedWitnessProvider::committed_oracle_order`,
assign stable slots in that order, request streaming materialization, and hand
the request to the selected backend. Real Dory commitments and opening hints
are then added as additional slot-keyed outputs of this same request.

Slot order is a backend scheduling/debugging contract, not a proof-layout
contract. The CPU backend returns `CommittedPolynomialOutput { slot, oracle,
rows, PCS::Output, PCS::OpeningHint }`; `jolt-prover` assembles verifier proof
fields by the `oracle`'s logical committed-polynomial ID. This prevents the
legacy proof payload order from being confused with the Stage 8 final opening
order:

```text
proof/transcript payload:
  RdInc, RamInc, InstructionRa(*), RamRa(*), BytecodeRa(*)

final PCS opening:
  RamInc, RdInc, InstructionRa(*), BytecodeRa(*), RamRa(*)
```

With `field-inline`, `FieldRdInc` participates only in the field-inline proof
surface and Stage 8 final opening order. Advice commitments remain in their
dedicated trusted/untrusted proof/API fields, not inside the base
`JoltCommitments` RA vectors.

The first PCS-capable CPU implementation should preserve representation at the
boundary:

- compact integer and dense field streams should feed fixed PCS rows
  incrementally through the streaming PCS trait and finish with both the
  commitment and opening hint; they should not build a full field-vector
  polynomial before commitment;
- one-hot streams must become a sparse one-hot polynomial representation, such
  as `jolt_poly::OneHotPolynomial`, so Dory can use generator lookup rather
  than a dense `T * K` field vector;
- the backend result must return both `PCS::Output` and `PCS::OpeningHint` by
  request slot, because Stage 8 needs the hint to avoid recomputing row
  commitments.

## Hard Invariants

- Concrete CPU backend code lives under `jolt_backends::cpu`, not as a
  production dependency of `jolt-prover`.
- CPU backend code must not append to or squeeze from the Fiat-Shamir transcript.
- CPU backend code must not choose transcript labels.
- CPU backend code must not choose sumcheck stage order or batching order.
- CPU backend code must not choose opening IDs or final-opening order.
- CPU backend code must not construct verifier-owned proof objects directly.
- CPU backend code must not rely on vector order to imply verifier proof fields;
  it returns slot-keyed, ID-bearing outputs and lets `jolt-prover` assemble.
- CPU backend code must not change claim formulas or BlindFold constraints.
- CPU backend code may be stateful, but its state must be algebraic/cache state,
  not hidden protocol schedule.
- Any backend-side global context, such as Dory layout context, must be selected
  from a protocol/backend configuration supplied by `jolt-prover`.
- Optimized and reference backends must agree on verifier-visible values for the
  same requests.
- CPU lookup fast paths must consume lookup semantics from `jolt-lookup-tables`
  directly or through `jolt-program::lookup` adapters. They may cache,
  precompute, batch, chunk, or device-upload canonical lookup results, but must
  not fork operand packing, lookup-index arithmetic, lookup-output formulas, or
  table routing. If a faster CPU path needs a new bulk lookup API, add that API
  to `jolt-lookup-tables` instead of reimplementing the formula in
  `jolt_backends::cpu`.

## CPU Fast Path Summary

This section summarizes the current optimizations in `jolt-core` that the
modular CPU backend must preserve or intentionally replace with measured
justification. The full accounting lives in
[`Jolt Core Prover Optimization Inventory`](./jolt-core-prover-optimization-inventory.md);
that inventory is the review checklist for specific optimization IDs.

### Trace Shape And Sizing

Current behavior:

- truncate trailing zero output bytes before preamble binding;
- enforce minimum padded trace length of `256`;
- pad trace length to a power of two;
- adjust padded trace length so the main Dory matrix can embed trusted and
  untrusted advice top-left blocks;
- compute `ram_K` from actual remapped RAM accesses and bytecode footprint in
  parallel;
- derive `ReadWriteConfig`, `OneHotParams`, `UniformSpartanKey`, initial RAM
  state, and final RAM state once.

Port requirement:

- move the policy into protocol/config code where it affects verifier-visible
  facts;
- keep the parallel and single-pass computations in the CPU witness/backend
  path;
- add parity tests for trace length, `ram_K`, one-hot params, read/write config,
  RAM initial/final states, and advice embedding dimensions against `jolt-core`.

### Streaming Witness Commitments

Current behavior:

- for CycleMajor Dory layout, avoid materializing all committed witness
  polynomials;
- stream padded trace rows in chunks;
- for each row chunk and each committed polynomial, call
  `stream_witness_and_commit_rows`;
- transpose row-major chunk commitments into per-polynomial tier-1 vectors;
- aggregate tier-1 chunks into final PCS commitments and opening hints;
- for AddressMajor layout, fall back to materialized parallel witness generation
  because the current streaming path assumes CycleMajor.

Port requirement:

- preserve streaming commitment as the canonical CPU path for CycleMajor;
- preserve the AddressMajor fallback until a measured replacement exists;
- represent this through `CommitmentRequest` plus witness streaming views, not by
  letting commitment code choose protocol commitment order;
- return slot-keyed PCS commitments and opening hints from backend commitment
  execution so `jolt-prover` can assemble verifier-owned proof objects and
  Stage 8 opening requests without backend-private maps;
- keep transcript absorption of commitments in `jolt-prover`, not in the CPU
  backend.

### Advice Commitment And Embedding

Current behavior:

- trusted and untrusted advice are committed in dedicated Dory contexts;
- advice vectors are sized from configured max advice byte sizes;
- `populate_memory_states` converts advice bytes to word polynomials;
- untrusted advice commitment is generated during proving and absorbed;
- trusted advice commitment is supplied from preprocessing and absorbed;
- advice polynomial values and opening hints are retained for Stage 8;
- Stage 8 applies advice Lagrange factors to embed smaller advice shapes into
  the main Dory matrix.

Port requirement:

- make advice shape and inclusion explicit in witness metadata and
  `CommitmentRequest`;
- keep the dedicated Dory contexts and retained hints in the CPU backend;
- keep transcript absorb ordering in `jolt-prover`;
- test no-advice, trusted-only, untrusted-only, and both-advice cases in both
  transparent and BlindFold modes.

### Polynomial Encodings

Current behavior:

- `MultilinearPolynomial` dispatches over dense field vectors, compact scalar
  vectors, one-hot polynomials, and RLC polynomials;
- `CompactPolynomial<T>` delays conversion from small scalars to field elements
  until binding;
- first bind uses small-scalar difference multiplication and equality shortcuts;
- later binds operate on field elements;
- evaluation uses split-eq and inside-out algorithms with serial/parallel
  thresholds;
- scalar multiplication exploits zero/one shortcuts.

Port requirement:

- `jolt-witness` oracles/views must preserve encoding capabilities;
- CPU kernels must be able to request compact/one-hot/RLC views directly through
  the standard witness/backend interop contract;
- do not normalize all committed or virtual witnesses into `Vec<F>`;
- add encoding parity tests comparing dense-reference evaluation to compact,
  one-hot, and streaming views.

### One-Hot And RA Polynomials

Current behavior:

- one-hot parameters split instruction lookup indices, bytecode PCs, and RAM
  addresses into compact chunks;
- `RaPolynomial` delays materialization through Round1/Round2/Round3/RoundN
  states;
- `SharedRaPolynomials` computes all RA chunk indices and pushforward `G`
  evaluations in one pass;
- RA indices are stored in fixed-size arrays to avoid hot-loop heap allocation;
- split-eq tables `E_hi`/`E_lo` improve pushforward computation;
- per-thread partial accumulators use unreduced field limbs and touched bitsets
  before reduction;
- eq tables and indices are shared across instruction, bytecode, and RAM RA
  families.

Port requirement:

- model RA/event-log views as first-class witness oracle views;
- derive instruction lookup RA indices and lookup outputs through
  `jolt-lookup-tables`, never through backend-local lookup packing formulas;
- `InstructionRa(d)` streams and materialized views should be produced by the
  witness/provider boundary using canonical `jolt-lookup-tables` semantics; the
  CPU backend may share RA state and pushforwards over those values, but it does
  not own their meaning;
- one-hot commitment execution must retain sparse row/index representation into
  the PCS layer; dense `Vec<F>` one-hot materialization is not acceptable for
  the canonical CPU fast path;
- keep shared RA computation as a CPU backend capability;
- preserve delayed materialization across sumcheck rounds;
- add tests that compare shared RA outputs to independent dense/one-hot
  evaluation for instruction, bytecode, and RAM families.

Regression guard:

- add a CPU-backend shell test that streams a real
  `jolt_witness::protocols::jolt_vm` `InstructionRa(d)` oracle through a
  `CommitmentRequest`-style requirement and verifies slot, rows, and one-hot chunk
  kinds;
- keep lookup adapter tests in `jolt-program` and dispatch tests in
  `jolt-lookup-tables`;
- audit new backend lookup code for direct formulas. Allowed operations are
  chunk extraction, caching, batching, pushforward, commitment, and evaluation
  over lookup values obtained from the canonical owner.

### Read/Write Sparse Matrices

Current behavior:

- read/write checking uses sparse matrix representations rather than dense
  `K x T` matrices;
- CycleMajor layout is sorted by `(row, col)` and is optimized for binding cycle
  variables first;
- AddressMajor layout is sorted by `(col, row)` and is optimized for binding
  address variables first;
- binding uses dry-run passes to compute exact output lengths;
- output buffers use spare capacity / `MaybeUninit` and set length after
  parallel writes;
- parallel recursive merge switches to sequential merge below a threshold;
- one-hot coefficient lookup tables avoid repeated coefficient work;
- sumcheck message accumulation uses unreduced products to reduce Montgomery
  reduction overhead.

Port requirement:

- keep these matrix types or direct descendants inside the CPU backend;
- expose only request/result boundaries to `jolt-prover`;
- test CycleMajor and AddressMajor bindings against a dense reference on small
  traces;
- benchmark both layouts before deleting or replacing either path.

### Streaming Sumcheck

Current behavior:

- `StreamingSumcheck` supports a streaming phase followed by materialized linear
  phase;
- schedules control window starts, switchover points, and unbound variable
  counts;
- `HalfSplitSchedule` computes cost-aware window sizes;
- `LinearOnlySchedule` disables streaming when appropriate;
- stage-specific shared state can initialize streaming windows and later cache
  openings after linearization.

Port requirement:

- preserve streaming sumcheck as a backend execution strategy;
- protocol supplies round count, degree, claims, and challenge sequence;
- backend owns windowing, materialization strategy, and scratch state;
- add tests that compare streaming and linear-only outputs for the same
  protocol-resolved request.

### Prefix/Suffix Decomposition

Current behavior:

- prefix/suffix decomposition represents table-style polynomials as
  `Σ P_i(prefix) · Q_i(suffix)`;
- `PrefixRegistry` shares prefix polynomials and final checkpoints;
- `CachedPolynomial` uses per-index `OnceLock` caches for repeated sumcheck
  evals;
- cache clearing and per-round binding avoid duplicate work across related
  decompositions.

Port requirement:

- keep prefix/suffix decomposition as a CPU relation-kernel optimization;
- expose required table/lookup witness views through `jolt-witness`;
- do not move prefix/suffix protocol meaning into the backend.

### Stage Batching And State Reuse

Current behavior:

- stages assemble multiple `SumcheckInstanceProver`s into one batched sumcheck;
- standard and ZK modes select `BatchedSumcheck::prove` vs `prove_zk`;
- univariate-skip rounds have separate clear/ZK paths;
- stage-local prover instances are dropped in background threads after use;
- advice claim reduction spans Stage 6 and Stage 7, so prover state is cached
  between stages;
- `ProverOpeningAccumulator` collects openings for Stage 8 and BlindFold.

Port requirement:

- `jolt-prover` owns batching order and clear/ZK selection;
- CPU backend may retain relation state and caches across request executions;
- backend returns opening/evaluation events by request slot, and `jolt-prover`
  owns accumulation under typed opening IDs;
- preserve background drop behavior where it measurably reduces peak memory.

### Stage 8 Streaming RLC And Dory Openings

Current behavior:

- Stage 8 builds the joint Dory opening from committed opening claims;
- dense `RamInc`/`RdInc` claims are scaled by `eq(r_address, 0)` because they
  are zero-padded in the Dory matrix;
- advice claims are scaled by advice embedding Lagrange factors;
- RLC powers are sampled after clear claims are transcript-bound in standard
  mode;
- the joint RLC polynomial is built directly from trace and advice polynomials,
  avoiding witness regeneration;
- opening hints from streaming commitments are reused;
- ZK mode extracts `y_com` and stores hidden evaluation/blinding data for
  BlindFold.

Port requirement:

- `jolt-prover` owns final opening order, claim binding mode, and gamma sampling;
- CPU backend owns efficient joint polynomial construction and PCS proof
  generation from an `OpeningRequest`;
- preserve streaming RLC construction and hint reuse;
- add parity tests for opening order, scaling factors, joint claim, and verifier
  acceptance.

### BlindFold Witness Construction

Current behavior:

- committed sumcheck rounds and output-claim rows retain coefficients,
  commitments, and blindings;
- output-claim row commitments from committed sumchecks are reused as Hyrax rows;
- regular non-coefficient rows are committed in parallel;
- final PCS evaluation commitment is included in the BlindFold instance;
- witness assignment and row layout are built after all sumcheck stages and
  Stage 8 data are available.

Port requirement:

- `jolt-prover` and `jolt-blindfold` own BlindFold protocol metadata and R1CS
  construction;
- CPU backend may provide private round/output/evaluation witness material by
  request slot;
- preserve row commitment reuse and parallel non-coefficient row commitments;
- test committed-proof shape, output-claim row ordering, and BlindFold verifier
  acceptance.

### Memory And Parallelism Policy

Current behavior:

- Rayon is used throughout witness generation, sparse matrix binding, RA
  pushforwards, polynomial binding, commitment generation, and BlindFold row
  commitments;
- `unsafe_allocate_zero_vec`, `MaybeUninit`, exact-capacity allocation, and
  spare-capacity writes avoid repeated initialization;
- `mem::take`, `Arc`, and background drop threads reduce retained memory;
- instrumentation via tracing/profiling scopes and optional allocative
  flamegraphs helps locate regressions.

Port requirement:

- preserve unsafe fast paths with local safety comments and tests;
- keep instrumentation names or add stable replacements so perf comparisons are
  stage-aligned;
- make memory-release behavior part of the CPU backend, not protocol logic.

## Porting Strategy

### Phase 0: Baseline And Inventory

- Record current `jolt-core` prove time and peak memory for representative
  guests: `muldiv`, `sha2`, `sha3`, `fibonacci`, and an advice-consuming guest.
- Record standard and ZK mode separately.
- Capture stage timings, Stage 8 timing, committed-witness timing, and peak
  memory.
- Add a checklist mapping each optimization in this spec to a target module.
- Use the optimization IDs from
  [`Jolt Core Prover Optimization Inventory`](./jolt-core-prover-optimization-inventory.md)
  in implementation plans and PR descriptions.

### Phase 1: Representation-Preserving `jolt-witness`

- Add witness oracles/views that can represent compact scalars, one-hot RA/event
  views, sparse read/write events, advice blocks, and streaming trace chunks.
- Add dense-reference evaluation tests for every view type.
- Avoid adding APIs that force full field-vector materialization.

### Phase 2: CPU Backend Shell

- Scaffold `crates/jolt-backends` with backend traits, request/result structs,
  and `jolt_backends::cpu` implementing the initial request inputs and
  slot-keyed backend output structs.
- Wire it from SDK/CLI/tests as a selected backend rather than making
  `jolt-prover` depend on it.
- Initially allow backend internals to mirror `jolt-core` closely.
- Do not expose backend internals as protocol APIs.

### Phase 3: Commitments And Advice

- Port streaming witness commitments first, because they exercise trace,
  witness, PCS, layout, and hint boundaries.
- Port advice commitment contexts and advice embedding metadata.
- Verify commitment ordering and transcript absorption through `jolt-prover`.

### Phase 4: Stages 1-2

- Port the smallest useful stage frontier while preserving current sumcheck
  instance implementations where possible.
- Compare stage outputs, challenges, opening events, and verifier acceptance.

### Phase 5: Stages 3-7

- Port remaining relation kernels with current optimized representations.
- Preserve cross-stage cached state for advice reductions.
- Keep background-drop and memory-release behavior.

### Phase 6: Stage 8

- Port typed opening output construction in `jolt-prover`.
- Port streaming RLC and Dory proof generation in the CPU backend.
- Add direct parity checks for joint claims and opening proof verification.

### Phase 7: BlindFold

- Port committed-round witness material and output-claim row material.
- Preserve row commitment reuse.
- Add deterministic-RNG tests where exact proof parity is useful; otherwise test
  accepted proof shape and verifier acceptance.

### Phase 8: Field Inline, Dory Assist, Wrapper

- Extend witness oracles/views and CPU kernels only where required by selected
  protocol requests.
- For field inline, preserve the verifier/protocol split: `FieldRdInc` is the
  committed FR surface, while `FieldRs1Ra`, `FieldRs2Ra`, and `FieldRdWa` are
  virtual openings anchored through field-inline bytecode metadata and the
  ordinary committed `BytecodeRa(i)` path.
- Keep non-VM witness providers out of the Jolt VM CPU fast path.
- Use the same backend trait shape to add experimental CUDA, Metal, or hybrid
  backend slices without weakening the CPU perf-parity requirement.

## Testing Strategy

### Correctness Tests

Required at each implemented frontier:

- modular CPU backend proof accepted by matching `jolt-verifier`;
- `jolt-core` parity for commitments, stage outputs, opening claims, and final
  verifier-visible proof fields where deterministic;
- transparent and BlindFold modes;
- no advice, trusted-only, untrusted-only, both-advice;
- CycleMajor and AddressMajor layouts where supported;
- dense-reference checks for compact, one-hot, RA, sparse matrix, streaming,
  and RLC witness views;
- tampering tests for transcript-bound values, opening events, output claims,
  advice commitments, and BlindFold private/public binding metadata.

Primary e2e checks remain:

```bash
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host,zk
```

As modular crates gain E2E entry points, add equivalent `jolt-prover` checks and
run them side by side with `jolt-core` until the modular CPU backend becomes the
canonical path.

### Performance Tests

Performance must be tracked as a first-class acceptance criterion.

Suggested benchmark set:

```text
muldiv                 // primary correctness/perf smoke test
fibonacci              // small/simple trace
sha2 / sha2-chain      // lookup-heavy
sha3 / sha3-chain      // hash-heavy
advice-demo / advice-consumer
memory-ops             // RAM/read-write stress
```

Metrics:

- total prove time;
- committed witness generation time;
- per-stage sumcheck time;
- Stage 8 opening time;
- BlindFold time in ZK mode;
- peak RSS;
- committed witness memory;
- Stage 8/RLC memory;
- proof size where relevant.

Regression policy:

- Any unexplained regression over `5%` in prove time or peak memory on the
  canonical CPU backend requires investigation.
- Any regression over `10%` requires explicit approval and a documented reason.
- Temporary regressions during bring-up must be tracked in the PR description
  with a cleanup issue or follow-up milestone.

### Backend Equivalence Tests

For each relation where a reference backend exists:

```text
same protocol request
same witness oracles/views
reference backend output == CPU backend output
```

For randomized ZK paths, use deterministic RNG fixtures when comparing exact
artifacts. Otherwise compare verifier acceptance, commitment/opening shape, and
BlindFold metadata consistency.

## Review Checklist

- Concrete CPU code lives in `jolt_backends::cpu`, and `jolt-prover` does not
  production-depend on it.
- Protocol decisions are visible in `jolt-prover`, not hidden in CPU kernels.
- CPU backend receives protocol-resolved requests and returns slot-keyed
  outputs.
- Transcript operations are not performed by compute kernels.
- Opening IDs and claim formulas are not selected by compute kernels.
- Existing fast representations are preserved unless a benchmarked replacement
  exists.
- No hot path was forced through dense `Vec<F>` materialization for abstraction
  convenience.
- Stage timing and memory instrumentation are sufficient to compare against
  `jolt-core`.
- `jolt-core` parity and performance baselines are reported for the changed
  frontier.

## References

- [`jolt-prover` model crate spec](./jolt-prover-model-crate.md)
- [`jolt-prover` frontier harness spec](./jolt-prover-frontier-harness.md)
- [`jolt-witness` crate spec](./jolt-witness-crate.md)
- [`Jolt Core Prover Optimization Inventory`](./jolt-core-prover-optimization-inventory.md)
- `jolt-core/src/zkvm/prover.rs`
- `jolt-core/src/zkvm/witness.rs`
- `jolt-core/src/poly/compact_polynomial.rs`
- `jolt-core/src/poly/one_hot_polynomial.rs`
- `jolt-core/src/poly/ra_poly.rs`
- `jolt-core/src/poly/shared_ra_polys.rs`
- `jolt-core/src/poly/prefix_suffix.rs`
- `jolt-core/src/poly/rlc_polynomial.rs`
- `jolt-core/src/poly/opening_proof.rs`
- `jolt-core/src/subprotocols/streaming_sumcheck.rs`
- `jolt-core/src/subprotocols/streaming_schedule.rs`
- `jolt-core/src/subprotocols/read_write_matrix/`
