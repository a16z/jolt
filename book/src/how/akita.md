# Akita

Akita is Jolt's lattice-based [polynomial commitment](./appendix/pcs.md) backend. Jolt integrates the [Akita PCS](https://github.com/LayerZero-Labs/akita) through the `jolt-akita` crate. It is an alternative to the elliptic-curve [Dory](./dory.md) backend and uses native trace column batching and grouped openings.

The `akita` Cargo feature selects this protocol at compile time. The current implementation supports clear proofs, including trusted and untrusted advice and committed programs. It does **not** support zero knowledge: `akita` and `zk` are mutually exclusive, and enabling both produces a compile error. [BlindFold](./blindfold.md) currently applies to the Dory backend.

## Field and witness representation

Jolt's Akita path uses `AkitaField`, the upstream `fp128` configuration's alias for `jolt_field::Prime128OffsetA7F7`. Its modulus is

$$
p = 2^{128} - 2^{32} + 22537.
$$

The RISC-V execution model and sumcheck-based approach remain shared with Dory, but the commitment layout changes. Akita does not provide the additive commitment homomorphism used by Dory's [random-linear-combination opening](./optimizations/batched-openings.md). Jolt instead commits the trace's one-hot columns as distinct polynomials in one native commitment group, called `OneHotTrace`.

This object contains the instruction, bytecode, and RAM address columns, together with balanced-digit increment columns and a signed carry. The increment columns encode a fused increment value derived by the sumcheck relations; Akita does not separately commit the `RdInc` and `RamInc` polynomials used by Dory. Production Akita traces use four-bit address chunks (K=16). Eight-bit chunks (K=256) are supported by the shipped catalogs only for explicit fixture and benchmark shapes listed in the [schedule policy](../../../crates/jolt-akita/schedules/README.md). Other K=256 shapes require a deployment-owned catalog containing the exact shape. The prover supplies selected row indices to Akita's native one-hot commitment path.

Advice and committed-program data have their own commitments. Trusted and untrusted advice use dense word polynomials. In committed-program mode, the whole bytecode and the initial program image are committed as bounded dense objects and opened directly. These objects are separate from `OneHotTrace` because they have their own sizes, layouts, and commitment lifetimes.

## Native batching and opening

For trace columns $P_0, \ldots, P_{m-1}$, Stage 8 assembles the ordered claims

$$
P_0(\mathbf{x}) = v_0, \ldots, P_{m-1}(\mathbf{x}) = v_{m-1}.
$$

Every column has `log_T + log_K` variables and uses the same point
$\mathbf{x}$ in `(cycle || address)` order. The columns share their trace
source so Akita's streaming kernels can process them together. There are no
trace slot-selector variables or selector-reduction challenges.

Native batching makes the first fold cycle-local: each chunk owns the same
cycle range in every column. Recursive folds inherit those owners, align each
chunk body to the successor's source block width, and retain the shared tail
in the last owner. When the schedule reduces the chunk count, it merges owners.
This behavior is provided by
[Akita #175](https://github.com/LayerZero-Labs/akita/pull/175).

The canonical layout checks column order and dimensions. Commitments, their
polynomial counts, group points, and ordered evaluations are bound to the
transcript before Akita derives batching challenges. The streaming witness
uses a `u64` mask to distinguish selected address zero from an empty row, so
the layout rejects more than 64 trace columns.

At stage 8, the trace group joins the advice and committed-program groups in
one native Akita opening proof. Each group can have its own evaluation point
and shape. The canonical order is:

1. Untrusted advice, when present.
2. Trusted advice, when present.
3. One whole-bytecode object (`ProgramBytecode`), followed by the initial program image, in committed-program mode.
4. `OneHotTrace`.

The bytecode group has `9 + log2(padded_bytecode_rows)` logical variables and a physical arity of at least 14, independent of the trace size. Each native group keeps its own opening point, so a short execution does not require splitting bytecode. Dory also commits one whole-bytecode table and embeds its balanced matrix into the joint opening grid. This is separate from Akita response chunk profiles and the address-bit chunks of `BytecodeRa`. Dory has no bytecode-table partitioning parameter: when the bytecode matrix exceeds the trace grid, it enlarges the joint opening grid and can increase proving time and memory.

The verifier derives the expected layouts and roles from preprocessing and protocol configuration. It checks the assembled statement before invoking Akita verification. This replaces Dory's commitment-level linear combination with a grouped proof over the original commitments.

## Setup and protocol selection

Akita uses transparent setup. Jolt supplies versioned schedule catalogs as `.aks` artifacts under `crates/jolt-akita/schedules/`. Preprocessing provisions the grouped schedules needed by the program and advice configuration; the resulting verifier setup carries the catalog used during verification. The proof selects a schedule by digest from that catalog rather than supplying new schedule parameters.

The workspace pins Akita to `55a40270d880110b423b9f94c3a52e005583d2da`.
Jolt delegates to Akita's adaptive ring-dimension schedule configuration
and uses the CPU backend for every fold.
The checked-in schedule catalogs match this revision. The default K=16
catalog covers traces through `2^31` cycles.

Akita's planner admits a Module-SIS instance only when its ADPS16 quantum
Core-SVP estimate is at least 169 bits, the cost of the ML-DSA-65 forgery
attack (NIST security category 3). Fiat-Shamir grinding and the challenge
spaces separately target 128 bits. Akita's
[policy spec](https://github.com/LayerZero-Labs/akita/blob/dfdbd7674cab5cfbe95243e947d1b10a982f232b/specs/sis-nist-level3-policy.md)
states both targets and their evidence.

`ProverConfig::derive_from_dimensions` selects the single-chunk geometry
`AkitaChunkProfile { chunks: 1, levels: 0 }`. Set `config.akita_chunk_profile`
before preprocessing to choose another chunk count and leading chunked fold
count, and use the same config for proving. Akita validates the geometry and
catalog together; proving rejects a profile that differs from preprocessing.

Applications supply catalogs covering their selected geometry and exact shapes.
`AkitaScheduleArtifacts` stores a collection keyed by `(K, chunks, levels)`.
Its directory loader requires the two dense files and discovers one-hot files
named `jolt-fp128-onehot-k<K>-w<chunks>r<levels>.aks`; an unsuffixed filename denotes
`w1r0`. The checked-in catalogs cover repository fixtures. Applications generate
and distribute their own production coverage.

The same upstream verifier consumes the selected catalog and geometry embedded
in public verifier preprocessing. Every one-hot setup binds both
`akita_chunk_profile` (chunks) and `akita_chunked_levels` (levels) into the
transcript. Verification does not read source catalog files.

Both prover and verifier must be built for the same protocol. The `akita` feature selects native grouped commitments and little-endian scalar challenges. A compiled verifier accepts only its selected protocol configuration and rejects a mismatching proof. Dory and Akita proofs therefore require matching preprocessing and verifier builds.

## Implementation

The main implementation paths are:

- `crates/jolt-akita/`: PCS adapter, native one-hot commitments, grouped openings, and schedule catalogs.
- `crates/jolt-claims/src/protocols/jolt/lattice/`: canonical layouts and Akita-specific sumcheck relations.
- `crates/jolt-openings/src/schemes.rs`: native group claims and metadata checks.
- `crates/jolt-openings/src/prefix.rs`: zero-prefix embeddings for auxiliary objects.
- `crates/jolt-kernels/src/reference/akita/`: native trace assembly and commitment kernels.
- `crates/jolt-prover/src/akita/`: preprocessing, advice objects, and final grouped opening orchestration.
- `crates/jolt-verifier/src/stages/stage8/akita.rs`: verifier assembly of the final opening claims.

The modular prover's end-to-end suite covers arithmetic execution, both one-hot chunk sizes, advice, committed programs, and rejection of modified claims. Run it with:

```bash
cargo nextest run -p jolt-prover --features prover-fixtures,akita --test akita_e2e --cargo-quiet
```

Plain prove-and-verify acceptance of the example guests under Akita runs in
the guest × mode matrix (`crates/jolt-prover/tests/e2e_matrix.rs`, see
[Testing gates](../dev/testing-gates.md)).
