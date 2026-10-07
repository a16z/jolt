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

Advice and committed-program data have their own commitments. Trusted and untrusted advice use dense word polynomials. In committed-program mode, bytecode chunks and the initial program image are committed as bounded dense objects and opened directly. These objects are separate from `OneHotTrace` because they have their own sizes, layouts, and commitment lifetimes.

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
3. Bytecode chunks in index order, followed by the initial program image, in committed-program mode.
4. `OneHotTrace`.

The verifier derives the expected layouts and roles from preprocessing and protocol configuration. It checks the assembled statement before invoking Akita verification. This replaces Dory's commitment-level linear combination with a grouped proof over the original commitments.

## Setup and protocol selection

Akita uses transparent setup. Jolt supplies versioned schedule catalogs as `.aks` artifacts under `crates/jolt-akita/schedules/`. Preprocessing provisions the grouped schedules needed by the program and advice configuration; the resulting verifier setup carries the catalog used during verification. The proof selects a schedule by digest from that catalog rather than supplying new schedule parameters.

The workspace pins Akita to `d98400c555a7fc779bb4e29c35fbd4adad3232d1`,
including batched source evaluation and opening preparation from
[Akita #169](https://github.com/LayerZero-Labs/akita/pull/169) and recursive
ownership alignment from
[Akita #175](https://github.com/LayerZero-Labs/akita/pull/175).
The checked-in schedule catalogs are regenerated with this revision.

`ProverConfig::derive_from_dimensions` selects the single-chunk Akita profile. To use
two, four, or eight chunks, set `config.akita_chunk_profile` before Akita
preprocessing and use that config for proving. Proving rejects a profile that
differs from the prepared setup. The selected profile is carried
by the verifier setup. Four-file schedule directories containing the updated
dense catalogs remain valid for the default profile; a nondefault profile needs
its matching companion catalog for the selected one-hot domain size.

Nondefault chunk profiles bind the `akita_chunk_profile` transcript label.

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
