# Akita

Akita is Jolt's lattice-based [polynomial commitment](./appendix/pcs.md) backend. Jolt integrates the [Akita PCS](https://github.com/LayerZero-Labs/akita) through the `jolt-akita` crate. It is an alternative to the elliptic-curve [Dory](./dory.md) backend and uses a packed witness layout and native batched openings.

The `akita` Cargo feature selects this protocol at compile time. The current implementation supports clear proofs, including trusted and untrusted advice and committed programs. It does **not** support zero knowledge: `akita` and `zk` are mutually exclusive, and enabling both produces a compile error. [BlindFold](./blindfold.md) currently applies to the Dory backend.

## Field and witness representation

Jolt's Akita path uses `AkitaField`, the upstream `fp128` configuration's alias for `jolt_field::Prime128OffsetA7F7`. Its modulus is

$$
p = 2^{128} - 2^{32} + 22537.
$$

The RISC-V execution model and sumcheck-based approach remain shared with Dory, but the commitment layout changes. Akita does not provide the additive commitment homomorphism used by Dory's [random-linear-combination opening](./optimizations/batched-openings.md). Jolt instead packs the trace's one-hot columns into one physical polynomial, called `OneHotTrace`.

This object contains the instruction, bytecode, and RAM address columns, together with balanced-digit increment columns and a signed carry. The increment columns encode a fused increment value derived by the sumcheck relations; Akita does not separately commit the `RdInc` and `RamInc` polynomials used by Dory. The supported address chunk widths are four or eight bits, corresponding to one-hot domains of size 16 or 256. The prover supplies selected row indices to Akita's native one-hot commitment path.

Advice and committed-program data have their own commitments. Trusted and untrusted advice use dense word polynomials. In committed-program mode, bytecode chunks and the initial program image are committed as bounded dense objects and opened directly. These objects are separate from `OneHotTrace` because they have their own sizes, layouts, and commitment lifetimes.

## Packing and opening

Prefix packing assigns each logical polynomial a fixed slot in a larger multilinear polynomial. For logical polynomials $P_0, \ldots, P_{m-1}$ evaluated at a common point $\mathbf{x}$, the packed polynomial is

$$
P(\mathbf{s}, \mathbf{x})
  = \sum_{i=0}^{m-1} \operatorname{eq}(\mathbf{s}, i)\,P_i(\mathbf{x}),
$$

where $\mathbf{s}$ selects a slot and $\operatorname{eq}$ is the [multilinear equality polynomial](./optimizations/eq.md). Jolt binds the layout and logical evaluation claims to the transcript before sampling the selector point. This reduces the trace's logical claims to one evaluation of the already committed physical polynomial.

At stage 8, that evaluation joins the advice and committed-program evaluations in one native Akita grouped opening. Each group can have its own evaluation point and shape. The canonical order is:

1. Untrusted advice, when present.
2. Trusted advice, when present.
3. Bytecode chunks in index order, followed by the initial program image, in committed-program mode.
4. `OneHotTrace`.

The verifier derives the expected layouts and roles from preprocessing and protocol configuration. It checks the assembled statement before invoking Akita verification. This replaces Dory's commitment-level linear combination with a grouped proof over the original commitments.

## Setup and protocol selection

Akita uses transparent setup. Jolt supplies versioned schedule catalogs as `.aks` artifacts under `crates/jolt-akita/schedules/`. Preprocessing provisions the grouped schedules needed by the program and advice configuration; the resulting verifier setup carries the catalog used during verification. The proof selects a schedule by digest from that catalog rather than supplying new schedule parameters.

Both prover and verifier must be built for the same protocol. The `akita` feature selects packed commitments and little-endian scalar challenges. A compiled verifier accepts only its selected protocol configuration and rejects a mismatching proof. Dory and Akita proofs therefore require matching preprocessing and verifier builds.

## Implementation

The main implementation paths are:

- `crates/jolt-akita/`: PCS adapter, native one-hot commitments, grouped openings, and schedule catalogs.
- `crates/jolt-claims/src/protocols/jolt/lattice/`: canonical layouts and Akita-specific sumcheck relations.
- `crates/jolt-openings/src/prefix.rs`: prefix packing and claim reduction.
- `crates/jolt-prover/src/akita/`: witness assembly, commitments, and the final grouped opening.
- `crates/jolt-verifier/src/stages/stage8/packed.rs`: verifier assembly of the final opening claims.

The modular prover's end-to-end suite covers arithmetic execution, both one-hot chunk sizes, advice, committed programs, and rejection of modified claims. Run it with:

```bash
cargo nextest run -p jolt-prover --features prover-fixtures,akita --test akita_e2e --cargo-quiet
```
