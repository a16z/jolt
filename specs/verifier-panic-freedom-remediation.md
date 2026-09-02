# Spec: Panic-Free Modular `jolt-verifier` and Dory Verification

| Field | Value |
|---|---|
| Authors | Quang Dao, Codex audit |
| Created | 2026-07-15 |
| Status | proposed remediation plan |
| Primary target | New modular `crates/jolt-verifier` and its SDK byte entry points |
| Dependency scope | Dory/PCS/transcript paths reached by the primary target; broader direct Dory hazards are labeled separately and do not delay Jolt fixes |
| Explicitly excluded | `jolt-prover-legacy` verifier and helper APIs |
| Jolt baseline | [`78ae842c2cbc016d3b8f0e8dfeb30880aebaba8d`](https://github.com/a16z/jolt/commit/78ae842c2cbc016d3b8f0e8dfeb30880aebaba8d) |
| Dory baseline | `dory-pcs 0.4.0`, [`56a7243aceb4d74e19a576a0245028d8d117507e`](https://github.com/a16z/dory/commit/56a7243aceb4d74e19a576a0245028d8d117507e), crates.io checksum `5c6a5b6aa708db35ced06eeef3d2b4408004c76279091e0bc387f8fd42796555` |
| Arkworks baseline | `a16z/arkworks-algebra`, `dev/twist-shout`, [`76bb3a4518928f1ff7f15875f940d614bb9845e6`](https://github.com/a16z/arkworks-algebra/commit/76bb3a4518928f1ff7f15875f940d614bb9845e6) |
| Primary existing remediation PR | [a16z/jolt#1609](https://github.com/a16z/jolt/pull/1609), tip [`60b75a3a4c655aa9d8eb028dd13d573fba2abe6e`](https://github.com/a16z/jolt/commit/60b75a3a4c655aa9d8eb028dd13d573fba2abe6e) |

## Summary

The new modular `jolt-verifier` stack at the pinned commits is not panic-free. No malformed-proof-only panic was found on native 64-bit when preprocessing and the Dory verifier setup are honestly generated, but the complete supported surface remains unsafe because its SDK entry points accept serialized proof, preprocessing, and I/O objects; preprocessing embeds a deserializable Dory setup; and Dory trusts several structural and algebraic invariants that its public types and deserializers do not enforce.

The remediation should not be one undifferentiated PR. The canonical ownership boundaries are:

1. `a16z/dory`: make Dory deserialization and verification fail closed.
2. `a16z/jolt`: revive and extend existing PR #1609 for bounded wire decoding; validate preprocessing; make SDK wrappers fallible; consume the fixed Dory release; remove remaining verifier-path invariant panics.
3. `a16z/arkworks-algebra`: optional type-invariant hardening for `PairingOutput`; Jolt does not need to block on it if Dory makes its wrappers invariant-preserving.
4. `a16z/jolt`: add adversarial regression and fuzz infrastructure after the API boundaries stabilize.

This document distinguishes confirmed external triggers, typed/direct API triggers, resource-exhaustion hazards, and internal invariant panics for which no malformed-input route is currently known.

## Audit method and confidence labels

The audit followed serialized native/WASM entry points into the new modular verifier, statically inventoried verifier-reachable panic/overflow/index/allocation sites, inspected the pinned crates.io Dory source and its Arkworks patch, and compared exact current PR-tip diffs. Three independent reviewers first covered Jolt, Dory/dependencies, and adversarial/PR overlap; a second cross-review read the complete remediation draft and corrected reachability, ownership, and tradeoffs.

“Confirmed external” means a concrete caller-controlled byte or proof/preprocessing value reaches the failure. “Direct typed API” means public Rust construction can reach it, even if checked wire decoding cannot. “Conditional/deep” means the operation is unsafe but earlier proof checks make a practical external trigger unproven. “Latent” means no current caller-controlled route was established. Resource hazards are reported separately because Rust allocation failure may abort rather than unwind.

## Goal and contract

Every supported new modular `jolt-verifier` entry point must reject malformed or degenerate inputs by returning an error or `false`. It must not unwind, abort, trap, index out of bounds, overflow, or intentionally call `panic!`, `assert!`, `unwrap`, `expect`, or `unreachable!` as a consequence of caller-controlled proof, preprocessing, public-I/O, setup, transcript, or commitment data.

The contract applies at four boundaries:

1. **Wire boundary:** arbitrary bytes supplied to native or WASM verification.
2. **Typed verification boundary:** arbitrary values of public proof, setup, preprocessing, and public-I/O types, including values constructed directly rather than by checked deserialization.
3. **Cryptographic adapter boundary:** arbitrary values passed to public Dory, PCS, vector-commitment, and transcript verifier APIs.
4. **Operational verifier boundary:** verifier-key/setup loading and cache access must return errors rather than panic on corrupt data or I/O failure.

Allocation failure cannot be reliably caught in Rust and may abort the process or trap WASM. Therefore panic freedom requires byte and structural limits before attacker-directed allocation. A library-side limit is not a substitute for an HTTP/RPC body limit: a service must reject an oversized request before buffering it in full.

## Non-goals

- Making `jolt-prover-legacy` or any of its verifier/helper APIs panic-free. It may still be exercised by repository-wide regression tests, but it is not part of this security contract or remediation scope.
- Changing proof-system algebra except where required to define zero-challenge handling or validated cryptographic types.
- Treating a digest stored beside attacker-controlled preprocessing as authentication. A digest is useful only if it is recomputed from a canonical, validated object or checked against a separately trusted value.
- Silently truncating, saturating, zipping-to-the-shorter-input, or mapping malformed values to identities. Such behavior avoids a panic by changing the statement being verified and is not an acceptable fix.
- Catching panics as the primary defense. `catch_unwind` is useful as a regression oracle and an outer FFI safeguard, but it does not catch allocator aborts and does not repair partially executed protocol logic.

## Pinned open-PR inventory

The open-PR review covered every `a16z/jolt` PR created or updated from 2026-07-01 through 2026-07-15, plus the older serialization PR remembered in the audit request. No open Dory PR exists, and no Dory or Arkworks PR was created or updated in that window. Arkworks' two older open PRs, #33 and #36, are unrelated.

| PR | Tip inspected | Relevant work | Action |
|---|---|---|---|
| [#1609 — Mitigate proof-deserialization DOS vector](https://github.com/a16z/jolt/pull/1609) | `60b75a3a4c655aa9d8eb028dd13d573fba2abe6e` | Michael Zhu (`moodlezoup`), created 2026-06-09. Adds a 128 MiB bounded `JoltProof` codec, exact-consumption checking, typed errors, and tests. | **Revive, rebase, and extend; do not duplicate.** It is an open conflicting draft and its API is not called by production SDK/WASM verification. |
| [#1657 — Migrate Jolt core transcripts to Spongefish](https://github.com/a16z/jolt/pull/1657) | `b9dcdec2a6afe24ab70958bec75f66fc2f451652` | Uses NARG frames, parses canonical sequences from actual frame bytes rather than a declared element count, and requires NARG EOF. | Open and non-draft, but conflicting; maintainers stated in discussion that the migration is paused. Preserve its checks if it returns. |
| [#1659 — addr2line bump](https://github.com/a16z/jolt/pull/1659) | `2d0e44ea223669b215816594acd2f097f30b6dcf` | No verifier or serialization changes. | No coordination needed. |
| [#1666 — Rust-to-Lean bytecode expansions](https://github.com/a16z/jolt/pull/1666) | `91e09a16bb9678925b24494e0ce35b4632846332` | Explicitly outside the live prover/verifier. | No coordination needed. |
| [#1669 — Feat/jolt prover](https://github.com/a16z/jolt/pull/1669) | `02a764d5252e6751712d54b57ca536998d308732` | Verifier edits are visibility/value refactors; Jolt's Dory adapter canonicalizes odd setup sizes and changes prover hints. | No duplicated hardening, but it textually overlaps J1/J2/J3/J4. Fix its new unchecked `max_num_vars + 1` before merge. |

### Existing PR #1609: exact coverage and gaps

At its current tip, #1609 provides:

- `MAX_PROOF_BYTES = 128 MiB` and `JoltProof::from_canonical_bytes[_bounded]` in [`crates/jolt-verifier/src/decode.rs`](https://github.com/a16z/jolt/blob/60b75a3a4c655aa9d8eb028dd13d573fba2abe6e/crates/jolt-verifier/src/decode.rs#L36-L85).
- A fixed 128 MiB bincode read limit, a caller-supplied physical-input-length precheck, exact top-level consumption, trailing-byte rejection, a matching encoder, typed errors, and round-trip/truncation/oversize tests.

It does **not** currently close the production boundary:

- The codec is opt-in and used only by tests. Generated WASM verification still calls the generic unlimited decoder at [`jolt-sdk/src/host_utils.rs`](https://github.com/a16z/jolt/blob/78ae842c2cbc016d3b8f0e8dfeb30880aebaba8d/jolt-sdk/src/host_utils.rs#L86-L97) from [`jolt-sdk/macros/src/lib.rs`](https://github.com/a16z/jolt/blob/78ae842c2cbc016d3b8f0e8dfeb30880aebaba8d/jolt-sdk/macros/src/lib.rs#L1395-L1422).
- It bounds only `JoltProof`, not preprocessing or `JoltDevice`.
- An outer byte limit does not stop a small Dory setup buffer from declaring an enormous **inner** canonical vector length.
- Its 128 MiB default is not derived from preprocessing and is likely too permissive for some WASM deployments.
- It is based on `235e7df8135b140b7fb94567393503439d614118` and conflicts with the current modular-verifier layout.

The correct no-duplication strategy is to retain #1609 and its attribution, rebase it, and turn it into the typed production wire boundary described in PR J1 below.

### Open PR #1657: exact coverage and gaps

At its current tip, #1657:

- Replaces most structured stage payloads with `JoltProof::narg: Vec<u8>` in [`crates/jolt-verifier/src/proof.rs`](https://github.com/a16z/jolt/blob/b9dcdec2a6afe24ab70958bec75f66fc2f451652/crates/jolt-verifier/src/proof.rs#L20-L35).
- Checks NARG frame lengths against available bytes and uses fallible `u64 -> usize` conversion in [`crates/jolt-transcript/src/codec.rs`](https://github.com/a16z/jolt/blob/b9dcdec2a6afe24ab70958bec75f66fc2f451652/crates/jolt-transcript/src/codec.rs#L33-L41).
- Decodes canonical sequences until the bounded frame body is exhausted, without trusting an element count, in [`messages.rs`](https://github.com/a16z/jolt/blob/b9dcdec2a6afe24ab70958bec75f66fc2f451652/crates/jolt-transcript/src/messages.rs#L92-L108).
- Requires NARG EOF on clear and ZK verifier paths in [`verifier.rs`](https://github.com/a16z/jolt/blob/b9dcdec2a6afe24ab70958bec75f66fc2f451652/crates/jolt-verifier/src/verifier.rs#L167-L179).

It does not bound the outer bincode object, preprocessing, I/O, or Dory setup, and it does not address any Dory verifier arithmetic or setup-shape issue. If revived, its frame and EOF checks should be preserved; PR #1609's bounded outer decode should be ported to the simplified proof type rather than replaced by a second NARG mechanism.

### Open PR #1669: prospective setup regression

At its current tip, `canonical_setup_log_n` computes `max_num_vars + 1` for odd values in [`crates/jolt-dory/src/scheme.rs`](https://github.com/a16z/jolt/blob/02a764d5252e6751712d54b57ca536998d308732/crates/jolt-dory/src/scheme.rs#L88-L93). `DoryScheme::setup_verifier(usize::MAX)` therefore panics in overflow-checked builds and wraps to zero otherwise.

The canonical fix is a fallible setup constructor using `checked_add`, combined with a documented protocol maximum. Saturation and rounding down are incorrect because they can change the generator bucket or provision insufficient generators.

## Finding matrix

| ID | Boundary | Failure | Confidence/reachability | Canonical owner | Existing PR overlap |
|---|---|---|---|---|---|
| J-SER-1 | Serialized proof | No production total-size/work limit | Confirmed resource-exhaustion boundary | `a16z/jolt` | #1609 direct but opt-in; #1657 changes representation/frame parsing only |
| J-SER-2 | Serialized preprocessing/I/O | Generic unlimited bincode decode | Confirmed resource-exhaustion boundary | `a16z/jolt` | None |
| J-PRE-1 | Preprocessing/domain helper APIs | Raw program/domain dimensions reach unchecked power-of-two/shift arithmetic | Confirmed direct helper hazard; top-level committed-program path currently rejects first; full-bytecode ZK path is conditional/deep | `a16z/jolt` | #1669 adds a trace-order guard, not general validation |
| J-IO-1 | Serialized/typed public I/O | Advice vectors are accepted and the verifier clones the whole `JoltDevice` | Confirmed memory-amplification boundary | `a16z/jolt` | None |
| D-SER-1 | Serialized Dory setup | Attacker `u64` passed to `Vec::with_capacity` | Confirmed deterministic panic/allocation failure | `a16z/dory` | None |
| D-SER-2 | Direct Dory proof decode | Attacker round count passed to two `Vec::with_capacity` calls | Confirmed upstream API hazard; Jolt inner wrapper caps rounds after outer buffer exists | `a16z/dory` | None |
| D-SETUP-1 | Typed or serialized setup | Unvalidated `chi`/`delta_*` vectors indexed by `max_log_n`/proof rounds | Confirmed verifier panic | `a16z/dory` | None |
| D-DIM-1 | Typed or serialized proof | Unchecked `nu + sigma` | Confirmed on wasm32 overflow-checked builds and direct typed APIs | `a16z/dory` | None |
| D-GROUP-1 | Direct typed proof/commitment | Public wrappers permit zero/invalid GT; subtraction inverses with `unwrap` | Confirmed direct/unchecked-construction panic | `a16z/dory` and Jolt adapter; optional arkworks hardening | None |
| D-STATE-1 | Direct low-level verifier API | Constructor/final-state invariants are only `debug_assert`ed; later coordinate indexing is unchecked | Confirmed direct typed API hazard | `a16z/dory` | None |
| J-SDK-1 | Native generated verifier | Reconstructs `MemoryLayout` using asserts and overflow `expect`s | Confirmed malformed-preprocessing panic | `a16z/jolt` | None |
| J-SDK-2 | Native generated verifier | `postcard::to_stdvec(...).unwrap()` on public inputs/output | Confirmed for a fallible/custom `Serialize` implementation | `a16z/jolt` | None |
| J-PCS-1 | Direct PCS API | Dory commitment combine asserts equal lengths | Confirmed direct typed API panic; current stage 8 constructs equal lengths | `a16z/jolt` API, with trait-level design decision | None |
| J-VC-1 | Direct VC API | Pedersen `verify` calls panicking `commit` above capacity | Confirmed direct typed API panic; current BlindFold callers precheck | `a16z/jolt` | None |
| J-CORE-1 | Typed verifier internals | Subtraction before `.get()`, unchecked slices, dimension asserts | Latent; no proof-only route found with honest preprocessing | `a16z/jolt` | None |
| D-TR-1 | Direct Dory transcript | Zero Fiat-Shamir challenge calls `panic!`; serialization uses `expect` | Theoretical random-oracle event/no practical trigger known; generic API/allocator hazards | `a16z/dory` | None |
| D-OPS-1 | Setup/cache loading | Corrupt cache, poisoned lock, path/I/O/serialization failures, unchecked shifts/allocations, and cache/setup identity assumptions | Confirmed direct Dory operational/setup hazard; optimized cache is not in Jolt's current final verifier | `a16z/dory` | None |
| J-MAL-1 | Nested canonical object | Dory wrappers do not require inner EOF | Confirmed malleability/payload-amplification issue, not itself a panic | `a16z/jolt` and upstream top-level helpers | #1609 outer EOF only; #1657 NARG EOF only |
| J-FUZZ-1 | Assurance | Dory fuzz workspace does not build and mutation strategy rarely reaches verification | Confirmed coverage failure | `a16z/jolt` | None |

## Detailed findings and changes

### J-SER-1: bounded production proof decoding

**Current surface.** The generic SDK decoder uses `bincode::config::standard()` with no limit and is the public WASM proof boundary. `JoltProof` contains many vectors and nested canonical byte buffers. Large valid-sized input bytes can drive proportional allocation and CPU work; WASM may trap on memory exhaustion.

**Canonical resolution.** Implement this by reviving #1609:

- Keep a hard process ceiling and caller-supplied physical-length precheck. If a genuinely caller-tightenable internal bincode limit is desired, parameterize the bincode configuration rather than implying #1609 already does so.
- Add a protocol/preprocessing-derived bound where the proof geometry is known. The derived bound should count exact or conservative maxima for sumcheck rounds, commitments, BlindFold rows, Dory rounds, and fixed group/field encodings.
- Wire the codec into every native/WASM byte-verifier path. There must be no production path that decodes a `JoltProof` through the generic helper.
- Require exact outer consumption and, separately, exact consumption for nested top-level Dory encodings.
- Enforce transport/request size before buffering the request.

**Policy tradeoff.** A static cap is easy to audit and protects unknown future fields, but 128 MiB is a large WASM budget and can hide unexpectedly bloated proofs. A geometry-derived cap is tighter but can drift when proof formats change. The recommended design uses both: a derived per-instance maximum clamped by a documented absolute native/WASM ceiling, with compile/test fixtures that fail when the serialized proof exceeds the estimate.

### J-SER-2: separate preprocessing and public-I/O codecs

**Current surface.** `deserialize_verifier_object<T>` applies one unlimited policy to proof, preprocessing, and `JoltDevice`. These objects have different trust models and safe sizes.

**Canonical resolution.** Replace use of the generic boundary with explicit APIs:

```text
decode_proof(bytes, ProofDecodeLimits)
decode_preprocessing(bytes, PreprocessingDecodeLimits)
decode_public_io(bytes, PublicIoDecodeLimits)
```

- `ProofDecodeLimits` is derived from verifier geometry plus a hard ceiling.
- `PreprocessingDecodeLimits` limits bytecode/program image/commitment counts and delegates PCS/VC setup decoding to bounded, semantically validating decoders.
- `PublicIoDecodeLimits` is derived from the validated memory layout and applies before allocating input/output/advice vectors.

The generic helper may remain for trusted fixture tooling, but it must be clearly named `deserialize_trusted_object` and must not be re-exported as a verifier API.

**Trust-model tradeoff.** The cleanest deployment model is to provision verifier preprocessing out of band and expose only a key identifier at the untrusted request boundary. That reduces attack surface but does not remove the typed API obligation: a public `verify(&preprocessing, ...)` must still return an error for a malformed object. Supporting arbitrary serialized preprocessing, as the current WASM API does, requires full structural and semantic validation.

### J-PRE-1: preprocessing-derived domain arithmetic

**Committed-program helper hazard, but current top-level precheck.** `ProgramMetadata` exposes serde-derived `usize` lengths, including `program_image_len_words`, in [`program.rs`](https://github.com/a16z/jolt/blob/78ae842c2cbc016d3b8f0e8dfeb30880aebaba8d/crates/jolt-program/src/preprocess/program.rs#L38-L52). [`precommitted_candidate`](https://github.com/a16z/jolt/blob/78ae842c2cbc016d3b8f0e8dfeb30880aebaba8d/crates/jolt-claims/src/protocols/jolt/geometry/claim_reductions/program_image.rs#L25-L35) calls `usize::next_power_of_two()`, so direct public construction of `PrecommittedSchedule` with `program_image_len_words = usize::MAX` panics in overflow-checked builds. However, this is **not** currently a serialized-preprocessing-to-`verify` panic: top-level validation first calls `compute_min_ram_k`, whose checked addition and `checked_next_power_of_two` return `InvalidMemoryLayout`. The regression should preserve that ordering while fixing the helper itself. Advice-domain padding divides the byte limit by eight first, so it does not overflow `next_power_of_two` for any `usize`; extreme advice values remain a size/work-policy and SDK `MemoryLayout::new` problem, not this panic.

**Conditional full-program trigger.** `BytecodePreprocessing` similarly exposes `code_size` and an independently sized bytecode vector in [`bytecode.rs`](https://github.com/a16z/jolt/blob/78ae842c2cbc016d3b8f0e8dfeb30880aebaba8d/crates/jolt-program/src/preprocess/bytecode.rs#L10-L28). The ZK BlindFold public-value path eventually computes `1usize << r_address.len()` in [`bytecode::read_raf_public_values`](https://github.com/a16z/jolt/blob/78ae842c2cbc016d3b8f0e8dfeb30880aebaba8d/crates/jolt-claims/src/protocols/jolt/geometry/bytecode.rs#L313-L331). A hostile `code_size` can derive a word-sized address length. This path runs only after stages 1–8 have verified, so it is a real unchecked verifier operation but not an easy immediate malformed-preprocessing trigger; reaching it requires a sufficiently consistent ZK proof. It should be fixed without overstating exploitability.

**Canonical resolution.** Validate preprocessing before any schedule or formula construction:

- committed program image and bytecode lengths are nonzero where required, bounded by the protocol/deployment maximum, and checked before padding, logarithms, shifts, additions, or conversion;
- full bytecode `code_size`, padded domain, bytecode vector length, PC map, and entry address/index are mutually consistent;
- advice byte/word counts are bounded as a resource/work policy and their derived domain uses checked arithmetic;
- memory regions, trace length, RAM geometry, PCS setup capacity, and optional VC capacity agree;
- all domain helpers use `checked_next_power_of_two`, `checked_shl`, `checked_add`, and fallible logarithm helpers even after validation.

The validated-preprocessing type should store checked derived dimensions so later stages do not recompute them from raw fields.

### J-IO-1: public-I/O allocation and clone amplification

`JoltDevice` serializes public `inputs`, `outputs`, **and both advice vectors** in [`common/src/jolt_device.rs`](https://github.com/a16z/jolt/blob/78ae842c2cbc016d3b8f0e8dfeb30880aebaba8d/common/src/jolt_device.rs#L44-L56). The modular verifier then clones the entire value merely to trim output padding in [`validate_inputs`](https://github.com/a16z/jolt/blob/78ae842c2cbc016d3b8f0e8dfeb30880aebaba8d/crates/jolt-verifier/src/verifier.rs#L353-L360). Large, irrelevant advice data can therefore survive decoding and approximately double peak memory before cryptographic rejection.

**Recommended resolution.** Define a verifier-specific wire/typed public-I/O type containing only statement-bearing fields. If compatibility requires accepting `JoltDevice`, reject nonempty advice vectors and validate input/output limits before cloning; normalize outputs into a borrowed view or clone only the bounded output prefix. This is both a codec fix (J1) and a semantic boundary fix (J2).

### D-SER-1 and D-SER-2: allocation-safe Dory canonical decoding

**Current surface.** Dory's generic `Vec<T>` decoder reads a `u64`, casts it to `usize`, and immediately preallocates in [`src/primitives/serialization.rs`](https://github.com/a16z/dory/blob/56a7243aceb4d74e19a576a0245028d8d117507e/src/primitives/serialization.rs#L289-L300). A tiny serialized setup can therefore panic with capacity overflow or request an enormous allocation before discovering EOF. The Arkworks proof decoder similarly allocates message vectors from an attacker-controlled round count in [`ark_serde.rs`](https://github.com/a16z/dory/blob/56a7243aceb4d74e19a576a0245028d8d117507e/src/backends/arkworks/ark_serde.rs#L440-L471).

The canonical `usize` decoder and the generic vector decoder both use unchecked `u64 as usize`. On wasm32, values above `u32::MAX` silently truncate, so identical setup bytes can acquire different dimensions or sequence lengths across architectures even when allocation does not fail.

Jolt's `DoryProof` wrapper preflights the inner round count at 64, but `DoryVerifierSetup` has no corresponding preflight, and upstream `ArkDoryProof::deserialize_compressed` remains unsafe for direct callers.

**Canonical resolution.** This belongs upstream in Dory:

1. Add a decode context carrying total bytes, maximum sequence length, and nesting/depth limits.
2. Use checked integer conversion and `try_reserve[_exact]`, mapping failure to `SerializationError`.
3. Do not preallocate the declared length when the remaining input cannot support it. At minimum, start empty or cap speculative preallocation; for fixed-size canonical types, bound count by remaining bytes divided by minimum encoded size.
4. Give proof/setup top-level decoders protocol-specific limits. A generic sequence decoder cannot know that Dory has at most a small number of rounds.
5. Require exact EOF in convenience `from_bytes` helpers; keep stream-oriented `CanonicalDeserialize` composable where concatenation is intentional.

**Compatibility tradeoff.** Adding a decode context to `DoryDeserialize` is the principled but breaking design and affects the derive crate. A compatibility release can first remove attacker-sized preallocation, add checked conversion/`try_reserve`, and add setup/proof-specific preflight. `try_reserve` prevents capacity-overflow unwind and reports many allocation failures, but is not a resource policy: it imposes no work limit and may succeed under virtual-memory overcommit. Count/byte limits remain mandatory. A later major release can make limits part of the trait.

### D-SETUP-1: validated Dory verifier setup

**Current surface.** `VerifierSetup` exposes five vectors and `max_log_n`. `ArkValid for ArkworksVerifierSetup` returns `Ok(())` unconditionally. Verification checks only `sigma <= max_log_n / 2`, then directly indexes:

- `chi[num_rounds]` and all `delta_*[num_rounds]` in [`reduce_and_fold.rs`](https://github.com/a16z/dory/blob/56a7243aceb4d74e19a576a0245028d8d117507e/src/reduce_and_fold.rs#L687-L714).
- `chi[0]` in both final-check modes at [lines 934 and 963](https://github.com/a16z/dory/blob/56a7243aceb4d74e19a576a0245028d8d117507e/src/reduce_and_fold.rs#L928-L966).

Generated setups have even `max_log_n` and every vector has exactly `max_log_n / 2 + 1` elements. Serialized empty or ragged vectors are nevertheless accepted.

**Canonical resolution.** Add structural `VerifierSetup::validate` and preferably a `ValidatedVerifierSetup` newtype. Dory's generic `Group` trait currently has no validity method, so D2 should pair generic structural checks with an Arkworks-specific recursive value validator and explicitly invoke both before transcript absorption. D3 can later make validity part of the backend type/trait invariant. Validate once at construction/deserialization:

- `max_log_n` is even and within a documented protocol/backend maximum.
- All five vectors have exactly `max_log_n / 2 + 1` elements and are nonempty.
- Backend G1/G2 values pass curve/subgroup validity; GT's underlying representation is nonzero and in the target subgroup. Group identity itself is valid and must not be generically rejected.
- Fixed generators and `ht` are valid.
- Cheap internal consistency holds: all four delta vectors have identity at index zero; `delta_1l == delta_2l`; `delta_1l[k] == chi[k - 1]` for `k >= 1`; `chi[0] == e(g1_0, g2_0)`; and `ht == e(h1, h2)`.

Verification must still use `.get()` and return `DoryError::InvalidSetup`; a validated type is defense in depth, not permission for unchecked indexing at an untrusted public API.

**Semantic-validation tradeoff.** Shape and group validity prevent panics but do not prove that `chi` and `delta_*` were honestly derived from the URS. Perform G1/G2 validity checks before the two consistency pairings so validation itself cannot reach an infallible backend on malformed direct values. The equalities above cheaply catch malformed or internally inconsistent setup and cost two pairings once at setup load, but still are not authentication. If preprocessing is an authenticated verifier key, full algebraic recomputation is unnecessary per proof. If arbitrary preprocessing bytes are accepted as a statement, the setup must be bound to a separately trusted digest or regenerated from a deterministic trusted URS. Recomputing all pairings during every verification is expensive and should not be the default.

### D-DIM-1: checked Dory dimensions

**Current surface.** Dory calculates `nu + sigma` twice in the verifier's point-length error path before checking `nu <= sigma` or setup capacity in [`evaluation_proof.rs`](https://github.com/a16z/dory/blob/56a7243aceb4d74e19a576a0245028d8d117507e/src/evaluation_proof.rs#L338-L400). Serialized values are `u32`, so this overflows `usize` on wasm32 with overflow checks. Direct callers can construct arbitrary `usize` values on any target.

**Canonical resolution.** Validate `nu <= sigma` and `sigma <= setup.max_log_n / 2` first, then use `nu.checked_add(sigma)` and return `InvalidSize`/`InvalidPointDimension` on failure. Audit every `1 << dimension`, `2^dimension`, addition, and multiplication in public setup/prove/verify APIs and replace it with checked arithmetic plus documented maximums.

This change is wire-compatible and has no meaningful verifier cost.

### D-GROUP-1: invariant-bearing group wrappers

**Current surface.** Dory's Arkworks wrappers and proof fields are publicly constructible, while `ArkValid for ArkDoryProof` returns `Ok(())`. An in-memory proof can set `vmv_message.d2` to `ArkGT(PairingOutput(Fq12::ZERO))`. ZK final verification subtracts its scaled value, reaching the patched Arkworks [`PairingOutput::sub_assign`](https://github.com/a16z/arkworks-algebra/blob/76bb3a4518928f1ff7f15875f940d614bb9845e6/ec/src/pairing.rs#L248-L251), which calls `cyclotomic_inverse().unwrap()`.

Checked canonical Jolt and upstream Dory decoding reject zero GT because field/group validation is forwarded to `PairingOutput::check`. This is a direct public construction/mutation, `Validate::No`, or malformed custom-backend trigger, not an ordinary checked proof-byte trigger. The no-op composite `ArkValid` implementations remain defective because directly constructed or post-mutated composite values pass `.check()`, and cross-field shape is never validated.

Jolt also exposes `DoryCommitment(pub Bn254GT)`, while `Bn254GT: From<Fq12>` permits `Fq12::ZERO`. A zero directly constructed commitment becomes verifier state `d1` and can reach the same inverse-based subtraction. The verifier must validate commitments as well as proof fields, and the arbitrary-field conversion should become checked or crate-private.

**Recommended resolution.** Fix the invariant at the narrowest owning type:

- Make `ArkGT`'s inner value private.
- Provide checked constructors and an explicit `unsafe`/crate-private unchecked constructor for internal pairing outputs whose validity is proven by construction.
- Implement recursive `ArkValid` for Dory proofs/setups instead of unconditional success.
- Explicitly invoke recursive checks at deserialization and verification entry; implementing `.check()` alone does not protect manual composite construction.
- Validate proof mode, dimensions, group values, and setup before transcript absorption.

**Arkworks tradeoff.** Making `PairingOutput` itself invariant-preserving in Arkworks is architecturally clean but broadly breaking. Defining subtraction of zero as zero or identity is algebraically wrong. Returning `Result` from group subtraction is incompatible with standard operator traits. Dory-level private checked wrappers are therefore the recommended immediate fix; open an Arkworks issue/PR separately if the ecosystem is willing to accept a major API change.

Final-exponentiation calls present the same design choice. Patched Arkworks' default `multi_pairing` unwraps `final_exponentiation` in [`pairing.rs`](https://github.com/a16z/arkworks-algebra/blob/76bb3a4518928f1ff7f15875f940d614bb9845e6/ec/src/pairing.rs#L143-L157), and Dory's backend delegates `pair` to that infallible API. Once inputs are validated, failure is an internal backend error, but a literal panic-free contract still requires Dory verifier/setup validation to use `try_pair`/`try_multi_pair` and propagate the existing `Option`/failure. That change belongs in a separate fallible-backend Dory PR because a trait-wide conversion propagates through pairing and prover code.

### D-STATE-1: lower-level verifier state and backend primitives

The one-shot `verify_evaluation` API is not Dory's only public verifier surface. `DoryVerifierState::new` checks coordinate lengths only with `debug_assert` at [`reduce_and_fold.rs:638–651`](https://github.com/a16z/dory/blob/56a7243aceb4d74e19a576a0245028d8d117507e/src/reduce_and_fold.rs#L638-L651), later indexes those coordinates directly at [lines 728–730](https://github.com/a16z/dory/blob/56a7243aceb4d74e19a576a0245028d8d117507e/src/reduce_and_fold.rs#L728-L730), and `verify_final` checks that all rounds were consumed only with `debug_assert` at [lines 863–878](https://github.com/a16z/dory/blob/56a7243aceb4d74e19a576a0245028d8d117507e/src/reduce_and_fold.rs#L863-L878). Release builds therefore turn short direct coordinate vectors into indexing panic and premature finalization into semantically incomplete verification.

The Arkworks backend also exposes MSM/multi-pairing methods that assert equal input lengths, and final exponentiation uses `expect`. Jolt's current top-level path supplies fixed equal-length arrays and valid points, so these are direct backend API hazards rather than demonstrated proof-byte triggers.

**Resolution options.** Prefer making the state type and individual transition methods crate-private, exposing one checked verifier entry point. If incremental verification is a supported API, replace `new` with `try_new`, encode the remaining-round state in private fields, and have every transition/finalization return `Result`. Add verifier-specific `try_msm`/`try_multi_pair` methods rather than making mismatched inputs zip or map to identity. A trait-wide `Result` conversion is cleaner but breaking; checked verifier variants can ship first.

### J-SDK-1 and J-SDK-2: fallible generated verifier wrappers

**Current surface.** The native generated verifier copies fields from `preprocessing.program.memory_layout()` into a `MemoryConfig`, then calls `JoltDevice::new` in [`jolt-sdk/macros/src/lib.rs`](https://github.com/a16z/jolt/blob/78ae842c2cbc016d3b8f0e8dfeb30880aebaba8d/jolt-sdk/macros/src/lib.rs#L272-L290). `MemoryLayout::new` contains power-of-two assertions and checked-arithmetic `expect`s in [`common/src/jolt_device.rs`](https://github.com/a16z/jolt/blob/78ae842c2cbc016d3b8f0e8dfeb30880aebaba8d/common/src/jolt_device.rs#L295-L413). For example, `max_trusted_advice_size = 24` reaches an assertion; extreme sizes reach alignment or region overflows.

The generated wrapper also calls `postcard::to_stdvec(...).unwrap()` for public arguments and output. A custom or fallible `Serialize` implementation can panic the verifier before canonical verification begins.

Every native generated verification also clones the entire captured preprocessing value in [`jolt-sdk/macros/src/lib.rs`](https://github.com/a16z/jolt/blob/78ae842c2cbc016d3b8f0e8dfeb30880aebaba8d/jolt-sdk/macros/src/lib.rs#L270-L274). This duplicates full bytecode/RAM preprocessing and the Dory verifier setup once per proof, before validation. It is an avoidable memory/latency amplifier even when preprocessing is trusted, and is especially dangerous while serialized preprocessing remains public.

**Canonical resolution.** Add:

- `MemoryLayout::try_new(&MemoryConfig) -> Result<MemoryLayout, MemoryLayoutError>` with checked alignment, powers, address ranges, ordering, and non-overlap.
- `JoltVerifierPreprocessing::validate() -> Result<ValidatedPreprocessing<'_>, VerifierError>` covering program metadata, memory layout, trace/RAM bounds, PCS setup, and optional VC setup.
- A generated fallible builder/closure API. Validation happens once when building the verifier; retain `Arc<ValidatedPreprocessing>` (or owned checked preprocessing) and borrow it rather than cloning raw preprocessing per proof. Verification returns a typed error for serialization or proof rejection.

**API tradeoff.** Three migration choices exist:

1. Change generated `build_verifier` and its closure to return `Result`. This is explicit and recommended, but breaking.
2. Add `try_build_verifier`/`try_verify` and deprecate the boolean API. The old boolean wrapper can map every error to `false`, preserving source compatibility without panics.
3. Capture preprocessing failure and return an always-`false` closure. This avoids a signature change but hides configuration errors and is not recommended as the primary API.

Public-I/O serialization errors should be distinct from cryptographic rejection in the typed API, even if the final WASM `bool` export maps both to `false`.

### J-PCS-1 and J-VC-1: direct cryptographic helper APIs

**Dory combination.** `DoryScheme::combine` asserts commitment/scalar length equality in [`crates/jolt-dory/src/scheme.rs`](https://github.com/a16z/jolt/blob/78ae842c2cbc016d3b8f0e8dfeb30880aebaba8d/crates/jolt-dory/src/scheme.rs#L244-L253). Current stage 8 derives both arrays from the same entries, so malformed proofs do not control the mismatch. The public trait API is still not panic-free.

The canonical solution is `try_combine(...) -> Result<Output, OpeningsError>` and use it in verification. Changing `AdditivelyHomomorphic::combine` itself to return `Result` is cleaner but affects every PCS implementation and prover call. A staged migration can add `try_combine`, make verifier code use it, and retain `combine` only for checked prover internals until the trait is revised. Zipping to the shorter input is unsound.

**Pedersen verification.** `Pedersen::verify` calls a `commit` that asserts `values.len() <= capacity` in [`crates/jolt-crypto/src/ec/pedersen.rs`](https://github.com/a16z/jolt/blob/78ae842c2cbc016d3b8f0e8dfeb30880aebaba8d/crates/jolt-crypto/src/ec/pedersen.rs#L87-L104). The verifier-side fix is nonbreaking: return `false` immediately when `values.len() > capacity`. Prover-side `commit` can later become fallible; current BlindFold verification already performs capacity checks.

### J-CORE-1: latent verifier invariant panics

No malformed-proof route was demonstrated for these sites after current round-count and dimension validation, but their attempted error handling is not actually panic-free:

- `point.get(point.len() - log_t..)` underflows before `.get()` in [`instruction_ra_virtualization.rs`](https://github.com/a16z/jolt/blob/78ae842c2cbc016d3b8f0e8dfeb30880aebaba8d/crates/jolt-verifier/src/stages/stage6b/instruction_ra_virtualization.rs#L119-L129), [`ram_ra_virtualization.rs`](https://github.com/a16z/jolt/blob/78ae842c2cbc016d3b8f0e8dfeb30880aebaba8d/crates/jolt-verifier/src/stages/stage6b/ram_ra_virtualization.rs#L112-L120), and [`bytecode_read_raf.rs`](https://github.com/a16z/jolt/blob/78ae842c2cbc016d3b8f0e8dfeb30880aebaba8d/crates/jolt-verifier/src/stages/stage6b/bytecode_read_raf.rs#L144-L150).
- Instruction address reconstruction slices through `point.len() - cycle_len` in [`stage5/instruction_read_raf.rs`](https://github.com/a16z/jolt/blob/78ae842c2cbc016d3b8f0e8dfeb30880aebaba8d/crates/jolt-verifier/src/stages/stage5/instruction_read_raf.rs#L82-L93).
- RAM/register helpers slice at `ram_log_k` or `REGISTER_ADDRESS_BITS` without `.get()` in [`ram_val_check.rs`](https://github.com/a16z/jolt/blob/78ae842c2cbc016d3b8f0e8dfeb30880aebaba8d/crates/jolt-verifier/src/stages/stage4/ram_val_check.rs#L254-L262) and [`registers_read_write_checking.rs`](https://github.com/a16z/jolt/blob/78ae842c2cbc016d3b8f0e8dfeb30880aebaba8d/crates/jolt-verifier/src/stages/stage4/registers_read_write_checking.rs#L108-L113).
- `OutputClaims::opening_values` uses `expect` on a derive-maintained invariant in [`crates/jolt-claims/src/claim_data.rs`](https://github.com/a16z/jolt/blob/78ae842c2cbc016d3b8f0e8dfeb30880aebaba8d/crates/jolt-claims/src/claim_data.rs#L46-L63).
- Sumcheck constructors assert positive protocol degree; `LtPolynomial::evaluate` asserts equal dimensions; stage-8 has nested exhaustive `unreachable!` arms; transcript label helpers assert fixed label lengths.

**Resolution.** Use `checked_sub` and `.get()` before every slice. Change helpers that cannot report failure, such as `reconstruct_r_address`, to return `Result`. Replace derive invariants with generated direct iteration or a fallible method. Replace enum `unreachable!` with typed errors even when the outer match appears exhaustive. Make transcript labels const-validated or constructed through fallible/compile-time checked types.

**Tradeoff.** These checks add negligible work relative to cryptographic verification. Retaining panics because values are “verifier-derived” makes refactors and alternate backends silently expand the attack surface. The recommended policy is no unchecked arithmetic, slice, or enum assumption in verifier crates, regardless of current reachability.

### D-TR-1: fallible transcript behavior

**Current surface.** Dory's default Blake2b transcript panics if a scalar challenge is zero. This is a theoretical verifier panic under the random-oracle model—no practical adversarial trigger is known without a field-sized preimage search—but it violates the absolute API contract. Its `expect`s when serializing group/protocol messages into a `Vec` are generic/custom-serializer or allocator-failure hazards, not malformed-proof triggers for the concrete Arkworks serializers. Jolt's adapter does not panic on zero challenge and Dory's verifier inversions return `InvalidProof`. Jolt's unused `reset` implementation is `unreachable!`; Dory 0.4 does not call it, making it a dependency-upgrade hazard rather than a current proof trigger.

**Zero-challenge options.**

1. **Deterministic rejection sampling with a counter/domain tag — recommended.** If the reduced challenge is zero, hash/squeeze again with an unambiguous counter until nonzero. Both prover and verifier use the same rule. This changes only transcripts for which the current implementation aborts, so it does not invalidate any successfully generated proof.
2. Return `Result` from `challenge_scalar` and reject zero. This is explicit but makes the transcript and every protocol call fallible.
3. Map zero to one. This introduces bias and is not recommended.

**Serialization/reset options.** A literal no-panic API requires transcript append methods to return `Result`, allowing serialization errors to propagate. Treating `Vec` serialization as infallible is operationally reasonable but still leaves `expect` in a public verifier call path. Remove `reset` from the Dory transcript trait if it is unused; making it a no-op or emulating reset by appending a label changes semantics and is unsafe without a protocol decision.

### D-OPS-1: setup and cache operations

Dory panics when setup storage paths cannot be determined, directories/files cannot be created, serialization fails, cached setup loading fails, or cache locks are poisoned. `ProverSetup::new` also performs unchecked shifts/allocations for extreme dimensions; `to_verifier_setup` assumes nonempty, equal-sized, power-of-two generator vectors. Evaluation verification currently clones the whole verifier setup into state, unnecessarily duplicating a potentially large malformed object.

The optimized Arkworks cache has an additional correctness and safety problem: global prepared-generator entries are selected primarily by length. A different random setup with the same dimensions can reuse preparations for the wrong generators, while undersized cached arrays can later be sliced/indexed. This is not merely an availability concern; cache identity must include the setup identity. It is a direct Dory-consumer hazard, but not currently in Jolt's normal final-verifier path: Jolt enables the feature without calling `init_cache`, and final verification uses ordinary multi-pairing.

**Resolution.** Introduce `SetupError` and make generation, derivation, load, store, and cache operations return `Result`. Validate cached prover and verifier setups before insertion/use. On corrupt cache bytes, either return an error or atomically quarantine and regenerate; never continue with partially decoded data. Use checked dimension arithmetic and a documented maximum before allocating generators. Borrow or hold an `Arc<ValidatedVerifierSetup>` in verifier state instead of cloning it.

For prepared data, the robust design is setup-local `Arc`/`OnceLock` state, which makes identity unambiguous and lifetime-bounded. A lower-impact compatibility fix can key the global cache by a cryptographic fingerprint plus dimensions and verify all lengths before reuse, but it retains global memory/lifetime complexity.

**Regeneration tradeoff.** Automatic regeneration improves availability but can hide persistent corruption and cause expensive repeated work. Recommended behavior: quarantine the corrupt file, regenerate once under a lock, write atomically, and surface telemetry. Security-sensitive callers should be able to disable regeneration and require an authenticated setup digest.

### J-MAL-1: nested exact-consumption checks

`DoryProof` and `DoryVerifierSetup` deserialize their inner canonical buffer from `&buf[..]` without checking that the inner decoder consumed all bytes. The SDK's top-level `consumed == bytes.len()` check cannot see garbage inside that nested byte vector.

This is not a demonstrated panic, but it permits multiple accepted encodings and payload amplification. Use a `Cursor`, require `position == buf.len()`, and add trailing-byte tests for each nested canonical wrapper. Preserve #1657's separate NARG EOF logic if that PR returns.

### J-FUZZ-1: adversarial assurance gap

The existing `crates/jolt-dory/fuzz` standalone workspace misses the root arkworks patch and currently fails from duplicate registry/git Arkworks types. `verify_tampered.rs` also mishandles the `Result` returned by `commit`. Even once it builds, arbitrary bytes almost never deserialize into valid curve points, so the target rarely exercises deep verification. It does not fuzz verifier setup, preprocessing, the generated SDK boundary, or WASM.

**Resolution.** Seed from authentic clear and ZK proofs, then structurally mutate:

- every vector length and optional proof-mode field;
- `nu`, `sigma`, round counts, point lengths, and setup `max_log_n`;
- every `chi`/`delta_*` vector independently;
- valid group elements, zero/identity/invalid unchecked direct values, and commitment layouts;
- preprocessing memory layout, bytecode/program metadata, VC capacity, public-I/O lengths;
- nested and outer trailing bytes, truncation, hostile length prefixes, and maximum accepted payloads.

Run the typed verifier and every deserialize-plus-verify boundary under `catch_unwind` as a panic oracle. Run allocation-amplification cases in subprocesses with memory/time limits because allocator abort cannot be caught. Include native 64-bit, a 32-bit overflow-checked target, and WASM runtime tests.

## Proposed PR sequence

### Existing Jolt PR J1: rebase and extend #1609 — typed verifier wire boundaries

**Repository:** `a16z/jolt`

**Scope:**

- Preserve #1609's proof codec, typed errors, exact-consumption checks, and tests.
- Rebase onto the pinned modular verifier.
- **J1a:** wire absolute proof/public-I/O caps and exact-consumption codecs into production SDK/WASM paths, using separate documented native/WASM ceilings.
- **J1b:** after validated geometry and safe Dory decoding exist, add production preprocessing decoding and derive tighter proof/I/O limits from `ValidatedPreprocessing`.
- Add hostile-prefix, nested-container, and real proof/preprocessing/I/O regression tests.

**Dependency boundary:** Dory's full inner decoder and semantic setup validation belong upstream, but J1 must not expose production preprocessing decoding while a tiny nested Dory setup can still trigger attacker-sized allocation. Either land D1/J3 first, or include a narrow temporary prefix/length preflight in J1 and remove it only after delegation to the fixed release. Do not fork Dory's full decoding logic.

**Dependency/coordination:** If #1657 is revived first, retain its NARG frame/EOF work and port J1 to its `JoltProof<PCS>` model. Otherwise target pinned main. J1 should not add a second NARG parser.

### Dory PR D1 — allocation-safe canonical decoding

**Repository:** `a16z/dory`

**Scope:**

- Checked `u64 -> usize` conversions (including standalone serialized `usize`) and non-panicking reserve behavior in generic sequence decoding.
- No attacker-directed preallocation; bounded URS/setup decode primitives and explicit total count/byte policies.
- Proof/setup-specific sequence/round limits.
- Direct `ArkDoryProof` round allocation protection.
- Bounded top-level decode helpers and exact-consumption APIs.
- Tiny-input/huge-prefix, truncation, trailing, and memory-budget tests.

**Decision:** Prefer a wire-compatible compatibility fix now; design a decode-context trait for the next breaking release.

### Dory PR D2 — fail-closed verifier validation

**Repository:** `a16z/dory`

**Scope:**

- Generic structural `VerifierSetup::validate`/`ValidatedVerifierSetup`, plus Arkworks-specific recursive value validation.
- Recursive proof/setup `ArkValid` implementations explicitly invoked at decode/verify entry.
- Checked `nu + sigma`, setup capacity, message counts, and point dimensions before transcript absorption.
- `.get()` for all setup accesses with `InvalidSetup`/`InvalidProof` errors.
- Checked or private incremental `DoryVerifierState` construction/transitions/finalization.
- Move the already-owned setup into verifier state instead of cloning it; reserve borrowed/`Arc` redesign for D4 if useful.
- Clear and ZK regression tests for empty/ragged/short/long setup vectors, all base-delta identities, left-delta/previous-`chi` relations, `chi[0]`/`ht` pairing consistency, zero rounds, maximum dimensions, and mixed proof modes.

**Dependency:** Can be reviewed independently of D1, but release both together so validated deserialization and safe verification agree.

### Dory PR D3 — invariant-preserving/fallible backend and transcript APIs

**Repository:** `a16z/dory`; optional follow-up in `a16z/arkworks-algebra`

**Scope:**

- Private checked `ArkGT` and other invariant-bearing wrappers.
- Make group validity part of the backend contract so D2's repeated runtime validation eventually becomes structurally unnecessary.
- Propagate final-exponentiation and transcript serialization failures.
- Deterministic nonzero challenge derivation.
- Remove or redesign unused transcript `reset`.
- Add fallible verifier MSM/multi-pairing variants and propagate final-exponentiation failure.

**Tradeoff:** This is the most API-disruptive upstream PR. Keep it separate from D1/D2 so urgent external-input fixes can ship first.

### Dory PR D4 — fallible setup/cache/storage

**Repository:** `a16z/dory`

**Scope:**

- `SetupError` and fallible setup generation/load/store/cache APIs.
- Checked setup dimension arithmetic and resource maximums.
- Cache validation, atomic writes, poison handling, corrupt-file quarantine/regeneration policy.
- Setup-local prepared-data cache, or a compatibility cache keyed by a cryptographic setup fingerprint and dimensions.
- Optional borrowed/`Arc` validated setup and setup-local prepared state for long-lived consumers.

**Tradeoff/dependency:** Setup construction signatures change. A compatibility wrapper may remain for applications, but Jolt must use only the fallible API. D4 should reuse D1's bounded setup/URS decoding and D2's validated setup type.

### Jolt PR J2 — preprocessing validation and fallible generated verifier

**Repository:** `a16z/jolt`

**Scope:**

- `MemoryLayout::try_new` and semantic layout validation.
- `JoltVerifierPreprocessing::validate`, including full/committed program metadata, bytecode/vector/domain consistency, advice bounds, trace/RAM geometry, PCS setup, and optional VC setup.
- Checked derived domain dimensions stored in `ValidatedPreprocessing`; no raw `next_power_of_two` or shifts on caller fields.
- A verifier-specific public-I/O type, or early rejection of advice fields plus borrowed/bounded output normalization.
- `try_build_verifier`/`try_verify` generated APIs; remove verifier-path postcard unwraps and retain checked preprocessing without per-proof cloning.
- Require validated preprocessing before canonical verification on native and WASM.
- Tests for every arithmetic, alignment, ordering, overlap, capacity, and serialization failure, including `program_image_len_words = usize::MAX` and `code_size = usize::MAX`.

**Decision required:** choose breaking `Result` API versus additive fallible API plus deprecated boolean wrapper. The additive migration is recommended.

**Dependency:** J2 can define the PCS validation hook and validate all Jolt-owned structure immediately, but complete Dory setup semantics require D2/J3. Likewise, preprocessing-derived proof/I/O limits should be added only from `ValidatedPreprocessing`; J1a can ship absolute caps first and J1b derives tighter limits after J2.

### Jolt PR J3 — consume fixed Dory and harden adapters

**Repository:** `a16z/jolt`

**Scope:**

- Upgrade to the D1/D2 Dory release and pin the exact version/checksum.
- Delegate setup/proof validation and map all Dory errors to `OpeningsError`/`VerifierError` without collapsing diagnostic classes.
- Require nested EOF for Dory proof/setup/commitment wrappers.
- Apply the fixed dependency path to every modular `jolt-verifier` Dory adapter.
- Add direct-wrapper tests for empty/ragged setup, dimension overflow, invalid group values, mixed modes, and trailing bytes.

**Urgency option:** If upstream release timing is slow, add a temporary Jolt setup preflight. Mark it as defense in depth and later reduce it to delegation; do not fork Dory's full decoder/verification logic.

### Jolt PR J4 — remove latent verifier invariant panics

**Repository:** `a16z/jolt`

**Scope:**

- Checked slicing/arithmetic and fallible helper signatures across `jolt-verifier`/`jolt-claims`.
- `try_combine` for verifier commitment aggregation.
- Pedersen verifier capacity rejection.
- Typed errors instead of verifier `unreachable!` and derive-maintained `expect`.
- Compile-time/fallible transcript labels.
- Audit all remaining public methods in `jolt-verifier` and its verifier-side `jolt-claims`/PCS/VC helpers.

**Tradeoff:** Do not force the broad PCS trait migration into the urgent P0 work. Add fallible verifier methods first, then revise shared traits in a focused follow-up.

### Jolt PR J5 — full no-panic regression and fuzz program

**Repository:** `a16z/jolt`

**Scope:**

- Repair the Dory fuzz workspace and target.
- Add native/WASM SDK deserialize-plus-verify fuzz targets and typed-verifier structured mutation targets.
- Maintain valid modular clear/ZK/advice/committed-program corpora.
- Add subprocess memory/time-budget tests, 32-bit overflow-check tests, and `catch_unwind` regression tests.
- Add a CI policy inventorying every verifier-crate `#[expect(clippy::panic|unwrap_used|expect_used)]`, unchecked index/slice, and intentionally infallible adapter.

**Dependency:** Land after J1-J4 APIs stabilize, while adding issue-specific regression tests in each earlier PR.

### Optional Arkworks PR A1 — invariant-preserving pairing outputs

**Repository:** `a16z/arkworks-algebra`

**Scope:** Explore making `PairingOutput` impossible to construct with zero and eliminating inverse unwraps from group subtraction.

**Decision:** Do not block Jolt/Dory remediation on this. Dory private checked wrappers close the verifier boundary with much less ecosystem disruption. Pursue A1 only with an explicit major-version/API-compatibility plan.

## Execution order and dependencies

```text
D1 bounded decode + D2 verifier checks ─> urgent Dory release ─> J3 adapter/bump ─┐
J1a #1609 absolute proof/I/O bounds ──────────────────────────────────────────────┤
J2 preprocessing validation + fallible SDK ──────────────────────────────────────┤
                                                                                  └─> J1b derived/preprocessing limits

J1b + J4 modular latent cleanup ─> J5 integrated fuzz/CI

D3 backend/transcript ─> independent follow-up Dory release ┐
D4 setup/cache ─────────> independent follow-up Dory release ┴─> later Jolt bump as needed

Optional A1 Arkworks hardening can proceed independently after D3's type decision.
```

J1a's absolute proof and public-I/O caps, D1/D2, and J2's Jolt-owned validation can proceed in parallel. J1b's production preprocessing decoder waits for D1/J3 unless it carries the narrowly scoped temporary setup preflight; its geometry-derived limits also wait for J2's validated dimensions. D3 and D4 should not delay the urgent D1/D2 release. J5 is a final integrated campaign, not a substitute for regression tests in each fix. Coordinate J1/J2/J4 rebases and setup-dimension changes with #1669 because it touches the same Jolt files even though it does not contain the hardening.

## Acceptance criteria

- [ ] Every production serialized verifier entry point has an enforced object-specific byte/work limit and exact top-level consumption.
- [ ] A tiny malformed Dory setup declaring `u64::MAX` returns a decode error without panic or large allocation.
- [ ] Empty, ragged, short, long, and `max_log_n`-inconsistent Dory verifier setups return `InvalidSetup` in clear and ZK verification.
- [ ] Dory dimension addition and every public setup/domain shift use checked arithmetic on native, 32-bit, and WASM targets.
- [ ] Arbitrary directly constructed Dory proof/setup/commitment values cannot inject zero/invalid GT representations into group operations.
- [ ] Generated native/WASM verification returns failure for invalid preprocessing and serialization errors; it never reconstructs an invalid memory layout through a panicking constructor.
- [ ] `program_image_len_words = usize::MAX`, extreme advice bounds, and `code_size = usize::MAX` return preprocessing/domain errors before schedule or BlindFold construction.
- [ ] Public I/O advice fields cannot amplify verifier memory, and output normalization does not clone an attacker-sized `JoltDevice`.
- [ ] Pedersen verification above capacity returns `false`/error; commitment aggregation length mismatch returns an error.
- [ ] Dory incremental verifier state cannot be constructed/finalized out of order without an error; mismatched backend MSM/pairing inputs return errors.
- [ ] Dory prepared-generator cache entries are bound to the exact setup, not only their dimensions.
- [ ] Nested Dory canonical wrappers reject trailing bytes.
- [ ] Clear and ZK authentic proofs continue to verify in native and WASM builds.
- [ ] `cargo nextest`, both required clippy modes, and `cargo fmt` pass in Jolt; upstream Dory has equivalent clear/ZK tests.
- [ ] Structured mutation corpora cover proof, preprocessing, setup, I/O, and direct typed APIs under panic and resource-budget oracles.
- [ ] The verifier-path panic inventory has no unreviewed `panic!`, `assert!`, `unwrap`, `expect`, `unreachable!`, unchecked subtraction-before-slice, or caller-controlled indexing.

## Test plan

For every Jolt implementation PR:

```bash
cargo nextest run -p jolt-dory --cargo-quiet
cargo nextest run -p jolt-verifier --cargo-quiet
cargo nextest run -p jolt-verifier --cargo-quiet --features zk
cargo nextest run -p jolt-prover-legacy muldiv --cargo-quiet --features host
cargo nextest run -p jolt-prover-legacy muldiv --cargo-quiet --features host,zk
cargo clippy --all --features host -q --all-targets -- -D warnings
cargo clippy --all --features host,zk -q --all-targets -- -D warnings
cargo fmt -q
```

The two `jolt-prover-legacy muldiv` commands above remain because the repository's `AGENTS.md` requires them as integration checks; passing them does **not** make the legacy verifier part of this panic-freedom target.

Add focused nextest cases for each modular-verifier finding rather than relying only on full-suite success. Use subprocess tests for allocation-abort risks and a real WASM runtime for the exported verifier. Fuzzing must begin from valid proof/setup corpora so mutations reach deep verifier stages.

At minimum, add named regressions for setup/vector length prefixes of `u64::MAX`, `u32::MAX + 1`, and values that truncate to plausible small wasm32 values; empty and independently ragged `chi`/`delta_*`; `nu + sigma` overflow on a 32-bit target; top-level committed `program_image_len_words = usize::MAX` (which must continue returning its current checked error) plus a direct `PrecommittedSchedule::new` no-panic/error regression for the currently panicking helper; full-program `code_size = usize::MAX`; and public I/O with large/nonempty advice. Exercise both clear and ZK modular verification wherever the affected path exists.

## Performance considerations

- Setup shape/group validation is linear in setup size. Perform it once when constructing validated preprocessing, not once per proof.
- Two setup-consistency pairings at trusted setup load are preferable to per-proof recomputation; cache the validated result with the exact setup identity.
- Proof/preprocessing/I/O byte caps reduce work and should not affect valid verification.
- Checked arithmetic and `.get()` in stage helpers are negligible relative to sumcheck and pairings.
- Recursive group/subgroup validation can be expensive if repeated. Checked deserialization plus an invariant-preserving validated wrapper should allow verification to avoid duplicate subgroup checks.
- Geometry-derived proof limits should be computed without allocating proof-sized tables.
- NARG/frame parsing should borrow from the input where possible; #1657 currently copies a frame body, increasing peak memory to the full proof plus the largest frame.
- Avoid cloning `JoltDevice` or Jolt preprocessing in verification; borrow checked data or retain preprocessing in an `Arc`. Move the already-owned Dory setup into its state immediately, and consider borrowed/`Arc` setup only as a longer-lived API optimization.

## Deployment guidance

Even after library fixes:

- Cap request bodies before buffering.
- Store verifier preprocessing under a trusted identifier rather than accepting arbitrary setup bytes where possible.
- Validate and cache the verifier key once, binding it to an authenticated digest/version.
- Run verification in a process or WASM instance with memory and time quotas.
- Record structured rejection reasons internally without exposing sensitive transcript state.
- Treat a caught panic as a security defect and terminate/quarantine that worker; do not resume verification from partially mutated state.

## References

- [a16z/jolt#1609 — Mitigate proof-deserialization DOS vector](https://github.com/a16z/jolt/pull/1609)
- [a16z/jolt#1657 — Migrate Jolt core transcripts to Spongefish](https://github.com/a16z/jolt/pull/1657)
- [a16z/jolt#1669 — Feat/jolt prover](https://github.com/a16z/jolt/pull/1669)
- [Unrelated older Arkworks PR #33](https://github.com/a16z/arkworks-algebra/pull/33) and [#36](https://github.com/a16z/arkworks-algebra/pull/36)
- [Dory verifier setup definition](https://github.com/a16z/dory/blob/56a7243aceb4d74e19a576a0245028d8d117507e/src/setup.rs#L42-L80)
- [Dory verification shape gate](https://github.com/a16z/dory/blob/56a7243aceb4d74e19a576a0245028d8d117507e/src/evaluation_proof.rs#L338-L400)
- [Jolt modular verifier entry point](https://github.com/a16z/jolt/blob/78ae842c2cbc016d3b8f0e8dfeb30880aebaba8d/crates/jolt-verifier/src/verifier.rs#L37-L170)
- [Jolt SDK WASM verifier boundary](https://github.com/a16z/jolt/blob/78ae842c2cbc016d3b8f0e8dfeb30880aebaba8d/jolt-sdk/macros/src/lib.rs#L1395-L1422)
