# Spec: Aligned Borrowed Byte Inputs

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @quangvdao                     |
| Created     | 2026-08-21                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

Jolt currently uses Postcard for every function argument. This is a good default for small typed values, but it is not a complete input model for large encoded objects. A borrowed byte slice receives a variable size Postcard length prefix, so the address of its payload depends on its length. Applications that need aligned word loads must understand this outer frame and repair alignment inside their own wire format. This spec adds an explicit byte codec for `#[jolt::provable]` parameters. Jolt will encode a fixed width length and footer, place the payload at an eight byte aligned address, and give the guest a borrowed slice without copying or allocating. The same codec will work for public input, trusted advice, untrusted advice, and private input. Postcard will remain the default.

## Intent

### Goal

Add one Jolt owned input codec that gives guest code a bounded, borrowed byte slice with an eight byte aligned payload while keeping input provenance separate from input representation.

### Terms

An **input channel** is one of the existing Jolt memory regions for public input, trusted advice, or untrusted advice. `PrivateInput<T>` uses the untrusted advice channel and adds the existing zero knowledge requirement.

A **codec** defines how one function argument becomes bytes in its channel and how the generated guest entry point reconstructs the argument. Postcard is the existing default codec. The byte codec in this spec is an explicit second codec.

A **byte frame** is the canonical length, padding, and payload representation defined below. It is part of the bytes that Jolt commits or exposes. It is not application data.

### Invariants

1. **Postcard remains the default.** A parameter without the new attribute must produce the same channel bytes and generated guest behavior as it does before this change.

2. **The byte codec is explicit.** The SDK must not automatically change the representation of every `&[u8]` parameter. Existing programs must opt in.

3. **Provenance and representation are independent.** The byte codec must work with public input, `TrustedAdvice<&[u8]>`, `UntrustedAdvice<&[u8]>`, and `PrivateInput<&[u8]>`. Channel selection must continue to control proof and verifier semantics. Codec selection must only control byte framing and the guest view.

4. **The frame is canonical.** The length uses one fixed width little endian `u64`. Every alignment byte is zero. An eight byte footer equals the bitwise complement of the declared length. The decoder rejects nonzero alignment bytes, truncated headers, lengths that do not fit `usize`, lengths beyond the remaining channel region, a missing or incorrect footer, and checked arithmetic overflow.

5. **The payload is eight byte aligned.** Every Jolt channel starts at an address divisible by eight. Before a byte frame, the encoder adds zero bytes until the channel cursor is divisible by eight. It then writes the eight byte length, the payload, and the eight byte footer. The payload therefore starts at an address divisible by eight.

6. **Guest decode is borrowed and constant work.** Decoding the frame must not allocate, copy, or scan the payload. Its work may depend on at most seven alignment bytes, the fixed length header, and the fixed footer. Application access to the payload remains ordinary guest execution.

7. **The declared region limit includes framing.** `max_input_size`, `max_trusted_advice_size`, and `max_untrusted_advice_size` include any alignment bytes, the eight byte length header, and the eight byte footer. No new hidden memory allowance is added.

8. **Every generated path uses one encoder.** Analyze, trace, trace to file, prove, generated verification, trusted advice commitment, Wasm support, and any preprocessing path that materializes arguments must call the same channel framing primitive. The procedural macro must not emit independent copies of the framing formula.

9. **Every guest channel uses one decoder.** Public input and both advice channels must call one bounded byte frame decoder. Channel specific code may choose a cursor, but it must not reimplement framing.

10. **Mixed codecs preserve argument order within each channel.** A channel may contain Postcard arguments and byte codec arguments. The host encoder and guest decoder must process them in the same declaration order for that channel.

11. **Native execution preserves the contract.** Calling the generated function natively must give the guest body an eight byte aligned slice for a byte codec argument. Native execution may use temporary aligned storage because it is a development and sanity check path. It must produce the same function result as RV64 execution.

12. **The codec does not create typed memory.** Jolt returns bytes. It must not cast application data into Rust structs, field elements, or generic plain data types. Applications remain responsible for endianness, canonical encodings, validity checks, and domain bounds.

13. **Alignment is an optimization, not a validity assumption.** Application libraries should use a safe parser that checks lengths and values before any typed read. If an application also supports non Jolt byte sources, that parser must retain a safe unaligned path.

14. **Public input binding does not weaken.** The existing public input transcript binding covers the complete byte frame. A different length, payload, or alignment byte is a different public input.

15. **Advice binding does not weaken.** Trusted advice commitment and untrusted advice witness construction cover the complete byte frame. The codec does not change who supplies or authenticates advice.

16. **Malformed frames cannot cause undefined behavior.** The generated guest may follow Jolt's existing panic behavior for an invalid function argument, but it must detect bounds and conversion errors before it constructs a slice or performs an aligned read.

17. **Program identity records the choice.** Adding or removing the byte codec attribute changes generated guest code and therefore changes the committed program identity. No separate global wire version is required.

18. **Output encoding does not change.** This spec does not alter Postcard output encoding or output normalization.

### Non-Goals

- This spec does not replace Postcard for ordinary typed arguments.
- This spec does not define a generic user supplied codec trait.
- This spec does not expose raw Rust object layouts as proof input.
- This spec does not add an fp32, fp64, fp128, or other field codec to Jolt.
- This spec does not guarantee alignment above eight bytes.
- This spec does not change the memory checking protocol or the public input transcript format outside the new opt in frame bytes.
- This spec does not remove the need for applications to validate untrusted or public data.
- This spec does not decide whether a large verifier setup belongs in public input, trusted advice, or the program image.
- This spec does not change runtime advice, which uses `AdviceTapeIO` rather than function argument serialization.
- This spec does not add host side zero copy ownership to `JoltDevice`. The host may still copy the payload once into the selected channel vector.
- This spec does not add an end of channel marker. It makes each declared byte frame canonical at its cursor. Generated APIs emit no unused suffix, but low level `JoltDevice` callers may still supply extra channel bytes. Existing public input and advice binding covers those bytes even when the guest does not consume them.

## Evaluation

### Acceptance Criteria

- [ ] A guest can declare a public byte parameter with `#[jolt::bytes] input: &[u8]`.
- [ ] A guest can apply `#[jolt::bytes]` to `TrustedAdvice<&[u8]>`, `UntrustedAdvice<&[u8]>`, and `PrivateInput<&[u8]>`.
- [ ] The macro rejects `#[jolt::bytes]` on any unsupported type with a compile error that names the accepted forms.
- [ ] The macro removes the consumed parameter attribute from emitted Rust so the compiler does not see an unknown attribute.
- [ ] Existing parameters without `#[jolt::bytes]` produce byte identical Postcard input, advice, and verifier encoding.
- [ ] A byte frame uses zero padding to the next eight byte channel offset, followed by `len: u64` in little endian order, followed by exactly `len` payload bytes, followed by `!len: u64` in little endian order.
- [ ] Empty payloads and payload lengths around every eight byte boundary decode correctly.
- [ ] For every starting cursor offset from zero through seven, the payload address observed by the RV64 guest is divisible by eight.
- [ ] Native execution gives the guest body an eight byte aligned slice even when the caller supplies a deliberately misaligned subslice.
- [ ] Decoding a byte frame performs no guest heap allocation and no input sized copy.
- [ ] Guest entry decode cycles do not grow with payload length when the guest only returns the payload length.
- [ ] A channel can mix Postcard and byte codec parameters before and after each other without changing declaration order.
- [ ] Public input, trusted advice, and untrusted advice all use the same framing implementation.
- [ ] The generated verifier reconstructs exactly the public bytes used by trace and prove.
- [ ] The generated trusted advice commitment function commits exactly the trusted advice bytes used by trace and prove.
- [ ] Standard and zero knowledge proofs accept for valid examples that cover each supported channel.
- [ ] Mutating a public byte frame causes verification against the original public input to fail.
- [ ] Mutating committed trusted advice bytes causes verification against the original trusted advice commitment to fail.
- [ ] Nonzero alignment bytes, truncated padding, truncated length, oversized length, truncated payload, missing footer, and incorrect footer fail without an out of bounds slice or load.
- [ ] A maximum size payload either fits with its complete frame or is rejected before execution. Framing may not silently exceed the configured channel limit.
- [ ] No output bytes, output normalization rule, or generated output API changes.
- [ ] The Jolt book explains when to use Postcard and when to use the byte codec.
- [ ] The implementation adds no application specific field or proof system type to `jolt-sdk`, `common`, or the procedural macro crate.

### Testing Strategy

#### Pure frame tests

Put the canonical host encoder and bounded guest decoder behind functions that unit tests can call directly. Test the following inputs:

- Empty payload.
- Payload lengths 1, 7, 8, 9, 15, 16, 17, 255, and 256.
- Starting channel offsets zero through seven.
- A payload whose length is `u64::MAX` in the encoded header.
- A payload length one byte larger than the available region.
- Nonzero values at every possible alignment byte position.
- A missing footer and an incorrect footer.
- A frame that ends exactly at the channel limit.
- A frame whose footer would exceed the channel limit.

The tests must assert the exact encoded bytes from the format in this spec. A production encoder followed by a production decoder is not an independent oracle.

#### Macro tests

Add compile tests for every accepted parameter form and for representative rejected forms. Inspect the expanded argument model through unit tests in `jolt-sdk/macros` rather than relying only on end to end compilation.

The macro tests must cover mixed argument declarations in each channel. One example should place a Postcard value before a byte argument and another should place a Postcard value after it.

#### Jolt evaluation invariant

Add `aligned_byte_frame` under `jolt-eval/src/invariant/` with `Test`, `Fuzz`, and `RedTeam` targets. Its input should contain a starting cursor residue, payload bytes, and a malformed frame mutation. The invariant should check the frame against an independent format oracle:

- The prefix position is the first offset at or after the cursor that is divisible by eight.
- Every skipped byte is zero.
- The next eight bytes equal the payload length in little endian order.
- The payload bytes are exact.
- The payload offset is divisible by eight.
- The eight bytes after the payload equal the bitwise complement of the declared length in little endian order.
- The decoder consumes exactly one frame and returns the expected remainder.
- Each malformed mutation returns an error before it can expose a borrowed payload.

The seed corpus must include all cursor residues, the boundary lengths listed above, a nonzero padding byte, a truncated header, an oversized declared length, a missing footer, and an incorrect footer.

#### End to end guests

Add one small guest package or extend an existing SDK fixture with functions that cover:

- Public aligned bytes.
- Trusted advice aligned bytes.
- Untrusted advice aligned bytes.
- Private input aligned bytes under the `zk` feature.
- Mixed Postcard and aligned byte arguments.

The guest should return a checksum that reads the first byte, last byte, length, and every complete `u64` word. This prevents the test from passing when the payload is not actually read.

Run the primary standard and zero knowledge tests required by `CLAUDE.md`:

```bash
cargo nextest run -p jolt-prover-legacy muldiv --cargo-quiet --features host
cargo nextest run -p jolt-prover-legacy muldiv --cargo-quiet --features host,zk
cargo nextest run -p jolt-prover --features prover-fixtures --cargo-quiet
cargo nextest run -p jolt-prover --features prover-fixtures,zk --cargo-quiet
```

Run focused SDK, macro, common, and `jolt-eval` tests before the full repository suite. Use `cargo nextest`, not `cargo test`.

#### Compatibility test

Before changing the macro, generate golden channel bytes for existing Postcard fixtures that include public input and both advice channels. After implementation, assert those exact bytes for every parameter without `#[jolt::bytes]`. Do not keep a copy of the old macro as the oracle.

### Performance

The byte codec targets guest startup cost and downstream word access. It does not claim to remove application parsing or validity checks.

The implementation must meet these requirements:

- Guest frame decode performs no allocation and no payload copy.
- Guest frame decode performs constant work with respect to payload length.
- The host performs at most the one channel vector copy that the existing SDK already performs. It must not build a second input sized staging vector merely to align the payload.
- Existing Postcard guest traces and host encoding benchmarks must not regress outside normal measurement noise.
- A word scanning RV64 guest must be able to compile to aligned `ld` instructions for complete `u64` words. Object code inspection is required during implementation, but it does not need to become a brittle CI text match.

Add a targeted benchmark that traces a guest which returns only the byte slice length. Measure payloads of 0 bytes, 4 KiB, 1 MiB, and 64 MiB. The cycle count between guest entry and the start of the function body must remain constant across these sizes. Report RV64IMAC cycles and virtual cycles separately.

Add a second benchmark that scans 1 MiB as little endian `u64` words. Compare the new byte codec against the existing Postcard `&[u8]` path. Report guest entry cycles, scan cycles, complete trace cycles, and peak host allocation. This comparison is diagnostic. Acceptance depends on the fixed alignment contract and absence of extra guest copies, not on a large speedup over Postcard's already borrowed slice decode.

No existing `jolt-eval` objective directly measures SDK argument decode. Add an `aligned_bytes_entry_cycles` objective only if the implementation can read deterministic emulator cycle counts without adding guest compilation to every objective run. Otherwise keep the targeted benchmark as an explicit SDK command and document it in the Jolt book.

## Design

### Current State

The procedural macro currently groups parameters by input channel. Every generated host path appends `postcard::to_stdvec` output to the selected channel. The generated RV64 entry point constructs a slice for the complete configured region and calls `postcard::take_from_bytes` for each parameter.

Jolt already rounds every channel size to eight bytes. As a result, `trusted_advice_start`, `untrusted_advice_start`, and `input_start` are all divisible by eight. Postcard then adds a variable size length prefix for a borrowed byte slice. The slice remains borrowed, but its payload does not retain the alignment of its channel start.

This distinction explains the scope of the feature. Postcard does not copy a borrowed byte slice during guest decode. The problem is that the SDK exposes only one representation policy and does not preserve a useful address property for large binary inputs.

### Public API

The first version adds one consumed parameter attribute:

```rust
#[jolt::provable]
fn verify(#[jolt::bytes] input: &[u8]) -> u32 {
    verify_encoded_object(input)
}
```

The same attribute composes with existing channel wrappers:

```rust
#[jolt::provable]
fn verify(
    #[jolt::bytes] public: &[u8],
    #[jolt::bytes] trusted: jolt::TrustedAdvice<&[u8]>,
    #[jolt::bytes] witness: jolt::UntrustedAdvice<&[u8]>,
) -> u32 {
    verify_all(&public, &trusted, &witness)
}
```

`PrivateInput<&[u8]>` is accepted because it is the existing alias for `UntrustedAdvice<&[u8]>`. The existing compile time check that requires the `zk` feature for private input remains in force.

The function body sees the same Rust types it declared. The attribute only changes generated transport code. It does not add a Jolt type to application logic.

The macro rejects these uses:

- `#[jolt::bytes] Vec<u8>` because it would require a guest allocation and would not be borrowed.
- `#[jolt::bytes] &[u32]` because Jolt does not define typed element endianness or validity.
- `#[jolt::bytes] SomeStruct` because this spec does not define user codecs.
- Any alignment argument because the first format has one fixed guarantee.

### Canonical Byte Frame

Each channel has an independent byte cursor. Channel start address and vector offset zero refer to the same location.

For a byte codec argument with payload `P`, the host encoder performs these steps:

1. Compute `padding = (8 - (cursor mod 8)) mod 8` with checked arithmetic.
2. Append exactly `padding` zero bytes.
3. Convert `P.len()` to `u64`. Reject a length that does not fit.
4. Append the eight byte little endian length.
5. Append the payload bytes.
6. Append the bitwise complement of the encoded length as an eight byte little endian footer.

The frame is therefore:

```text
+------------------+----------------------+-------------------+----------------------+
| 0 to 7 zero bytes| u64 length, little   | payload bytes     | !length, little      |
|                  | endian               |                   | endian               |
+------------------+----------------------+-------------------+----------------------+
                   ^                      ^
                   8 byte aligned         8 byte aligned
```

No trailing padding follows the footer. A later byte codec argument adds its own leading padding. A later Postcard argument starts immediately after the footer.

The footer is required because the generated guest currently sees the complete configured channel region, not the actual host vector length. `JoltDevice` returns zero for an address inside the configured region that lies beyond the vector. Without a nonzero footer, a truncated payload whose missing suffix consists of zero bytes could look complete to the guest. For every valid frame, the declared length is smaller than `u64::MAX`, so its bitwise complement is nonzero.

The guest decoder receives the current channel cursor and remaining configured bytes. It performs these steps:

1. Compute the same padding from the cursor address.
2. Check that the padding bytes exist and are all zero.
3. Check that eight length bytes remain.
4. Read the length as little endian `u64` without assuming the source header is aligned until the padding check has established it.
5. Convert the length to `usize` with a checked conversion.
6. Check that the complete payload and eight footer bytes remain.
7. Read the footer and compare it with the bitwise complement of the declared length.
8. Construct the borrowed payload slice.
9. Return the payload and the channel remainder after the footer.

The decoder never reads or validates bytes beyond the declared payload. The application parser owns those bytes.

### Macro Argument Model

Replace the three parallel argument lists with one ordered descriptor per declared parameter:

```rust
enum InputChannel {
    Public,
    TrustedAdvice,
    UntrustedAdvice,
}

enum InputCodec {
    Postcard,
    AlignedBytes,
}

struct InputArgument {
    name: Ident,
    ty: Box<Type>,
    channel: InputChannel,
    codec: InputCodec,
}
```

The exact private representation may differ, but it must represent channel and codec once. Helper iterators may select a channel without cloning policy into separate sources of truth.

One macro helper must emit host encoding for an `InputArgument`. Another must emit guest decoding. Analyze, trace, trace to file, prove, generated verification, trusted advice commitment, and Wasm generation must use those helpers.

This refactor is part of the feature because the current macro emits Postcard calls at many independent sites. Adding another branch independently at each site would make it easy for one generated API to disagree with another.

### SDK Framing Owner

Put the framing primitives in a small `jolt-sdk` input module that can compile in the required host and guest feature graphs. The host side owns append operations. The guest side owns bounded cursor decode. The procedural macro should select a codec and call these primitives. It should not contain byte offset arithmetic in generated tokens.

The input module should expose errors that distinguish invalid padding, truncated length, length conversion failure, truncated payload, missing footer, and incorrect footer. Generated guest startup may map those errors to the existing panic behavior. Tests and lower level users need the precise errors.

### Native Execution

The current native function path calls the original Rust body directly. That would let the caller choose an arbitrary slice address and would violate the new function contract.

For a function with a byte codec argument, the macro should emit a private body function and a public native wrapper. The wrapper copies each byte codec argument into temporary storage with at least eight byte alignment, constructs the same declared wrapper type when advice is used, and calls the private body. Ordinary Postcard parameters pass through unchanged.

This native copy is not part of guest execution or proving. It exists so native sanity checks exercise the same alignment contract. Tests must pass deliberately misaligned host subslices into the public wrapper.

### Guest Execution

The generated RV64 `main` already creates one cursor for each configured channel. It will decode each parameter according to its descriptor. For a byte codec parameter, it calls the shared frame decoder and receives a borrowed slice into the memory mapped channel.

The decoder must establish alignment before it constructs the function argument. Application code may still check the pointer and keep an unaligned fallback when it accepts bytes from sources other than this Jolt entry point.

### Proof and Transcript Binding

The public input frame lives in `JoltDevice.inputs`. Existing verifier validation and transcript absorption already bind that vector. No transcript function changes are required.

Trusted and untrusted advice frames live in their existing channel vectors. Existing advice witness and commitment logic therefore binds the frame. The generated trusted advice commitment function must call the same byte frame encoder as trace and prove.

Because padding is checked, length is fixed width, and the footer is checked, one semantic byte slice has one accepted frame at a given declaration position. A malformed frame cannot add ignored nonzero bytes before the payload or omit a zero suffix from the payload. This is frame canonicality, not channel exhaustion. The generated host APIs emit no bytes after the final declared argument. A low level caller that constructs `JoltDevice` directly can append unused bytes, but those bytes remain part of the existing public input or advice binding.

### Application Boundary

The SDK contract ends at an aligned byte slice. A cryptographic application should define a canonical byte format. For example, a field library may define each value as one or more fixed width little endian words. The application must check every residue against its modulus before constructing a field value.

Jolt must not offer `&[F]` or a generic plain data cast as part of this feature. Native Rust layout can change with compiler, target, and library representation. It can also admit invalid field values without a canonical check.

### Security and Failure Behavior

The frame header is controlled by the same party that controls its channel. Public input is supplied to the verifier. Untrusted advice is supplied by the prover. Trusted advice is authenticated through its existing commitment. The guest decoder must therefore treat every header as untrusted for memory safety.

The decoder must complete all length and bounds checks before `from_raw_parts` or any typed read. The proof system may attest to a guest panic for malformed data under low level APIs, but generated verification that expects normal completion must reject it under the existing panic flag rule.

The codec makes no secrecy or side channel claim. It does not change the existing advice threat model.

### Data Flow

```text
Rust argument
    |
    | generated host encoder selected by channel and codec
    v
canonical bytes in JoltDevice channel
    |
    | existing public input binding or advice commitment
    v
memory mapped RV64 channel with eight byte aligned start
    |
    | shared bounded byte frame decoder
    v
borrowed, eight byte aligned &[u8]
    |
    | application owned canonical parser
    v
validated application values
```

### Compatibility

The feature is opt in. A program without `#[jolt::bytes]` keeps its existing channel bytes, guest entry decoder, generated host signatures, and proof behavior.

Adding the attribute to an existing parameter changes the channel bytes and guest ELF. Existing preprocessing and proofs for that function are not compatible with the changed program. This is expected because the program identity commits to the generated entry point.

The new attribute should be rejected by older Jolt versions at compile time. No runtime negotiation or dual decoder is required.

### Alternatives Considered

#### Automatically specialize every borrowed byte slice

Rejected because it would silently change input bytes and program identity for existing guests. An explicit attribute makes migration reviewable.

#### Add alignment padding inside Postcard

Rejected because Postcard is a general serialization package and does not own Jolt memory addresses. An application would still need to reason about the outer prefix and mixed argument cursor.

#### Add a Jolt field input type

Rejected because field endianness, modulus checks, extension representation, and library layout belong to the field implementation. Jolt should provide bytes with a clear address contract.

#### Accept native Rust objects through a plain data trait

Rejected because proof inputs need a stable representation across host and guest targets. Plain data only proves that bit patterns are valid for a Rust type. It does not provide a protocol encoding or field canonicality.

#### Add a generic custom codec trait

Rejected for the first version. A user codec would expand the macro and safety surface before Jolt has one proven non Postcard use. A built in byte codec solves the known general problem and leaves semantic parsing in ordinary libraries.

#### Add a second bulk input memory region

Rejected because existing channels already provide the required proof semantics and eight byte aligned starts. A new region would change memory layout and proof protocol code without a need established by this feature.

#### Put the payload length in a register or virtual instruction

Rejected because it would add ISA and trace semantics for data that can be represented canonically in the existing committed channel bytes.

#### Expose actual channel lengths through new memory metadata

Rejected for the first version because it would change memory layout and proof input construction for every program. The fixed footer detects truncation inside the opt in frame without changing existing programs.

#### Let applications compute outer Postcard padding

Rejected because it exposes an SDK implementation detail and forces every large input format to track Jolt framing changes. This is the current workaround that the feature removes.

#### Require only the RV64 path to be aligned

Rejected because native execution is a documented sanity check path for guest functions. Giving the same function two address contracts would hide bugs until tracing.

## Documentation

Update `book/src/usage/guests_hosts/guests.md` with these points:

- Postcard remains the normal input codec.
- `#[jolt::bytes]` is for a large encoded object that the guest will parse itself.
- The payload is borrowed, has an eight byte aligned address, and includes framing in the configured channel limit.
- Alignment does not validate application data.
- Advice wrappers control provenance and can compose with the byte codec.
- Adding the attribute changes program identity and proof compatibility.

Add one complete public input example and one trusted advice example. The public example should parse fixed width little endian integers and check bounds rather than cast bytes into a Rust struct.

Update troubleshooting guidance for `max_input_size` so it includes byte frame overhead. Note that a maximum payload may need up to twenty three additional bytes, consisting of at most seven alignment bytes, the eight byte length, and the eight byte footer.

No protocol chapter changes are needed because the existing proof continues to bind the resulting channel vectors.

## Execution

### Slice 1: Argument model and format helpers

- Parse and remove `#[jolt::bytes]` from function parameters.
- Represent channel and codec in one argument descriptor.
- Add compile errors for unsupported attributed types.
- Add the canonical host encoder and bounded guest decoder.
- Add independent exact format tests and malformed frame tests.

### Slice 2: Generated host and guest paths

- Route analyze, trace, trace to file, prove, generated verification, trusted advice commitment, preprocessing helpers, and Wasm through one encoder emitter.
- Route the three guest channel cursors through one decoder emitter.
- Preserve exact Postcard generation for default arguments.
- Add mixed codec tests that would fail if any generated path uses a different frame.

### Slice 3: Native execution and end to end coverage

- Split attributed guest functions into a private body and public native wrapper.
- Add aligned native temporary storage.
- Add public, trusted, untrusted, private, and mixed argument guest fixtures.
- Run valid proof and tamper rejection tests in standard and zero knowledge modes.

### Slice 4: Evaluation and documentation

- Add the `aligned_byte_frame` `jolt-eval` invariant and fuzz target.
- Add entry decode and word scan benchmarks.
- Inspect RV64 object code for aligned word loads.
- Update the Jolt book and troubleshooting guide.
- Run formatting, both required Clippy feature graphs, focused tests, and the repository acceptance suites.

### Expected Files

The implementer should expect to touch these areas:

- `jolt-sdk/macros/src/lib.rs` for parameter parsing and generated path selection.
- `jolt-sdk/src/` for the canonical frame owner and native aligned storage.
- `jolt-sdk/tests/` or a focused example guest for generated API coverage.
- `common/src/jolt_device.rs` tests to keep the eight byte channel start premise explicit. The memory layout itself should not need to change.
- `jolt-eval/src/invariant/` and its fuzz target registration.
- `book/src/usage/guests_hosts/guests.md` and `book/src/usage/troubleshooting.md`.

The implementer should not change Jolt verifier transcript code, advice protocol semantics, field crates, or output normalization unless a spec review finds a missing requirement that requires such a change.

## References

- [`jolt-sdk/macros/src/lib.rs`](../jolt-sdk/macros/src/lib.rs), which currently emits Postcard encoding and decoding for every argument.
- [`common/src/jolt_device.rs`](../common/src/jolt_device.rs), which defines the three channel regions and their eight byte aligned memory layout.
- [`book/src/usage/guests_hosts/guests.md`](../book/src/usage/guests_hosts/guests.md), which defines current input provenance and Serde behavior.
- [`jolt-eval/README.md`](../jolt-eval/README.md), which defines invariant, fuzz, and objective workflows.
- [LayerZero Labs Akita PR #425](https://github.com/LayerZero-Labs/akita/pull/425), which demonstrates the current need to compensate for the outer Postcard frame before using aligned field loads.
- [Postcard](https://github.com/jamesmunns/postcard), the existing default Serde format.
