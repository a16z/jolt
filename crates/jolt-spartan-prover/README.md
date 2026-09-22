# Clear sparse-R1CS Spartan

`jolt-spartan-prover::prove::<PCS>` and
`SpartanKey::verify::<PCS>` implement the clear two-sumcheck Spartan argument
with one ordinary `jolt-openings::CommitmentScheme` witness opening. The
verifier crate owns the checked key, proof format, errors, transcript sequence,
and shared claim calculations. The prover crate supplies sumcheck kernels.
HyperKZG and Dory are exercised as independent PCS implementations.

This is a standalone library contract for applications supplying sparse R1CS.
It is not yet an Akita verifier circuit or an EVM verifier. Verification and
statement absorption traverse the matrices; the baseline has no succinct
matrix argument, no structured matrix evaluator, and no ZK masking.

## Key and accepted statement

Construct `SpartanKey::new(matrices, public_input_count, policy_id)` from an
authenticated application relation. Columns are `[1, public inputs, private
witness]`. The constructor checks all row counts and column bounds, nonempty
constraint/private-witness sets, partition arithmetic, and padding overflow.
The field must have characteristic greater than three, because the inherited
round interpolation uses the points zero through three. Invalid input returns
`SpartanError` in release builds as well.

The key owns its matrices immutably and is not deserializable. Applications may
deserialize `ConstraintMatrices`, then invoke the checked constructor. Shape
validation is not authentication: applications must select the intended
matrices, public partition, transcript, and PCS setup independently of the
proof. `policy_id` is the authenticated identifier of that application policy.
The exact matrix encoding and dimensions are also absorbed in each proof;
the protocol does not trust a prover-supplied matrix digest.

Rows and private witness coordinates pad independently to powers of two, each
at least two. Additional rows/columns have zero matrix coefficients. The honest
prover fills witness padding with zero; the verifier need not enforce those
unused coordinates to establish the unpadded R1CS. Public values are never
included in the privately committed witness column.

## Protocol and source map

Source: ordinary Spartan R1CS reduction, [Setty, ePrint 2019/550](https://eprint.iacr.org/2019/550),
with explicit sparse matrix evaluation. Arithmetic kernels adapt
`jolt-wrapper/src/spartan.rs` at donor
`56343ddbea18b5021b6971b7c2c3f17d1a67726f`; the donor's committed-round stream,
column carries, Dory verifier circuit, and shared-domain packing are excluded.

1. `SpartanKey::begin` absorbs protocol version, policy, matrix shape/content,
   public inputs, and `C_W`, then draws `tau` and separates the outer stage.
2. `OuterRounds` proves
   `sum_x eq(tau,x) * (Az(x)Bz(x)-Cz(x)) = 0` at individual degree three.
   Both sides use the existing compressed clear sumcheck transcript. At `rx`,
   `check_outer` checks the product of the three returned evaluations.
3. `begin_inner` absorbs all three evaluations before sampling three independent
   matrix weights. It subtracts the directly computed constant/public-column
   contribution from their weighted sum, then separates the inner stage.
4. `ConstraintMatrices::project_column_range` computes the private-column
   linear form in one pass per matrix. `InnerRounds` proves its inner product
   with `W` at degree two. Private columns are reindexed from zero.
5. The verifier computes the linear-form evaluation from its matrices. It
   checks the final product, absorbs `W(ry)`, and verifies a single PCS opening.

The existing sumcheck engine owns coefficient compression, round labels,
degree checks, transcript challenges, and binding order. Shared key methods
own all protocol-level transcript operations and public-input subtraction.
`SpartanProof` contains only the two compressed clear sumchecks, commitment,
outer evaluations, witness evaluation, and PCS proof. There is no alternate
proof mode or unchecked degree supplied by the prover.

## Evidence and limits

Permanent tests use the hand-computed relation `y=x^3+5` with `x=3,y=32`,
including its unaligned row/witness counts, plus a one-row relation with no
public input. Both PCS backends accept; transcript states match after proving
and verification. Tests reject wrong public inputs, unsatisfied witnesses,
malformed key shapes, altered claims/commitments/openings, changed matrix/policy
encoding, excess degrees, and missing rounds. An independently calculated
projection `[11,9,16]` checks the new sparse projection helper.

The consumer security assumption is a fixed witness extraction before `tau`
and PCS openings consistent with it. For HyperKZG, see its README's distinction
between imported capacity, full published SRS degree, and extracted coefficient
projection. Neither round-trip tests nor these formulas establish the complete
extraction/Fiat–Shamir composition theorem. Actual transcript sampling and that
composition remain separate proof-review obligations.

Applications own bounded proof-container decoding and trailing-byte rejection.
The prover is not constant-time. Clear proof messages reveal evaluations and
are not a zero-knowledge protocol. No proving-speed, proof-size, or gas claim
is made by this implementation slice.

```sh
cargo nextest run -p jolt-spartan-prover -p jolt-spartan-verifier -p jolt-r1cs -p jolt-hyperkzg --cargo-quiet
cargo clippy -p jolt-spartan-prover -p jolt-spartan-verifier -p jolt-hyperkzg --all-targets -- -D warnings
cargo fmt -p jolt-spartan-prover -p jolt-spartan-verifier -p jolt-r1cs -p jolt-hyperkzg --check
```

## Opt-in BN254 transcript policy

The standalone caller policy is
`jolt_transcript::Bn254WideBlake2bTranscript::new(application_label)`.
Pass that same transcript through the complete proof, including the PCS opening;
Spartan's PCS-generic API remains unchanged. Both parties must select this policy
and authenticate the same relation/setup policy identifier. The fixed session
`bn254-blake2b-wide384-v1` binds the sampler and Blake2b-512 spongefish backend
before any Spartan tau or PCS challenge; the application label is a big-endian u64 byte length followed by a padded
32-byte word (maximum label length 32). Old sampler proofs are incompatible.

`challenge`, `challenge_scalar`, and inherited vector draws all reduce three
128-bit scalar draws as `(a * 2^128 + b) * 2^128 + c` modulo BN254 Fr.
Conditional on independent uniform underlying byte blocks, maximum output mass
is at most `1/p + 2^-384`. Poseidon and arbitrary user transcript implementations
are outside this policy. This opt-in changes no shared Jolt transcript default.
The standalone acceptance, malformed-proof, and replay tests use this policy
with HyperKZG; Spartan also exercises Dory. Independent integer reduction vectors
and scalar-byte decoding tests live in `jolt-transcript/src/wide.rs`.
Online extraction, complete-SRS assumptions, and a backend-specific Fiat–Shamir
composition theorem remain separate unproved deployment obligations.
