# Single D64 candidate relation: implementation contract

Native source is published Akita epoch6 `28fc72021`, specifically
`sampler/xof.rs::next_usize_mod`, `position_sample.rs`'s partial Fisher–Yates,
and `signed_sparse.rs::sample`. Blake stream framing is already constrained.
This packet starts from authenticated byte variables, not an unbound byte hint.

Public shape is `(count_pm1, count_pm2, K)`, with weight in 1..=64 and K>=1
position trials per nontrivial position. Tape length is exactly
`K * (weight - (weight == 64)) + weight`; the last position when n=1 reads no
byte. The production selective-L2 constructor imports native counts. Arbitrary
counts describe the byte sampler only and do not authenticate a protocol policy.

For each public position i, active starts at one. Each of K slots selects the
byte at a private one-hot cursor. Low ceil(log2(64-i)) bits form v; acceptance
is exactly `active * [v < 64-i]`. Every active trial advances the cursor once,
including rejects. After the first acceptance, later slots are inactive and do
not advance it. Completion within K is mandatory. The selected v determines
j=i+v; constraints perform the actual swap of P[i] and P[j] starting from the
identity permutation. Merely distinct positions are insufficient.

Only after all positions, weight fresh bytes are consumed for signs in order.
Coefficient i is (1-2*low_bit(byte)) times its public magnitude. Outputs retain
ordered positions/coefficient pairing and additionally scatter into 64 signed
coefficients. The private cursor and consumed-byte expression are returned.
All routing and activity derives from constrained Boolean bits; no unchecked
witness address or stop flag is accepted. Masked-off high bits remain input
bytes but do not affect the native draw.

ONE must be fixed externally; all handles use one builder. A fully known
assignment exhausting a per-position budget returns typed CapacityExceeded;
unknown assignment synthesis has the same rows and forces each completion.
No truncated candidate is accepted. K is outer circuit capacity, not a change
to native sampling, and has a completeness/privacy policy obligation. Native
position retries have no separate iteration cap: they continue until acceptance
or the Blake stream's 2^70-byte exhaustion. K is a stricter circuit profile.

Scope: one candidate at local tape offset zero. Connecting that tape to a
private variable offset in the authenticated global Blake stream, linking
candidate rounds/first norm acceptance, and applying the separately reviewed
norm predicate remain unimplemented. Restarting this API on stream prefix zero
for a rejected candidate would be incorrect. This is not native acceptance or
a completed wrapper.

## Constraint argument and source ownership

The initial cursor is the constant vector e0. Given a one-hot cursor and Boolean
active, each next selector is `(1-active)*current + active*previous`; the
`active*end=0` row prevents mass escaping the tape. Thus the selected byte bits
are Boolean linear selections of constrained input bits. The binary prefix
comparator returns exactly `[v<n]`. The recurrence
`active' = active - active*[v<n]` is Boolean and can never reactivate; forcing
its final value to zero enforces the first valid draw within capacity.
The accepted-bit sum therefore has exactly one active term. Its equality
selectors are one-hot over 0..n; swapping with those selectors preserves an
identity-initialized permutation. Position outputs are range-constrained bytes,
coefficients are bounded SignedVars, and dense outputs are equality-bound to
the ordered scatter. All identities are small exact Boolean/integer expressions
in BN254; no quotient or modular reduction is used for position sampling.

`jolt-akita::r1cs::D64CandidateProfile` owns the source-specific transition.
`ByteVar::bit_expressions` exposes existing Boolean-constrained bits, and
`R1csBuilder::evaluate` only derives witness hints from existing assignments;
it contributes no enforcement and does not select circuit shape.
The external contract is eventual Akita wrapper composition with the accepted
Blake counter-stream and operator-norm relations in these existing crates.

## Validation and measurement

`r1cs_sparse_candidate` contains four distinct checks: production native sampler
parity composed with the constrained Blake stream; a hand-checkable rejected
byte/ordered swap/sign case with input, position, coefficient, dense and cursor
mutations; equal matrices for unknown and different rejection-count assignments
plus mandatory capacity failure; and n=1 zero-consumption at weight64.
The native oracle is the live `sample_sparse_challenges` entry point with a fixed
root provider (it deliberately does not validate the root's transcript origin).
Root0, coordinate0, counts31+11 consume 47 position bytes and42 sign bytes;
this89-byte count was independently computed from the canonical hashlib Blake
stream and native masked-rejection rule. The differential test uses K=4 and
compares native output order, signs and dense coordinates. No norm filter is
applied by that native entry point.

Reproduce from this checkout:

```bash
CARGO_INCREMENTAL=0 cargo nextest run --offline --locked -p jolt-akita --no-default-features --features r1cs --test r1cs_sparse_candidate --cargo-quiet
cargo run --offline --locked -q -p jolt-akita --no-default-features --features r1cs --example sparse_candidate_r1cs_cost
```

The intentional diagnostic includes allocated input tape bits and candidate
outputs; it excludes Blake hashing, norm, root authentication and retry rounds.
Measured selective-L2 profile sizes:

| Trials per position | Tape bytes | Rows | Variables | Nonzeros |
| --- | --- | --- | --- | --- |
| 1 | 84 | 110356 | 109893 | 476180 |
| 2 | 126 | 200094 | 199589 | 915847 |
| 4 | 210 | 485410 | 484821 | 2313797 |

These are relation sizes, not complete-wrapper proving measurements or claims
that K=1/K=2 can encode the root-zero native candidate.
