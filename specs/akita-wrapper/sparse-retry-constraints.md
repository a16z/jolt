# Fixed-capacity D64 first norm acceptance

Source: published epoch6 Akita `e24f3b6f54bfcfb378a9ad65a0609e1bdd1ccdc1`, indexed
sampling in `sampler/mod.rs`, D64 position/sign sampling, and the reviewed
native epsilon1 operator-norm table. This extends the reviewed single-candidate
relation, retaining one owner for each sampler and norm formula.

Public shape is R candidate slots, K trials per position, one coordinate index,
and a fixed root32 -> contiguous Blake tape of R*(42*K+42) bytes. R is in
1..=4096 (native outer retry cap); practical R/K can reject otherwise valid
native executions. No negligible failure claim is made. A candidate exceeding
K fails; rejected position bytes are never skipped. Unused authenticated tape
suffixes and inactive candidate tails are allowed and remain private.

Each candidate starts from the previous candidate's private one-hot cursor.
Activity initially equals 1. Position reads and signs advance that shared cursor
only while the candidate is active. Inactive slots still synthesize the same
sampler and norm rows, but leave the cursor fixed and cannot select an output.
Their masked draw sums are zero, so the existing permutation/shell constraints
remain satisfiable without an arbitrary accepted dummy shell.

For every candidate, all 32 frequency upper bounds U are constrained using the
native epsilon1 formula and table. A Boolean a is exactly U<=T: constrain
U-T=(1-2a)*m+(1-a), with m a bounded nonnegative integer. Thus a=0 requires
U>=T+1; a=1 requires U<=T. Residuals remain below 2^113, far below BN254.
The AND A of all 32 a values is the exact native norm predicate, including for
previous rejected candidates (at least one actual inequality must fail).
Take=active*A; next_active=active-take. Final activity is constrained to zero.
These identities force selection of the first accepted candidate, exactly once.
Outputs are equality-bound to the take-weighted dense coefficients. No claimed
retry index, norm decision or rejected-candidate witness is trusted.

The public consumer binds the Blake tape to root32 and public coordinate. The
root itself still needs binding to the authenticated FoldDraw transcript; ONE
and same-builder provenance remain external obligations. This is one indexed
challenge, not complete FoldDraw domain/grinding or whole-proof acceptance.

Tests cover live native first acceptance including an actually rejected
candidate, coherent alternate-output/selection attacks, position-budget failure,
retry-capacity failure, and exact matrix equality across private retry outcomes
and unknown assignments.

## Ownership and finite-capacity contract

`D64RetryProfile::sample` is the actual indexed-stream consumer: it imports the
reviewed `AkitaSparseStreamVar`, hashes one fixed prefix, then calls the sampler
on one shared tape. `D64CandidateProfile::sample_at` is crate-internal and requires
an already constrained one-hot cursor and Boolean activity; only the initial e0
and inductively preserved outputs are supplied. The standalone candidate API
retains its original offset-zero contract. Inactive candidate signs read without
advancing; their ordered positions become identity, giving a valid mixed shell
that need not pass the norm test. Its acceptance cannot influence activity.

`D64ShellVar::visit_frequencies` owns the native accumulators/table validation;
`frequency_upper` owns the epsilon1 formula shared by existing acceptance-only
checks and the new exact classifier. Classification checks every frequency even
for rejected candidates, rather than using a claimed failing index. The private
native constant `MAX_OP_NORM_ATTEMPTS=4096` is not exported; the public profile
therefore explicitly checks that pinned-source ceiling. No native crate changes
or dependency changes are part of this packet.

Native mask retries have no per-position cap except eventual 2^70-byte stream
exhaustion. Circuit K and R are stricter public capacities. Each active candidate
must complete within K at every position; the circuit cannot abandon a long draw
and try another candidate. A failure leaves partial constraints and the builder
must be discarded. Finite capacity, the authenticated FoldDraw root/domain and
grinding transitions, policy selection, and all other verifier relations remain
whole-wrapper obligations. The API covers one coordinate starting at its native
stream offset zero; candidate2 and later start at constrained private offsets.

## Producer validation

Root `[13;32]`, coordinate0, has a rejected first candidate and accepted second
candidate in the live native FoldDraw path. R2/K4 produces exactly its accepted
dense coefficients and consumes 190 bytes. Root `[0;32]` accepts its first
candidate, consumes 89 bytes, and has identical matrices to root13 and unknown
assignments. The native root provider deliberately bypasses transcript replay;
this oracle validates indexed sampling and norm filtering, not FoldDraw framing.
The fixture seeds were selected using a temporary approximate-complex-norm search;
all committed assertions use the live exact native integer table and FoldDraw,
not that approximate search. No independently derived norm oracle is claimed.

A coherent attack replaces all output coefficients with the actual rejected
first shell and changes selection bits; it fails. A separate frequency test
forges both the comparison decision and its range auxiliaries in each direction;
the final equation rejects false acceptance and false rejection. Boundary U=T
accepts and U=T+1 rejects. Existing native epsilon1 boundary and later-frequency
regressions also pass. R1/root13 and R2/root5 exceed retry capacity; K3/root13
exceeds position capacity. All emit an unsatisfied completion constraint before
returning their typed error.

Both minimal/default focused suites passed 11 tests (27 unrelated lib tests
filtered out); scoped all-target clippy passes in both feature configurations,
and feature-off lib clippy passes. Reproduce from this checkout:

```bash
CARGO_INCREMENTAL=0 cargo nextest run --offline --locked -p jolt-akita --no-default-features --features r1cs --lib --test r1cs_sparse_candidate -E 'test(r1cs::sparse_retry) | test(r1cs::operator_norm) | binary(r1cs_sparse_candidate)' --test-threads 1 --cargo-quiet
CARGO_INCREMENTAL=0 cargo clippy --offline --locked -p jolt-akita --all-targets --no-default-features --features r1cs -- -D warnings
CARGO_INCREMENTAL=0 cargo run --offline --locked -q -p jolt-akita --no-default-features --features r1cs --example sparse_retry_r1cs_cost
```

The R2/K4 diagnostic measures 2,268,605 rows, 2,263,709 variables and 11,189,260
nonzeros. It includes 32 allocated root bytes, 420 authenticated tape bytes,
private routing, two candidate slots, all norm decisions and selected outputs.
It excludes authenticating the FoldDraw root and all other verifier constraints;
it is a relation size, not a SNARK proving measurement. Larger profiles, practical
failure probability and full Jolt/native proof acceptance were not measured here.
