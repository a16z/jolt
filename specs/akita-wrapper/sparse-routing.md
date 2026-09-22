# Exact conditional tape routing

Source implementation of the independently reviewed conductor relation in
`stream-routing-proposal.md` / `stream-routing-review.md` (original workspace).
The private owner is `jolt-akita::r1cs::sparse_routing::ReadTape`; production
callers are the existing single-candidate and first-accepted-retry samplers.
No hash, field sampling, Fisher–Yates, norm predicate or rejection rule changes.

## Contract and soundness map

`ReadTape::constrain` creates the only read state and runs a private sampler
closure. It finalizes routing before releasing the result. Callers cannot create
an unfinalized read state; public sampler APIs never return before finalization.
Every retry candidate shares this same state. The scalar cursor starts at zero;
its materialized recurrence adds each Boolean activity exactly once. Read bytes
are eight allocated Boolean bits. Inactive bytes are constrained to zero.

Records `(activity, prefix_count, packed_byte)` pass through one fixed recursive
Beneš network. All three components share the same Boolean switch. Each switch
materializes both outputs: one selector Boolean row plus three product rows and
three sum rows. Therefore records are permuted exactly, and expressions remain
bounded rather than recursively expanding. The output active-prefix, index,
byte-to-tape and capacity equations are the reviewed relation verbatim.

BN254 Fr has characteristic greater than2^64 and255. Shape construction checks
that the padded power-of-two size and tape length fit u64, and that the switch
count fits usize. Thus all prefix counts are canonical integers below Fr, and
byte equality is injective. Empty reads use one inactive padded record; N1 has
no switch; N2 has one switch. No unknown assignment controls topology or loops.
Errors leave partial constraints, and callers discard the builder on error.
ONE must be fixed and all supplied handles must come from that builder.

The existing ending one-hot cursor APIs are retained because current sampler
consumers and adversarial tests use them. They are allocated only at candidate
boundaries, with Booleanity, sum-one and weighted-count equality to the shared
scalar cursor. They do not drive byte selection or reset progress.

## Total routing witness algorithm

For a permutation mapping input ports to output ports, pair input neighbors and
pair inputs whose destinations are neighboring output ports. The union of these
two perfect matchings consists of even alternating cycles (a doubled edge is
allowed). Two-color each connected component. Every input pair and output pair
has one member of each color. The first switch layer routes color0 upward and
color1 downward; each subnetwork permutation maps input-pair index to
output-pair index. Recursion halves N until N1/N2. The final switches route the
two colors to the requested even/odd destination ports. Thus every permutation
has a setting, and every executed switch network preserves all ports exactly.
Invalid non-permutations are rejected before indexing. Known reads choose the
stable active-prefix permutation; inactive records occupy the remaining ports.
Witness settings are hints only: the same equations constrain arbitrary settings.

The inactive-zero change is covered by the independent review's semantic proof:
inactive position trials contribute no selected bits; inactive sign reads form
a valid positive dummy mixed shell. Its norm acceptance is multiplied by zero,
so it cannot affect accepted output or advance the cursor. Every active rejected
position and rejected norm candidate still consumes the next tape bytes.

## Validation scope

Exhaustive host topology checks enumerate all permutations at N1/2/4/8; larger
prescribed rotations/bit-reversals check N16/128/1024. Separate small R1CS checks
cover padding, known/unknown layout, malformed settings, coherent wrong record
bytes/indices, tape/byte/activity/cursor mutations, every network auxiliary,
empty/singleton cases and capacity. Native candidate/retry parity and an actual
R3/K7 coordinate remain required before promotion. No full terminal-context
matrix, full wrapper proof, negligible capacity-overflow or ZK claim follows.

## Preregistered native coordinate and resource plan

The intentional `sparse_retry_r1cs_cost` tool accepts explicit R/K and optional
root/index inputs. With a known root it compares all dense coefficients to the
live native FoldDraw sampler, checks the entire R1CS witness, reports consumed
bytes and candidate selectors, and fingerprints A/B/C matrices. With unknown
root it constructs shape only and explicitly reports no witness/native check.
It is not a transcript authentication or proof benchmark.

The selected run is R3/K7, coordinate4, root
`f27f73fef09b695e8a5355ea576583a146c5edd9764a4ad5464b6279d03eae2b`.
Before execution, `/private/tmp/sparse-routing-20260922/native-expectations.json`
freezes the native observer's293-byte consumption, third-candidate selection,
and all ordered positions/coefficients. They are production-native expectations
independent of the constrained sampler implementation, not a constraint oracle
or independently regenerated hash vector. Final matrices cannot specialize on
these private assignments.

Approved scope: serial candidate/retry/FoldDraw regression tests and one known
coordinate construction,4GiB sampled process-group RSS,600s,12GiB disk-free
floor, one Cargo build job, no incremental compilation. No full terminal or proof
allocation is authorized. Before measurement, rough total geometry is estimated
at1.0–1.2M rows and6–8M nonzeros; this estimate is not a result. Initial compile
failures (array inference, checked-result/index lints, diagnostic u32 conversion)
are retained with repaired-source commands. Tests and measured results follow
in the final evidence packet rather than being inferred from successful builds.

## Producer validation and measured geometry

Frozen implementation commit: `80697e094` (full hash in the evidence manifest).
The subsequent documentation commit does not change code. Public native source
is f5f75335eae18241681fd24ca0a60fca8f0512af, inherited unchanged from the frozen
terminal-context checkpoint. Evidence: `/private/tmp/sparse-routing-20260922`.

- Scoped all-target Clippy passed. Both crates involved have no default feature
  set that adds another routing implementation; `jolt-akita/r1cs` is explicit.
- Final nextest selected15 tests across the library and candidate integration
  binary: all passed,32 filtered. This includes four routing tests, all existing
  retry tests, all candidate integration tests, and all FoldDraw tests. The added
  unknown empty-tape adversarial assignment satisfies every preceding row and
  fails exactly the final capacity row.
- Known/unknown matrix equality is tested for the routing network, candidate
  sampler, R2/K4 retry composition and FoldDraw. No second full R3/K7 unknown
  allocation was performed or inferred from the known run.
- Formatting and whitespace checks passed. No full terminal matrix or proof ran.

Final selected commands (under the pinned governor, with
`CARGO_INCREMENTAL=0`, `CARGO_BUILD_JOBS=1`, and the existing wrapper target):

```text
cargo clippy --offline --locked --profile test -p jolt-akita --features r1cs --all-targets -- -D warnings
cargo nextest run --offline --locked --cargo-profile test -p jolt-akita --features r1cs --lib --test r1cs_sparse_candidate -E 'test(sparse_routing) | test(sparse_retry) | test(fold_draw) | binary(r1cs_sparse_candidate)' --test-threads 1 --cargo-quiet
cargo build --offline --locked --profile test -p jolt-akita --features r1cs --example sparse_retry_r1cs_cost
<target>/debug/examples/sparse_retry_r1cs_cost 3 7 f27f73fef09b695e8a5355ea576583a146c5edd9764a4ad5464b6279d03eae2b 4
```

The single known-coordinate run produced1,125,080 rows,1,111,159 variables
including ONE, and5,997,218 nonzeros. Matrix SHA256:
`ecd8208b81feae0b7ed2816bbdafbdc006ea1fdb4418c48219241bd3adadfea1`.
All native dense coefficients matched and the full witness satisfied the R1CS.
Consumption was293 bytes; selectors were[0,0,1]. These match the preregistered
native expectations without being baked into constraints.

The governor recorded2.163310667 seconds for the complete diagnostic, including
native sampling, witness construction/checking, matrix counting and fingerprinting.
Sampled process-group peak RSS was844,333,056 bytes (two one-second samples);
child maximum RSS was844,398,592 bytes. Minimum free disk was14,284,865,536 bytes.
These are measured diagnostic costs, not standalone synthesis timings or proof
costs. Regression compilation plus execution took305.805s, with3,872,931,840
sampled peak bytes; the actual15-test run took3.083s. All final commands exited0
with no remaining process-group members.

Next boundary: independent review of this implementation, then the frozen
conditional terminal-context native-root/output and coherent mutation controls.
Using this measured coordinate size, all seven coordinates contribute about7.88M
rows; with canonical transcript hashing and the terminal assembler the full
fragment is still roughly21–23M rows. That is an estimate, not a completed matrix
measurement or authorization to allocate it. Prefix authentication, whole-proof
semantic verification and finite-capacity completeness remain unresolved.
