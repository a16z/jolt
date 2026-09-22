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
