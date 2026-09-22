# Complete fixed-profile terminal relation assembler

Status: implemented and locally validated; not a complete Akita verifier or an authenticated
transcript prefix. Frozen input is the reviewed epoch-six census at
`/private/tmp/akita-epoch6-census-20260922/observed-replay`, source
`1e20d62234de5780463dbaee9758264abc6271c3`, public native e24. The assembler
extends the existing terminal/scalar owners; it must not bake private positions,
coefficients, point coordinates, z/e/t or quotient assignments into the key.

## API and invariant

A checked public profile consumes native `TerminalFoldParams` and canonical
setup A coefficients. It admits only D64, rank3, width256, seven live blocks,
one inner digit, seventeen point coordinates and the exact supported full-L2
cap546507225. The caller must authenticate the setup/profile and A constants.
The owner checks native dimension accessors and response layout, not a new
schedule formula. The public A matrix alone is specialized into coefficients.

`TerminalZ` owns the single16384-coordinate signed vector, exact ranges and
complete physical squared-L2 cap. Canonical e[448] and t[1344] handles are
supplied once; their unique centered representatives are constrained once and
shared across every A/consistency row. Scalar verification uses those same
canonical e handles. A field-owner byte constructor binds sixteen existing
Boolean bytes to the canonical q element; it does not prove their source is the
accepted proof segment. Golomb decoding and predecessor t equality remain
upstream obligations.

Seven dense64 challenge arrays remain private handles. The existing D64ShellVar
owner constrains each exact31+11 shell; the assembler binds signed range-two
handles to those same coefficients. No position-dependent host routing occurs.
Every row expands all64 positions, with public negacyclic wrap signs, including
zero coefficients. A later caller passes the actual accepted retry outputs into
these handles. Shell membership does not establish norm acceptance, sampling,
byte consumption, first acceptance or transcript authentication.

The point owner derives all Lagrange position/block/inner weights from seventeen
canonical input handles. Position weights are centered once, then shared by
all64 consistency rows. Scalar reduction uses the existing bounded modular q
arithmetic; no invalid whole-row scalar lift is substituted. Quotients are
private auxiliary assignments, constrained by the existing exact public bounds.
All handles belong to one builder and ONE is externally fixed. Unknown witnesses
must produce the identical matrices. Construction errors can leave partial rows.

## Source map and integer bounds

Native `terminal_direct::verify_terminal_ring_relations` maps to all192 A
coefficients and64 consistency coefficients. `verify_terminal_trace` maps to
the existing complete scalar opening relation. `prepare_opening_point` maps to
`TerminalPointVar::prepare`. Canonical bytes use the native field modulus owner.
The integer-row owner checks its no-wrap certificate from every declared term.
Existing quotient bounds use the enforced shell mass7*(31+2*11)=371 and full
z cap. The dense product representation uses the looser per-term absolute mass
7*64*2=896 for the release no-wrap check; the actual shell still justifies the
existing quotient ranges. No formula is evaluated using private host routing.

## Preregistered resource and validation plan

Static A expansion: 192*16384=3145728 terms, not that many constraints.
Dense challenge products: (192+64)*7*64=114688; consistency weight*z products:
64*256=16384. Reuse all field/centering allocations and point weights.
Estimated complete construction:1.8–2.2million rows,12–15million nonzeros,
less than2GiB for a single builder/matrix, conservative sampled process cap6GiB.
These are estimates, not measurements. One actual construction is authorized
only after coordinator release, with900s deadline and12GiB disk floor.

Required evidence: actual accepted all-row/scalar assignment satisfies; coherent
private e/t/target mutations preserve the matrices and fail their equations;
byte/centering/range mutations reject; malformed public geometry rejects before
large construction; witness-independent layout. Reuse the reviewed census and
native codecs; do not generate a new proof. No full-wrapper acceptance claim.


## Frozen validation packet (2026-09-22)

Tested implementation: `9e4f9c85a14daaa944ceaa81a8e65d4dd125ebf6`.
Evidence and exact command JSONs: `/private/tmp/terminal-relation-20260922/manifest.json`.
The final report commit changes only this document. The consumer resolves all
native crates to public `f5f75335eae18241681fd24ca0a60fca8f0512af`, with one
workspace field owner and no local native patches. Preserve the distinction from
the input's local28fc producer and accepted public-e24 replay; this packet does
not regenerate a native proof or relabel its provenance.

Actual geometry is 2,126,926 constraints, 2,096,074 variables and 13,060,079
matrix nonzeros. The 3,145,728 static A terms are counted separately. Positive,
coherent-e, coherent-t, claim and unknown modes have identical matrix SHA256
`c68f857f65935222606d236b7db19883e9913f315e525a664c05c3134e83663b`.
The positive assignment satisfies every row. Coherent e and t controls update
the field value and matching canonical sixteen bytes together, retain the
original quotient inputs and construct auxiliary witnesses successfully; actual
R1CS evaluation rejects at rows 1,859,668 and 1,760,548 respectively. The claim
control rejects at row 2,126,925. These are constraint failures, not host witness
construction errors. Unknown mode proves only matrix equality: its `None`
result is not a satisfying assignment, regardless of the common diagnostic
construction-success label.

The positive construction, check and matrix fingerprint took 4.25 seconds with
1,630,666,752 bytes sampled process-group peak RSS. This is not a proof timing.
All five runs stayed within the reviewed governor's 6GiB sampled RSS, 900-second
and 12GiB free-disk limits; sampling is not a hard memory cap. Exact per-run
results, source/lock/executable/input hashes and the preserved executable are in
the packet. Initial example compile failures and their corrected passing run
remain archived alongside successful fmt and clippy logs.

Validation used `CARGO_INCREMENTAL=0`, the shared wrapper target and:

```sh
cargo clippy --offline --locked --profile test -p jolt-akita --features r1cs --all-targets -- -D warnings
cargo build --offline --locked --profile test -p jolt-akita --features r1cs --example terminal_relation_cost
cargo nextest run --offline --locked --cargo-profile test -p jolt-r1cs -p jolt-akita --features jolt-akita/r1cs --lib --test-threads 1 --cargo-quiet
```

The complete owner suites passed 78/78 with none skipped, including the existing
integer row, centering, exact range, physical norm and terminal regressions.
The two new focused byte-linkage/shared-shell tests also passed independently.
No separate minimal-feature validation or outer proof was run in this packet.

The next integration boundary is a caller supplying authenticated terminal proof
byte handles and decoded z, actual accepted retry coefficient handles, shared
constrained point/claim handles and authenticated setup A/profile. Local shell,
canonical byte, point-weight, z range and complete physical norm constraints are
already present. Proof-source positions, Golomb decoding, predecessor t linkage,
challenge norm/first acceptance/root consumption, setup authentication and the
preceding transcript/folds remain outside this relation. ONE and same-builder
obligations remain explicit. This packet establishes no full-wrapper, EVM or
security claim.
