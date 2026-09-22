# Fixed-profile point preparation and scalar opening

This component constrains the degree-one K16/D64 terminal scalar algebra and
connects point-derived position weights to the existing bounded consistency-row
component. It is not a complete verifier. It contains no hash or sampler changes.

The reference artifact `/private/tmp/akita-scalar-census-20260922` is historical:
its Blake2b/SHAKE protocol is not the required Blake-only protocol. Only its
independently reviewed field algebra is used here. A new Blake-only accepted
fixture and review remain necessary. The branch starts at Blake-only local Jolt
head `f7b84c863dc62dd0315ebb94833cf737f86fdbf0`; algebraic ownership does not depend
on the pending native sampler migration.

## Source map and supported specialization

The conductor's `scalar-census-review.md` and `consistency-census-review.md` trace
Akita `252abb895046cc1d5b9955a26a2ad2318148ac26`. Census evidence head
`b4cb4fbf9`, code `758286f98b2f7ee85319ba9dae50cc4ad03327c1`, and terminal SHA256
`a6297fda73342e0f961b66620fc058fd547d2ac7f20bb13fffa2f4e038fe3ca6` identify the
historical reference. This implementation follows:

- `akita-types/src/proof/batch.rs::prepare_opening_point`, degree-one branch:
  split six inner coordinates from the outer coordinates, use Lagrange basis,
  256 positions, and seven live blocks.
- `ring_opening_point_from_field`: eight position coordinates followed by three
  block coordinates, low-bit-first Boolean indexing, retain seven of eight
  block weights without renormalizing them.
- `field_reduction.rs::recover_ring_subfield_inner_product` and
  `recover_psi_inner_product`: degree one and D=64 reduce to the ordinary
  64-term dot product, without a half factor, twist or other normalization.
- `akita-verifier/src/protocol/core/terminal_direct.rs::verify_terminal_trace`:
  seven block-scaled e rings, followed by that functional. The supported suffix
  has one row, global scale one and row coefficient one.

`jolt-akita::r1cs::scalar::TerminalPointVar` owns this fixed geometry. Unlike the
more general native point preparation, it requires exactly 17 canonical
coordinates; it does not implicitly pad a shorter point, support another basis,
extension degree, row scale, block count or profile. Those broader native cases
must not be silently routed through this component.

## Primitive ownership and constraints

`jolt-poly::r1cs::lagrange_weights_fp128_bn254` owns the reusable Boolean Lagrange
table construction. From initial constrained one, for each low-to-high coordinate
x and existing weight w, it emits right=w*x and left=w-right modulo q, then places
all left weights before all right weights. Induction gives table entry i equal
to `product_j(bit_j(i) ? x_j : 1-x_j)` in low-bit-first order. Prefix sharing
requires 2^n-1 products and subtractions, hence 63+255+7=325 for this profile.

The new `Fp128Var::subtract` lives beside existing add/multiply. It allocates a
canonical difference c and a Boolean borrow b and enforces `a+q*b=other+c`.
Canonical ranges bound the residual magnitude below 2q<r, so the BN254 equation
is an integer equation. Borrow and output assignments are generated honestly for
convenience but remain independently constrained. Unknown assignments emit the
same gates. This is the helper's first production use.

`TerminalPointVar::enforce_scalar_opening` takes the existing canonical e handles
in block-major [7][64] order and the canonical claim handle. For each j it uses
seven CRT products and six modular additions to compute `outer[j]`. It then
uses 64 more CRT products and 63 additions to compute the inner dot product and
binds that canonical result to the supplied claim. Thus scalar recovery contains
exactly 512 full-width q products and 447 additions, before accounting for point
preparation and canonical input allocation. There is no whole-row BN254 lift.
The scalar review exhibits a concrete residual equal to r for the range-only
shortcut; the existing CRT multiplication instead binds the residual modulo r
and modulo eight using linked low bits and canonical operands/quotient.

`enforce_consistency_row` converts all 256 derived position weights to centered
signed handles using existing `SignedVar::centered`, then calls the existing
`TerminalZ::enforce_consistency_row`. Supplied optional signed assignments are
not trusted weights: the centered handles are constrained to the canonical
point-derived values. The z vector still comes from the complete norm-enforcing
`TerminalZ` constructor. Calling this method repeatedly currently reallocates
centered weight handles; no amortized multirow cost is claimed.

## Boundary obligations

Point/e/claim handles must already have their canonical constraints from
`Fp128Var`, share this builder, and later be bound to the actual proof bytes,
Fiat–Shamir outputs and predecessor fold state. ONE must be fixed externally.
Handle index validation does not prove builder provenance when indices coincide.
Canonicality and all arithmetic are constrained; correspondence to an accepted
proof is not established merely by honest constructor assignments.

The consistency bridge does not constrain sparse challenge sampling, support,
signs, or routed e selection. Its caller must pass centered handles linked to
the same e values used by scalar recovery and constrain the sparse routing. The
diagnostic does this linkage for its public frozen routing, not as a dynamic
routing circuit. Setup/schedule identity, A rows, proof decoding, hash/sampler
constraints, earlier fold checks and all preceding Jolt acceptance remain outside
this component. It must not be described as a complete terminal or wrapper proof.

Circuit shape depends only on public dimensions and fixed supported profile,
not field assignments. Witness generation is not constant-time; an eventual
outer ZK SNARK may hide canonical values, CRT quotients, borrows, centered signs,
and integer norm witnesses. This component supplies no ZK proof or timings.

## Evidence and reproduction

Permanent tests cover independent subtraction vectors with corrupted borrows;
a two-coordinate Lagrange vector that distinguishes bit order; full-profile
Boolean selector weights; the omitted eighth block weight without normalization;
input, point, block weight and claim tampering; wrong shape and foreign indices;
and exact known/unknown matrix equality for the complete point/scalar circuit.
Existing CRT range and low-bit tamper tests remain in the same targeted suite.

The intentional `scalar_r1cs_cost` example checks the historical assignment and
reports two actual matrices: point preparation plus scalar opening and canonical
inputs; then that same circuit plus the complete z norm and one point-bound
consistency coefficient row. It consumes trusted local JSON, not untrusted proof
bytes, and does not authenticate artifact hashes. Counts are component sizes,
not complete-verifier estimates or proof-time/gas evidence. It generates no proof.

```
cargo nextest run -p jolt-r1cs -p jolt-poly -p jolt-akita --features jolt-akita/r1cs -j 1 --cargo-quiet
cargo nextest run -p jolt-r1cs -p jolt-poly -p jolt-akita --no-default-features --features jolt-akita/r1cs -j 1 --cargo-quiet
cargo clippy -p jolt-r1cs -p jolt-poly -p jolt-akita --features jolt-akita/r1cs --all-targets -- -D warnings
cargo clippy -p jolt-r1cs -p jolt-poly -p jolt-akita --no-default-features --features jolt-akita/r1cs --all-targets -- -D warnings
cargo run -q -p jolt-akita --features r1cs --example scalar_r1cs_cost -- /private/tmp/akita-scalar-census-20260922
```

Focused new test filters are `subtraction_tests` in jolt-r1cs, `lagrange_tests` in
jolt-poly, and `r1cs::scalar::tests` in jolt-akita, with the features above.
Local raw logs are `/private/tmp/scalar-r1cs-{nextest,minimal-nextest,clippy,minimal-clippy,cost}.log`.
The inactive sibling `crypto-r1cs/target` was reused for build artifacts. No
field-inline feature or workspace-wide host/ZK suite is part of this packet.

Measured matrices, including all allocations and bindings described above:

| Component | Rows | Variables (including ONE) | A/B/C nonzeros |
|---|---:|---:|---:|
| Canonical inputs, point preparation, scalar opening | 434,238 | 428,874 | 2,379,951 |
| Same, plus complete z norm and point-bound consistency row 0 | 1,151,051 | 1,128,199 | 5,528,736 |

Both completed historical assignments satisfied their full emitted matrices.
These measurements include no A row and only one of the 64 consistency rows.
The default targeted nextest suite passed 280 tests with the pre-existing
schedule-DP regeneration test ignored (run
`b1cb9f4a-cb02-428b-b011-fada822e0220`). Default all-target clippy, the diagnostic,
Rust formatting/diff checks and Taplo validation of the changed manifest passed.
Taplo ran outside the macOS sandbox because of its system-configuration access.
The coherent-claim regression additionally replaces the claim's entire canonical
assignment, preserving all its range constraints, and requires rejection at the
final scalar equality. This separates claim binding from mere malformed-bit
rejection.

Reduced-feature validation also passed 280 tests with the same existing ignored
test (run `34b14003-0446-420f-8ace-ba2d21a4f85c`), and its all-target clippy exited
zero. The optional thiserror dependency is activated only with Akita's r1cs
feature. Initial tool invocations encountered an example registered before its
file was written and a duplicate nextest thread option; both were corrected
before these successful checks. Neither was a circuit or assignment failure.
