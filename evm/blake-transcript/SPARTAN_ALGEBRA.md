# Incomplete outer/public/inner algebra

This extends the reviewed authenticated input boundary without changing its wire
contract or claiming complete proof acceptance. `checkIncompleteAlgebra` returns
an `IncompleteCheckpoint`; it is not a verification API. In particular, a changed
inner polynomial can still pass this stage: its terminal relation depends on the
unimplemented sparse matrix and witness checks.

## Exact production schedule

| Native owner | Solidity check or retained obligation |
| --- | --- |
| `ComputationKey::begin` | Existing decoder initializes tau and outer marker |
| `CompressedSumcheckProof::verify` | `sumcheck`: width-tagged coefficient absorption, omitted-linear reconstruction, Horner evaluation |
| `check_outer_relation` | eq(tau,rx)(a*b-c) equals outer final claim |
| `ComputationKey::outer_weights` | counted3 evaluations, then3 wide384 challenges |
| `public_opening_query` | public-column claims before selector, MSB-first eq weights, selector concatenated with rx |
| `HyperKZGScheme::verify_opening` | exact authenticated setup/statement/compression bytes; nonzero challenge and every binary folding identity |
| KZG `absorb_evaluations`, `absorb_witnesses` | exact3 row-major evaluation vectors, polynomial batching challenge,3 compressed witnesses, opening batching challenge |
| KZG pairing equation | **PendingPublicKzg**, not checked |
| `verify_public` contribution | weighted matrix-major public columns times [1,inputs] |
| `inner_claim` and compressed inner verify | exact labeled initial claim and quadratic rounds |
| `verify_sparse`, witness product and opening | **Not executed**, required before proof acceptance |

The pending opening records its proof byte offset, point, evaluation and all three
PCS challenges. The authenticated key/setup and parsed proof remain in the same
execution context. This is a continuation boundary, not an attestation that the
public claims are authentic. No external caller may supply or mark an obligation
complete. The final verifier must consume the pairing and all remaining algebra.

## Resolved details

Compressed widths are1..degree with zero fixed-slot padding, as in accepted wire
v1. For incoming sum s, reconstruct c1=s-2c0-sum(c2..cd). Absorb only the stored
width coefficients with `sumcheck_poly` counted framing, then draw the challenge.
A shorter polynomial does not silently absorb its padded zero slots. This is not
an extra algebraic restriction on the native producer.

Eq indexing is first-coordinate-most-significant. Public slab padding contributes
zero; no padded claims are transmitted. Fr wire bytes are LE while transcript
field elements and integer words are BE32. Group transcript messages retain the
original checked compressed bytes. Public HyperKZG consumes the setup ID, imported
power capacity AND global public degree, G1/G2/betaG2, commitment, counted point,
evaluation, fold commitments, and all downstream opening messages before inner SC.
The prior externally validated G2/setup trust boundary is unchanged.

Outer/inner supported key shapes have at least two padded entries, so their
round counts are never zero. The helper's empty-loop identity preserves its input
claim, but no production outer/inner zero-round capability is asserted. Zero-round
GKR networks belong to the next sparse slice and must be tested there.

## Evidence and limits

Native tiny fixtures execute production full verification first, then production
outer sumcheck, public query, real public PCS verification, and inner sumcheck to
emit challenges/claims and post-public/post-inner transcript states. Toy, empty
N=2, and a coherent empty relation with different nonzero public inputs cover
ordinary/short rounds and identity commitments. No norm proof or large proving
run is needed. EVM mutations target outer coefficients/terminal claims, public
claims and fold evaluations; separate controls document still-unchecked inner and
private relations. Tests compare native intermediate values, not a second JS
sumcheck implementation.

No pairing, private GKR, final witness product or remaining PCS is claimed.
No deployment, full-verifier gas, operational extraction/ROM or ZK claim follows
from these checks. A valid-looking partial checkpoint is not a valid proof.

## Composition and validation

The complete verifier distributes the remaining checks across fixed, code-hash-authenticated modules; see [SPARTAN_VERIFIER.md](SPARTAN_VERIFIER.md). Every deployed runtime is checked against EIP-170. A partial algebra checkpoint is never accepted as a complete proof or supplied externally to that verifier.

`npm run test:spartan-algebra` exercises native checkpoints, targeted algebra reverts, and explicit incomplete-obligation controls. It compares native intermediate values, not another JavaScript implementation of the formulas. The production native fixture generator fully verifies its proofs before recording their intermediate values.

Current complete-verifier outcomes, frozen input hashes and gas scope are recorded in [validation/README.md](validation/README.md). Historical partial-verifier runtime/gas observations are not full-verifier measurements.
