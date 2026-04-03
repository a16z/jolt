# Extended Prover and Verifier Protocol

## What

Extend the Jolt proof from a single proof π to a pair (π, π') where π' is an auxiliary SNARK that certifies the expensive Dory PCS verification. The extended verifier checks π's information-theoretic component (sumcheck — pure field arithmetic) directly, then verifies π' to confirm the cryptographic component (Dory pairings, MSMs, GT exponentiations) would have accepted.

Based on "Efficient Recursion for the Jolt zkVM."

## Why

The Jolt verifier has two components:
1. **Information-theoretic:** sumcheck round verification, claim formula evaluation — all native BN254 scalar field arithmetic, already cheap
2. **Cryptographic:** Dory opening proof verification — GT exponentiations, G1/G2 scalar multiplications, a multi-pairing — dominates cost

Wrapping the full verifier in an R1CS SNARK (the wrapper approach in 05) requires encoding pairing arithmetic as constraints, which is expensive (~billions of RV64 cycles for hash-based approaches). The extended P/V approach sidesteps this: it constructs sumcheck-based PIOPs for each EC operation over a curve 2-cycle (BN254/Grumpkin), keeping all arithmetic native-field.

Result: the auxiliary proof adds ~1.5–1.9s to proving regardless of trace length, and the extended verifier costs ~170–200M cycles (8–10x fewer than wrapping the full verifier).

## Core Design

**Curve 2-cycle:** BN254 and Grumpkin share the same field pair (Fr/Fq) but swap which is the scalar field vs. base field. The auxiliary proof π' operates over Fq (BN254's base field = Grumpkin's scalar field), so Dory's group arithmetic is native-field in π'.

**PCS for π':** Hyrax over Grumpkin. No pairings needed — the Hyrax verifier only does G1 MSMs over Grumpkin, which are cheap native operations. This avoids the recursive PCS verification problem entirely.

**Sumcheck PIOPs for EC operations:**
Each Dory verifier operation becomes a sumcheck PIOP:

| Operation | Technique | Rounds | Degree |
|-----------|-----------|--------|--------|
| GT exponentiation | Quotient identity zero-check + shift sumcheck (EqPlusOne) | ~11 | 8 |
| GT multiplication | Quotient identity zero-check | 4 | ~8 |
| G1/G2 scalar mul | Double-and-add recurrence, denominator-free affine Weierstrass | varies | ≤8 |
| G1/G2 addition | Complete affine group law (27 constraints for G1, 47 for G2) | varies | ≤8 |

All instances of each operation type are batched via a single zero-check over a constraint index.

**Three-stage auxiliary proof:**
1. **Stage 1:** Packed GT exponentiation zero-check (~11 rounds, degree 8)
2. **Stage 2:** All remaining constraints batched (claim reductions, shifts, EC scalar mul, EC add, wiring) — ~25 rounds, degree ≤8
3. **Stage 3:** Prefix-packing reduction to a single dense Hyrax opening claim

**Wiring:** The Dory verification computation is a fixed DAG of ~649 group operations (for σ=19). Inter-PIOP data flow uses sumcheck-based copy constraints — direct O(E) wiring checks, not committed permutation accumulators.

**Prefix packing:** All witness polynomials are packed into a single 21-variable multilinear polynomial. This minimizes Hyrax commitment cost (one commit over √2²¹ ≈ 1024 elements) and opening cost.

## How It Fits

The auxiliary proof maps naturally onto the existing architecture:

**Protocol IR:** Each EC-PIOP (GT exp, GT mul, G1 smul, G1 add, G2 smul, G2 add) can be expressed as Protocol IR vertices with their specific compositions. The wiring topology is deterministic from the Dory verification graph (fixed given σ).

**Module:** The compiler produces a Module for the auxiliary proof, just as it does for the main Jolt proof. The auxiliary Module has ~3 stages with batched sumcheck instances.

**Schedule-driven execution:** The same `execute()` runtime in `jolt-zkvm` runs the auxiliary proof. The only difference is the field type (Fq instead of Fr) and PCS (Hyrax/Grumpkin instead of Dory/BN254).

**Dual-field / dual-PCS:** This is the main architectural requirement. The prover pipeline must support:
- Inner proof: `prove::<CpuBackend, Fr, Blake2bTranscript, DoryPCS>(inner_module, ...)`
- Auxiliary proof: `prove::<CpuBackend, Fq, Blake2bTranscript, HyraxPCS>(aux_module, ...)`

The current generic structure (`prove` is generic over `F` and `PCS`) already supports this. The concrete types just differ between the two invocations.

**Transcript sharing:** π and π' share the same Fiat-Shamir transcript. The auxiliary proof extends the transcript from where the inner proof left off, making π' an extension (not a separate proof).

## Scope

1. **Grumpkin field + curve implementation** — new field type Fq, new curve Grumpkin in `jolt-crypto`
2. **Hyrax PCS** — new crate `jolt-hyrax` implementing `CommitmentScheme` for Hyrax over Grumpkin
3. **EC-PIOP Protocol IR definitions** — express GT exp, GT mul, G1/G2 scalar mul, G1/G2 add as Protocol IR
4. **Dory verification graph → wiring topology** — given σ, enumerate the ~649 operations and their data dependencies
5. **Prefix packing** — pack all auxiliary witness polynomials into a single multilinear polynomial
6. **Extended prover** — after `prove()` returns π, construct auxiliary witnesses and run `prove()` again for π'
7. **Extended verifier** — verify π's IT component, then verify π'

## Dependencies

- Compiler full protocol (01) — to express the EC-PIOPs in the Protocol IR
- Working jolt-zkvm pipeline — the auxiliary proof uses the same runtime

## Unblocks

- Efficient on-chain verification without a heavy wrapper
- Proof aggregation (the auxiliary proof structure is recursion-friendly)

## Open Questions

- The Dory verification graph is fixed for a given σ. Should the graph be hard-coded per σ value, or generated from a description of the Dory protocol?
- Prefix packing requires careful variable layout (21-variable polynomial with sub-polynomial regions for each PIOP). Should this be a compiler concern or done manually?
- The paper targets σ=19. How does the auxiliary proof scale with different σ values? Is there a sweet spot?
- Transcript sharing between π and π' requires the auxiliary prover to replay the inner transcript. Should the inner prover output transcript checkpoints?
