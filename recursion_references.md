# Recursion References

Checked: 2026-05-17

## Local Paper

Local directory:

```text
/Users/markos/recursion-paper
```

Important files:

- `/Users/markos/recursion-paper/paper.pdf`
- `/Users/markos/recursion-paper/recursion.tex`
- `/Users/markos/recursion-paper/dory.tex`
- `/Users/markos/recursion-paper/techniques.tex`
- `/Users/markos/recursion-paper/protocol.tex`

Paper title:

```text
Efficient Recursion for the Jolt zkVM
```

High-level paper state:

- The paper describes a Dory-assist recursion protocol for Jolt.
- The expensive Dory verifier work is moved into an auxiliary proof over BN254 base field arithmetic.
- The auxiliary proof uses Hyrax over Grumpkin, taking advantage of the BN254/Grumpkin cycle.
- The extended verifier mostly avoids GT exponentiations and pairings inside the zkVM.
- The paper reports extended verifier costs around `171M-198M` RV64 cycles, versus roughly `1.4B-1.9B` cycles for the standard verifier.
- The paper caveat is that, in the described implementation, the final pairing check was not yet fully proved inside the recursion SNARK.

## Main Branch To Inspect

The most relevant branch I found is in Quang's fork:

```text
https://github.com/quangvdao/jolt
```

Branch:

```text
quang/recursion-temp
```

Tip commit:

```text
87fcafe44602562d5dce593acd61afa66d6bc4c4
feat(recursion): snapshot MML alignment + GT wiring debug
committer date: 2026-02-03
```

GitHub link:

```text
https://github.com/quangvdao/jolt/tree/quang/recursion-temp
```

Useful local checkout command that avoids touching the current worktree:

```bash
git fetch https://github.com/quangvdao/jolt.git \
  refs/heads/quang/recursion-temp:refs/remotes/quangvdao/recursion-temp
git worktree add ../jolt-quang-recursion-temp refs/remotes/quangvdao/recursion-temp
```

## Related Branches

Branches checked in `quangvdao/jolt`:

| Branch | Tip commit | Date | Notes |
| --- | --- | --- | --- |
| `quang/recursion-temp` | `87fcafe4` | 2026-02-03 | Newest recursion branch found; includes Miller loop and pairing-boundary work. |
| `quang/voj-recursion` | `812f7418` | 2026-01-23 | Older than `recursion-temp`; appears to be an ancestor of it. |
| `quang/feat/recursion-2` | `2c49a0fa` | 2026-01-07 | Earlier Dory-assist recursion branch. |
| `prover-backend-final-final` | `882ebd99` | 2026-02-01 | Related performance/recursion work, but `recursion-temp` is newer. |
| `prover-backend-final` | `56c2e5a8` | 2026-01-30 | Related backend branch. |

Important relation:

```text
quang/voj-recursion is an ancestor of quang/recursion-temp.
```

So `quang/recursion-temp` contains the VOJ recursion work plus later commits.

## Dory Dependency

The recursion branch depends on LayerZero's Dory recursion fork:

```text
https://github.com/LayerZero-Research/dory
```

Branch:

```text
lz-recursion
```

Tip seen:

```text
22e2b14974563befab94eb81a43b892a3253d27e
```

The `quang/recursion-temp` workspace points Dory at:

```toml
dory = { package = "dory-pcs", git = "https://github.com/LayerZero-Research/dory", branch = "lz-recursion", default-features = false }
```

## Implementation Map In `quang/recursion-temp`

Core recursion API:

```text
jolt-core/src/zkvm/recursion/api.rs
```

Important exports:

- `prove_recursion(...) -> RecursionArtifact`
- `verify_recursion(...) -> Result<()>`
- `RecursionArtifact`

Main recursion protocol implementation:

```text
jolt-core/src/zkvm/recursion/prover.rs
jolt-core/src/zkvm/recursion/verifier.rs
jolt-core/src/zkvm/recursion/mod.rs
```

Operation families:

```text
jolt-core/src/zkvm/recursion/g1/
jolt-core/src/zkvm/recursion/g2/
jolt-core/src/zkvm/recursion/gt/
jolt-core/src/zkvm/recursion/pairing/
```

Miller loop / pairing-specific files:

```text
jolt-core/src/zkvm/recursion/pairing/multi_miller_loop.rs
jolt-core/src/zkvm/recursion/pairing/shift.rs
jolt-core/src/poly/commitment/dory/witness/multi_miller_loop.rs
```

Dory recursion glue:

```text
jolt-core/src/poly/commitment/dory/recursion.rs
jolt-core/src/poly/commitment/dory/instance_plan.rs
jolt-core/src/poly/commitment/dory/witness/
```

Hyrax PCS used by recursion:

```text
jolt-core/src/poly/commitment/hyrax.rs
```

Grumpkin inline support:

```text
jolt-inlines/grumpkin/
jolt-inlines/msm/
```

Recursion example harness:

```text
examples/recursion/
examples/recursion/guest/
```

## Protocol State

The `quang/recursion-temp` branch looks like the most complete implementation found so far.

It includes:

- Base Jolt verifier stages 1-7 replay before recursion verification.
- Stage 8 Dory recursion preparation.
- A standalone `RecursionArtifact`.
- Hyrax-over-Grumpkin commitment/opening for the recursion witness.
- G1 add/scalar-mul constraints.
- G2 add/scalar-mul constraints.
- GT exponentiation/multiplication constraints.
- AST-derived wiring constraints from the Dory verifier computation graph.
- Prefix packing to reduce many virtual openings into one dense Hyrax opening.
- Multi-Miller-loop witness generation and sumcheck constraints.
- Guest-side `verify_recursion` support in the recursion example.

The key status change versus the paper is:

- The paper says the pairing check was not yet proved.
- `quang/recursion-temp` adds Miller loop proving and wires a final-exponentiation pairing boundary.
- The final exponentiation remains outside the recursion SNARK: `verify_recursion` checks that `final_exponentiation(miller_rhs) == rhs`.

So the branch is closer to a "full recursion SNARK" than the paper snapshot, but it is not literally proving every pairing-related operation inside the recursion SNARK. It proves the Miller loop and binds the boundary; final exponentiation is still an external verifier check.

## Current Checkout State

Current repo:

```text
/Users/markos/jolt
```

Current checkout does not contain the full `quang/recursion-temp` implementation.

Current local branch had:

- `origin/feat/recursion-2`
- `origin/feat/snark-composition`
- a stale/local remote-tracking ref `quang/pr/19`

Current `examples/recursion` in this checkout is mostly the naive "run verifier in guest" recursion harness, not the full Dory-assist/Miller-loop recursion protocol from `quang/recursion-temp`.

## Wrapping / Groth16 Status In Current Checkout

The active Groth16 wrapping path in the current checkout is the `transpiler` crate:

```text
transpiler/
```

Status:

- Symbolically verifies Jolt verifier stages 1-7.
- Emits gnark code and witness JSON.
- Uses Poseidon for a circuit-friendly transcript.
- Has Go-side Groth16 tests.
- Does not implement Stage 8 PCS/Dory verification in-circuit.
- Uses an `AstCommitmentScheme` stub for the PCS boundary.

This is separate from the recursion SNARK branch above. The old historical `crates/jolt-wrapper` prototype is not present in the current workspace.

## Groth16 Pipeline Infra In Current Jolt

The direct Groth16 implementation is not in `jolt-core`. The split is:

- `jolt-core` provides verifier code that can be run symbolically.
- `transpiler` records that symbolic verifier run as an arithmetic AST.
- `transpiler/go` compiles the generated gnark circuit and runs Groth16.

Current pipeline:

```text
Jolt proof + IO + verifier preprocessing
  -> symbolize proof fields as MleAst witness variables
  -> run TranspilableVerifier over stages 1-7
  -> record arithmetic and equality checks as AST constraints
  -> emit gnark circuit + witness JSON
  -> gnark frontend.Compile -> groth16.Setup -> groth16.Prove -> groth16.Verify
```

Important files:

```text
jolt-core/src/zkvm/transpilable_verifier.rs
jolt-core/src/transcripts/poseidon.rs
transpiler/src/main.rs
transpiler/src/symbolic_proof.rs
transpiler/src/symbolize.rs
transpiler/src/symbolic_traits/opening_accumulator.rs
transpiler/src/symbolic_traits/ast_commitment_scheme.rs
transpiler/src/gnark_codegen.rs
transpiler/go/stages_circuit_test.go
transpiler/go/e2e_test.go
```

The main `jolt-core` piece is `TranspilableVerifier`. It mirrors the normal verifier path, but is designed to work with a symbolic opening accumulator.

It covers:

- Fiat-Shamir preamble and transcript appends.
- Stages 1-7 sumcheck verification.
- Spartan, RAM, register, instruction lookup, bytecode, booleanity, hamming/advice claim-reduction checks.

It does not cover:

- Stage 8 PCS/Dory verification.
- ZK/BlindFold proofs.
- Dory pairing logic in-circuit.

The PCS boundary is stubbed through `AstCommitmentScheme` because stages 1-7 do not require actual PCS verification. This is why the current Groth16 path is best described as "wrapping the sumcheck verifier portion of Jolt", not a complete recursive verifier.

Poseidon is the circuit-friendly transcript path. Proof generation and transpilation must use matching Poseidon transcript behavior, otherwise the Fiat-Shamir challenges diverge.

## Modular Crates Compared To Groth16 Transpiler

The current Groth16 transpiler is a wrapper around today's `jolt-core` verifier behavior:

```text
current jolt-core verifier
  -> symbolic execution with MleAst
  -> low-level arithmetic AST
  -> gnark/Groth16
```

The modular crates are a replacement architecture for the verifier/protocol model:

```text
current jolt-core proof artifact
  -> compatibility decode
  -> typed verifier model
  -> generic field/transcript/sumcheck/opening/PCS crates
  -> explicit stage dataflow and claim formulas
  -> future verifier, circuit, or generated backend target
```

Relevant modular crates:

```text
crates/jolt-field
crates/jolt-transcript
crates/jolt-poly
crates/jolt-sumcheck
crates/jolt-openings
crates/jolt-dory
crates/jolt-hyperkzg
crates/jolt-r1cs
crates/jolt-claims
crates/jolt-blindfold
crates/jolt-verifier
```

Practical difference:

- Groth16/transpiler records what the current verifier computed.
- Modular crates aim to represent what the verifier is checking.

Current modular state:

- `jolt-claims` has semantic Jolt claim formulas and tracking for mirrored formulas from `jolt-core`.
- `jolt-r1cs` has a small R1CS builder and lowering from claim expressions.
- `jolt-sumcheck` has generic verifier logic and an R1CS lowering module for sumcheck round checks.
- `jolt-blindfold` composes claims and sumcheck-R1CS pieces for BlindFold-style verifier equations.
- `jolt-verifier` is still early scaffolding, not a complete standalone verifier.

## How `jolt-claims` Could Impact Groth16

`jolt-claims` does not change the current Groth16 pipeline automatically. Today the Groth16 path records `TranspilableVerifier` arithmetic with `MleAst`.

The long-term alternative is:

```text
typed stage proof + transcript challenges + openings
  -> jolt-claims semantic expressions
  -> jolt-r1cs / backend lowering
  -> Groth16-compatible R1CS
```

That would give the wrapper a better IR than a raw arithmetic trace:

- Claim formulas are semantic: RAM read-write input claim, Spartan output claim, bytecode claim reduction, etc.
- Required openings/challenges/public values are explicit.
- The same formulas can feed native verification, BlindFold constraints, and circuit generation.
- The opening plan can become typed dataflow instead of accumulator side effects.
- A Bolt/generated verifier target can emit claim expressions directly rather than replaying `jolt-core` control flow.

What `jolt-claims` does not solve by itself:

- It does not run Groth16.
- It does not generate gnark code today.
- It does not constrain Poseidon transcript construction.
- It does not solve Stage 8 PCS/Dory verification.
- It does not replace generic sumcheck round verification.

So the likely future direction is not "make the current transpiler use `jolt-claims` directly" as a small patch. It is a retargeting from:

```text
execute verifier Rust and record arithmetic
```

to:

```text
lower explicit verifier formulas and protocol checks to R1CS
```

## `jolt-r1cs` And Groth16

Groth16 proves R1CS, so the modular R1CS pieces are a plausible route to a future Groth16 wrapper.

Current lowering path:

```text
jolt-claims Expr
  -> jolt-r1cs::lowering::{lower_claim_expr, assert_claim_expr_eq}
  -> R1csBuilder
  -> ConstraintMatrices + witness Vec<F>
  -> future Groth16 backend adapter
```

What `crates/jolt-r1cs/src/lowering.rs` already does:

- `Opening(id)` becomes an R1CS variable.
- `Challenge(id)` is read from `ClaimSources` as a field value.
- `Public(id)` is read from `ClaimSources` as a field value.
- Products allocate intermediate variables and constraints.
- `builder.witness()` gives the assignment.
- `builder.into_matrices()` gives sparse matrices.

The important limitation: challenges and publics are currently field constants from `ClaimSources`, not variables constrained by a transcript circuit. That is enough for formula-lowering tests and BlindFold-style baked coefficients, but not enough for a standalone Groth16 verifier unless those values are public inputs or are tied to in-circuit Poseidon outputs.

## `jolt-sumcheck` R1CS Support

`crates/jolt-sumcheck/src/r1cs.rs` is the generic piece that `jolt-claims` does not cover.

It allocates and constrains the verifier equations for a sumcheck proof:

```text
s_i(0) + s_i(1) = claim_in
s_i(r_i)        = claim_out
```

API shape:

```rust
let layout = allocate_sumcheck_r1cs_layout(&mut builder, shape, &rounds)?;
append_sumcheck_r1cs_constraints(&mut builder, shape, &rounds, &layout)?;
```

The intended composition is:

```text
jolt-claims input expression
  -> assert equals sumcheck input_claim variable

jolt-sumcheck R1CS
  -> enforce round-by-round sumcheck transitions

jolt-claims output expression
  -> assert equals sumcheck output_claim variable
```

This helps a lot for a future Groth16 wrapper, but it still assumes the challenges are already available as field values. A full wrapper must additionally prove or publicly bind the transcript construction that produced those challenges.

## BlindFold R1CS Vs Full Groth16 Wrapper

The main difference is challenge binding.

BlindFold-style R1CS:

```text
given transcript-derived challenges and committed sumcheck messages,
prove the verifier equations hold
```

The native verifier/protocol code replays Fiat-Shamir outside the R1CS and bakes those challenge values into the verifier-equation system.

Full Groth16 wrapper:

```text
given proof data, commitments, IO, and preprocessing digest,
prove the transcript absorbs exactly those values,
prove Poseidon derives the challenges,
prove the verifier equations use those same challenges
```

Extra requirements for a full wrapper:

- Proof-data absorption inside the circuit, or public challenge binding outside it.
- Public/private input layout for Groth16.
- Sumcheck coefficients must be the same values both hashed into the transcript and checked by the sumcheck equations.
- Commitment/field/byte encoding must match the native verifier.
- IO and preprocessing/config values must be bound into the transcript.
- Stage 8 PCS/Dory verification still needs a plan.

So the short version:

- BlindFold R1CS proves verifier equations with baked challenges.
- Full Groth16 wrapper must also prove where the challenges came from.

## Jolt R1CS Matrix Evaluation Inside A Wrapper

There are two different R1CS layers:

1. Jolt guest R1CS: the uniform Spartan/R1CS relation proving the guest trace is valid.
2. Groth16 wrapper R1CS: the circuit proving that the Jolt verifier ran correctly.

The wrapper may need to reproduce the Jolt verifier's matrix-evaluation logic, especially around Spartan outer. The native verifier computes values like the expected Spartan claim from:

- static public Jolt R1CS tables,
- transcript challenges,
- openings of R1CS input polynomials.

For a self-contained Groth16 wrapper, this computation should be constrained in-circuit.

It is also possible to split it:

```text
inside Groth16:
  derive challenges from transcript
  assert public challenge inputs equal derived challenges
  verify sumcheck round transitions
  assert final claim equals expected_spartan_claim_public

outside Groth16:
  compute expected_spartan_claim_public from public R1CS table,
  public challenges, and public/bound opening evals
```

That split can be sound if all boundary values are public and bound, but it is no longer a fully self-contained wrapper. The outer verifier still performs Jolt-specific work.

Current Jolt R1CS shape in this checkout:

- `NUM_R1CS_INPUTS = 35`
- `NUM_R1CS_CONSTRAINTS = 19`
- univariate-skip split is effectively `10 + 9` rows

Very rough structural constraint budget for doing the inner Jolt R1CS matrix eval inside Groth16:

```text
compute L_i(r0) weights:        ~8-9 constraints
weight 19 A-row evals:           19 constraints
weight 19 B-row evals:           19 constraints
blend row groups:                 2 constraints
final Az * Bz:                    1 constraint
-----------------------------------------------
subtotal:                       ~50 constraints
```

For the full Spartan outer output-claim expression, including `Eq` and scaling terms, a rough budget is more like `70-120` constraints, depending on trace size and lowering strategy. This excludes transcript/hash constraints.

Caveat: this assumes a structured lowering that keeps row evals as linear combinations and only multiplies by challenge-dependent weights. A naive expanded sum-of-products lowering could be much larger.

## Quick Commands

List relevant Quang fork branches:

```bash
git ls-remote --heads https://github.com/quangvdao/jolt.git \
  | rg -i 'recurs|snark|wrap|groth|gnark|hyrax|grumpkin|onchain|spartan|prover-backend|voj|pairing|miller'
```

Compare branch tip dates:

```bash
git show --no-patch --format='%H %ci %cn <%ce>%n%s' \
  87fcafe44602562d5dce593acd61afa66d6bc4c4 \
  812f7418ef092928537dc7d38c9f496a9fc721c4 \
  2c49a0fa7c462e567a96ab8abe7a19665cf4f433
```

Check ancestry:

```bash
git merge-base --is-ancestor \
  812f7418ef092928537dc7d38c9f496a9fc721c4 \
  87fcafe44602562d5dce593acd61afa66d6bc4c4
```

Exit code `0` means `voj-recursion` is contained in `recursion-temp`.
