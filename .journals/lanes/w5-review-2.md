# W5 assembly review #2

Target: `5b3dced07` (`wrap/spartan-hyperkzg`). Scope: wrapper assembly, stream,
relation table, relation, keys, adapters, cost accounting, and real e2e. T1/T2
table internals were treated as reviewed dependencies except at their assembly
boundaries.

## Verdict

**0 blockers / 1 major / 2 minors.** The current fibonacci key is fully linked
and the real gate passes. Key construction is still a caller/test concern, and
its commitment-order arithmetic is wrong as soon as RAM-RA and bytecode-RA
family lengths differ.

## Findings

### Major 1 — canonical key/link assembly is test-owned and its commitment permutation is profile-specific

**Files:**

- `crates/jolt-wrapper/src/wrap.rs:230-267`
- `crates/jolt-wrapper/tests/wrap_real_t1_r.rs:471-527,1087-1095`

`verify_wrapped_with_key` now has no free exporter argument, so one constructed
`WrapVerifierKey` fixes the accepted relation. Its public constructor still
takes a caller-built `AssemblyStatement`, `WrapAssemblyPlan`, and arbitrary
relation/CopyLink pinned commitments. The only full builder is the real test.
It manually converts T2's final-opening order
`ram_inc, rd_inc, instruction, bytecode-RA, ram-RA` into T1's absorption order
`rd_inc, ram_inc, instruction, ram-RA, bytecode-RA`.

The two tail families use each other's lengths at lines 1093-1094. This happens
to work for fibonacci because both lengths are four. With four RAM-RA and three
bytecode-RA commitments, the added regression gets transcript positions
`[36, 37]` where `[38, 39]` is required. A key for such a profile cross-links
T1 commitment bytes to different T2 operands, so R's commitment weights and
T2's deferred opening no longer name the same polynomial list. The raw key
constructor also permits omitted relation/CopyLink pins; soundness therefore
depends on unvalidated setup code outside the crate.

**Fix:** make one production key builder own the link plan. Have the Dory
adapter return commitments tagged by `JoltCommittedPolynomial` (or expose a
canonical tagged order), join those tags against T1's schedule identities, and
make the raw plan/pin constructor private. For the local arithmetic, the
bytecode range starts at `2 + instruction + ram`; the RAM range starts at
`2 + instruction`.

Regression: `.journals/lanes/w5-review-2-tests.patch`.

### Minor 1 — the degree-six claim remains one degree above every current round polynomial

**Files:**

- `crates/jolt-wrapper/src/stream/protocol.rs:94-102,261-267`
- `crates/jolt-wrapper/src/stream/term_stage.rs:21-35,102-125`
- `crates/jolt-wrapper/src/limb_table/relation.rs:710-758`
- `crates/jolt-wrapper/src/stream/shared_rounds.rs:177-228,370-433`

The real proof sends four factor evaluations (`ell=128 B`), not five. T1 has at
most two factors; R and every CopyLink cap at four; T2's widest terms are
`helper * f0 * f1 * f2` and the four-factor digit-range term. The term-table
coefficient MLE adds one factor, so its actual maximum degree is five. Stage A
also has maximum degree five.

`TermStageProver` nevertheless interpolates at seven points unconditionally,
and both prover and verifier pass degree six. The shared check therefore uses
`beta^(N-7)` (`N - 1 - 6`) and admits malicious degree-six round polynomials.
This raises the per-round soundness term without enabling a deterministic
forgery. The report's five-factor-T2 explanation is contradicted by both the
exported formulas and the four serialized evaluations.

**Fix:** interpolate through `0..=self.factors.len() + 1` (or trim the zero top
coefficient), pass degree five for this key, and use the degree-five shift
`beta^(N-6)` (`N - 1 - 5`). If a later exporter really emits five factors,
select degree six from key-owned exporter metadata.

### Minor 2 — requested cleanup left stale docs, test-only public surfaces, and two oversized tests

**Files:**

- `crates/jolt-wrapper/src/lib.rs:1-7`
- `crates/jolt-wrapper/src/wrap.rs:178-236,583-646,689`
- `crates/jolt-wrapper/src/relation_table/mod.rs:21-27`
- `crates/jolt-wrapper/tests/wrap_real_t1_r.rs:1`
- `crates/jolt-wrapper/tests/limb_table_e2e.rs:1`

The crate docs still describe R as Spartan and advertise a Groth16 layer that
the program dropped. `WrapPreparation` still says "T1/T2/Spartan". The raw
exporter plans, `DoryLinkedProver`, and relation-term context builders remain
public with only test callers. Production source files are now under 1,000
lines (`stream.rs` 933; limb relation 931), but the real wrapper test is 1,731
lines and the limb-table e2e is 1,310.

**Fix:** move canonical key construction into `wrap`, make its component plans
private, rewrite the crate/preparation docs for the R row table, and split the
real test into key/link, tamper, and reporting modules.

## Link ledger

| Inner-verifier value | Count | Binding | Result |
|---|---:|---|---|
| T1 squeeze outputs | 376 | one key-pinned CopyLink into R challenge anchors; decoder kind checked (`Challenge125` vs `Scalar128`) | complete |
| Fr absorbs before the final squeeze | 1,199 | one key-pinned CopyLink; aligned rows use `fr_word`, 22-byte-tail rows use `fr_word_shifted`; R anchor index is the schedule index | complete |
| Fr absorbs after the final squeeze | 23 | no T1 link, by design: 22 opening coordinates plus the joint evaluation are R variables; no later challenge reads them | complete |
| Dory elements | 45,152 bytes / 1,526 T2 input rows | eight key-pinned CopyLinks; GT limb order, commitment byte reversal, compressed G1/G2 x limbs, sign bit, and zero infinity flag checked | complete for fibonacci; Major 1 for unequal commitment-family lengths |
| Dory scalars | 173 named values / 230 occurrences | R anchors in `DoryScalar::link_order`; one occurrence-weighted scalar-link member; T2's constant/offset/window terms share that member | complete |
| External R inputs | 7 Fr | public CopyLink: key-owned selector/id columns plus verifier-evaluated sparse public MLE | complete |
| T1 initial state and first-block public tail | 54 bytes / 4 Fr | injective 16-byte packing in `hash_public_statement`; checked against the key's `PublicInputs`, then used in T1's verifier-side final relation | complete |
| Profile digest | 32 bytes | `AssemblyStatement.key_digest == WrapHashKey.profile_digest`; replay rejects a proof-shape mismatch | complete |
| Program digest and IO preamble | hashed preamble | fixed transitively by the 32-byte T1 `state_in` and 22-byte tail in the statement, under Blake3 collision resistance | complete under reference-key trust |

The 23 post-final-squeeze values are not split between R and T2. Evaluation is
a named scalar linked directly. The point feeds R's Dory scalar derivations;
the resulting `Ht`, pairing coefficients, and related named values are the
ones occurrence-linked into T2. `from_jolt` sources T2 `WireValues` from those
R witness cells, not a second point/evaluation copy.

## Public pins and key trust

- All seven statement fields occupy R anchor cells. The public CopyLink's
  selected-column evaluation is recomputed from `statement.public_inputs` and
  the stage point; it is not proof data.
- The existing real mismatch test changes `ValIo` and rejects. The same
  `values: 0..7` plan covers `InitEval` and all five bytecode stage values; no
  field-specific branch exists.
- Hash, R, CopyLink, and T2 selector/sigma columns are verifier-key
  commitments. Proof shape and profile mismatch fail before algebraic checks.
- Trust assumption: `WrapHashKey::from_reference` verifies one known inner
  proof to derive the schedule; the caller must also derive the outer plan,
  T2 layout, public statement, and non-T1/T2 pins correctly. Major 1 is the
  missing checked owner of that second step.

## Fiat-Shamir order

Verified order:

1. wrapper domain, profile key digest, 11 statement Fr values;
2. phase 1a commitments, R/CopyLink challenges, T1 challenges, theta, scalar-link rho;
3. T2 phase 1b, then `xi, alpha`; phase 2a, then `fp_root`; phase 2b, then `beta, fp_combine, copy_root`;
4. phase 2c plus R/CopyLink helpers, then T2 row challenges and R/CopyLink tau/weights;
5. stage-A input claims and member batching coefficients; each round commitment precedes its point and next claim;
6. term input claim and ten committed rounds; shared-round batching coefficients, both `S(0)` values, then BDFG shift/quotient/witness;
7. four factor evaluations, post-evaluation lambdas, 11 clear stage-B rounds, reduced claim;
8. HyperKZG fold commitments/challenges, both signed evaluation rows, `w`, and `P0(r^2)` in the PCS order.

`verify_wrapped_with_key` constructs one `CountingKeccakTranscript`, derives
the statement against it, and passes that live transcript into assembly
verification. No detached replay remains.

## Term, stage B, and HyperKZG geometry

- Terms: `232 T1 + 15 R + 110 CopyLink + 177 T2 + 1 scalar = 535`.
- Term rounds: `ceil(log2(535)) = 10`.
- Full packed groups at k=32: 36, padded to 64; slot variables: 5; stage B: 11 rounds.
- Packed opening variables: 18 row + 5 slot = 23.
- Shared BDFG shift exponent matches the declared degree-six bound (`N-7`), but Minor 1 shows that declaration is loose.

## Tamper matrix

`tamper_suite` visits every element of every nonempty serialized value vector:
all 21 wire commitments, all committed/clear round data, both `S(0)` values,
all three shared BDFG elements, four term evaluations, the reduced claim, every
HyperKZG fold commitment, `w`, both evaluation rows, and `P0(r^2)`. Empty
legacy vectors and option/container shapes are rejected by
`validate_assembly_proof`; they are not each toggled in the test.

Witness/statement checks cover phase commitments, relation/CopyLink helpers,
T2 VK pins, a unique-recoding window row, sign, psi-chain input, a digit
occurrence, an absorbed R word, a T2 input row, public `ValIo`, proof shape,
and the profile digest. The new scratch regression covers the missing unequal
commitment-family identity.

## Cleanup result

Removed: carry, Spark, standalone Spartan, tensor stream, standalone R proof,
`NativeParity`, duplicate relation evaluators, manual timing probes, and their
tests. No `#[allow]` remains in scope; `#[expect]` use is narrow. Imports follow
the nominal-path rule in the reviewed assembly. Residuals are Minor 2.

## Performance note

Confirmed, not a finding: each `CopyLinkProver` materializes and binds 20 full
`2^18` columns even though selectors are sparse; eleven members repeat that
work. A follow-up should introduce one batched CopyLink member with shared
physical-column bind state and per-link random coefficients, so each referenced
column binds once per round and the eleven final relations are accumulated in
one pass. Key-owned selector/id commitments and independent beta/gamma domains
must remain unchanged.

## Verification

- Real k=32 gate: passed, 81.84 s test time.
- Payload/bincode/statement: 5,728 / 5,836 / 352 bytes.
- Geometry: T=535; 10 term rounds; 11 stage-B rounds; opening dimension 23.
- Cost: 185 ecMul / 184 ecAdd / 8 pairing pairs / 40,722 Fr mul / 8 Fr inv / 755 Keccak; gas model 2,883,641.
- New unequal-family regression: failed as intended in 0.27 s (`left [36,37]`, expected `[38,39]`).
- No full suite, clippy, or fmt run; the task requested the real gate and review policy leaves broad checks to CI.
