# Stage-4 RegistersRW radix-4 Phase-2 pin

**Verdict: NO-GO for the scoped `[P4,P4,P4,S,S×27]` implementation.** The
first seven RegistersRW variables are cycle variables, not register-address
variables. The current Metal prefix folds adjacent cycle rows. A real virtual-
address prefix requires address-first reordering and a different state algorithm;
the claimed unchanged-CSR/pass-count cut is therefore false.

No Cargo, build, tests, or production edits were run for this audit.

## Invariant ledger

| # | invariant | verdict | pin |
|---:|---|---|---|
| 1 | centered domain / extension convention | **GO with correction** | `D=[-1,0,1,2]`; evaluation windows are ascending centered integers, not domain-first |
| 2 | RegistersRW degree, message, CSR stride / bit order | **NO-GO** | address P4 degree is 6, not exact degree 9; current prefix and CSR rows are cycles |
| 3 | every stage-4 address-factor consumer enumerated | **NO-GO for proposed cut** | stage 6a/6b and committed-program folds were omitted; there is no opening-accumulator object |
| 4 | packed register-address factors never enter PCS | **GO** | factors are consumed by virtual folds or discarded before cycle-only reductions; both final opening paths use fresh canonical points |
| 5 | `[P4,P4,P4,S,S×27]` batch math | **GO algebra / NO-GO engine** | 34 semantic variables, 31 messages; current engine equates variables, messages, and scalar challenges |
| 6 | transcript / config / ZK fail-closed seams | **NO-GO today** | no schedule axis or mixed-round wire; config validation is correctly placed before transcript creation |
| 7 | proposed minimal cut | **NO-GO** | it omits reordering, dense-Val/address-state work, generated batch plumbing, stage 6 consumers, and fallback parity |

## 1. Domain, cell convention, wire degree

- Centered size 4 is exactly `D={-1,0,1,2}` because
  `start=-floor((n-1)/2)`; nodes are stored in ascending consecutive order
  ([`crates/jolt-poly/src/lagrange.rs:465-484`](../../crates/jolt-poly/src/lagrange.rs),
  [`crates/jolt-poly/src/lagrange.rs:668-673`](../../crates/jolt-poly/src/lagrange.rs)).
- Registers bind LowToHigh and the address fold pairs `table[2y]` with
  `table[2y+1]`; therefore the first local bit is the LSB. For a packed pair
  `(b_i,b_{i+1})`, `w=b_i+2b_{i+1}=2*x_hi+x_lo`, `z_w=w-1`, and output factor
  order is high-to-low `[B6,Q45,Q23,Q01]`
  ([`crates/jolt-kernels/src/optimized/registers_read_write.rs:1210-1219`](../../crates/jolt-kernels/src/optimized/registers_read_write.rs),
  [`crates/jolt-kernels/src/optimized/registers_read_write.rs:1281-1297`](../../crates/jolt-kernels/src/optimized/registers_read_write.rs),
  [`crates/jolt-kernels/src/optimized/registers_read_write.rs:1348-1362`](../../crates/jolt-kernels/src/optimized/registers_read_write.rs)).
- Existing uni-skip evaluates an ascending centered window. Thus the exact
  degree-6 RegistersRW address message uses seven points `[-3,-2,-1,0,1,2,3]`.
  A deliberately conservative degree-9 wire would use ten points `[-4,-3,-2,
  -1,0,1,2,3,4,5]`; the proposed domain-first order is not the existing
  convention ([`crates/jolt-kernels/src/reference/spartan_outer.rs:135-188`](../../crates/jolt-kernels/src/reference/spartan_outer.rs),
  [`crates/jolt-poly/src/lagrange.rs:11-25`](../../crates/jolt-poly/src/lagrange.rs)).
- The relation's global binary-round bound is 3, but during address rounds
  `EqCycle` and `RdInc` are address-constant. The address summand is
  `EqCycle*(Wa*(RdInc+Val)+Ra*Val)`: two degree-3 quaternary input extensions
  multiply, so `deg_Z q <= 6`, seven full coefficients. Degree 9 is sound but
  neither exact nor minimal
  ([`crates/jolt-claims/src/protocols/jolt/relations/registers/read_write_checking.rs:85-113`](../../crates/jolt-claims/src/protocols/jolt/relations/registers/read_write_checking.rs),
  [`crates/jolt-kernels/src/optimized/registers_read_write.rs:1202-1238`](../../crates/jolt-kernels/src/optimized/registers_read_write.rs)).
- The verifier sum functional already exists as
  `CenteredIntegerDomain::new(4)` power sums. A packed round must use a distinct
  label, full `c0..c6`, one squeeze, and the same four `L_w(r)` weights for every
  oracle bind ([`crates/jolt-sumcheck/src/domain.rs:84-117`](../../crates/jolt-sumcheck/src/domain.rs),
  [`crates/jolt-sumcheck/src/round_proof.rs:75-86`](../../crates/jolt-sumcheck/src/round_proof.rs)).

## 2. Relation order and Metal CSR — blocking contradiction

```text
current RegistersRW local variables:  cycle[0..logT) | address[0..7)
current stage-4 batch globals:         R-only[0..7) | R+RamVal[7..34)
proposed packed globals:               P4 P4 P4 S   | S × 27
                                      ^^^^^^^^^^^
                                      cycle bits, not address bits
```

- Default config explicitly assigns all cycle variables to phase 1 and all
  seven register-address variables to phase 2
  ([`crates/jolt-prover/src/config.rs:142-151`](../../crates/jolt-prover/src/config.rs)).
  The exact existing dimension setting for address-first is
  `registers_rw_phase1_num_rounds=0`,
  `registers_rw_phase2_num_rounds=7`: the point builder then consumes phase 2
  address challenges before phase 3 cycle challenges and still emits canonical
  `address || cycle` order
  ([`crates/jolt-claims/src/protocols/jolt/geometry/dimensions.rs:144-179`](../../crates/jolt-claims/src/protocols/jolt/geometry/dimensions.rs)).
  Both reference and optimized kernels reject any phase 1 other than all
  cycles; optimized dispatches cycle messages until `log_t`, then address
  messages ([`crates/jolt-kernels/src/reference/registers_read_write.rs:30-45`](../../crates/jolt-kernels/src/reference/registers_read_write.rs),
  [`crates/jolt-kernels/src/optimized/registers_read_write.rs:728-739`](../../crates/jolt-kernels/src/optimized/registers_read_write.rs),
  [`crates/jolt-kernels/src/optimized/registers_read_write.rs:1426-1444`](../../crates/jolt-kernels/src/optimized/registers_read_write.rs)).
- The Metal guard has the same requirement and transitions to the host only
  after every cycle bind; it never runs the address phase
  ([`crates/jolt-kernels/src/metal/slots/registers_read_write.rs:633-664`](../../crates/jolt-kernels/src/metal/slots/registers_read_write.rs),
  [`crates/jolt-kernels/src/metal/slots/registers_read_write.rs:785-854`](../../crates/jolt-kernels/src/metal/slots/registers_read_write.rs)).
- CSR `row_offsets[t..t+1]` are cycle rows; `col` is the register. The shader
  reads rows `2*gid` and `2*gid+1`, unions equal columns, and halves the row
  count. This is a cycle bind, not “four child register rows”
  ([`crates/jolt-kernels/src/metal/slots/registers_read_write.rs:77-167`](../../crates/jolt-kernels/src/metal/slots/registers_read_write.rs),
  [`crates/jolt-kernels/src/metal/shaders/registers_read_write.metal:41-119`](../../crates/jolt-kernels/src/metal/shaders/registers_read_write.metal),
  [`crates/jolt-kernels/src/metal/shaders/registers_read_write.metal:208-267`](../../crates/jolt-kernels/src/metal/shaders/registers_read_write.metal)).
- Address-first would group `col>>2` within each unchanged cycle row, keep `T`
  rows, and emit `col>>2`; it would not quarter CSR rows. More importantly,
  quaternary binding needs
  `Val_hat(g,j)=sum_w L_w(r)*Val(4g+w,j)`. Current rows store at most the three
  accessed registers, while absent-register `Val` values become cross terms
  after `Ra/Wa` folding. **Inference:** current sparse entries cannot compute
  that fold without a new register-state/history algorithm or dense `128*T`
  state. The “unchanged CSR, no representation change” premise is refuted by
  the row construction itself
  ([`crates/jolt-kernels/src/metal/slots/registers_read_write.rs:77-129`](../../crates/jolt-kernels/src/metal/slots/registers_read_write.rs),
  [`crates/jolt-kernels/src/optimized/registers_read_write.rs:872-887`](../../crates/jolt-kernels/src/optimized/registers_read_write.rs)).

Consequences: packing the measured Metal prefix packs cycle factors; moving the
address prefix first invalidates the current kernel and performance model;
packing the existing address tail is host-only and overlaps active RamVal.

## 3. Complete consumer inventory

| consumer | exact obligation | evidence |
|---|---|---|
| stage-4 point derivation / final claim | all five outputs share one point; `expected_final_claim` evaluates the symbolic output expression. Its only derived equality is **cycle-only**—there is no proposed address-Eq factor here | [`stage4/registers_read_write_checking.rs:84-118`](../../crates/jolt-verifier/src/stages/stage4/registers_read_write_checking.rs), [`jolt-verifier-derive/src/lib.rs:782-850`](../../crates/jolt-verifier-derive/src/lib.rs) |
| stage-4 output extraction | optimized/Metal extraction currently reconstructs seven ordinary address scalars and calls ordinary Eq tables; packed factor expansion must replace that path | [`optimized/registers_read_write.rs:1348-1387`](../../crates/jolt-kernels/src/optimized/registers_read_write.rs), [`optimized/registers_read_write.rs:1453-1469`](../../crates/jolt-kernels/src/optimized/registers_read_write.rs) |
| stage-4 point storage | generated point cells are `Vec<F>` and stage output exposes one scalar slice; no typed radix-4 factor exists | [`stages/relations.rs:62-86`](../../crates/jolt-verifier/src/stages/relations.rs), [`stage4/outputs.rs:83-121`](../../crates/jolt-verifier/src/stages/stage4/outputs.rs) |
| stage-5 RegistersValEvaluation | carries the stage-4 address prefix forward, builds `RdWa`'s address fold / Eq table, and produces stage-5 `RdInc/RdWa` points with the same address factor | [`stage5/registers_val_evaluation.rs:78-116`](../../crates/jolt-verifier/src/stages/stage5/registers_val_evaluation.rs), [`reference/registers_val_evaluation.rs:36-64`](../../crates/jolt-kernels/src/reference/registers_val_evaluation.rs), [`optimized/registers_val_evaluation.rs:123-158`](../../crates/jolt-kernels/src/optimized/registers_val_evaluation.rs) |
| stage-6a BytecodeReadRAF | carries full stage-4/stage-5 points; stage-value folds consume both address prefixes and stage-4 `RdWa/Rs1Ra/Rs2Ra` values | [`stage6a/bytecode_read_raf.rs:40-122`](../../crates/jolt-verifier/src/stages/stage6a/bytecode_read_raf.rs), [`stage6a/bytecode_read_raf.rs:137-188`](../../crates/jolt-verifier/src/stages/stage6a/bytecode_read_raf.rs), [`geometry/bytecode.rs:414-453`](../../crates/jolt-claims/src/protocols/jolt/geometry/bytecode.rs) |
| stage-6b full / committed-program folds | full-program table fold and committed lane weights independently expand the two register-address Eq vectors | [`stage6b/batch.rs:250-349`](../../crates/jolt-verifier/src/stages/stage6b/batch.rs), [`geometry/claim_reductions/bytecode.rs:381-427`](../../crates/jolt-claims/src/protocols/jolt/geometry/claim_reductions/bytecode.rs), [`geometry/claim_reductions/bytecode.rs:465-479`](../../crates/jolt-claims/src/protocols/jolt/geometry/claim_reductions/bytecode.rs) |
| stage-6b IncClaimReduction | consumes both register `RdInc` values but explicitly retains only their cycle slices; output is a fresh binary cycle point | [`stage6b/batch.rs:271-321`](../../crates/jolt-verifier/src/stages/stage6b/batch.rs), [`stage6b/inc_claim_reduction.rs:28-55`](../../crates/jolt-verifier/src/stages/stage6b/inc_claim_reduction.rs), [`stage6b/inc_claim_reduction.rs:58-145`](../../crates/jolt-verifier/src/stages/stage6b/inc_claim_reduction.rs) |
| BlindFold | reconstructs one scalar point per semantic variable and a Boolean-domain batch; packed mode must be rejected before this stage | [`stages/zk/blindfold/stage4.rs:51-73`](../../crates/jolt-verifier/src/stages/zk/blindfold/stage4.rs), [`stages/zk/blindfold/stage4.rs:136-151`](../../crates/jolt-verifier/src/stages/zk/blindfold/stage4.rs) |
| stage-8 homomorphic / Dory | `RdInc` comes from IncClaimReduction; final point is assembled only from stage-6 cycle plus stage-7 address points and passed as an ordinary scalar slice to PCS/Dory | [`geometry/committed_openings.rs:23-77`](../../crates/jolt-claims/src/protocols/jolt/geometry/committed_openings.rs), [`geometry/committed_openings.rs:124-197`](../../crates/jolt-claims/src/protocols/jolt/geometry/committed_openings.rs), [`stage8/verify.rs:105-149`](../../crates/jolt-verifier/src/stages/stage8/verify.rs), [`jolt-dory/src/scheme.rs:239-270`](../../crates/jolt-dory/src/scheme.rs) |
| stage-8 Akita batch | four inc claims join lattice ReadRAF by **value** with cycle points, produce `FusedInc`, then stage 7 creates the canonical leaf point used by `verify_batch` | [`lattice/relations/read_raf.rs:1-17`](../../crates/jolt-claims/src/protocols/jolt/lattice/relations/read_raf.rs), [`lattice/relations/read_raf.rs:76-125`](../../crates/jolt-claims/src/protocols/jolt/lattice/relations/read_raf.rs), [`stage7/hamming_weight_claim_reduction.rs:40-63`](../../crates/jolt-verifier/src/stages/stage7/hamming_weight_claim_reduction.rs), [`stage8/packed.rs:193-225`](../../crates/jolt-verifier/src/stages/stage8/packed.rs) |

No `opening_accumulator` exists in the modular path. The accumulator is the
typed stage-output graph plus the two stage-8 builders above.

## 4. PCS invariant proof

```text
stage4 RdInc(address-factor || cycle) ─┐
stage5 RdInc(address-factor || cycle) ─┴─ slice cycle only
                                         │
                    non-Akita: IncClaimReduction → fresh cycle point → PCS/Dory
                    Akita: ReadRAF value fold → FusedInc → stage-7 point → PCS batch

stage4/5 virtual Ra/Wa/Val → factor-expanded Eq/value folds → scalar claims/weights
                                                        └→ later canonical points
```

`RegistersVal`, `Rs1Ra`, `Rs2Ra`, and `RdWa` are virtual; `RdInc` is committed
([`geometry/registers.rs:24-70`](../../crates/jolt-claims/src/protocols/jolt/geometry/registers.rs)).
The committed claim's address factor is dummy and is removed before either
opening route. **Invariant GO:** a correctly typed register-address factor never
enters `PCS::open/verify/open_batch/verify_batch`. This does not license direct
`EqPolynomial::evals` on an encoded factor prefix; every virtual consumer in
§3 must use the same 128-weight expansion.

## 5. Batch math

For semantic schedule `[P4,P4,P4,S,S×27]`:

```text
semantic variables = 2+2+2+1+27 = 34
protocol messages  = 1+1+1+1+27 = 31
RegistersRW window = semantic [0,34), active in every message
RamVal window       = semantic [7,34), inactive for P4,P4,P4,S

RamVal initial padding = C * 2^(34-27) = 128C
inactive recurrence    = 128C /4 /4 /4 /2 = C at the first shared S
inactive P4 polynomial = current_claim/4 (constant over four nodes)
```

The `2^7` padding is semantic-variable-based, not message-based. Current code
computes exactly that padding but hard-codes one Boolean message and one `/2`
per variable ([`crates/jolt-sumcheck/src/batch.rs:13-29`](../../crates/jolt-sumcheck/src/batch.rs),
[`crates/jolt-sumcheck/src/batch.rs:34-64`](../../crates/jolt-sumcheck/src/batch.rs),
[`crates/jolt-sumcheck/src/prover.rs:210-250`](../../crates/jolt-sumcheck/src/prover.rs),
[`crates/jolt-sumcheck/src/prover.rs:295-316`](../../crates/jolt-sumcheck/src/prover.rs)).

Required schedule bounds are `[6,6,6,2,3×27]` (a uniform 3 bound on the single
address round is harmless). `BatchMember.rounds`, `max_num_vars`, proof-round
count, member-local round, and `Vec<F>` point length are currently the same
number; verifier and derive code also require one compressed Boolean polynomial
per variable ([`crates/jolt-sumcheck/src/proof.rs:118-151`](../../crates/jolt-sumcheck/src/proof.rs),
[`crates/jolt-sumcheck/src/verifier.rs:87-131`](../../crates/jolt-sumcheck/src/verifier.rs),
[`crates/jolt-verifier-derive/src/lib.rs:525-582`](../../crates/jolt-verifier-derive/src/lib.rs),
[`crates/jolt-verifier/src/stages/relations.rs:196-233`](../../crates/jolt-verifier/src/stages/relations.rs)).
The engine needs separate semantic arity, message count, per-message domain /
degree, and member windows. The clear proof/recorder also need a mixed full-P4 /
compressed-S representation; today they store only compressed Boolean rounds
([`crates/jolt-sumcheck/src/proof.rs:33-45`](../../crates/jolt-sumcheck/src/proof.rs),
[`crates/jolt-sumcheck/src/recorder.rs:82-101`](../../crates/jolt-sumcheck/src/recorder.rs),
[`crates/jolt-sumcheck/src/recorder.rs:118-148`](../../crates/jolt-sumcheck/src/recorder.rs)).

## 6. Transcript, config, and ZK seams

1. Add proof-carried `RegistersRwSchedule::{BinaryV1,AddressFirstRadix4V1}` to
   `JoltProtocolConfig`; validate exact build policy plus
   `AddressFirstRadix4V1 => Transparent` in `validate_proof_config`.
2. Validate the schedule before transcript construction. Current validation is
   already at the correct seam ([`crates/jolt-verifier/src/verifier.rs:276-311`](../../crates/jolt-verifier/src/verifier.rs),
   [`crates/jolt-verifier/src/config.rs:97-118`](../../crates/jolt-verifier/src/config.rs)).
3. Absorb the schedule in both consensus-critical preambles, beside the four RW
   phase fields ([`crates/jolt-verifier/src/verifier.rs:555-627`](../../crates/jolt-verifier/src/verifier.rs),
   [`crates/jolt-verifier/src/verifier.rs:753-843`](../../crates/jolt-verifier/src/verifier.rs),
   [`crates/jolt-prover/src/stages/stage0.rs:150-159`](../../crates/jolt-prover/src/stages/stage0.rs)).
4. Use a distinct packed-round label and schedule-driven mixed proof shape;
   reject wrong variant/count/degree before absorbing that stage's first round.
5. Literal “reject before parsing” is not available: `validate_proof_config`
   receives an already-deserialized `JoltProof` whose protocol field is at
   [`crates/jolt-verifier/src/proof.rs:53-71`](../../crates/jolt-verifier/src/proof.rs).
   If pre-deserialization rejection is mandatory, add a versioned proof envelope;
   otherwise the enforceable seam is before transcript/stage work.

## 7. Corrected minimum production cut after an algebraic GO

An oracle GO is necessary but insufficient: first choose and prove a performant
address-first `Val_hat` state algorithm. With that decision fixed, the minimum
safe production cut is:

| area | exact files |
|---|---|
| schedule/wire engine | `crates/jolt-sumcheck/src/{batch,prover,proof,recorder,verifier,round_proof,error,lib}.rs`; `crates/jolt-verifier-derive/src/lib.rs`; `crates/jolt-verifier/src/stages/relations.rs`; `crates/jolt-prover/src/driver.rs` |
| factor geometry / relation | `crates/jolt-claims/src/protocols/jolt/geometry/{dimensions,bytecode}.rs`; `crates/jolt-claims/src/protocols/jolt/geometry/claim_reductions/bytecode.rs`; `crates/jolt-claims/src/protocols/jolt/relations/registers/read_write_checking.rs` |
| fail-closed config / preamble | `crates/jolt-verifier/src/{config,proof,verifier}.rs`; `crates/jolt-prover/src/{config,prover}.rs`; `crates/jolt-prover/src/stages/stage0.rs` |
| stage 4 / factor carrier | `crates/jolt-prover/src/stages/{stage4,stage5,stage6a,stage6b}.rs`; `crates/jolt-verifier/src/stages/stage4/{registers_read_write_checking,outputs,verify}.rs`; `crates/jolt-verifier/src/stages/stage5/{registers_val_evaluation,verify}.rs`; `crates/jolt-verifier/src/stages/stage6a/{bytecode_read_raf,verify}.rs`; `crates/jolt-verifier/src/stages/stage6b/{batch,bytecode_read_raf,committed_reduction_cycle_phase,verify}.rs` |
| CPU kernels / consumers | `crates/jolt-kernels/src/reference/{views,registers_read_write,registers_val_evaluation,bytecode_read_raf}.rs`; `crates/jolt-kernels/src/optimized/{registers_read_write,registers_val_evaluation,bytecode_read_raf}.rs` |
| Metal | `crates/jolt-kernels/src/metal/slots/registers_read_write.rs`; `crates/jolt-kernels/src/metal/shaders/registers_read_write.metal`; `crates/jolt-kernels/src/metal/runtime.rs` |

No Dory, Akita opening, committed-sumcheck, or BlindFold production file should
change; those paths are respectively downstream-canonical or fail-closed. Any
implementation touching fewer areas is missing a live consumer or protocol seam.

### Required test matrix

| layer | mandatory cases |
|---|---|
| algebra | exact `w/bit` map; seven-node degree-6 interpolation; four node sums; random `q(r)` vs direct four-way folds; `U=XY` regression proving factor != two ordinary MLE coordinates; leading coefficients above 6 reject/nonzero detect |
| batch/wire | 34 semantic / 31 message shape; RamVal `128/4/4/4/2=1` join; full-P4 + compressed-S transcript twin; extra/missing/wrong round variant; degree 7; P4 crossing semantic join; schedule replay both directions |
| factor consumers | stage-4 final claim; all five shared output points; stage-5 `RdWa` fold; stage-6a stage values; stage-6b full table fold and committed lane weights; address-factor expansion equals brute-force 128 weights |
| backend parity | reference ↔ optimized ↔ Metal for every packed round; empty/1/2/3-access CSR rows; accesses spanning different 4-register groups; unaccessed-neighbor `Val` cross terms; device failure after each packed message/bind resumes the same wire schedule |
| protocol/e2e | transparent Dory reference/optimized/Metal accept; final Dory point contains no packed factor; committed-program accept; Akita accept + `verify_batch` point audit; witness/coeff/schedule/bit-order tamper reject; every packed+BlindFold combination rejects before transcript creation |

## Hidden blockers

1. The measured seven-round Metal prefix is cycle work; the virtual-address
   legality argument and performance target refer to different coordinates.
2. Address-first folding needs dense register-state information absent from the
   sparse cycle rows; no performant algorithm is specified.
3. Exact packed address degree is 6, invalidating the ten-evaluation cost model.
4. Current proof, derive, point, and kernel APIs assume one scalar challenge per
   semantic variable; a four-weight factor is not representable as two scalars.
5. The scoped consumer list misses stage 6a/6b full and committed-program paths;
   its stage-4 address-Eq consumer and opening accumulator do not exist.
