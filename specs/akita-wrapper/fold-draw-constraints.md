# D64 FoldDraw root binding

Native codec revision is f5f75335eae18241681fd24ca0a60fca8f0512af,
with byte-preserving FoldChallengeFrame extraction based on
7649443f44fa8c3b6d6ad3edd374bf14ea149df3. The native frame is the sole owner
of validation and prefix/nonce/suffix layout. Production FoldDraw now calls it.
Jolt uses its prefix/suffix directly, inserting four constrained LE nonce bytes.

Selected route: EvaluationTrace, D64 selective-L2 counts 31+11 and its certified
rejection domain (including certificate error 4/margin 600). The actual runtime
norm remains the reviewed epsilon1 table; these are intentionally different
source contracts. Public metadata is group index, exact live-block/claim counts,
and callback coordinate index. The native frame validates counts and total;
callback index must be below that total and canonically fit LE-u64.

The inherited AkitaTranscriptVar is mutated in place: append_bytes applies its
native LE64 message-length frame to the complete FoldDraw payload, then squeeze
one native 32-byte block. That constrained root feeds the reviewed indexed Blake
stream and bounded first-norm-acceptance relation. No caller-supplied root hint
is accepted. This does not prove that the inherited state already includes the
instance descriptor, proof preamble, earlier folds or previous group draws.

Grinding is not moved into FoldDraw. verify_fold reads a FoldResponse nonce
before group challenge derivation. Native TranscriptNonceReader only consumes
its 12-bit packed stream slot; it performs no absorb/squeeze or PoW predicate.
The adapter selects those 12 constrained bits at a public bit offset, zero-extends
them to canonical u32 LE bytes, and binds the same nonce into the payload.
Packed-stream plan ownership, site/order/exhaustion and final padding validation
remain caller obligations. Other ProofOfWork sites are separate transcript
transitions and are not invented here. Response correctness/norm checks that
justify the prover's nonce search remain downstream verifier obligations; the
verifier does not require the first prover-search nonce.

Native payload before the transcript length prefix is canonical sample label
(group/blocks/claims) || totalLE64 || DLE64 || config domain || nonceLE32 ||
rejection domain. Constants and codecs are imported from their native owners.
The native framing extraction must retain all existing validation errors and
challenge vectors; a separate Python struct-packed payload vector checks layout.

Public R/K remains finite capacity, not a negligible-failure claim. ONE is fixed
externally and every handle must use one builder. This packet binds one selected
group root and one indexed output, not complete proof acceptance.


## Source-to-constraint map

- Native `fold_draw/frame.rs::FoldChallengeFrame` owns the checked public frame;
  existing `FoldDraw::draw_folding_challenges_with_rejection` is its first native
  production caller. `D64FoldDrawShape` calls the same codec with the fixed route.
- Native `transcript_grinding.rs::GrindingRun` pins FoldResponse slots to exported
  `FOLD_RESPONSE_NONCE_BITS`; `FoldResponseNonceVar::from_packed` links that many
  low-bit-first source bits to 32 Boolean output bits, forcing the unused high bits
  to zero. The selected public offset remains the caller's plan-replay obligation.
- Native `LiveFoldDraw::absorb_and_squeeze` appends one payload and consumes one
  challenge block. `D64FoldDrawShape::draw` uses the constrained transcript's same
  length-framed append and native-owned seed length, mutating the inherited state.
- `D64FoldDrawVar::sample_coordinate` checks the public callback index against the
  exact frame count, then calls the accepted bounded retry relation on that root.
  The claim-major flat index must be the one used by the rest of the verifier.
- Native `verify_fold` reads a FoldResponse before `derive_multi_group_stage1_challenges`;
  `draw_group_fold_challenges` selects this route for EvaluationTrace+D64+L2.
  `physical_l2_norm.rs` and terminal response checks remain outside this packet.

The nonce is not required to be the first successful prover-search nonce: native
verification checks the supplied bounded nonce and the ensuing proof relations.
No unrelated PoW predicate is substituted for those response relations.

## Producer validation and dependency boundary

Native codec commit: `f5f75335eae18241681fd24ca0a60fca8f0512af`. Six native
FoldDraw tests pass on its canonical public dependency graph, including existing
golden challenges. Scoped native clippy, formatting and file-line checks pass.
The consumer passes three focused tests in both default and minimal `r1cs`
profiles and both all-target clippy profiles. Tests cover native indexed output
and transcript continuation, domain separation, coherent nonce/proof tampering,
canonical nonce high bits, invalid shapes and witness-independent matrices.

Original local-overlay evidence and exact commands are preserved in
`/private/tmp/fold-draw-r1cs-20260922/manifest.json`; those checks patched all
14 native crates to the codec source and retain their resolved lock/config.
The codec is now published in https://github.com/markosg04/akita/pull/3, and
all tracked Akita dependencies pin its exact public revision. Locked offline
metadata resolves 14 public Akita packages and one workspace jolt-field without
local overlays. Only native revision substitutions changed in Cargo.lock.

Final public-pin nextest passed all three focused tests, run
`39611a5f-1e51-4f3a-ae2d-c743054d1206`, log
`/private/tmp/fold-draw-public-tests.log`. The command is:

```
CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=../wrapper/target cargo nextest run \
  --offline --locked -p jolt-akita --no-default-features --features r1cs \
  --lib -E 'test(r1cs::fold_draw)' --test-threads 1 --cargo-quiet
```

Final public-pin all-target clippy (`--features r1cs -D warnings`), formatting
and diff checks passed. Clippy log: `/private/tmp/fold-draw-public-clippy.log`.
These checks do not establish complete wrapper acceptance.

## Next verifier boundary

Update the historical terminal observer and census to the current Blake epoch
before composing a complete fixed-profile terminal verifier. Reuse its actual
proof/preprocessing replay and terminal coefficient export, then regenerate
assignments from current proof/setup/transcript bytes. Historical epoch-five
observations cannot be relabeled as epoch-six assignments. The complete target
includes all 192 A rows, 64 consistency rows and the separate scalar check,
alongside canonical decoding and the complete physical norm cap.

At the native `verify_fold`/`verify_terminal_suffix` boundary, bind the opening
layout and grinding plan to the public profile, replay the real preamble and
prior folds, consume every required nonce slot/group/coordinate, and link these
outputs to all response equations. This packet supplies only the FoldDraw
transition inside that replay; it does not authenticate the inherited state,
check the complete nonce stream or establish whole-proof acceptance. No new
constraint-cost or full-proof performance measurement is claimed here.

