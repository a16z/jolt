//! `jolt_core_transcript_consistency` — mechanizes Invariant #1 of the
//! jolt-core transcript→spongefish migration below the full `muldiv` e2e.
//!
//! For a representative jolt-core-shaped operation sequence (`public_message`
//! of shared values + `prover_message` of proof values +
//! `verifier_message`/`challenge_128`), the prover produces a NARG byte-string
//! and an **independently-built verifier replays that NARG**
//! (`prover_message::<T>()` reads in order), deriving **identical challenges**
//! and passing **`check_eof`**. A NARG with **trailing garbage** must be
//! **rejected** by `check_eof` (the soundness-critical malleability guard).
//! Finally, the same ops under a **different instance digest** must derive
//! **different challenges** — confirming the instance is bound into the sponge
//! (a symmetric instance-drop the replay check alone cannot see).
//!
//! This catches role/order drift and the malleability hole with a fast
//! `jolt-eval` test rather than only the end-to-end prover. It uses the
//! Blake2b512 sponge — jolt-core's default.

use ark_bn254::Fr as ArkFr;
use jolt_field::Fr as JFr;
use spongefish::instantiations::Blake2b512;

use jolt_transcript::{prover_transcript, verifier_transcript, BytesMsg, OptimizedChallenge};

use crate::invariant::transcript_symmetry::{Input, Op};
use crate::invariant::{CheckError, Invariant, InvariantViolation};

const SESSION: &[u8] = b"jolt-eval/jolt-core-transcript-consistency/v1";

/// Runs the prover side, returning the NARG byte-string and the challenges it
/// derived in order.
fn prove(ops: &[Op], instance: [u8; 32]) -> (Vec<u8>, Vec<JFr>) {
    let mut prover = prover_transcript(SESSION, instance, Blake2b512::default());
    let mut challenges = Vec::new();
    for op in ops {
        match op {
            Op::PublicBytes(b) => prover.public_message(&BytesMsg(b.clone())),
            Op::PublicScalar(f) => prover.public_message(&ArkFr::from(*f)),
            Op::ProverBytes(b) => prover.prover_message(&BytesMsg(b.clone())),
            Op::ProverScalar(f) => prover.prover_message(&ArkFr::from(*f)),
            Op::Challenge => {
                let c: ArkFr = prover.verifier_message();
                challenges.push(JFr::from(c));
            }
            Op::OptimizedChallenge => challenges.push(prover.challenge_128()),
        }
    }
    (prover.narg_string().to_vec(), challenges)
}

/// Replays `ops` against `narg`: reads back every prover message in order,
/// checks each challenge equals `expected`, and finally asserts `check_eof`.
/// Returns `Err(reason)` on any read failure, challenge divergence, or a
/// non-empty NARG tail.
fn replay(ops: &[Op], instance: [u8; 32], narg: &[u8], expected: &[JFr]) -> Result<(), String> {
    let mut verifier = verifier_transcript(SESSION, instance, Blake2b512::default(), narg);
    let mut idx = 0usize;
    for (op_idx, op) in ops.iter().enumerate() {
        match op {
            Op::PublicBytes(b) => verifier.public_message(&BytesMsg(b.clone())),
            Op::PublicScalar(f) => verifier.public_message(&ArkFr::from(*f)),
            Op::ProverBytes(want) => {
                let got: BytesMsg = verifier
                    .prover_message()
                    .map_err(|e| format!("op {op_idx}: read BytesMsg failed: {e:?}"))?;
                if got.as_slice() != want.as_slice() {
                    return Err(format!("op {op_idx}: ProverBytes round-trip mismatch"));
                }
            }
            Op::ProverScalar(want) => {
                let got: ArkFr = verifier
                    .prover_message()
                    .map_err(|e| format!("op {op_idx}: read Fr failed: {e:?}"))?;
                if JFr::from(got) != *want {
                    return Err(format!("op {op_idx}: ProverScalar round-trip mismatch"));
                }
            }
            Op::Challenge => {
                let c: ArkFr = verifier.verifier_message();
                if JFr::from(c) != expected[idx] {
                    return Err(format!("op {op_idx}: Challenge diverged"));
                }
                idx += 1;
            }
            Op::OptimizedChallenge => {
                if verifier.challenge_128() != expected[idx] {
                    return Err(format!("op {op_idx}: OptimizedChallenge diverged"));
                }
                idx += 1;
            }
        }
    }
    verifier
        .check_eof()
        .map_err(|e| format!("check_eof failed: {e:?}"))
}

fn seed_corpus() -> Vec<Input> {
    let scalar = JFr::from_le_bytes_mod_order(&[0x3Cu8; 32]);
    let op_sequences: Vec<Vec<Op>> = vec![
        vec![],
        vec![Op::ProverScalar(scalar), Op::Challenge],
        // A jolt-core-shaped stage: absorb shared statement, write proof
        // payload, squeeze optimized + full challenges.
        vec![
            Op::PublicBytes(b"statement".to_vec()),
            Op::PublicScalar(scalar),
            Op::ProverBytes(vec![1, 2, 3, 4, 5, 6, 7, 8]),
            Op::OptimizedChallenge,
            Op::ProverScalar(JFr::from(7u64)),
            Op::Challenge,
            Op::ProverBytes(vec![]),
            Op::OptimizedChallenge,
        ],
    ];

    // Index 0 keeps the degenerate all-zeros instance; the rest are non-zero so
    // the corpus exercises instance binding, not just the zero fixture.
    op_sequences
        .into_iter()
        .enumerate()
        .map(|(i, ops)| Input {
            instance: [i as u8; 32],
            ops,
        })
        .collect()
}

/// NARG-replay + malleability-guard invariant for the jolt-core transcript flow.
#[jolt_eval_macros::invariant(Test, Fuzz)]
#[derive(Default)]
pub struct JoltCoreTranscriptConsistencyInvariant;

impl Invariant for JoltCoreTranscriptConsistencyInvariant {
    type Setup = ();
    type Input = Input;

    fn name(&self) -> &str {
        "jolt_core_transcript_consistency"
    }

    fn description(&self) -> String {
        "A jolt-core-shaped NARG (public_message + prover_message + \
         verifier_message/challenge_128) replayed by an independently-built \
         verifier derives identical challenges and passes check_eof; a NARG \
         with trailing garbage is rejected by check_eof; and the same ops under \
         a different instance digest derive different challenges (the instance \
         is bound into the sponge)."
            .to_string()
    }

    fn setup(&self) {}

    fn check(&self, _setup: &(), input: Input) -> Result<(), CheckError> {
        let (narg, challenges) = prove(&input.ops, input.instance);

        // Honest replay: identical challenges + clean check_eof.
        replay(&input.ops, input.instance, &narg, &challenges).map_err(|reason| {
            CheckError::Violation(InvariantViolation::with_details(
                "honest NARG replay diverged or failed check_eof".to_string(),
                reason,
            ))
        })?;

        // Malleability guard: appending a trailing byte must be rejected
        // (either a read fails or, more typically, check_eof sees the unread
        // tail). If it verifies cleanly, the proof bytes are malleable.
        let mut tampered = narg.clone();
        tampered.push(0xFF);
        if replay(&input.ops, input.instance, &tampered, &challenges).is_ok() {
            return Err(CheckError::Violation(InvariantViolation::with_details(
                "trailing-garbage NARG accepted — proof bytes are malleable".to_string(),
                "check_eof returned Ok on narg ‖ 0xFF".to_string(),
            )));
        }

        // Domain separation: the same ops under a *different* instance digest
        // must derive *different* challenges — otherwise the instance isn't
        // bound into the sponge. The replay/symmetry checks above cannot catch
        // a symmetric instance-drop: if both sides ignored the instance they
        // would still agree. (Skipped when the ops squeeze no challenges.)
        if !challenges.is_empty() {
            let mut other_instance = input.instance;
            other_instance[0] ^= 0xFF;
            let (_, other_challenges) = prove(&input.ops, other_instance);
            if other_challenges == challenges {
                return Err(CheckError::Violation(InvariantViolation::with_details(
                    "instance digest not bound — distinct instances derive identical challenges"
                        .to_string(),
                    "domain separation violated: prove(instance) == prove(instance ^ 0xFF)"
                        .to_string(),
                )));
            }
        }

        Ok(())
    }

    fn seed_corpus(&self) -> Vec<Input> {
        seed_corpus()
    }
}
