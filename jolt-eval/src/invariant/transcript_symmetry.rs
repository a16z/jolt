//! `transcript_prover_verifier_consistency` — for each spongefish sponge,
//! a `ProverState` / `VerifierState` pair driven by the same operation
//! sequence must round-trip every prover message and produce the same
//! verifier challenges. The same ops under a different instance digest must
//! also derive different challenges, confirming the instance is bound into
//! the sponge (a symmetric instance-drop the symmetry check alone can't see).

use arbitrary::{Arbitrary, Unstructured};
use ark_bn254::Fr as ArkFr;
use jolt_field::Fr as JFr;
use rand::rngs::StdRng;
use spongefish::instantiations::{Blake2b512, Keccak};

use jolt_transcript::{
    prover_transcript, verifier_transcript, BytesMsg, OptimizedChallenge, PoseidonSponge,
    ProverState, VerifierState,
};

use crate::invariant::{CheckError, Invariant, InvariantViolation};

const SESSION: &[u8] = b"jolt-eval/transcript-symmetry/v1";

/// One operation in the prover/verifier sequence.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub enum Op {
    /// Both sides absorb the same public bytes.
    PublicBytes(Vec<u8>),
    /// Both sides absorb the same public BN254 `Fr` scalar.
    PublicScalar(#[schemars(with = "[u8; 32]")] JFr),
    /// Prover absorbs + emits bytes; verifier reads them back from the NARG.
    ProverBytes(Vec<u8>),
    /// Prover absorbs + emits a BN254 `Fr` scalar; verifier reads it back.
    ProverScalar(#[schemars(with = "[u8; 32]")] JFr),
    /// Both sides squeeze a verifier challenge.
    Challenge,
    /// Both sides squeeze a 128-bit optimized challenge (`challenge_128`).
    ///
    /// Only the byte sponges (Blake2b/Keccak) implement this; the Poseidon
    /// invariant filters it out (Poseidon uses full-field `challenge-254-bit`
    /// and leaves `challenge_128` `unimplemented!()` — #1586 reviewer).
    OptimizedChallenge,
}

/// Sequence of operations replayed in lockstep by both sides.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct Input {
    /// 32-byte instance digest — the per-statement `DomainSeparator` binding.
    /// Varied (not fixed to zero) so the corpus/fuzzer exercises instance
    /// binding: an all-zeros-only fixture would mask a one-sided instance drop,
    /// since `prover(0)` and `verifier(0-or-ignored)` agree regardless.
    pub instance: [u8; 32],
    /// Operations to apply in order.
    pub ops: Vec<Op>,
}

impl<'a> Arbitrary<'a> for Input {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let instance: [u8; 32] = u.arbitrary()?;
        let n = u.int_in_range(0u8..=20)? as usize;
        let mut ops = Vec::with_capacity(n);
        for _ in 0..n {
            let tag = u.int_in_range(0u8..=5)?;
            ops.push(match tag {
                0 => Op::PublicBytes(arb_bytes(u)?),
                1 => Op::PublicScalar(arb_scalar(u)?),
                2 => Op::ProverBytes(arb_bytes(u)?),
                3 => Op::ProverScalar(arb_scalar(u)?),
                4 => Op::Challenge,
                _ => Op::OptimizedChallenge,
            });
        }
        Ok(Self { instance, ops })
    }
}

fn arb_bytes(u: &mut Unstructured<'_>) -> arbitrary::Result<Vec<u8>> {
    let len = u.int_in_range(0u8..=64)? as usize;
    (0..len).map(|_| u.arbitrary()).collect()
}

fn arb_scalar(u: &mut Unstructured<'_>) -> arbitrary::Result<JFr> {
    let bytes: [u8; 32] = u.arbitrary()?;
    Ok(JFr::from_le_bytes_mod_order(&bytes))
}

/// Runs the prover side over `input.ops` under `instance`, returning the NARG
/// byte-string and the challenges squeezed in order.
fn prover_run<H>(
    input: &Input,
    instance: [u8; 32],
    build_sponge: &impl Fn() -> H,
) -> (Vec<u8>, Vec<JFr>)
where
    H: spongefish::DuplexSpongeInterface<U = u8>,
    ProverState<H, StdRng>: OptimizedChallenge,
{
    let mut prover = prover_transcript(SESSION, instance, build_sponge());
    let mut challenges: Vec<JFr> = Vec::new();
    for op in &input.ops {
        match op {
            Op::PublicBytes(b) => prover.public_message(&BytesMsg(b.clone())),
            Op::PublicScalar(f) => prover.public_message(&ArkFr::from(*f)),
            Op::ProverBytes(b) => prover.prover_message(&BytesMsg(b.clone())),
            Op::ProverScalar(f) => prover.prover_message(&ArkFr::from(*f)),
            Op::Challenge => {
                let c: ArkFr = prover.verifier_message();
                challenges.push(JFr::from(c));
            }
            Op::OptimizedChallenge => {
                challenges.push(prover.challenge_128());
            }
        }
    }
    (prover.narg_string().to_vec(), challenges)
}

fn replay<H>(
    input: &Input,
    instance: [u8; 32],
    narg: &[u8],
    prover_challenges: &[JFr],
    build_sponge: &impl Fn() -> H,
) -> Result<(), CheckError>
where
    H: spongefish::DuplexSpongeInterface<U = u8>,
    for<'a> VerifierState<'a, H>: OptimizedChallenge,
{
    let mut verifier = verifier_transcript(SESSION, instance, build_sponge(), narg);
    let mut challenge_idx = 0usize;

    for (op_idx, op) in input.ops.iter().enumerate() {
        match op {
            Op::PublicBytes(b) => verifier.public_message(&BytesMsg(b.clone())),
            Op::PublicScalar(f) => verifier.public_message(&ArkFr::from(*f)),
            Op::ProverBytes(expected) => {
                let got: BytesMsg = verifier
                    .prover_message()
                    .map_err(|e| violation("prover_message<BytesMsg>", op_idx, e))?;
                if got.0.as_slice() != expected.as_slice() {
                    return Err(mismatch("ProverBytes round-trip", op_idx));
                }
            }
            Op::ProverScalar(expected) => {
                let got: ArkFr = verifier
                    .prover_message()
                    .map_err(|e| violation("prover_message<Fr>", op_idx, e))?;
                if JFr::from(got) != *expected {
                    return Err(mismatch("ProverScalar round-trip", op_idx));
                }
            }
            Op::Challenge => {
                let verifier_c: ArkFr = verifier.verifier_message();
                if JFr::from(verifier_c) != prover_challenges[challenge_idx] {
                    return Err(mismatch("Challenge", op_idx));
                }
                challenge_idx += 1;
            }
            Op::OptimizedChallenge => {
                if verifier.challenge_128() != prover_challenges[challenge_idx] {
                    return Err(mismatch("OptimizedChallenge", op_idx));
                }
                challenge_idx += 1;
            }
        }
    }

    verifier
        .check_eof()
        .map_err(|e| violation("check_eof", input.ops.len(), e))
}

fn run_check<H>(input: &Input, build_sponge: impl Fn() -> H) -> Result<(), CheckError>
where
    H: spongefish::DuplexSpongeInterface<U = u8>,
    ProverState<H, StdRng>: OptimizedChallenge,
    for<'a> VerifierState<'a, H>: OptimizedChallenge,
{
    let (narg, prover_challenges) = prover_run(input, input.instance, &build_sponge);

    replay(
        input,
        input.instance,
        &narg,
        &prover_challenges,
        &build_sponge,
    )?;

    let mut tampered = narg.clone();
    tampered.push(0xFF);
    if replay(
        input,
        input.instance,
        &tampered,
        &prover_challenges,
        &build_sponge,
    )
    .is_ok()
    {
        return Err(CheckError::Violation(InvariantViolation::with_details(
            "trailing-garbage NARG accepted".to_string(),
            "check_eof returned Ok on narg || 0xFF".to_string(),
        )));
    }

    // Domain separation: the same ops under a *different* instance digest must
    // derive *different* challenges — otherwise the instance isn't bound into
    // the sponge. The symmetry check above can't see a *symmetric* instance
    // drop: if both sides ignored the instance they would still agree.
    // (Skipped when the ops squeeze no challenges — nothing to compare.)
    if !prover_challenges.is_empty() {
        let mut other_instance = input.instance;
        other_instance[0] ^= 0xFF;
        let (_, other_challenges) = prover_run(input, other_instance, &build_sponge);
        if other_challenges == prover_challenges {
            return Err(CheckError::Violation(InvariantViolation::with_details(
                "instance digest not bound — distinct instances derive identical challenges"
                    .to_string(),
                "domain separation violated: prove(instance) == prove(instance ^ 0xFF)".to_string(),
            )));
        }
    }
    Ok(())
}

fn violation(what: &str, op_idx: usize, err: spongefish::VerificationError) -> CheckError {
    CheckError::Violation(InvariantViolation::with_details(
        format!("{what} failed on verifier"),
        format!("op_idx={op_idx}, err={err:?}"),
    ))
}

fn mismatch(what: &str, op_idx: usize) -> CheckError {
    CheckError::Violation(InvariantViolation::with_details(
        format!("{what} mismatch between prover and verifier"),
        format!("op_idx={op_idx}"),
    ))
}

fn seed_corpus_shared() -> Vec<Input> {
    let scalar = JFr::from_le_bytes_mod_order(&[0xABu8; 32]);
    let mut staged = vec![Op::PublicBytes(b"statement".to_vec())];
    for stage in 0u64..8 {
        staged.push(Op::PublicScalar(JFr::from(stage + 1)));
        staged.push(Op::ProverBytes(vec![stage as u8; (stage % 5 + 1) as usize]));
        for _ in 0..(stage % 3 + 1) {
            staged.push(Op::OptimizedChallenge);
        }
        staged.push(Op::ProverScalar(JFr::from(stage.wrapping_mul(0x9E37_79B9))));
        staged.push(Op::Challenge);
        staged.push(Op::PublicScalar(JFr::from(stage ^ 0xA5)));
    }

    let mut mixed_1k = Vec::with_capacity(1000);
    for i in 0..1000u64 {
        mixed_1k.push(match i % 6 {
            0 => Op::PublicBytes(vec![i as u8; (i % 13) as usize]),
            1 => Op::PublicScalar(JFr::from(i)),
            2 => Op::ProverBytes(vec![(i ^ 0x5A) as u8; (i % 11) as usize]),
            3 => Op::ProverScalar(JFr::from(i.wrapping_mul(2_654_435_761))),
            4 => Op::Challenge,
            _ => Op::OptimizedChallenge,
        });
    }

    let op_sequences: Vec<Vec<Op>> = vec![
        vec![],
        vec![Op::Challenge],
        vec![Op::PublicBytes(b"hello".to_vec())],
        vec![Op::PublicScalar(scalar)],
        vec![Op::ProverBytes(b"prover-data".to_vec())],
        vec![Op::ProverScalar(scalar)],
        vec![Op::OptimizedChallenge],
        vec![
            Op::ProverBytes(vec![0xAA; 48]),
            Op::ProverBytes(vec![]),
            Op::ProverBytes(vec![0xBB; 3]),
            Op::OptimizedChallenge,
        ],
        vec![
            Op::PublicBytes(b"setup".to_vec()),
            Op::ProverScalar(scalar),
            Op::Challenge,
            Op::ProverBytes(vec![1, 2, 3, 4, 5]),
            Op::OptimizedChallenge,
            Op::PublicScalar(scalar),
            Op::Challenge,
            Op::ProverScalar(JFr::from(42u64)),
            Op::OptimizedChallenge,
            Op::PublicBytes(vec![]),
        ],
        staged,
        mixed_1k,
    ];

    // Pair each op-sequence with a distinct instance digest: index 0 keeps the
    // degenerate all-zeros case; the rest are non-zero so the corpus exercises
    // instance binding (a zero-only fixture would mask a one-sided instance drop).
    op_sequences
        .into_iter()
        .enumerate()
        .map(|(i, ops)| Input {
            instance: [i as u8; 32],
            ops,
        })
        .collect()
}

fn description_for(label: &str) -> String {
    format!(
        "spongefish ProverState/VerifierState pair ({label} sponge) replaying \
         the same operation sequence must round-trip every prover message \
         and agree on every challenge; and the same ops under a different \
         instance digest must derive different challenges (the instance is \
         bound into the sponge)."
    )
}

/// Spongefish symmetry invariant for the Blake2b512 sponge.
#[jolt_eval_macros::invariant(Test, Fuzz, RedTeam)]
#[derive(Default)]
pub struct TranscriptConsistencyBlake2bInvariant;

impl Invariant for TranscriptConsistencyBlake2bInvariant {
    type Setup = ();
    type Input = Input;

    fn name(&self) -> &str {
        "transcript_prover_verifier_consistency_blake2b"
    }

    fn description(&self) -> String {
        description_for("Blake2b512")
    }

    fn setup(&self) {}

    fn check(&self, _setup: &(), input: Input) -> Result<(), CheckError> {
        run_check::<Blake2b512>(&input, Blake2b512::default)
    }

    fn seed_corpus(&self) -> Vec<Input> {
        seed_corpus_shared()
    }
}

/// Spongefish symmetry invariant for the Keccak sponge.
#[jolt_eval_macros::invariant(Test, Fuzz, RedTeam)]
#[derive(Default)]
pub struct TranscriptConsistencyKeccakInvariant;

impl Invariant for TranscriptConsistencyKeccakInvariant {
    type Setup = ();
    type Input = Input;

    fn name(&self) -> &str {
        "transcript_prover_verifier_consistency_keccak"
    }

    fn description(&self) -> String {
        description_for("Keccak")
    }

    fn setup(&self) {}

    fn check(&self, _setup: &(), input: Input) -> Result<(), CheckError> {
        run_check::<Keccak>(&input, Keccak::default)
    }

    fn seed_corpus(&self) -> Vec<Input> {
        seed_corpus_shared()
    }
}

/// Spongefish symmetry invariant for the Poseidon sponge.
#[jolt_eval_macros::invariant(Test, Fuzz, RedTeam)]
#[derive(Default)]
pub struct TranscriptConsistencyPoseidonInvariant;

impl Invariant for TranscriptConsistencyPoseidonInvariant {
    type Setup = ();
    type Input = Input;

    fn name(&self) -> &str {
        "transcript_prover_verifier_consistency_poseidon"
    }

    fn description(&self) -> String {
        description_for("Poseidon")
    }

    fn setup(&self) {}

    fn check(&self, _setup: &(), input: Input) -> Result<(), CheckError> {
        // Poseidon has no 128-bit `challenge_128` (it's `unimplemented!()`); its optimized
        // challenge IS the full-field one. Map `OptimizedChallenge -> Challenge` (don't drop
        // it) so the op still squeezes — keeping the domain-separation check alive.
        let ops = input
            .ops
            .into_iter()
            .map(|op| match op {
                Op::OptimizedChallenge => Op::Challenge,
                other => other,
            })
            .collect();
        let input = Input {
            instance: input.instance,
            ops,
        };
        run_check::<PoseidonSponge>(&input, PoseidonSponge::new)
    }

    fn seed_corpus(&self) -> Vec<Input> {
        seed_corpus_shared()
    }
}
