//! `transcript_prover_verifier_consistency` — for each spongefish sponge,
//! a `ProverState` / `VerifierState` pair driven by the same operation
//! sequence must round-trip every prover message and produce the same
//! verifier challenges.

use arbitrary::{Arbitrary, Result as ArbitraryResult, Unstructured};
use jolt_field::{CanonicalEncoding, Fr as JFr};
use spongefish::instantiations::Keccak;
use spongefish::{DuplexSpongeInit, Encoding, VerificationError};

use jolt_transcript::{
    prover_transcript, verifier_transcript, Blake2b512, BytesMsg, ChallengeSponge, FieldMsg,
    PoseidonSponge, ProverTranscript, VerifierTranscript,
};

use crate::invariant::{CheckError, Invariant, InvariantViolation};

const SESSION: &[u8] = b"jolt-eval/transcript-symmetry/v1";
const INSTANCE_DIGEST: [u8; 32] = [0u8; 32];

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
}

/// Sequence of operations replayed in lockstep by both sides.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct Input {
    /// Operations to apply in order.
    pub ops: Vec<Op>,
}

impl<'a> Arbitrary<'a> for Input {
    fn arbitrary(u: &mut Unstructured<'a>) -> ArbitraryResult<Self> {
        let n = u.int_in_range(0u8..=20)? as usize;
        let mut ops = Vec::with_capacity(n);
        for _ in 0..n {
            let tag = u.int_in_range(0u8..=4)?;
            ops.push(match tag {
                0 => Op::PublicBytes(arb_bytes(u)?),
                1 => Op::PublicScalar(arb_scalar(u)?),
                2 => Op::ProverBytes(arb_bytes(u)?),
                3 => Op::ProverScalar(arb_scalar(u)?),
                _ => Op::Challenge,
            });
        }
        Ok(Self { ops })
    }
}

fn arb_bytes(u: &mut Unstructured<'_>) -> ArbitraryResult<Vec<u8>> {
    let len = u.int_in_range(0u8..=64)? as usize;
    (0..len).map(|_| u.arbitrary()).collect()
}

fn arb_scalar(u: &mut Unstructured<'_>) -> ArbitraryResult<JFr> {
    let bytes: [u8; 32] = u.arbitrary()?;
    Ok(JFr::from_bytes_le_reduced(&bytes))
}

/// Scalars travel as [`FieldMsg`] and challenges are one-element `Fr`
/// messages, so the same sequence runs on the byte sponges and on Poseidon,
/// which absorbs and squeezes `Fr` natively.
fn run_check<H>(input: &Input) -> Result<(), CheckError>
where
    H: DuplexSpongeInit + ChallengeSponge<JFr>,
    BytesMsg: Encoding<H::U>,
    FieldMsg<JFr>: Encoding<H::U>,
    (BytesMsg, [u8; 32]): Encoding<H::U>,
{
    let mut prover = prover_transcript::<H>(SESSION, INSTANCE_DIGEST);
    let mut prover_challenges: Vec<JFr> = Vec::new();

    for op in &input.ops {
        match op {
            Op::PublicBytes(b) => prover.public_message(&BytesMsg(b.clone())),
            Op::PublicScalar(f) => prover.public_message(&FieldMsg(*f)),
            Op::ProverBytes(b) => prover.prover_message(&BytesMsg(b.clone())),
            Op::ProverScalar(f) => prover.prover_message(&FieldMsg(*f)),
            Op::Challenge => {
                let c = ProverTranscript::field_challenges::<JFr>(&mut prover, 1)
                    .map_err(|e| prover_violation("field_challenges", e))?;
                prover_challenges.extend(c);
            }
        }
    }

    let narg: Vec<u8> = prover.narg_string().to_vec();
    let mut verifier = verifier_transcript::<H>(SESSION, INSTANCE_DIGEST, &narg);
    let mut challenge_idx = 0usize;

    for (op_idx, op) in input.ops.iter().enumerate() {
        match op {
            Op::PublicBytes(b) => verifier.public_message(&BytesMsg(b.clone())),
            Op::PublicScalar(f) => verifier.public_message(&FieldMsg(*f)),
            Op::ProverBytes(expected) => {
                let got: BytesMsg = verifier
                    .prover_message()
                    .map_err(|e| violation("prover_message<BytesMsg>", op_idx, e))?;
                if got.as_slice() != expected.as_slice() {
                    return Err(mismatch("ProverBytes round-trip", op_idx));
                }
            }
            Op::ProverScalar(expected) => {
                let got: FieldMsg<JFr> = verifier
                    .prover_message()
                    .map_err(|e| violation("prover_message<FieldMsg<Fr>>", op_idx, e))?;
                if got.0 != *expected {
                    return Err(mismatch("ProverScalar round-trip", op_idx));
                }
            }
            Op::Challenge => {
                let verifier_c = VerifierTranscript::field_challenges::<JFr>(&mut verifier, 1)
                    .map_err(|e| violation("field_challenges", op_idx, e))?;
                if verifier_c.first() != prover_challenges.get(challenge_idx) {
                    return Err(mismatch("Challenge", op_idx));
                }
                challenge_idx += 1;
            }
        }
    }

    verifier
        .check_eof()
        .map_err(|e| violation("check_eof", input.ops.len(), e))?;
    Ok(())
}

fn prover_violation(what: &str, err: VerificationError) -> CheckError {
    CheckError::Violation(InvariantViolation::with_details(
        format!("{what} failed on prover"),
        format!("err={err:?}"),
    ))
}

fn violation(what: &str, op_idx: usize, err: VerificationError) -> CheckError {
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
    let scalar = JFr::from_bytes_le_reduced(&[0xABu8; 32]);
    let mut mixed_1k = Vec::with_capacity(1000);
    for i in 0..1000u64 {
        mixed_1k.push(match i % 5 {
            0 => Op::PublicBytes(vec![i as u8; (i % 13) as usize]),
            1 => Op::PublicScalar(JFr::from(i)),
            2 => Op::ProverBytes(vec![(i ^ 0x5A) as u8; (i % 11) as usize]),
            3 => Op::ProverScalar(JFr::from(i.wrapping_mul(2_654_435_761))),
            _ => Op::Challenge,
        });
    }

    vec![
        Input { ops: vec![] },
        Input {
            ops: vec![Op::Challenge],
        },
        Input {
            ops: vec![Op::PublicBytes(b"hello".to_vec())],
        },
        Input {
            ops: vec![Op::PublicScalar(scalar)],
        },
        Input {
            ops: vec![Op::ProverBytes(b"prover-data".to_vec())],
        },
        Input {
            ops: vec![Op::ProverScalar(scalar)],
        },
        Input {
            ops: vec![
                Op::PublicBytes(b"setup".to_vec()),
                Op::ProverScalar(scalar),
                Op::Challenge,
                Op::ProverBytes(vec![1, 2, 3, 4, 5]),
                Op::Challenge,
                Op::PublicScalar(scalar),
                Op::Challenge,
                Op::ProverScalar(JFr::from(42u64)),
                Op::Challenge,
                Op::PublicBytes(vec![]),
            ],
        },
        Input { ops: mixed_1k },
    ]
}

fn description_for(label: &str) -> String {
    format!(
        "spongefish ProverState/VerifierState pair ({label} sponge) replaying \
         the same operation sequence must round-trip every prover message \
         and agree on every challenge."
    )
}

/// Sponge selector for the merged fuzz target. Blake2b512 and Keccak are
/// upstream spongefish instantiations over the same generic layer, so per-
/// sponge fuzz targets duplicated coverage; one target fuzzes all three with
/// the fuzzer choosing the sponge.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub enum SpongeKind {
    Blake2b,
    Keccak,
    Poseidon,
}

/// Input for the merged fuzz target: a sponge choice plus the op sequence.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct SpongeInput {
    pub sponge: SpongeKind,
    pub ops: Vec<Op>,
}

impl<'a> Arbitrary<'a> for SpongeInput {
    fn arbitrary(u: &mut Unstructured<'a>) -> ArbitraryResult<Self> {
        let sponge = match u.int_in_range(0u8..=2)? {
            0 => SpongeKind::Blake2b,
            1 => SpongeKind::Keccak,
            _ => SpongeKind::Poseidon,
        };
        let Input { ops } = Input::arbitrary(u)?;
        Ok(Self { sponge, ops })
    }
}

/// Merged fuzz-facing symmetry invariant over all three sponges.
///
/// The per-sponge invariants below keep their deterministic Test/RedTeam
/// coverage; this is the only one synthesized into a fuzz target.
#[jolt_eval_macros::invariant(Fuzz)]
#[derive(Default)]
pub struct TranscriptConsistencyInvariant;

impl Invariant for TranscriptConsistencyInvariant {
    type Setup = ();
    type Input = SpongeInput;

    fn name(&self) -> &str {
        "transcript_prover_verifier_consistency"
    }

    fn description(&self) -> String {
        description_for("fuzzer-selected")
    }

    fn setup(&self) {}

    fn check(&self, _setup: &(), input: SpongeInput) -> Result<(), CheckError> {
        let ops = Input { ops: input.ops };
        match input.sponge {
            SpongeKind::Blake2b => run_check::<Blake2b512>(&ops),
            SpongeKind::Keccak => run_check::<Keccak>(&ops),
            SpongeKind::Poseidon => run_check::<PoseidonSponge>(&ops),
        }
    }

    fn seed_corpus(&self) -> Vec<SpongeInput> {
        [
            SpongeKind::Blake2b,
            SpongeKind::Keccak,
            SpongeKind::Poseidon,
        ]
        .into_iter()
        .flat_map(|sponge| {
            seed_corpus_shared()
                .into_iter()
                .map(move |input| SpongeInput {
                    sponge,
                    ops: input.ops,
                })
        })
        .collect()
    }
}

#[jolt_eval_macros::invariant(Test, RedTeam)]
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
        run_check::<Blake2b512>(&input)
    }

    fn seed_corpus(&self) -> Vec<Input> {
        seed_corpus_shared()
    }
}

#[jolt_eval_macros::invariant(Test, RedTeam)]
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
        run_check::<Keccak>(&input)
    }

    fn seed_corpus(&self) -> Vec<Input> {
        seed_corpus_shared()
    }
}

#[jolt_eval_macros::invariant(Test, RedTeam)]
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
        run_check::<PoseidonSponge>(&input)
    }

    fn seed_corpus(&self) -> Vec<Input> {
        seed_corpus_shared()
    }
}
