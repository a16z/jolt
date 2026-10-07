//! `transcript_prover_verifier_consistency` — for each sponge, a NARG
//! `ProverTranscript` / `VerifierTranscript` pair driven by the same operation
//! sequence must round-trip every prover message, agree on every challenge,
//! and consume the argument string exactly.

use arbitrary::{Arbitrary, Result as ArbitraryResult, Unstructured};
use jolt_field::{CanonicalEncoding, Fr as JFr};
use jolt_transcript::{
    Blake2b512, Channel, Keccak, PoseidonSponge, ProtocolId, ProverTranscript, Sponge,
    TranscriptError, VerifierTranscript,
};

use crate::invariant::{CheckError, Invariant, InvariantViolation};

const SESSION: &[u8] = b"jolt-eval/transcript-symmetry/v1";
const PROTOCOL: &str = "jolt-eval/transcript-symmetry";
/// The longest byte message an [`Op::ProverBytes`] carries.
const MAX_PROVER_BYTES: usize = 64;

/// One operation in the prover/verifier sequence.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub enum Op {
    /// Both sides absorb the same public bytes.
    PublicBytes(Vec<u8>),
    /// Both sides absorb the same public BN254 `Fr` scalar.
    PublicScalar(#[schemars(with = "[u8; 32]")] JFr),
    /// Prover sends length-prefixed bytes; verifier reads them back.
    ProverBytes(Vec<u8>),
    /// Prover sends a BN254 `Fr` scalar; verifier reads it back.
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
    let len = u.int_in_range(0..=MAX_PROVER_BYTES)?;
    (0..len).map(|_| u.arbitrary()).collect()
}

fn arb_scalar(u: &mut Unstructured<'_>) -> ArbitraryResult<JFr> {
    let bytes: [u8; 32] = u.arbitrary()?;
    Ok(JFr::from_bytes_le_reduced(&bytes))
}

fn run_check<H: Sponge>(input: &Input) -> Result<(), CheckError> {
    let protocol = ProtocolId::new::<H>(PROTOCOL);
    let mut prover = ProverTranscript::<H>::new(&protocol, SESSION);
    let mut prover_challenges: Vec<[u8; 32]> = Vec::new();

    for (op_idx, op) in input.ops.iter().enumerate() {
        match op {
            Op::PublicBytes(b) => prover.public_bytes(b),
            Op::PublicScalar(f) => prover.public(f),
            Op::ProverBytes(b) => prover
                .send_bounded_bytes(b, MAX_PROVER_BYTES)
                .map_err(|e| violation("send_bounded_bytes", op_idx, e))?,
            Op::ProverScalar(f) => prover.send(f),
            Op::Challenge => prover_challenges.push(prover.challenge_bytes()),
        }
    }

    let narg = prover.finish();
    let mut verifier = VerifierTranscript::<H>::new(&protocol, SESSION, &narg);
    let mut challenge_idx = 0usize;

    for (op_idx, op) in input.ops.iter().enumerate() {
        match op {
            Op::PublicBytes(b) => verifier.public_bytes(b),
            Op::PublicScalar(f) => verifier.public(f),
            Op::ProverBytes(expected) => {
                let got = verifier
                    .receive_bounded_bytes(MAX_PROVER_BYTES)
                    .map_err(|e| violation("receive_bounded_bytes", op_idx, e))?;
                if got != expected.as_slice() {
                    return Err(mismatch("ProverBytes round-trip", op_idx));
                }
            }
            Op::ProverScalar(expected) => {
                let got: JFr = verifier
                    .receive()
                    .map_err(|e| violation("receive<Fr>", op_idx, e))?;
                if got != *expected {
                    return Err(mismatch("ProverScalar round-trip", op_idx));
                }
            }
            Op::Challenge => {
                let verifier_c: [u8; 32] = verifier.challenge_bytes();
                if verifier_c != prover_challenges[challenge_idx] {
                    return Err(mismatch("Challenge", op_idx));
                }
                challenge_idx += 1;
            }
        }
    }

    verifier
        .finish()
        .map_err(|e| violation("finish", input.ops.len(), e))?;
    Ok(())
}

fn violation(what: &str, op_idx: usize, err: TranscriptError) -> CheckError {
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
        "NARG prover/verifier transcript pair ({label} sponge) replaying \
         the same operation sequence must round-trip every prover message, \
         agree on every challenge, and consume the argument string exactly."
    )
}

/// Sponge selector for the merged fuzz target. The three sponges share one
/// transcript layer, so per-sponge fuzz targets duplicated coverage; one
/// target fuzzes all three with the fuzzer choosing the sponge.
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

/// Transcript symmetry invariant for the Blake2b512 sponge.
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

/// Transcript symmetry invariant for the Keccak sponge.
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

/// Transcript symmetry invariant for the Poseidon sponge.
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
