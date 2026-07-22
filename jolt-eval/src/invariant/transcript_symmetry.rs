//! `transcript_prover_verifier_consistency` — for each spongefish sponge,
//! a `ProverState` / `VerifierState` pair driven by the same operation
//! sequence must round-trip every prover message and produce the same
//! verifier challenges.

use arbitrary::{Arbitrary, Unstructured};
use jolt_field::{CanonicalRepr, Fr as JFr};
use spongefish::instantiations::{Blake2b512, Keccak};

use jolt_transcript::{prover_transcript, verifier_transcript, BytesMsg, PoseidonSponge};

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
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
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

fn arb_bytes(u: &mut Unstructured<'_>) -> arbitrary::Result<Vec<u8>> {
    let len = u.int_in_range(0u8..=64)? as usize;
    (0..len).map(|_| u.arbitrary()).collect()
}

fn arb_scalar(u: &mut Unstructured<'_>) -> arbitrary::Result<JFr> {
    let bytes: [u8; 32] = u.arbitrary()?;
    Ok(JFr::from_le_bytes_mod_order(&bytes))
}

fn run_check<H>(input: &Input, build_sponge: impl Fn() -> H) -> Result<(), CheckError>
where
    H: spongefish::DuplexSpongeInterface<U = u8>,
{
    let mut prover = prover_transcript(SESSION, INSTANCE_DIGEST, build_sponge());
    let mut prover_challenges: Vec<[u8; 32]> = Vec::new();

    for op in &input.ops {
        match op {
            Op::PublicBytes(b) => prover.public_message(&BytesMsg(b.clone())),
            Op::PublicScalar(f) => prover.public_message(&scalar_bytes(*f)),
            Op::ProverBytes(b) => prover.prover_message(&BytesMsg(b.clone())),
            Op::ProverScalar(f) => prover.prover_message(&scalar_bytes(*f)),
            Op::Challenge => {
                let c: [u8; 32] = prover.verifier_message();
                prover_challenges.push(c);
            }
        }
    }

    let narg: Vec<u8> = prover.narg_string().to_vec();
    let mut verifier = verifier_transcript(SESSION, INSTANCE_DIGEST, build_sponge(), &narg);
    let mut challenge_idx = 0usize;

    for (op_idx, op) in input.ops.iter().enumerate() {
        match op {
            Op::PublicBytes(b) => verifier.public_message(&BytesMsg(b.clone())),
            Op::PublicScalar(f) => verifier.public_message(&scalar_bytes(*f)),
            Op::ProverBytes(expected) => {
                let got: BytesMsg = verifier
                    .prover_message()
                    .map_err(|e| violation("prover_message<BytesMsg>", op_idx, e))?;
                if got.as_slice() != expected.as_slice() {
                    return Err(mismatch("ProverBytes round-trip", op_idx));
                }
            }
            Op::ProverScalar(expected) => {
                let got: [u8; 32] = verifier
                    .prover_message()
                    .map_err(|e| violation("prover_message<[u8; 32]>", op_idx, e))?;
                if got != scalar_bytes(*expected) {
                    return Err(mismatch("ProverScalar round-trip", op_idx));
                }
            }
            Op::Challenge => {
                let verifier_c: [u8; 32] = verifier.verifier_message();
                if verifier_c != prover_challenges[challenge_idx] {
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

fn scalar_bytes(value: JFr) -> [u8; 32] {
    let mut out = [0u8; 32];
    value.to_bytes_le(&mut out);
    out
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
        run_check::<PoseidonSponge>(&input, PoseidonSponge::new)
    }

    fn seed_corpus(&self) -> Vec<Input> {
        seed_corpus_shared()
    }
}
