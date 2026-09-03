//! Full-wrapper input preparation shared by the prover and integration tests.

use common::jolt_device::JoltDevice;
use jolt_crypto::Bn254;
use jolt_field::Fr;
use jolt_hyperkzg::{HyperKZGProverSetup, HyperKZGVerifierSetup};
use jolt_r1cs::Variable;
use jolt_transcript::Blake3Transcript;
use thiserror::Error;

use crate::hash_table::{HashTable, JoltSchedule, RecordingTranscript, ScheduleError};
use crate::profile::{ProfileError, WrapperProfile};
use crate::relation::{
    build_relation, generate_witness, Preprocessing, Proof, Relation, RelationError, Witness,
};
use crate::spartan::{ChallengeDecoder, PublicChallenge, SharedWitnessColumn, SpartanError};
use crate::stream::{
    commit_packed, prove_assembly, verify_assembly_with_cost, AssemblyStatement, Column,
    StageMember, StageResult, StreamError, VerifierCost, WrapperProof,
};

pub const DEFAULT_COMMON_LOG_ROWS: usize = 18;
pub const DEFAULT_PACKING_FACTOR: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WrapConfig {
    pub common_log_rows: usize,
    pub packing_factor: usize,
}

impl Default for WrapConfig {
    fn default() -> Self {
        Self {
            common_log_rows: DEFAULT_COMMON_LOG_ROWS,
            packing_factor: DEFAULT_PACKING_FACTOR,
        }
    }
}

#[derive(Debug, Error)]
pub enum WrapError {
    #[error("wrapper profile: {0}")]
    Profile(#[from] ProfileError),
    #[error("original proof: {0}")]
    OriginalProof(#[from] jolt_verifier::VerifierError),
    #[error("transcript table: {0}")]
    Schedule(#[from] ScheduleError),
    #[error("verifier relation: {0}")]
    Relation(#[from] RelationError),
    #[error("Spartan witness: {0}")]
    Spartan(#[from] SpartanError),
    #[error("wrapper stream: {0}")]
    Stream(#[from] StreamError),
    #[error("packing factor must be a nonzero power of two, got {0}")]
    InvalidPacking(usize),
    #[error("T1 needs 2^{required} rows, common domain is 2^{configured}")]
    CommonRowDomain { required: usize, configured: usize },
    #[error("common row exponent {0} does not fit usize")]
    CommonRowExponent(usize),
    #[error("relation witness fails at row {0}")]
    UnsatisfiedRelation(usize),
    #[error("relation witness is missing variable {0}")]
    MissingWitness(usize),
}

/// Commits the adapters' common-domain columns and proves their stage-A
/// members, stage-B reductions, and one final HyperKZG opening.
pub fn wrap<F>(
    columns: &[Column],
    statement: &AssemblyStatement,
    members: &mut [StageMember<'_>],
    setup: &HyperKZGProverSetup<Bn254>,
    final_claims: F,
) -> Result<WrapperProof, WrapError>
where
    F: FnOnce(&StageResult, &[Fr]) -> Result<Vec<Fr>, StreamError>,
{
    let packed = commit_packed(columns, statement.k, setup)?;
    Ok(prove_assembly(
        &packed,
        statement,
        members,
        setup,
        final_claims,
    )?)
}

/// Verifies the generic member list and returns execution-derived EVM
/// operation counts with the stage results.
pub fn verify_wrapped<F>(
    statement: &AssemblyStatement,
    proof: &WrapperProof,
    setup: &HyperKZGVerifierSetup<Bn254>,
    final_claims: F,
) -> Result<(Vec<StageResult>, VerifierCost), WrapError>
where
    F: Fn(&StageResult, &[Fr], &mut VerifierCost) -> Result<Vec<Fr>, StreamError>,
{
    Ok(verify_assembly_with_cost(
        proof,
        statement,
        setup,
        final_claims,
    )?)
}

/// Verified inputs needed before the T1/T2/Spartan sumchecks can be assembled.
pub struct WrapPreparation {
    pub config: WrapConfig,
    pub profile: WrapperProfile,
    pub profile_digest: [u8; 32],
    pub relation: Relation,
    pub relation_witness: Witness,
    pub hash_schedule: JoltSchedule,
    pub hash_table: HashTable,
    pub public_known: Vec<Fr>,
    pub public_challenges: Vec<PublicChallenge>,
    pub shared_witness: SharedWitnessColumn,
}

impl WrapPreparation {
    pub fn new(
        preprocessing: &Preprocessing,
        public_io: &JoltDevice,
        proof: &Proof,
        config: WrapConfig,
    ) -> Result<Self, WrapError> {
        if !config.packing_factor.is_power_of_two() {
            return Err(WrapError::InvalidPacking(config.packing_factor));
        }
        let profile = WrapperProfile::new(preprocessing, proof)?;
        let profile_digest = profile.digest()?;
        let relation = build_relation(&profile)?;
        let relation_witness = generate_witness(&profile, preprocessing, public_io, proof)?;
        relation
            .matrices
            .check_witness(&relation_witness.values)
            .map_err(WrapError::UnsatisfiedRelation)?;

        let _ = RecordingTranscript::<Blake3Transcript>::take_log();
        jolt_verifier::verify::<
            Fr,
            crate::relation::Pcs,
            crate::relation::Vc,
            RecordingTranscript<Blake3Transcript>,
        >(preprocessing, public_io, proof, None)?;
        let records = RecordingTranscript::<Blake3Transcript>::take_log();
        let natural = JoltSchedule::new(&records, None)?;
        if natural.symbolic.log_rows > config.common_log_rows {
            return Err(WrapError::CommonRowDomain {
                required: natural.symbolic.log_rows,
                configured: config.common_log_rows,
            });
        }
        let hash_schedule = if natural.symbolic.log_rows == config.common_log_rows {
            natural
        } else {
            JoltSchedule::new(&records, Some(config.common_log_rows))?
        };
        let hash_table = HashTable::build(&hash_schedule);

        let (public_known, public_challenges) = public_values(&relation, &relation_witness.values)?;
        let private_start = 1 + relation.public.num_public;
        let private_witness = relation_witness
            .values
            .get(private_start..)
            .ok_or(WrapError::MissingWitness(private_start))?;
        let common_rows = 1usize
            .checked_shl(config.common_log_rows as u32)
            .ok_or(WrapError::CommonRowExponent(config.common_log_rows))?;
        let shared_witness = SharedWitnessColumn::new(private_witness, common_rows)?;

        Ok(Self {
            config,
            profile,
            profile_digest,
            relation,
            relation_witness,
            hash_schedule,
            hash_table,
            public_known,
            public_challenges,
            shared_witness,
        })
    }
}

fn public_values(
    relation: &Relation,
    values: &[Fr],
) -> Result<(Vec<Fr>, Vec<PublicChallenge>), WrapError> {
    let read = |variable: Variable| {
        values
            .get(variable.index())
            .copied()
            .ok_or(WrapError::MissingWitness(variable.index()))
    };
    let mut known = Vec::with_capacity(7);
    known.push(read(relation.public.val_io)?);
    known.push(read(relation.public.init_eval)?);
    for variable in relation.public.stage_values {
        known.push(read(variable)?);
    }
    let outputs = &relation.public.outputs;
    let mut challenges = Vec::with_capacity(relation.public.num_public - known.len());
    for &variable in outputs.ram_address.iter().chain(&outputs.bytecode_address) {
        challenges.push(PublicChallenge {
            value: read(variable)?,
            decoder: ChallengeDecoder::Challenge125,
        });
    }
    for &variable in &outputs.bytecode_gammas {
        challenges.push(PublicChallenge {
            value: read(variable)?,
            decoder: ChallengeDecoder::Scalar128,
        });
    }
    for &variable in &outputs.register_address {
        challenges.push(PublicChallenge {
            value: read(variable)?,
            decoder: ChallengeDecoder::Challenge125,
        });
    }
    Ok((known, challenges))
}
