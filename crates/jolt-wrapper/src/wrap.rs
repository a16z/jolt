//! Full-wrapper input preparation shared by the prover and integration tests.

use common::jolt_device::JoltDevice;
use jolt_crypto::Bn254;
use jolt_field::Fr;
use jolt_hyperkzg::{HyperKZGProverSetup, HyperKZGVerifierSetup};
use jolt_r1cs::Variable;
use jolt_transcript::Blake3Transcript;
use jolt_verifier::VerifierError;
use thiserror::Error;

use crate::hash_table::schedule::preamble;
use crate::hash_table::{
    HashTable, HashTableKey, JoltSchedule, LinkMap, PublicInputs, Recorded, RecordingTranscript,
    ScheduleError, SymbolicSchedule, T1Challenges,
};
use crate::profile::{ProfileError, WrapperProfile};
use crate::relation::{
    build_relation, generate_witness, Pcs, Preprocessing, Proof, Relation, RelationError, Vc,
    Witness,
};
use crate::spartan::{ChallengeDecoder, PublicChallenge, SharedWitnessColumn, SpartanError};
use crate::stream::{
    combine_packed_phases, commit_packed, commitment_prefix_challenges, prove_assembly,
    verify_assembly_with_cost, AssemblyStatement, Column, Commitment, PackedColumns, StageMember,
    StageResult, StreamError, TermExporter, VerifierCost, WrapperProof,
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
    OriginalProof(#[from] VerifierError),
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
    #[error("wrapper assembly requires exactly two commitment phases")]
    CommitmentPhases,
    #[error("T1 verifier-key member layout is invalid")]
    T1MemberLayout,
    #[error("proof profile does not match the wrapper verifier key")]
    ProfileMismatch,
}

/// T1 schedule and public-input derivation fixed during trusted setup.
#[derive(Clone)]
pub struct WrapHashKey {
    profile_digest: [u8; 32],
    table: HashTableKey,
    links: LinkMap,
    group_offset: usize,
    challenge_offset: usize,
    members: [usize; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct T1Placement {
    pub group_offset: usize,
    pub challenge_offset: usize,
    pub members: [usize; 2],
}

impl WrapHashKey {
    pub fn from_reference(
        preprocessing: &Preprocessing,
        public_io: &JoltDevice,
        proof: &Proof,
        config: WrapConfig,
        placement: T1Placement,
        setup: &HyperKZGProverSetup<Bn254>,
    ) -> Result<Self, WrapError> {
        validate_config(config)?;
        let profile = WrapperProfile::new(preprocessing, proof)?;
        let profile_digest = profile.digest()?;
        let records = verified_transcript(preprocessing, public_io, proof)?;
        let natural = SymbolicSchedule::from_reference(&records, None)?;
        if natural.log_rows > config.common_log_rows {
            return Err(WrapError::CommonRowDomain {
                required: natural.log_rows,
                configured: config.common_log_rows,
            });
        }
        let schedule = if natural.log_rows == config.common_log_rows {
            natural
        } else {
            SymbolicSchedule::from_reference(&records, Some(config.common_log_rows))?
        };
        let table = HashTableKey::new(schedule, config.packing_factor, setup)?;
        let links = LinkMap::new(&table.schedule);
        Ok(Self {
            profile_digest,
            table,
            links,
            group_offset: placement.group_offset,
            challenge_offset: placement.challenge_offset,
            members: placement.members,
        })
    }

    pub fn schedule(&self) -> &SymbolicSchedule {
        &self.table.schedule
    }

    pub fn links(&self) -> &LinkMap {
        &self.links
    }

    pub fn pinned_commitments(&self) -> Vec<(usize, Commitment)> {
        self.table.pinned_commitments(self.group_offset)
    }
}

/// Profile-fixed data consumed by wrapper verification.
pub struct WrapVerifierKey {
    statement: AssemblyStatement,
    hash: WrapHashKey,
    hash_public: PublicInputs,
}

impl WrapVerifierKey {
    pub fn new(
        mut statement: AssemblyStatement,
        hash: WrapHashKey,
        hash_public: PublicInputs,
        mut pinned_commitments: Vec<(usize, Commitment)>,
    ) -> Self {
        pinned_commitments.extend(hash.pinned_commitments());
        statement.pinned_commitments = pinned_commitments;
        Self {
            statement,
            hash,
            hash_public,
        }
    }

    pub fn hash_schedule(&self) -> &SymbolicSchedule {
        self.hash.schedule()
    }

    pub fn hash_links(&self) -> &LinkMap {
        self.hash.links()
    }

    fn statement(&self, challenges: &[Fr]) -> Result<AssemblyStatement, WrapError> {
        let schedule = self.hash.schedule();
        let count = T1Challenges::count(schedule.log_rows);
        let end = self
            .hash
            .challenge_offset
            .checked_add(count)
            .ok_or(WrapError::T1MemberLayout)?;
        let t1 = challenges
            .get(self.hash.challenge_offset..end)
            .ok_or(WrapError::T1MemberLayout)?;
        let claims =
            T1Challenges::from_challenges(t1, schedule.log_rows).input_claims(&self.hash_public);
        let mut statement = self.statement.clone();
        for (member, claim) in self.hash.members.into_iter().zip(claims) {
            statement
                .members
                .get_mut(member)
                .ok_or(WrapError::T1MemberLayout)?
                .input_claim = claim;
        }
        Ok(statement)
    }

    fn full_commitments(&self, wire: &[Commitment]) -> Result<Vec<Commitment>, WrapError> {
        let groups = self
            .statement
            .commitment_phases
            .iter()
            .map(|phase| phase.group_count)
            .sum();
        let mut wire = wire.iter();
        let mut full = Vec::with_capacity(groups);
        for group in 0..groups {
            let commitment = self
                .statement
                .pinned_commitments
                .iter()
                .find_map(|&(pinned, commitment)| (pinned == group).then_some(commitment))
                .or_else(|| wire.next().copied())
                .ok_or(WrapError::CommitmentPhases)?;
            full.push(commitment);
        }
        if wire.next().is_some() {
            return Err(WrapError::CommitmentPhases);
        }
        Ok(full)
    }

    fn challenges(&self, commitments: &[Commitment]) -> Result<Vec<Fr>, WrapError> {
        let mut start = 0usize;
        let mut phases = Vec::with_capacity(self.statement.commitment_phases.len());
        for phase in &self.statement.commitment_phases {
            let end = start
                .checked_add(phase.group_count)
                .ok_or(WrapError::CommitmentPhases)?;
            phases.push((
                commitments
                    .get(start..end)
                    .ok_or(WrapError::CommitmentPhases)?,
                phase.challenge_count,
            ));
            start = end;
        }
        if start != commitments.len() {
            return Err(WrapError::CommitmentPhases);
        }
        Ok(commitment_prefix_challenges(
            &self.statement.key_digest,
            &self.statement.public_inputs,
            &phases,
        ))
    }
}

/// Phase-1 commitments and their Fiat–Shamir challenges.
pub struct WrapPhaseOne {
    packed: PackedColumns,
    challenges: Vec<Fr>,
}

/// Both commitment phases and all pre-stage Fiat–Shamir challenges.
pub struct WrapCommitted {
    packed: PackedColumns,
    challenges: Vec<Fr>,
}

impl WrapCommitted {
    pub fn challenges(&self) -> &[Fr] {
        &self.challenges
    }
}

impl WrapPhaseOne {
    pub fn challenges(&self) -> &[Fr] {
        &self.challenges
    }
}

/// Commits phase 1 before exposing the challenges used to construct table members.
pub fn commit_wrap_phase_one(
    columns: &[Column],
    key: &WrapVerifierKey,
    setup: &HyperKZGProverSetup<Bn254>,
) -> Result<WrapPhaseOne, WrapError> {
    let [phase_one, _] = key.statement.commitment_phases.as_slice() else {
        return Err(WrapError::CommitmentPhases);
    };
    let packed = commit_packed(columns, key.statement.k, setup)?;
    if packed.layout.group_count != phase_one.group_count {
        return Err(WrapError::CommitmentPhases);
    }
    let challenges = commitment_prefix_challenges(
        &key.statement.key_digest,
        &key.statement.public_inputs,
        &[(&packed.commitments, phase_one.challenge_count)],
    );
    Ok(WrapPhaseOne { packed, challenges })
}

/// Commits phase 2 after helper construction and exposes the remaining member challenges.
pub fn commit_wrap_phase_two(
    phase_one: WrapPhaseOne,
    phase_two_columns: &[Column],
    key: &WrapVerifierKey,
    setup: &HyperKZGProverSetup<Bn254>,
) -> Result<WrapCommitted, WrapError> {
    let [_, phase_two_spec] = key.statement.commitment_phases.as_slice() else {
        return Err(WrapError::CommitmentPhases);
    };
    let phase_two = commit_packed(phase_two_columns, key.statement.k, setup)?;
    if phase_two.layout.group_count != phase_two_spec.group_count {
        return Err(WrapError::CommitmentPhases);
    }
    let packed = combine_packed_phases(vec![phase_one.packed, phase_two])?;
    let challenges = key.challenges(&packed.commitments)?;
    Ok(WrapCommitted { packed, challenges })
}

/// Proves stage A only after both commitment phases have fixed every member challenge.
pub fn wrap(
    committed: WrapCommitted,
    key: &WrapVerifierKey,
    members: &mut [StageMember<'_>],
    exporters: &[&dyn TermExporter],
    setup: &HyperKZGProverSetup<Bn254>,
) -> Result<WrapperProof, WrapError> {
    let statement = key.statement(&committed.challenges)?;
    Ok(prove_assembly(
        &committed.packed,
        &statement,
        members,
        exporters,
        setup,
    )?)
}

/// Verifies the generic member list and returns execution-derived EVM
/// operation counts with the stage results.
pub fn verify_wrapped(
    statement: &AssemblyStatement,
    proof: &WrapperProof,
    exporters: &[&dyn TermExporter],
    setup: &HyperKZGVerifierSetup<Bn254>,
) -> Result<(Vec<StageResult>, VerifierCost), WrapError> {
    Ok(verify_assembly_with_cost(
        proof, statement, exporters, setup,
    )?)
}

/// Verifies against commitments and the T1 schedule pinned during trusted setup.
pub fn verify_wrapped_with_key(
    key: &WrapVerifierKey,
    proof: &WrapperProof,
    exporters: &[&dyn TermExporter],
    setup: &HyperKZGVerifierSetup<Bn254>,
) -> Result<(Vec<StageResult>, VerifierCost), WrapError> {
    let commitments = key.full_commitments(&proof.commitments)?;
    let challenges = key.challenges(&commitments)?;
    let statement = key.statement(&challenges)?;
    verify_wrapped(&statement, proof, exporters, setup)
}

/// Verified inputs needed before the T1/T2/Spartan sumchecks can be assembled.
pub struct WrapPreparation {
    pub config: WrapConfig,
    pub profile: WrapperProfile,
    pub profile_digest: [u8; 32],
    pub relation: Relation,
    pub relation_witness: Witness,
    /// The verifier-key schedule (here from this run; key generation derives
    /// it once per profile with `SymbolicSchedule::from_reference`).
    pub hash_key: SymbolicSchedule,
    pub hash_public: PublicInputs,
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
        hash_key: &WrapHashKey,
    ) -> Result<Self, WrapError> {
        validate_config(config)?;
        let profile = WrapperProfile::new(preprocessing, proof)?;
        let profile_digest = profile.digest()?;
        if profile_digest != hash_key.profile_digest {
            return Err(WrapError::ProfileMismatch);
        }
        let relation = build_relation(&profile)?;
        let relation_witness = generate_witness(&profile, preprocessing, public_io, proof)?;
        relation
            .matrices
            .check_witness(&relation_witness.values)
            .map_err(WrapError::UnsatisfiedRelation)?;

        let records = verified_transcript(preprocessing, public_io, proof)?;
        let schedule = hash_key.schedule();
        let hash_public = PublicInputs::from_preamble(&preamble(&records), schedule)?;
        let hash_schedule = JoltSchedule::witness(&records, schedule)?;
        let hash_table = HashTable::build(&hash_schedule, &hash_public);

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
            hash_key: schedule.clone(),
            hash_public,
            hash_schedule,
            hash_table,
            public_known,
            public_challenges,
            shared_witness,
        })
    }
}

fn validate_config(config: WrapConfig) -> Result<(), WrapError> {
    if !config.packing_factor.is_power_of_two() {
        return Err(WrapError::InvalidPacking(config.packing_factor));
    }
    Ok(())
}

fn verified_transcript(
    preprocessing: &Preprocessing,
    public_io: &JoltDevice,
    proof: &Proof,
) -> Result<Vec<Recorded>, WrapError> {
    let _ = RecordingTranscript::<Blake3Transcript>::take_log();
    jolt_verifier::verify::<Fr, Pcs, Vc, RecordingTranscript<Blake3Transcript>>(
        preprocessing,
        public_io,
        proof,
        None,
    )?;
    Ok(RecordingTranscript::<Blake3Transcript>::take_log())
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
