//! Full-wrapper input preparation shared by the prover and integration tests.

use common::jolt_device::JoltDevice;
use jolt_crypto::Bn254;
use jolt_field::{Fr, Ring};
use jolt_hyperkzg::{HyperKZGProverSetup, HyperKZGVerifierSetup, NoopVerifierObserver};
use jolt_poly::UnivariatePoly;
use jolt_r1cs::Variable;
use jolt_sumcheck::prover::ProveRounds;
use jolt_sumcheck::SumcheckError;
use jolt_transcript::Blake3Transcript;
use jolt_verifier::VerifierError;
use thiserror::Error;

use crate::hash_table::schedule::preamble;
use crate::hash_table::{
    HashTable, HashTableKey, JoltSchedule, LinkMap, PublicInputs, Recorded, RecordingTranscript,
    ScheduleError, StreamTermExporter, SymbolicSchedule, T1Challenges,
};
use crate::limb_table::digit_link::LinkMember;
use crate::limb_table::schedule::Layout as LimbTableLayout;
use crate::limb_table::stream::LimbTableKey;
use crate::profile::{ProfileError, WrapperProfile};
use crate::relation::{
    build_relation, generate_witness, Pcs, Preprocessing, Proof, Relation, RelationError, Vc,
    Witness,
};
use crate::relation_table::DoryScalarLinkProver;
use crate::spartan::{ChallengeDecoder, PublicChallenge, SharedWitnessColumn, SpartanError};
use crate::stream::{
    combine_packed_phases, commit_packed, commitment_prefix_challenges, prove_assembly,
    verify_assembly_with_cost, AssemblyStatement, Column, Commitment, PackedColumns, StageMember,
    StageResult, StreamError, Term, TermContext, TermExporter, TermObserver, VerifierCost,
    WrapperProof,
};

pub const DEFAULT_COMMON_LOG_ROWS: usize = 18;
pub const DEFAULT_PACKING_FACTOR: usize = 32;

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
    #[error("wrapper commitment phases do not match the verifier key")]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DoryLinkPlacement {
    pub challenge: usize,
    pub theta: usize,
    pub member: usize,
}

pub struct WrapLimbKey {
    table: LimbTableKey,
    group_offset: usize,
}

impl WrapLimbKey {
    pub fn new(table: LimbTableKey, group_offset: usize) -> Self {
        Self {
            table,
            group_offset,
        }
    }
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
        let links = LinkMap::new(table.schedule());
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
        self.table.schedule()
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
    limb: WrapLimbKey,
    dory_link: Option<DoryLinkPlacement>,
}

impl WrapVerifierKey {
    pub fn new(
        mut statement: AssemblyStatement,
        hash: WrapHashKey,
        hash_public: PublicInputs,
        limb: WrapLimbKey,
        dory_link: Option<DoryLinkPlacement>,
        mut pinned_commitments: Vec<(usize, Commitment)>,
    ) -> Self {
        pinned_commitments.extend(hash.pinned_commitments());
        pinned_commitments.extend(limb.table.pinned_commitments(limb.group_offset));
        statement.pinned_commitments = pinned_commitments;
        Self {
            statement,
            hash,
            hash_public,
            limb,
            dory_link,
        }
    }

    pub fn hash_schedule(&self) -> &SymbolicSchedule {
        self.hash.schedule()
    }

    pub fn hash_links(&self) -> &LinkMap {
        self.hash.links()
    }

    pub fn limb_layout(&self) -> &LimbTableLayout {
        self.limb.table.layout()
    }

    fn statement(&self, challenges: &[Fr]) -> Result<AssemblyStatement, WrapError> {
        self.statement_observed(challenges, &mut NoopVerifierObserver)
    }

    fn statement_observed(
        &self,
        challenges: &[Fr],
        observer: &mut dyn TermObserver,
    ) -> Result<AssemblyStatement, WrapError> {
        let schedule = self.hash.schedule();
        let end = self
            .hash
            .challenge_offset
            .checked_add(T1Challenges::count(schedule.log_rows))
            .ok_or(WrapError::T1MemberLayout)?;
        if challenges.get(self.hash.challenge_offset..end).is_none() {
            return Err(WrapError::T1MemberLayout);
        }
        let exporter = StreamTermExporter {
            log_rows: schedule.log_rows,
            challenge_offset: self.hash.challenge_offset,
            public: &self.hash_public,
            columns: &[],
            row_member: self.hash.members[0],
            wiring_member: self.hash.members[1],
        };
        let claims = exporter.input_claims(challenges, observer);
        let mut statement = self.statement.clone();
        for (member, claim) in self.hash.members.into_iter().zip(claims) {
            statement
                .members
                .get_mut(member)
                .ok_or(WrapError::T1MemberLayout)?
                .input_claim = claim;
        }
        if let Some(link) = self.dory_link {
            let rho = challenges
                .get(link.challenge)
                .copied()
                .ok_or(WrapError::T1MemberLayout)?;
            let theta = challenges
                .get(link.theta)
                .copied()
                .ok_or(WrapError::T1MemberLayout)?;
            let claim = crate::limb_table::stream::link_input_claim_with(
                Fr::from_u64(0),
                rho,
                theta,
                self.limb.table.layout(),
                &mut |a, b| observer.fr_mul(a, b),
            );
            statement
                .members
                .get_mut(link.member)
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

/// Ordered commitment phases and the challenges exposed so far.
pub struct WrapCommitments {
    phases: Vec<PackedColumns>,
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

impl WrapCommitments {
    pub fn new() -> Self {
        Self {
            phases: Vec::new(),
            challenges: Vec::new(),
        }
    }

    pub fn challenges(&self) -> &[Fr] {
        &self.challenges
    }

    pub fn commit(
        mut self,
        columns: &[Column],
        statement: &AssemblyStatement,
        setup: &HyperKZGProverSetup<Bn254>,
    ) -> Result<Self, WrapError> {
        let spec = statement
            .commitment_phases
            .get(self.phases.len())
            .ok_or(WrapError::CommitmentPhases)?;
        let packed = commit_packed(columns, statement.k, setup)?;
        if packed.layout.group_count != spec.group_count {
            return Err(WrapError::CommitmentPhases);
        }
        self.phases.push(packed);
        let phases = self
            .phases
            .iter()
            .zip(&statement.commitment_phases)
            .map(|(packed, spec)| (packed.commitments.as_slice(), spec.challenge_count))
            .collect::<Vec<_>>();
        self.challenges =
            commitment_prefix_challenges(&statement.key_digest, &statement.public_inputs, &phases);
        Ok(self)
    }

    pub fn finish(self, statement: &AssemblyStatement) -> Result<WrapCommitted, WrapError> {
        if self.phases.len() != statement.commitment_phases.len() {
            return Err(WrapError::CommitmentPhases);
        }
        Ok(WrapCommitted {
            packed: combine_packed_phases(self.phases)?,
            challenges: self.challenges,
        })
    }
}

impl Default for WrapCommitments {
    fn default() -> Self {
        Self::new()
    }
}

pub struct DoryLinkedProver {
    digit: LinkMember,
    scalar: DoryScalarLinkProver,
    digit_claim: Fr,
    scalar_claim: Fr,
    pending: Option<(UnivariatePoly<Fr>, UnivariatePoly<Fr>)>,
}

impl DoryLinkedProver {
    pub fn new(digit: LinkMember, scalar: DoryScalarLinkProver, input_claim: Fr) -> Self {
        let digit_claim = digit.input_claim();
        let scalar_claim = scalar.input_claim();
        assert_eq!(digit_claim - scalar_claim, input_claim);
        Self {
            digit,
            scalar,
            digit_claim,
            scalar_claim,
            pending: None,
        }
    }
}

impl ProveRounds<Fr> for DoryLinkedProver {
    fn num_rounds(&self) -> usize {
        self.digit.num_rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<Fr>,
        round: usize,
        previous_claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        if let Some(challenge) = bind {
            let (digit, scalar) = self
                .pending
                .take()
                .unwrap_or_else(|| unreachable!("a prior round supplies the bind polynomial"));
            self.digit_claim = digit.evaluate(challenge);
            self.scalar_claim = scalar.evaluate(challenge);
        }
        if self.digit_claim - self.scalar_claim != previous_claim {
            return Err(SumcheckError::RoundCheckFailed {
                round,
                expected: previous_claim,
                actual: self.digit_claim - self.scalar_claim,
            });
        }
        let digit = self.digit.prove_round(bind, round, self.digit_claim)?;
        let scalar = self.scalar.prove_round(bind, round, self.scalar_claim)?;
        let combined = &digit - &scalar;
        self.pending = Some((digit, scalar));
        Ok(combined)
    }

    fn finish_rounds(&mut self, bind: Fr) -> Result<(), SumcheckError<Fr>> {
        self.digit.finish_rounds(bind)?;
        self.scalar.finish_rounds(bind)
    }
}

pub struct NegatingTermExporter<'a>(pub &'a dyn TermExporter);

impl TermExporter for NegatingTermExporter<'_> {
    fn terms(&self, context: &TermContext<'_>) -> Vec<Term> {
        let mut terms = self.0.terms(context);
        for term in &mut terms {
            term.coefficient = -term.coefficient;
        }
        terms
    }

    fn terms_observed(
        &self,
        context: &TermContext<'_>,
        observer: &mut dyn TermObserver,
    ) -> Vec<Term> {
        let mut terms = self.0.terms_observed(context, observer);
        for term in &mut terms {
            term.coefficient = -term.coefficient;
        }
        terms
    }
}

/// Proves stage A only after every commitment phase has fixed the member challenges.
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
    let mut statement_cost = VerifierCost::default();
    let statement = key.statement_observed(&challenges, &mut statement_cost)?;
    let (results, mut cost) = verify_wrapped(&statement, proof, exporters, setup)?;
    cost.fr_mul += statement_cost.fr_mul;
    Ok((results, cost))
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
