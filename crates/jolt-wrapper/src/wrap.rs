//! Full-wrapper input preparation shared by the prover and integration tests.

mod key;

use std::ops::Range;

use common::jolt_device::JoltDevice;
use jolt_crypto::Bn254;
use jolt_field::{CanonicalEncoding, Fr, Ring};
use jolt_hyperkzg::{HyperKZGProverSetup, HyperKZGVerifierSetup, NoopVerifierObserver};
use jolt_r1cs::Variable;
use jolt_transcript::Blake3Transcript;
use jolt_verifier::VerifierError;
use thiserror::Error;

use self::key::{build_key_assembly, hash_form_value};

use crate::hash_table::schedule::preamble;
use crate::hash_table::terms::AffineForm as HashAffineForm;
use crate::hash_table::{
    HashTable, HashTableKey, JoltSchedule, LinkMap, PublicInputs, Recorded, RecordingTranscript,
    ScheduleError, StreamTermExporter as HashStreamTermExporter, SymbolicSchedule, T1Challenges,
};
use crate::limb_table::columns::Columns as LimbColumns;
use crate::limb_table::schedule::Layout as LimbTableLayout;
use crate::limb_table::stream::{LimbTableKey, StreamTermExporter as LimbStreamTermExporter};
use crate::links::{
    batch_witnesses, CopyLink, CopyLinkTermExporter, CopyLinkTermSide, CopyLinkValueSource,
    CopyLinkWitness, DoryScalarLink, DoryScalarTermExporter, LinkError, WIRES,
};
use crate::profile::{ProfileError, WrapperProfile};
use crate::relation::{
    build_relation, generate_witness, Pcs, Preprocessing, Proof, Relation, RelationError, Vc,
    Witness,
};
use crate::stream::{
    assembly_transcript, combine_packed_phases, commit_packed, commitment_prefix_challenges,
    prove_spartan_assembly, verify_spartan_assembly_from_transcript, AssemblyStatement, Column,
    ColumnId, Commitment, CountingKeccakTranscript, PackedColumns, SpartanAssembly,
    SpartanVerifierAssembly, StageMember, StageResult, StreamError, Term, TermContext,
    TermExporter, TermObserver, VerifierCost, WrapperProof,
};
use crate::SpartanError;

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
    #[error("link: {0}")]
    Link(#[from] LinkError),
    #[error("Spartan: {0}")]
    Spartan(#[from] SpartanError),
    #[error("wrapper stream: {0}")]
    Stream(#[from] StreamError),
    #[error("packing factor must be a nonzero power of two, got {0}")]
    InvalidPacking(usize),
    #[error("T1 needs 2^{required} rows, common domain is 2^{configured}")]
    CommonRowDomain { required: usize, configured: usize },
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
    #[error("wrapper statement does not match its verifier key")]
    StatementMismatch,
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
struct DoryLinkPlacement {
    challenge: usize,
    theta: usize,
    member: usize,
}

struct WrapLimbKey {
    table: LimbTableKey,
}

impl WrapLimbKey {
    fn new(table: LimbTableKey) -> Self {
        Self { table }
    }
}

impl WrapHashKey {
    pub fn from_reference(
        preprocessing: &Preprocessing,
        public_io: &JoltDevice,
        proof: &Proof,
        config: WrapConfig,
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
            group_offset: 0,
            challenge_offset: 0,
            members: [0, 1],
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

#[derive(Clone)]
struct CopyExporterPlan {
    link: CopyLink,
    left: CopyLinkTermSide,
    right: CopyLinkTermSide,
    tau: Range<usize>,
    beta: usize,
    gamma: usize,
    weights: Range<usize>,
    member: usize,
}

#[derive(Clone)]
struct LimbExporterPlan {
    challenge_offset: usize,
    theta_offset: usize,
    rho_offset: usize,
    columns: Vec<ColumnId>,
    row_member: usize,
    link_member: usize,
}

#[derive(Clone)]
struct ScalarExporterPlan {
    rows: usize,
    positions: Vec<usize>,
    rho_offset: usize,
    wire: ColumnId,
    member: usize,
}

#[derive(Clone)]
struct WrapAssemblyPlan {
    hash_columns: Vec<ColumnId>,
    witness_column: ColumnId,
    carry_member: usize,
    copies: Vec<CopyExporterPlan>,
    limb: LimbExporterPlan,
    scalar: ScalarExporterPlan,
    max_factors: usize,
}

#[derive(Clone)]
enum LeftLinkValue {
    Hash(HashAffineForm),
    Zero,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum LimbLinkValue {
    Witness,
    Chunk(usize),
    Sign,
    Zero,
}

#[derive(Clone)]
struct CopyKey {
    link: CopyLink,
    left: [LeftLinkValue; WIRES],
    right: [LimbLinkValue; WIRES],
}

#[derive(Clone, Copy)]
enum CopyLinkValueInner<'a> {
    Hash(&'a HashAffineForm, &'a HashTable),
    Witness(&'a [Fr]),
    Chunk(&'a LimbColumns, usize),
    Sign(&'a [u8]),
    Zero(usize),
}

pub struct CopyLinkValue<'a> {
    source: CopyLinkValueInner<'a>,
}

impl CopyLinkValueSource for CopyLinkValue<'_> {
    fn len(&self) -> usize {
        match self.source {
            CopyLinkValueInner::Hash(_, table) => table.rows(),
            CopyLinkValueInner::Witness(values) => values.len(),
            CopyLinkValueInner::Chunk(columns, _) => columns.rows(),
            CopyLinkValueInner::Sign(values) => values.len(),
            CopyLinkValueInner::Zero(rows) => rows,
        }
    }

    fn value(&self, row: usize) -> Fr {
        match self.source {
            CopyLinkValueInner::Hash(form, table) => hash_form_value(form, table, row),
            CopyLinkValueInner::Witness(values) => values[row],
            CopyLinkValueInner::Chunk(columns, chunk) => columns.chunk(row, chunk),
            CopyLinkValueInner::Sign(values) => Fr::from_u64(u64::from(values[row])),
            CopyLinkValueInner::Zero(_) => Fr::from_u64(0),
        }
    }
}

pub type WrapCopyLinkWitness<'a> = CopyLinkWitness<CopyLinkValue<'a>, CopyLinkValue<'a>>;

/// Profile-fixed data consumed by wrapper verification.
pub struct WrapVerifierKey {
    statement: AssemblyStatement,
    hash: WrapHashKey,
    hash_public: PublicInputs,
    limb: WrapLimbKey,
    dory_link: Option<DoryLinkPlacement>,
    assembly: WrapAssemblyPlan,
    copies: Vec<CopyKey>,
    relation: Relation,
}

impl WrapVerifierKey {
    pub fn new(
        profile: &WrapperProfile,
        mut hash: WrapHashKey,
        hash_public: PublicInputs,
        limb_table: LimbTableKey,
        public_inputs: Vec<Fr>,
        setup: &HyperKZGProverSetup<Bn254>,
    ) -> Result<Self, WrapError> {
        if profile.digest()? != hash.profile_digest {
            return Err(WrapError::StatementMismatch);
        }
        let assembly = build_key_assembly(
            profile,
            &hash,
            &hash_public,
            limb_table,
            public_inputs,
            setup,
        )?;
        hash.challenge_offset = 0;
        Ok(Self {
            statement: assembly.statement,
            hash,
            hash_public,
            limb: assembly.limb,
            dory_link: Some(assembly.dory_link),
            assembly: assembly.plan,
            copies: assembly.copies,
            relation: assembly.relation,
        })
    }

    pub fn assembly_statement(&self) -> &AssemblyStatement {
        &self.statement
    }

    pub fn copy_count(&self) -> usize {
        self.copies.len()
    }

    pub fn copy_link(&self, index: usize) -> Option<&CopyLink> {
        self.copies.get(index).map(|copy| &copy.link)
    }

    pub fn copy_fixed_columns(&self) -> Vec<Column> {
        self.copies
            .iter()
            .flat_map(|copy| {
                copy.link
                    .left
                    .fixed_columns()
                    .into_iter()
                    .chain(copy.link.right.fixed_columns())
            })
            .map(Column::Fr)
            .collect()
    }

    fn copy_values<'a>(
        &'a self,
        index: usize,
        hash: &'a HashTable,
        witness: &'a [Fr],
        limb: &'a LimbColumns,
    ) -> Option<([CopyLinkValue<'a>; WIRES], [CopyLinkValue<'a>; WIRES])> {
        let copy = self.copies.get(index)?;
        let rows = witness.len();
        let left = copy.left.each_ref().map(|source| CopyLinkValue {
            source: match source {
                LeftLinkValue::Hash(form) => CopyLinkValueInner::Hash(form, hash),
                LeftLinkValue::Zero => CopyLinkValueInner::Zero(rows),
            },
        });
        let right = copy.right.map(|source| CopyLinkValue {
            source: match source {
                LimbLinkValue::Witness => CopyLinkValueInner::Witness(witness),
                LimbLinkValue::Chunk(chunk) => CopyLinkValueInner::Chunk(limb, chunk),
                LimbLinkValue::Sign => CopyLinkValueInner::Sign(&limb.flags),
                LimbLinkValue::Zero => CopyLinkValueInner::Zero(rows),
            },
        });
        Some((left, right))
    }

    pub fn copy_witnesses<'a>(
        &'a self,
        hash: &'a HashTable,
        witness: &'a [Fr],
        limb: &'a LimbColumns,
        challenges: &[(Fr, Fr)],
    ) -> Result<Vec<WrapCopyLinkWitness<'a>>, WrapError> {
        if challenges.len() != self.copies.len() {
            return Err(WrapError::T1MemberLayout);
        }
        let values = (0..self.copies.len())
            .map(|index| {
                self.copy_values(index, hash, witness, limb)
                    .ok_or(WrapError::T1MemberLayout)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(batch_witnesses(
            self.copies.iter().zip(values).zip(challenges).map(
                |((copy, (left, right)), &(beta, gamma))| (&copy.link, left, right, beta, gamma),
            ),
        )?)
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

    pub fn dory_scalar_link(&self, rho: Fr) -> DoryScalarLink<'_> {
        DoryScalarLink::new(
            self.assembly.scalar.rows,
            &self.assembly.scalar.positions,
            self.limb.table.layout(),
            rho,
        )
    }

    pub fn term_count(&self, context: &TermContext<'_>) -> usize {
        self.export_terms(context, &mut NoopVerifierObserver).len()
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
        let exporter = HashStreamTermExporter {
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

    fn export_terms(
        &self,
        context: &TermContext<'_>,
        observer: &mut dyn TermObserver,
    ) -> Vec<Term> {
        let plan = &self.assembly;
        let hash = HashStreamTermExporter {
            log_rows: self.hash.schedule().log_rows,
            challenge_offset: self.hash.challenge_offset,
            public: &self.hash_public,
            columns: &plan.hash_columns,
            row_member: self.hash.members[0],
            wiring_member: self.hash.members[1],
        };
        let mut terms = hash.terms_observed(context, observer);
        for copy in &plan.copies {
            let relation_weights = context.challenges[copy.weights.clone()]
                .try_into()
                .unwrap_or_else(|_| unreachable!("three copy-link weights"));
            let exporter = CopyLinkTermExporter {
                link: &copy.link,
                left: copy.left.clone(),
                right: copy.right.clone(),
                tau: &context.challenges[copy.tau.clone()],
                beta: context.challenges[copy.beta],
                gamma: context.challenges[copy.gamma],
                relation_weights,
                member_index: copy.member,
            };
            terms.extend(exporter.terms_observed(context, observer));
        }
        let limb = &plan.limb;
        let limb_exporter = LimbStreamTermExporter {
            layout: self.limb.table.layout(),
            challenge_offset: limb.challenge_offset,
            theta_offset: limb.theta_offset,
            rho_offset: limb.rho_offset,
            columns: &limb.columns,
            row_member: limb.row_member,
            link_member: limb.link_member,
        };
        terms.extend(limb_exporter.terms_observed(context, observer));
        let scalar = &plan.scalar;
        let scalar_link = DoryScalarLink::new(
            scalar.rows,
            &scalar.positions,
            self.limb.table.layout(),
            context.challenges[scalar.rho_offset],
        );
        let scalar_exporter = DoryScalarTermExporter {
            link: &scalar_link,
            wire: scalar.wire,
            member_index: scalar.member,
        };
        terms.extend(
            scalar_exporter
                .terms_observed(context, observer)
                .into_iter()
                .map(|mut term| {
                    term.coefficient = -term.coefficient;
                    term
                }),
        );
        terms
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
}

fn hash_public_statement(public: &PublicInputs) -> Vec<Fr> {
    let mut bytes = Vec::with_capacity(32 + public.tail.len());
    bytes.extend(public.state_in.iter().flat_map(|word| word.to_le_bytes()));
    bytes.extend_from_slice(&public.tail);
    bytes
        .chunks(16)
        .map(|chunk| {
            let mut packed = [0u8; 16];
            packed[..chunk.len()].copy_from_slice(chunk);
            Fr::from_u128_checked(u128::from_le_bytes(packed))
                .unwrap_or_else(|| unreachable!("128-bit statement chunk is canonical"))
        })
        .collect()
}

struct KeyTermExporter<'a> {
    key: &'a WrapVerifierKey,
}

impl TermExporter for KeyTermExporter<'_> {
    fn max_factors(&self) -> usize {
        self.key.assembly.max_factors
    }

    fn terms(&self, context: &TermContext<'_>) -> Vec<Term> {
        self.key.export_terms(context, &mut NoopVerifierObserver)
    }

    fn terms_observed(
        &self,
        context: &TermContext<'_>,
        observer: &mut dyn TermObserver,
    ) -> Vec<Term> {
        self.key.export_terms(context, observer)
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
/// Proves stage A only after every commitment phase has fixed the member challenges.
pub fn wrap(
    committed: WrapCommitted,
    key: &WrapVerifierKey,
    relation_witness: &Witness,
    members: Vec<StageMember<'_>>,
    setup: &HyperKZGProverSetup<Bn254>,
) -> Result<WrapperProof, WrapError> {
    let statement = key.statement(&committed.challenges)?;
    let exporter = KeyTermExporter { key };
    let public_columns = 1 + key.relation.public.num_public;
    let witness = relation_witness
        .values
        .get(public_columns..)
        .ok_or(WrapError::StatementMismatch)?;
    if relation_witness.values.get(1..public_columns)
        != statement
            .public_inputs
            .get(..key.relation.public.num_public)
    {
        return Err(WrapError::StatementMismatch);
    }
    Ok(prove_spartan_assembly(
        &committed.packed,
        &statement,
        members,
        &[&exporter],
        SpartanAssembly {
            matrices: &key.relation.matrices,
            public_inputs: &statement.public_inputs[..key.relation.public.num_public],
            witness,
            witness_column: key.assembly.witness_column,
            carry_member: key.assembly.carry_member,
        },
        setup,
    )?)
}

/// Verifies against commitments and the T1 schedule pinned during trusted setup.
pub fn verify_wrapped_with_key(
    key: &WrapVerifierKey,
    proof: &WrapperProof,
    setup: &HyperKZGVerifierSetup<Bn254>,
) -> Result<(Vec<StageResult>, VerifierCost), WrapError> {
    let commitments = key.full_commitments(&proof.commitments)?;
    let (mut transcript, challenges) = assembly_transcript::<CountingKeccakTranscript>(
        &key.statement.key_digest,
        &key.statement.public_inputs,
        &commitments,
        &key.statement,
    )?;
    let mut statement_cost = VerifierCost::default();
    let statement = key.statement_observed(&challenges, &mut statement_cost)?;
    let exporter = KeyTermExporter { key };
    Ok(verify_spartan_assembly_from_transcript(
        proof,
        &statement,
        &[&exporter],
        SpartanVerifierAssembly {
            matrices: &key.relation.matrices,
            public_inputs: &statement.public_inputs[..key.relation.public.num_public],
            witness_column: key.assembly.witness_column,
            carry_member: key.assembly.carry_member,
        },
        setup,
        (&mut transcript, &challenges, &commitments),
        statement_cost,
    )?)
}

/// Verified inputs needed before the T1/T2/R sumchecks can be assembled.
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

        let public_known = public_values(&relation, &relation_witness.values)?;

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

fn public_values(relation: &Relation, values: &[Fr]) -> Result<Vec<Fr>, WrapError> {
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
    Ok(known)
}
