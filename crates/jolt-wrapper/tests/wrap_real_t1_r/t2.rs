use jolt_field::{Fr, One, Ring};
use jolt_hyperkzg::{NoopVerifierObserver, VerifierObserver};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::prover::ProveRounds;
use jolt_sumcheck::SumcheckError;
use jolt_wrapper::limb_table::adapter::{
    from_jolt, ordered_commitments, AdapterError, JoltDoryInputs,
};
use jolt_wrapper::limb_table::columns::{operand_columns, Columns, CHUNK_COLUMNS};
use jolt_wrapper::limb_table::digit_link::link_term;
use jolt_wrapper::limb_table::digit_link::LinkMember;
use jolt_wrapper::limb_table::dory::FlattenedCheck;
use jolt_wrapper::limb_table::export::{free_column, pin_columns, ClaimedColumns};
use jolt_wrapper::limb_table::layout::LOG_ROWS;
use jolt_wrapper::limb_table::lookup::{public_evals, LookupColumns, PublicColumns};
use jolt_wrapper::limb_table::relation::col::{CLAIMED, COMMITTED, PHASE1_END, PHASE2_END, WIDTH};
use jolt_wrapper::limb_table::relation::{eq_tau_column, Challenges, RowRelation, SLOTS};
use jolt_wrapper::limb_table::schedule::{build, Layout};
use jolt_wrapper::limb_table::terms::Term as T2Term;
use jolt_wrapper::limb_table::wiring::{copy_kernel_table, fingerprint_columns};
use jolt_wrapper::relation::{DoryScalar, Preprocessing, Proof, Relation, Witness};
use jolt_wrapper::relation_table::DoryScalarLinkProver;
use jolt_wrapper::stream::{
    AffineForm, Column, ColumnId, Term, TermContext, TermExporter, TermObserver,
};

pub const CHALLENGE_COUNT: usize = LOG_ROWS + 11;
pub const PHASE_ONE_COLUMNS: usize = PHASE1_END;
pub const PHASE_TWO_COLUMNS: usize = PHASE2_END - PHASE1_END;

pub fn retain_used_links(proof: &Proof, relation: &mut Relation) {
    let check = FlattenedCheck::derive(
        proof.joint_opening_proof.0.sigma,
        ordered_commitments(&proof.commitments).len(),
    );
    let used = check.wires();
    for (wire, _) in &mut relation.link.dory.scalars {
        *wire = t2_name(wire.clone());
    }
    let available = relation
        .link
        .dory
        .scalars
        .iter()
        .map(|(wire, _)| wire)
        .collect::<Vec<_>>();
    let missing = used
        .iter()
        .filter(|wire| !available.contains(wire))
        .collect::<Vec<_>>();
    assert!(missing.is_empty(), "T2 wires missing from R: {missing:?}");
    relation
        .link
        .dory
        .scalars
        .retain(|(wire, _)| used.contains(wire));
}

fn t2_name(wire: DoryScalar) -> DoryScalar {
    match wire {
        DoryScalar::Delta1R(index) => DoryScalar::Delta1R(index - 1),
        DoryScalar::Delta2R(index) => DoryScalar::Delta2R(index - 1),
        other => other,
    }
}

pub struct Base {
    pub inputs: JoltDoryInputs,
    pub layout: Layout,
    pub columns: Columns,
    pub public: PublicColumns,
}

impl Base {
    pub fn new(
        preprocessing: &Preprocessing,
        proof: &Proof,
        relation: &Relation,
        witness: &Witness,
    ) -> Result<Self, AdapterError> {
        let inputs = from_jolt(
            &preprocessing.pcs_setup,
            &proof.commitments,
            &proof.joint_opening_proof,
            &relation.link.dory,
            &witness.values,
        )?;
        let layout = build(
            &inputs.check,
            &inputs.values,
            &inputs.setup,
            &inputs.wire_order,
        );
        let coordinates = inputs.witness.coordinates_in(&layout.input_order);
        let values = layout
            .program
            .evaluate(&coordinates)
            .unwrap_or_else(|_| unreachable!("verified Dory inputs satisfy the limb program"));
        let columns = Columns::generate(&layout.program, &values, LOG_ROWS);
        let public = PublicColumns::new(&layout);
        Ok(Self {
            inputs,
            layout,
            columns,
            public,
        })
    }

    pub fn phase_one(&self) -> Vec<Column> {
        let mut columns = (0..CHUNK_COLUMNS)
            .map(|column| Column::U16(self.columns.chunk_column(column)))
            .collect::<Vec<_>>();
        columns.extend(self.public.digits.iter().cloned().map(Column::Bits));
        columns.push(Column::Fr(self.public.digit_values.clone()));
        let (m_pos, m_neg) = lookup_multiplicities(&self.public);
        columns.push(Column::U32(m_pos));
        columns.push(Column::U32(m_neg));
        columns.push(Column::U32(range_multiplicities(
            &self.columns,
            &self.public,
        )));
        assert_eq!(columns.len(), PHASE1_END);
        bit_reverse_columns(&mut columns);
        columns
    }

    pub fn vk(&self) -> Vec<Column> {
        let (pin, limbs) = pin_columns(&self.layout);
        let mut columns = vec![Column::Fr(pin)];
        columns.extend(limbs.into_iter().map(Column::Fr));
        columns.push(Column::Fr(free_column(&self.layout)));
        assert_eq!(columns.len(), CLAIMED - COMMITTED);
        bit_reverse_columns(&mut columns);
        columns
    }

    pub fn claimed(&self, relation: &RowRelation) -> ClaimedColumns {
        let z_xi = self.columns.xi_values(relation.challenges.xi);
        let operands = operand_columns(&self.layout.program, &z_xi, SLOTS);
        let fingerprints = fingerprint_columns(&self.layout.table_reads, &z_xi, relation);
        let lookup = LookupColumns::new(
            &self.public,
            &operands,
            &fingerprints.0,
            &fingerprints.1,
            relation,
        );
        let (helpers, multiplicities) = self
            .columns
            .logup_columns(relation.challenges.alpha, &self.public.digits);
        ClaimedColumns::assemble(
            &self.columns,
            &self.public,
            operands,
            helpers,
            multiplicities
                .into_iter()
                .map(|value| Fr::from_u64(u64::from(value)))
                .collect(),
            PublicColumns::inverse_table(relation.challenges.alpha),
            lookup,
            fingerprints,
            pin_columns(&self.layout),
            free_column(&self.layout),
        )
    }

    pub fn matrix(&self, relation: &RowRelation, claimed: &ClaimedColumns) -> Vec<Vec<Fr>> {
        let eq_tau = eq_tau_column(&relation.challenges.tau);
        let copy = copy_kernel_table(
            &self.layout.program,
            &self.public.kinds,
            &self.layout.table_reads,
            &eq_tau,
            relation,
        );
        let constancy = self.public.constancy_weights(&eq_tau);
        let (small, id) = PublicColumns::small_and_id();
        let mut columns = claimed.columns.clone();
        columns.extend([
            eq_tau,
            copy,
            self.public.sel.clone(),
            self.public.is_gt.clone(),
            self.public.is_g1.clone(),
            self.public.is_g2.clone(),
            self.public.s0.clone(),
            self.public.coord.clone(),
            constancy,
            small,
            id,
        ]);
        assert_eq!(columns.len(), WIDTH);
        columns
    }
}

pub fn bit_reverse_columns(columns: &mut [Column]) {
    for column in columns {
        match column {
            Column::Bits(values) => bit_reverse(values),
            Column::U16(values) => bit_reverse(values),
            Column::U32(values) => bit_reverse(values),
            Column::Fr(values) => bit_reverse(values),
        }
    }
}

fn bit_reverse<T>(values: &mut [T]) {
    assert!(values.len().is_power_of_two());
    let shift = usize::BITS - values.len().trailing_zeros();
    for index in 0..values.len() {
        let reversed = index.reverse_bits() >> shift;
        if index < reversed {
            values.swap(index, reversed);
        }
    }
}

pub fn challenges(values: &[Fr]) -> (Challenges, Fr) {
    assert_eq!(values.len(), CHALLENGE_COUNT);
    let tau = values[..LOG_ROWS].to_vec();
    let scalars = &values[LOG_ROWS..];
    (
        Challenges {
            tau,
            xi: scalars[0],
            alpha: scalars[1],
            gamma: scalars[2],
            lambda: scalars[3],
            beta: scalars[4],
            fp_root: scalars[5],
            fp_combine: scalars[6],
            lambda_lookup: scalars[7],
            copy_root: scalars[8],
            constancy_root: scalars[9],
        },
        scalars[10],
    )
}

pub fn column_ids(
    phase_one_base: usize,
    phase_two_base: usize,
    vk_base: usize,
    packing: usize,
) -> Vec<ColumnId> {
    let mut ids = vec![ColumnId { group: 0, slot: 0 }; CLAIMED];
    for (range, base) in [
        (0..PHASE1_END, phase_one_base),
        (PHASE1_END..PHASE2_END, phase_two_base),
        (COMMITTED..CLAIMED, vk_base),
    ] {
        for local in range {
            let physical = base + local
                - if local < PHASE1_END {
                    0
                } else if local < PHASE2_END {
                    PHASE1_END
                } else {
                    COMMITTED
                };
            ids[local] = ColumnId {
                group: physical / packing,
                slot: physical % packing,
            };
        }
    }
    ids
}

pub struct Exporter<'a> {
    pub layout: &'a Layout,
    pub relation: &'a RowRelation,
    pub columns: &'a [ColumnId],
    pub rho: Fr,
    pub row_member: usize,
    pub digit_member: usize,
}

impl Exporter<'_> {
    fn export(&self, context: &TermContext<'_>, observer: &mut dyn TermObserver) -> Vec<Term> {
        let tau = self
            .relation
            .challenges
            .tau
            .iter()
            .rev()
            .copied()
            .collect::<Vec<_>>();
        let public = {
            let mut bridge = ObserverBridge(observer);
            public_evals(
                self.layout,
                self.relation,
                &tau,
                context.row_point,
                &mut bridge,
            )
        };
        let row_scale = context.batching_coefficients[self.row_member];
        let mut terms = self
            .relation
            .terms(&public)
            .into_iter()
            .map(|term| self.map(term, row_scale, observer))
            .collect::<Vec<_>>();
        let omega = {
            let mut bridge = ObserverBridge(observer);
            jolt_wrapper::limb_table::lookup::omega_eval(
                self.layout,
                self.rho,
                context.row_point,
                &mut bridge,
            )
        };
        terms.push(self.map(
            link_term(omega),
            context.batching_coefficients[self.digit_member],
            observer,
        ));
        terms
    }

    fn map(&self, term: T2Term, scale: Fr, observer: &mut dyn TermObserver) -> Term {
        Term {
            coefficient: observer.fr_mul(scale, term.coefficient),
            factors: term
                .factors
                .into_iter()
                .map(|form| AffineForm {
                    constant: form.constant,
                    weights: form
                        .weights
                        .into_iter()
                        .map(|(column, weight)| (self.columns[column.0 as usize], weight))
                        .collect(),
                })
                .collect(),
        }
    }
}

impl TermExporter for Exporter<'_> {
    fn terms(&self, context: &TermContext<'_>) -> Vec<Term> {
        self.export(context, &mut NoopVerifierObserver)
    }

    fn terms_observed(
        &self,
        context: &TermContext<'_>,
        observer: &mut dyn TermObserver,
    ) -> Vec<Term> {
        self.export(context, observer)
    }
}

pub struct LinkProver {
    digit: LinkMember,
    scalar: DoryScalarLinkProver,
    digit_claim: Fr,
    scalar_claim: Fr,
    pending: Option<(UnivariatePoly<Fr>, UnivariatePoly<Fr>)>,
}

impl LinkProver {
    pub fn new(digit: LinkMember, scalar: DoryScalarLinkProver, expected: Fr) -> Self {
        let digit_claim = digit.input_claim();
        let scalar_claim = scalar.input_claim();
        assert_eq!(digit_claim - scalar_claim, expected);
        Self {
            digit,
            scalar,
            digit_claim,
            scalar_claim,
            pending: None,
        }
    }
}

impl ProveRounds<Fr> for LinkProver {
    fn num_rounds(&self) -> usize {
        LOG_ROWS
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

pub struct NegatingExporter<'a>(pub &'a dyn TermExporter);

impl TermExporter for NegatingExporter<'_> {
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

struct ObserverBridge<'a>(&'a mut dyn TermObserver);

impl VerifierObserver for ObserverBridge<'_> {
    fn ec_mul(&mut self, _count: usize) {}

    fn ec_add(&mut self, _count: usize) {}

    fn pairing_pairs(&mut self, _count: usize) {}

    fn record_fr_mul(&mut self) {
        let _ = self.0.fr_mul(Fr::one(), Fr::one());
    }

    fn record_fr_inv(&mut self) {}
}

fn lookup_multiplicities(public: &PublicColumns) -> (Vec<u32>, Vec<u32>) {
    let rows = 1usize << LOG_ROWS;
    let mut positive = vec![0u32; rows];
    let mut negative = vec![0u32; rows];
    for &(row, conjugated) in public.keys.iter().flatten() {
        if conjugated {
            negative[row as usize] += 1;
        } else {
            positive[row as usize] += 1;
        }
    }
    (positive, negative)
}

fn range_multiplicities(columns: &Columns, public: &PublicColumns) -> Vec<u32> {
    let mut multiplicities = vec![0u32; 1usize << LOG_ROWS];
    for row in 0..columns.rows() {
        for value in columns.chunks[row] {
            multiplicities[value as usize] += 1;
        }
        for bits in &public.digits {
            multiplicities[usize::from(bits[row])] += 1;
        }
    }
    multiplicities
}

#[test]
fn t2_consumed_scalars_match_the_relation_links() {
    use std::collections::HashSet;

    let (sigma, commitments) = (11, 41);
    let consumed = FlattenedCheck::derive(sigma, commitments)
        .wires()
        .into_iter()
        .collect::<HashSet<_>>();
    let mut relation = vec![DoryScalar::Evaluation];
    relation.extend((0..commitments).map(DoryScalar::CommitmentWeight));
    for round in 0..sigma {
        relation.extend([
            DoryScalar::Beta(round),
            DoryScalar::BetaInv(round),
            DoryScalar::Alpha(round),
            DoryScalar::AlphaInv(round),
        ]);
    }
    relation.extend([
        DoryScalar::Gamma,
        DoryScalar::GammaInv,
        DoryScalar::D,
        DoryScalar::DInv,
        DoryScalar::DSquared,
        DoryScalar::D2Init,
        DoryScalar::Chi(sigma),
    ]);
    for round in 0..sigma {
        relation.extend([
            DoryScalar::U(round),
            DoryScalar::V(round),
            DoryScalar::UAlpha(round),
            DoryScalar::VAlphaInv(round),
            DoryScalar::Chi(sigma - 1 - round),
            DoryScalar::Delta1R(sigma - round),
            DoryScalar::Delta2R(sigma - round),
        ]);
    }
    relation.extend([
        DoryScalar::S1Acc,
        DoryScalar::S2Acc,
        DoryScalar::Ht,
        DoryScalar::PairingG2ZeroScalar,
        DoryScalar::PairingG1ZeroScalar,
    ]);
    let relation = relation.into_iter().map(t2_name).collect::<HashSet<_>>();
    let missing = consumed.difference(&relation).collect::<Vec<_>>();
    assert!(missing.is_empty(), "T2 inputs missing R links: {missing:?}");
    assert_eq!(consumed.len(), 172);
    assert_eq!(relation.len(), 175);
    assert_eq!(
        relation
            .difference(&consumed)
            .cloned()
            .collect::<HashSet<_>>(),
        HashSet::from([DoryScalar::Chi(sigma), DoryScalar::S1Acc, DoryScalar::S2Acc,])
    );
}
