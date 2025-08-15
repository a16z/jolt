//! This is a port of the sumcheck-based batch opening proof protocol implemented
//! in Nova: https://github.com/microsoft/Nova/blob/2772826ba296b66f1cd5deecf7aca3fd1d10e1f4/src/spartan/snark.rs#L410-L424
//! and such code is Copyright (c) Microsoft Corporation.
//! For additively homomorphic commitment schemes (including Zeromorph, HyperKZG) we
//! can use a sumcheck to reduce multiple opening proofs (multiple polynomials, not
//! necessarily of the same size, each opened at a different point) into a single opening.

use num_derive::FromPrimitive;
use rayon::prelude::*;
use std::{
    collections::{BTreeMap, HashMap},
    sync::{Arc, RwLock},
};

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use super::{
    commitment::commitment_scheme::CommitmentScheme,
    eq_poly::EqPolynomial,
    multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
    split_eq_poly::GruenSplitEqPolynomial,
};
use crate::{
    field::JoltField,
    poly::{
        multilinear_polynomial::PolynomialEvaluation,
        one_hot_polynomial::{OneHotPolynomialProverOpening, OneHotSumcheckState},
    },
    subprotocols::sumcheck::{BatchedSumcheck, SumcheckInstance, SumcheckInstanceProof},
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
    zkvm::witness::{CommittedPolynomial, VirtualPolynomial},
};

pub type Endianness = bool;
pub const BIG_ENDIAN: Endianness = false;
pub const LITTLE_ENDIAN: Endianness = true;

#[derive(Clone, Debug, PartialEq, Default)]
pub struct OpeningPoint<const E: Endianness, F: JoltField> {
    pub r: Vec<F>,
}

impl<const E: Endianness, F: JoltField> std::ops::Index<usize> for OpeningPoint<E, F> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        &self.r[index]
    }
}

impl<const E: Endianness, F: JoltField> std::ops::Index<std::ops::RangeFull>
    for OpeningPoint<E, F>
{
    type Output = [F];

    fn index(&self, _index: std::ops::RangeFull) -> &Self::Output {
        &self.r[..]
    }
}

impl<const E: Endianness, F: JoltField> OpeningPoint<E, F> {
    pub fn len(&self) -> usize {
        self.r.len()
    }

    pub fn split_at_r(&self, mid: usize) -> (&[F], &[F]) {
        self.r.split_at(mid)
    }

    pub fn split_at(&self, mid: usize) -> (Self, Self) {
        let (left, right) = self.r.split_at(mid);
        (Self::new(left.to_vec()), Self::new(right.to_vec()))
    }
}

impl<const E: Endianness, F: JoltField> OpeningPoint<E, F> {
    pub fn new(r: Vec<F>) -> Self {
        Self { r }
    }

    pub fn endianness(&self) -> &'static str {
        if E == BIG_ENDIAN {
            "big"
        } else {
            "little"
        }
    }

    pub fn match_endianness<const SWAPPED_E: Endianness>(&self) -> OpeningPoint<SWAPPED_E, F>
    where
        F: Clone,
    {
        let mut reversed = self.r.clone();
        if E != SWAPPED_E {
            reversed.reverse();
        }
        OpeningPoint::<SWAPPED_E, F>::new(reversed)
    }
}

impl<F: JoltField> From<Vec<F>> for OpeningPoint<LITTLE_ENDIAN, F> {
    fn from(r: Vec<F>) -> Self {
        Self::new(r)
    }
}

impl<F: JoltField> From<Vec<F>> for OpeningPoint<BIG_ENDIAN, F> {
    fn from(r: Vec<F>) -> Self {
        Self::new(r)
    }
}

impl<const E: Endianness, F: JoltField> Into<Vec<F>> for OpeningPoint<E, F> {
    fn into(self) -> Vec<F> {
        self.r
    }
}

impl<const E: Endianness, F: JoltField> Into<Vec<F>> for &OpeningPoint<E, F>
where
    F: Clone,
{
    fn into(self) -> Vec<F> {
        self.r.clone()
    }
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, FromPrimitive)]
#[repr(u8)]
pub enum SumcheckId {
    SpartanOuter,
    SpartanInner,
    SpartanShift,
    InstructionBooleanity,
    InstructionHammingWeight,
    InstructionReadRaf,
    RamReadWriteChecking,
    RamRafEvaluation,
    RamHammingWeight,
    RamHammingBooleanity,
    RamBooleanity,
    RamRaVirtualization,
    RamOutputCheck,
    RamValEvaluation,
    RamValFinalEvaluation,
    RegistersReadWriteChecking,
    RegistersValEvaluation,
    BytecodeReadRaf,
    BytecodeBooleanity,
    BytecodeHammingWeight,
    OpeningReduction,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord)]
pub enum OpeningId {
    Committed(CommittedPolynomial, SumcheckId),
    Virtual(VirtualPolynomial, SumcheckId),
}

pub type Openings<F> = BTreeMap<OpeningId, (OpeningPoint<BIG_ENDIAN, F>, F)>;

pub struct SharedEqPolynomial<F: JoltField> {
    num_variables_bound: usize,
    eq_poly: GruenSplitEqPolynomial<F>,
}

impl<F: JoltField> SharedEqPolynomial<F> {
    fn new_gruen(opening_point: &[F]) -> Self {
        Self {
            eq_poly: GruenSplitEqPolynomial::new(opening_point, BindingOrder::HighToLow),
            num_variables_bound: 0,
        }
    }
}

/// An opening (of a dense polynomial) computed by the prover.
///
/// May be a batched opening, where multiple dense polynomials opened
/// at the *same* point are reduced to a single polynomial opened
/// at the (same) point.
/// Multiple openings can be accumulated and further
/// batched/reduced using a `ProverOpeningAccumulator`.
#[derive(Clone)]
pub struct DensePolynomialProverOpening<F: JoltField> {
    /// The polynomial being opened. May be a random linear combination
    /// of multiple polynomials all being opened at the same point.
    pub polynomial: Option<MultilinearPolynomial<F>>,
    /// The multilinear extension EQ(x, opening_point). This is typically
    /// an intermediate value used to compute `claim`, but is also used in
    /// the `ProverOpeningAccumulator::prove_batch_opening_reduction` sumcheck.
    pub eq_poly: Arc<RwLock<SharedEqPolynomial<F>>>,
}

impl<F: JoltField> DensePolynomialProverOpening<F> {
    #[tracing::instrument(
        skip_all,
        name = "DensePolynomialProverOpening::compute_prover_message"
    )]
    fn compute_prover_message(&mut self, _round: usize, previous_claim: F) -> Vec<F> {
        let shared_eq = self.eq_poly.read().unwrap();
        let polynomial = self.polynomial.as_ref().unwrap();
        let gruen_eq = &shared_eq.eq_poly;

        // Compute q(0) = sum of polynomial(i) * eq(r, i) for i in [0, mle_half)
        let mle_half = polynomial.len() / 2;
        let q_0 = if gruen_eq.E_in_current_len() <= 1 {
            // E_in is fully bound
            (0..mle_half)
                .into_par_iter()
                .map(|j| {
                    let eq_eval = gruen_eq.E_out_current()[j];
                    let poly_eval = polynomial.get_bound_coeff(j);
                    eq_eval * poly_eval
                })
                .sum()
        } else {
            let num_x_out = gruen_eq.E_out_current_len();
            let num_x_in = gruen_eq.E_in_current_len();
            let d_e_in = gruen_eq.E_in_current();
            let d_e_out = gruen_eq.E_out_current();

            (0..num_x_in)
                .into_par_iter()
                .map(|x_in| {
                    let inner_sum: F = (0..num_x_out)
                        .into_par_iter()
                        .map(|x_out| {
                            let j = (x_in << num_x_out.log_2()) | x_out;
                            let poly_eval = polynomial.get_bound_coeff(j);
                            d_e_out[x_out] * poly_eval
                        })
                        .sum();
                    d_e_in[x_in] * inner_sum
                })
                .sum()
        };

        let gruen_univariate_evals = gruen_eq.gruen_evals_deg_2(q_0, previous_claim);

        vec![gruen_univariate_evals[0], gruen_univariate_evals[1]]
    }

    #[tracing::instrument(skip_all, name = "DensePolynomialProverOpening::bind")]
    fn bind(&mut self, r_j: F, round: usize) {
        let mut shared_eq = self.eq_poly.write().unwrap();
        if shared_eq.num_variables_bound <= round {
            shared_eq.eq_poly.bind(r_j);
            shared_eq.num_variables_bound += 1;
        }

        self.polynomial
            .as_mut()
            .unwrap()
            .bind_parallel(r_j, BindingOrder::HighToLow);
    }

    fn final_sumcheck_claim(&self) -> F {
        self.polynomial.as_ref().unwrap().final_sumcheck_claim()
    }
}

#[derive(derive_more::From, Clone)]
pub enum ProverOpening<F: JoltField> {
    Dense(DensePolynomialProverOpening<F>),
    OneHot(OneHotPolynomialProverOpening<F>),
}

#[derive(Clone)]
pub struct OpeningProofReductionSumcheck<F>
where
    F: JoltField,
{
    prover_state: Option<ProverOpening<F>>,
    /// Represents the polynomial(s) opened. May be a random linear combination
    /// of multiple polynomials, all being opened at the same point.
    polynomials: Vec<CommittedPolynomial>,
    /// The ID of the sumcheck these openings originated from
    sumcheck_id: SumcheckId,
    rlc_coeffs: Vec<F>,
    input_claims: Vec<F>,
    opening_point: Vec<F>,
    sumcheck_claim: Option<F>,
}

impl<F> OpeningProofReductionSumcheck<F>
where
    F: JoltField,
{
    fn new_prover_instance_dense(
        polynomials: Vec<CommittedPolynomial>,
        sumcheck_id: SumcheckId,
        eq_poly: Arc<RwLock<SharedEqPolynomial<F>>>,
        opening_point: Vec<F>,
        claims: Vec<F>,
    ) -> Self {
        let opening = DensePolynomialProverOpening {
            polynomial: None, // Defer initialization until opening proof reduction sumcheck
            eq_poly,
        };
        Self {
            polynomials,
            sumcheck_id,
            input_claims: claims,
            rlc_coeffs: vec![], // Populated later
            prover_state: Some(opening.into()),
            opening_point,
            sumcheck_claim: None,
        }
    }

    fn new_prover_instance_one_hot(
        polynomial: CommittedPolynomial,
        sumcheck_id: SumcheckId,
        eq_state: Arc<RwLock<OneHotSumcheckState<F>>>,
        opening_point: Vec<F>,
        claim: F,
    ) -> Self {
        let opening = OneHotPolynomialProverOpening::new(eq_state);
        Self {
            polynomials: vec![polynomial],
            sumcheck_id,
            input_claims: vec![claim],
            rlc_coeffs: vec![F::one()],
            prover_state: Some(opening.into()),
            opening_point,
            sumcheck_claim: None,
        }
    }

    fn new_verifier_instance(
        polynomials: Vec<CommittedPolynomial>,
        sumcheck_id: SumcheckId,
        opening_point: Vec<F>,
        claims: Vec<F>,
    ) -> Self {
        let rlc_coeffs = if polynomials.len() == 1 {
            vec![F::one()]
        } else {
            vec![] // Will be populated later
        };
        Self {
            polynomials,
            sumcheck_id,
            input_claims: claims,
            rlc_coeffs,
            prover_state: None,
            opening_point,
            sumcheck_claim: None,
        }
    }

    #[tracing::instrument(skip_all, name = "OpeningProofReductionSumcheck::prepare_sumcheck")]
    fn prepare_sumcheck(
        &mut self,
        polynomials_map: Option<&HashMap<CommittedPolynomial, MultilinearPolynomial<F>>>,
        gammas: &[F],
    ) {
        #[cfg(test)]
        {
            use crate::poly::multilinear_polynomial::PolynomialEvaluation;

            if let Some(polynomials_map) = polynomials_map {
                for (label, claim) in self.polynomials.iter().zip(self.input_claims.iter()) {
                    let poly = polynomials_map.get(label).unwrap();
                    debug_assert_eq!(
                        poly.evaluate(&self.opening_point),
                        *claim,
                        "Evaluation mismatch for {:?} {label:?}",
                        self.sumcheck_id
                    );
                }
            }
        }

        if self.polynomials.len() > 1 {
            assert_eq!(
                gammas.len(),
                self.polynomials.len(),
                "Expected {} gammas but got {}",
                self.polynomials.len(),
                gammas.len()
            );
            self.rlc_coeffs = gammas.to_vec();
        } else {
            assert_eq!(gammas.len(), 1, "Expected 1 gamma but got {}", gammas.len());
            self.rlc_coeffs = vec![F::one()];
        }

        if self.polynomials.len() > 1 {
            let reduced_claim = self
                .rlc_coeffs
                .par_iter()
                .zip(self.input_claims.par_iter())
                .map(|(gamma, claim)| *gamma * claim)
                .sum();
            self.input_claims = vec![reduced_claim];

            if let Some(prover_state) = self.prover_state.as_mut() {
                let polynomials_map = polynomials_map.unwrap();
                let polynomials: Vec<_> = self
                    .polynomials
                    .par_iter()
                    .map(|label| polynomials_map.get(label).unwrap())
                    .collect();

                let rlc_poly =
                    MultilinearPolynomial::linear_combination(&polynomials, &self.rlc_coeffs);
                debug_assert_eq!(rlc_poly.evaluate(&self.opening_point), reduced_claim);
                let num_vars = rlc_poly.get_num_vars();

                let opening_point_len = self.opening_point.len();
                debug_assert_eq!(
                    num_vars,
                    opening_point_len,
                    "{:?} have {num_vars} variables each but opening point from {:?} has length {opening_point_len}",
                    self.polynomials,
                    self.sumcheck_id,
                );

                match prover_state {
                    ProverOpening::Dense(opening) => opening.polynomial = Some(rlc_poly),
                    ProverOpening::OneHot(_) => {
                        panic!("Unexpected one-hot opening")
                    }
                };
            }
        } else if let Some(prover_state) = self.prover_state.as_mut() {
            let polynomials_map = polynomials_map.unwrap();
            let poly = polynomials_map.get(&self.polynomials[0]).unwrap();
            let num_vars = poly.get_num_vars();
            let opening_point_len = self.opening_point.len();
            debug_assert_eq!(
                    num_vars,
                    opening_point_len,
                    "{:?} has {num_vars} variables but opening point from {:?} has length {opening_point_len}",
                    self.polynomials[0],
                    self.sumcheck_id,
                );

            match prover_state {
                ProverOpening::Dense(opening) => opening.polynomial = Some(poly.clone()),
                ProverOpening::OneHot(opening) => {
                    if let MultilinearPolynomial::OneHot(poly) = poly {
                        opening.initialize(poly.clone());
                    } else {
                        panic!("Unexpected non-one-hot polynomial")
                    }
                }
            };
        }
    }

    fn cache_sumcheck_claim(&mut self) {
        debug_assert!(self.sumcheck_claim.is_none());
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        let claim = match prover_state {
            ProverOpening::Dense(opening) => opening.final_sumcheck_claim(),
            ProverOpening::OneHot(opening) => opening.final_sumcheck_claim(),
        };
        self.sumcheck_claim = Some(claim);
    }
}

impl<F> SumcheckInstance<F> for OpeningProofReductionSumcheck<F>
where
    F: JoltField,
{
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        self.opening_point.len()
    }

    fn input_claim(&self) -> F {
        assert_eq!(
            self.input_claims.len(),
            1,
            "Input claims should have been reduced by now"
        );
        self.input_claims[0]
    }

    fn compute_prover_message(&mut self, round: usize, previous_claim: F) -> Vec<F> {
        debug_assert!(round < self.num_rounds());
        let prover_state = self.prover_state.as_mut().unwrap();
        match prover_state {
            ProverOpening::Dense(opening) => opening.compute_prover_message(round, previous_claim),
            ProverOpening::OneHot(opening) => opening.compute_prover_message(round, previous_claim),
        }
    }

    fn bind(&mut self, r_j: F, round: usize) {
        debug_assert!(round < self.num_rounds());

        let prover_state = self.prover_state.as_mut().unwrap();
        match prover_state {
            ProverOpening::Dense(opening) => opening.bind(r_j, round),
            ProverOpening::OneHot(opening) => opening.bind(r_j, round),
        }
    }

    fn expected_output_claim(
        &self,
        _: Option<std::rc::Rc<std::cell::RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F],
    ) -> F {
        let eq_eval = EqPolynomial::mle(&self.opening_point, r);
        eq_eval * self.sumcheck_claim.unwrap()
    }

    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.to_vec())
    }

    fn cache_openings_prover(
        &self,
        _accumulator: std::rc::Rc<std::cell::RefCell<ProverOpeningAccumulator<F>>>,
        _opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        unimplemented!("Unused")
    }

    fn cache_openings_verifier(
        &self,
        _accumulator: std::rc::Rc<std::cell::RefCell<VerifierOpeningAccumulator<F>>>,
        _opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        unimplemented!("Unused")
    }
}

/// Accumulates openings computed by the prover over the course of Jolt,
/// so that they can all be reduced to a single opening proof using sumcheck.
#[derive(Clone)]
pub struct ProverOpeningAccumulator<F>
where
    F: JoltField,
{
    pub sumchecks: Vec<OpeningProofReductionSumcheck<F>>,
    pub openings: Openings<F>,
    #[cfg(test)]
    pub appended_virtual_openings: std::rc::Rc<std::cell::RefCell<Vec<OpeningId>>>,
    // #[cfg(test)]
    // joint_commitment: Option<PCS::Commitment>,
}

/// Accumulates openings encountered by the verifier over the course of Jolt,
/// so that they can all be reduced to a single opening proof verification using sumcheck.
pub struct VerifierOpeningAccumulator<F>
where
    F: JoltField,
{
    sumchecks: Vec<OpeningProofReductionSumcheck<F>>,
    pub openings: Openings<F>,
    /// In testing, the Jolt verifier may be provided the prover's openings so that we
    /// can detect any places where the openings don't match up.
    #[cfg(test)]
    prover_opening_accumulator: Option<ProverOpeningAccumulator<F>>,
    // #[cfg(test)]
    // pcs_setup: Option<PCS::ProverSetup>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug)]
pub struct ReducedOpeningProof<
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    ProofTranscript: Transcript,
> {
    pub sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    pub sumcheck_claims: Vec<F>,
    joint_opening_proof: PCS::Proof,
    #[cfg(test)]
    joint_poly: MultilinearPolynomial<F>,
    #[cfg(test)]
    joint_commitment: PCS::Commitment,
}

impl<F> Default for ProverOpeningAccumulator<F>
where
    F: JoltField,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F> ProverOpeningAccumulator<F>
where
    F: JoltField,
{
    pub fn new() -> Self {
        Self {
            sumchecks: vec![],
            openings: BTreeMap::new(),
            #[cfg(test)]
            appended_virtual_openings: std::rc::Rc::new(std::cell::RefCell::new(vec![])),
            // #[cfg(test)]
            // joint_commitment: None,
        }
    }

    pub fn len(&self) -> usize {
        self.sumchecks.len()
    }

    pub fn evaluation_openings(&self) -> &Openings<F> {
        &self.openings
    }

    pub fn evaluation_openings_mut(&mut self) -> &mut Openings<F> {
        &mut self.openings
    }

    /// Get the value of an opening by key
    pub fn get_opening(&self, key: OpeningId) -> F {
        self.openings.get(&key).unwrap().1
    }

    pub fn get_virtual_polynomial_opening(
        &self,
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
    ) -> (OpeningPoint<BIG_ENDIAN, F>, F) {
        let (point, claim) = self
            .openings
            .get(&OpeningId::Virtual(polynomial, sumcheck))
            .unwrap_or_else(|| panic!("opening for {sumcheck:?} {polynomial:?} not found"));
        #[cfg(test)]
        {
            let mut virtual_openings = self.appended_virtual_openings.borrow_mut();
            if let Some(index) = virtual_openings
                .iter()
                .position(|id| id == &OpeningId::Virtual(polynomial, sumcheck))
            {
                virtual_openings.remove(index);
            }
        }
        (point.clone(), *claim)
    }

    pub fn get_committed_polynomial_opening(
        &self,
        polynomial: CommittedPolynomial,
        sumcheck: SumcheckId,
    ) -> (OpeningPoint<BIG_ENDIAN, F>, F) {
        let (point, claim) = self
            .openings
            .get(&OpeningId::Committed(polynomial, sumcheck))
            .unwrap_or_else(|| panic!("opening for {sumcheck:?} {polynomial:?} not found"));
        (point.clone(), *claim)
    }

    /// Adds openings to the accumulator. The given `polynomials` are opened at
    /// `opening_point`, yielding the claimed evaluations `claims`.
    /// Multiple polynomials opened at a single point are batched into a single
    /// polynomial opened at the same point.
    #[tracing::instrument(skip_all, name = "ProverOpeningAccumulator::append_dense")]
    pub fn append_dense(
        &mut self,
        polynomials: Vec<CommittedPolynomial>,
        sumcheck: SumcheckId,
        opening_point: Vec<F>,
        claims: &[F],
    ) {
        assert_eq!(polynomials.len(), claims.len());

        // Use Gruen optimization for the eq polynomial
        let shared_eq = Arc::new(RwLock::new(SharedEqPolynomial::new_gruen(&opening_point)));

        // Add openings to map
        for (label, claim) in polynomials.iter().zip(claims.iter()) {
            let opening_point_struct = OpeningPoint::<BIG_ENDIAN, F>::new(opening_point.clone());
            let key = OpeningId::Committed(*label, sumcheck);
            self.openings
                .insert(key, (opening_point_struct.clone(), *claim));
        }

        let sumcheck = OpeningProofReductionSumcheck::new_prover_instance_dense(
            polynomials,
            sumcheck,
            shared_eq,
            opening_point,
            claims.to_vec(),
        );
        self.sumchecks.push(sumcheck);
    }

    #[tracing::instrument(skip_all, name = "ProverOpeningAccumulator::append_sparse")]
    pub fn append_sparse(
        &mut self,
        polynomials: Vec<CommittedPolynomial>,
        sumcheck: SumcheckId,
        r_address: Vec<F>,
        r_cycle: Vec<F>,
        claims: Vec<F>,
    ) {
        let r_concat = [r_address.as_slice(), r_cycle.as_slice()].concat();

        let shared_eq = Arc::new(RwLock::new(OneHotSumcheckState::new(&r_address, &r_cycle)));

        // Add openings to map
        for (label, claim) in polynomials.iter().zip(claims.iter()) {
            let opening_point_struct = OpeningPoint::<BIG_ENDIAN, F>::new(r_concat.clone());
            let key = OpeningId::Committed(*label, sumcheck);
            self.openings
                .insert(key, (opening_point_struct.clone(), *claim));
        }

        for (label, claim) in polynomials.into_iter().zip(claims.into_iter()) {
            let sumcheck = OpeningProofReductionSumcheck::new_prover_instance_one_hot(
                label,
                sumcheck,
                shared_eq.clone(),
                r_concat.clone(),
                claim,
            );
            self.sumchecks.push(sumcheck);
        }
    }

    pub fn append_virtual(
        &mut self,
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
        claim: F,
    ) {
        self.openings.insert(
            OpeningId::Virtual(polynomial, sumcheck),
            (opening_point, claim),
        );
        #[cfg(test)]
        self.appended_virtual_openings
            .borrow_mut()
            .push(OpeningId::Virtual(polynomial, sumcheck));
    }

    /// Reduces the multiple openings accumulated into a single opening proof,
    /// using a single sumcheck.
    #[tracing::instrument(skip_all, name = "ProverOpeningAccumulator::reduce_and_prove")]
    pub fn reduce_and_prove<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        &mut self,
        mut polynomials: HashMap<CommittedPolynomial, MultilinearPolynomial<F>>,
        mut opening_hints: HashMap<CommittedPolynomial, PCS::OpeningProofHint>,
        pcs_setup: &PCS::ProverSetup,
        transcript: &mut ProofTranscript,
    ) -> ReducedOpeningProof<F, PCS, ProofTranscript> {
        println!(
            "{} sumcheck instances in batched opening proof reduction",
            self.sumchecks.len()
        );

        let total_challenges_needed: usize = self
            .sumchecks
            .iter()
            .map(|sumcheck| {
                if sumcheck.polynomials.len() > 1 {
                    sumcheck.polynomials.len()
                } else {
                    1
                }
            })
            .sum();

        let all_gammas: Vec<F> = transcript.challenge_vector(total_challenges_needed);

        let prepare_span = tracing::span!(
            tracing::Level::INFO,
            "prepare_all_sumchecks",
            count = self.sumchecks.len()
        );
        let _enter = prepare_span.enter();

        let mut gamma_offsets = vec![0];
        for sumcheck in self.sumchecks.iter() {
            let num_gammas = if sumcheck.polynomials.len() > 1 {
                sumcheck.polynomials.len()
            } else {
                1
            };
            gamma_offsets.push(gamma_offsets.last().unwrap() + num_gammas);
        }

        self.sumchecks
            .par_iter_mut()
            .zip(gamma_offsets.par_iter())
            .for_each(|(sumcheck, &offset)| {
                let num_gammas = if sumcheck.polynomials.len() > 1 {
                    sumcheck.polynomials.len()
                } else {
                    1
                };
                let gammas_slice = &all_gammas[offset..offset + num_gammas];
                sumcheck.prepare_sumcheck(Some(&polynomials), gammas_slice);
            });

        drop(_enter);

        // Use sumcheck reduce many openings to one
        let (sumcheck_proof, r_sumcheck, sumcheck_claims) =
            self.prove_batch_opening_reduction(transcript);

        transcript.append_scalars(&sumcheck_claims);

        let gamma: F = transcript.challenge_scalar();
        let mut gamma_powers = vec![F::one()];
        for i in 1..self.sumchecks.len() {
            gamma_powers.push(gamma_powers[i - 1] * gamma);
        }

        // Combines the individual polynomials into the RLC that will be used for the
        // batched opening proof.
        let joint_poly = {
            let mut rlc_map = BTreeMap::new();
            for (gamma, sumcheck) in gamma_powers.iter().zip(self.sumchecks.iter()) {
                for (coeff, polynomial) in
                    sumcheck.rlc_coeffs.iter().zip(sumcheck.polynomials.iter())
                {
                    if let Some(value) = rlc_map.get_mut(&polynomial) {
                        *value += *coeff * gamma;
                    } else {
                        rlc_map.insert(polynomial, *coeff * gamma);
                    }
                }
            }

            let (coeffs, polynomials): (Vec<F>, Vec<MultilinearPolynomial<F>>) = rlc_map
                .into_iter()
                .map(|(k, v)| (v, polynomials.remove(k).unwrap()))
                .unzip();

            MultilinearPolynomial::linear_combination(
                &polynomials.iter().collect::<Vec<_>>(),
                &coeffs,
            )
        };

        #[cfg(test)]
        let joint_commitment = PCS::commit(&joint_poly, pcs_setup).0;

        // Compute the opening proof hint for the reduced opening by homomorphically combining
        // the hints for the individual sumchecks.
        let hint = {
            let mut rlc_map = BTreeMap::new();
            for (gamma, sumcheck) in gamma_powers.iter().zip(self.sumchecks.iter()) {
                for (coeff, polynomial) in
                    sumcheck.rlc_coeffs.iter().zip(sumcheck.polynomials.iter())
                {
                    if let Some(value) = rlc_map.get_mut(&polynomial) {
                        *value += *coeff * gamma;
                    } else {
                        rlc_map.insert(polynomial, *coeff * gamma);
                    }
                }
            }

            let (coeffs, hints): (Vec<F>, Vec<PCS::OpeningProofHint>) = rlc_map
                .into_iter()
                .map(|(k, v)| (v, opening_hints.remove(k).unwrap()))
                .unzip();
            debug_assert!(
                opening_hints.is_empty(),
                "Commitments to {:?} are not used",
                opening_hints.keys()
            );

            PCS::combine_hints(hints, &coeffs)
        };

        // Reduced opening proof
        let joint_opening_proof = PCS::prove(pcs_setup, &joint_poly, &r_sumcheck, hint, transcript);

        #[cfg(not(test))]
        {
            let sumchecks = std::mem::take(&mut self.sumchecks);
            crate::utils::thread::drop_in_background_thread(sumchecks);
        }

        ReducedOpeningProof {
            sumcheck_proof,
            sumcheck_claims,
            joint_opening_proof,
            #[cfg(test)]
            joint_poly,
            #[cfg(test)]
            joint_commitment,
        }
    }

    /// Proves the sumcheck used to prove the reduction of many openings into one.
    #[tracing::instrument(skip_all)]
    pub fn prove_batch_opening_reduction<ProofTranscript: Transcript>(
        &mut self,
        transcript: &mut ProofTranscript,
    ) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F>, Vec<F>) {
        let instances: Vec<&mut dyn SumcheckInstance<F>> = self
            .sumchecks
            .iter_mut()
            .map(|opening| {
                let instance: &mut dyn SumcheckInstance<F> = opening;
                instance
            })
            .collect();

        let (sumcheck_proof, r_sumcheck) = BatchedSumcheck::prove(instances, None, transcript);

        let claims: Vec<_> = self
            .sumchecks
            .iter_mut()
            .map(|opening| {
                opening.cache_sumcheck_claim();
                opening.sumcheck_claim.unwrap()
            })
            .collect();

        (sumcheck_proof, r_sumcheck, claims)
    }
}

impl<F> Default for VerifierOpeningAccumulator<F>
where
    F: JoltField,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F> VerifierOpeningAccumulator<F>
where
    F: JoltField,
{
    pub fn new() -> Self {
        Self {
            sumchecks: vec![],
            openings: BTreeMap::new(),
            #[cfg(test)]
            prover_opening_accumulator: None,
        }
    }

    /// Compare this accumulator to the corresponding `ProverOpeningAccumulator` and panic
    /// if the openings appended differ from the prover's openings.
    #[cfg(test)]
    pub fn compare_to(&mut self, prover_openings: ProverOpeningAccumulator<F>) {
        self.prover_opening_accumulator = Some(prover_openings);
    }

    pub fn len(&self) -> usize {
        self.sumchecks.len()
    }

    pub fn openings_mut(&mut self) -> &mut Openings<F> {
        &mut self.openings
    }

    pub fn get_virtual_polynomial_opening(
        &self,
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
    ) -> (OpeningPoint<BIG_ENDIAN, F>, F) {
        let (point, claim) = self
            .openings
            .get(&OpeningId::Virtual(polynomial, sumcheck))
            .unwrap_or_else(|| panic!("No opening found for {sumcheck:?} {polynomial:?}"));
        (point.clone(), *claim)
    }

    pub fn get_committed_polynomial_opening(
        &self,
        polynomial: CommittedPolynomial,
        sumcheck: SumcheckId,
    ) -> (OpeningPoint<BIG_ENDIAN, F>, F) {
        let (point, claim) = self
            .openings
            .get(&OpeningId::Committed(polynomial, sumcheck))
            .unwrap_or_else(|| panic!("No opening found for {sumcheck:?} {polynomial:?}"));
        (point.clone(), *claim)
    }

    /// Adds openings to the accumulator. The polynomials underlying the given
    /// `commitments` are opened at `opening_point`, yielding the claimed evaluations
    /// `claims`.
    /// Multiple polynomials opened at a single point can be batched into a single
    /// polynomial opened at the same point. This function performs the verifier side
    /// of this batching by homomorphically combining the commitments before appending
    /// to `self.openings`.
    pub fn append_dense(
        &mut self,
        polynomials: Vec<CommittedPolynomial>,
        sumcheck: SumcheckId,
        opening_point: Vec<F>,
    ) {
        #[cfg(test)]
        'test: {
            if self.prover_opening_accumulator.is_none() {
                break 'test;
            }
            let prover_opening =
                &self.prover_opening_accumulator.as_ref().unwrap().sumchecks[self.sumchecks.len()];
            assert_eq!(
                prover_opening.opening_point, opening_point,
                "opening point mismatch"
            );
            assert_eq!(
                prover_opening.polynomials.len(),
                polynomials.len(),
                "batch size mismatch"
            );
        }

        let claims = polynomials
            .iter()
            .map(|poly| {
                self.openings
                    .get(&OpeningId::Committed(*poly, sumcheck))
                    .unwrap()
                    .1
            })
            .collect();

        self.sumchecks
            .push(OpeningProofReductionSumcheck::new_verifier_instance(
                polynomials,
                sumcheck,
                opening_point,
                claims,
            ));
    }

    /// Adds openings to the accumulator. The polynomials underlying the given
    /// `commitments` are opened at `opening_point`, yielding the claimed evaluations
    /// `claims`.
    /// Multiple sparse polynomials opened at a single point are NOT batched into
    /// a single polynomial opened at the same point.
    pub fn append_sparse(
        &mut self,
        polynomials: Vec<CommittedPolynomial>,
        sumcheck: SumcheckId,
        opening_point: Vec<F>,
    ) {
        for label in polynomials.into_iter() {
            #[cfg(test)]
            'test: {
                if self.prover_opening_accumulator.is_none() {
                    break 'test;
                }
                let prover_opening = &self.prover_opening_accumulator.as_ref().unwrap().sumchecks
                    [self.sumchecks.len()];
                assert_eq!(
                    (prover_opening.polynomials[0], prover_opening.sumcheck_id),
                    (label, sumcheck),
                    "Polynomial mismatch"
                );
                assert_eq!(
                    prover_opening.polynomials.len(),
                    1,
                    "batch size mismatch for {sumcheck:?} {label:?}"
                );
                assert_eq!(
                    prover_opening.opening_point, opening_point,
                    "opening point mismatch for {sumcheck:?} {label:?}"
                );
            }

            let claim = self
                .openings
                .get(&OpeningId::Committed(label, sumcheck))
                .unwrap()
                .1;

            self.sumchecks
                .push(OpeningProofReductionSumcheck::new_verifier_instance(
                    vec![label],
                    sumcheck,
                    opening_point.clone(),
                    vec![claim],
                ));
        }
    }

    /// Populates the opening point for an existing claim in the evaluation_openings map.
    pub fn append_virtual(
        &mut self,
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let key = OpeningId::Virtual(polynomial, sumcheck);
        if let Some((_, claim)) = self.openings.get(&key) {
            let claim = *claim; // Copy the claim value
            self.openings.insert(key, (opening_point.clone(), claim));
        } else {
            panic!("Tried to populate opening point for non-existent key: {key:?}");
        }
    }

    /// Verifies that the given `reduced_opening_proof` (consisting of a sumcheck proof
    /// and a single opening proof) indeed proves the openings accumulated.
    pub fn reduce_and_verify<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        &mut self,
        pcs_setup: &PCS::VerifierSetup,
        commitment_map: &mut HashMap<CommittedPolynomial, PCS::Commitment>,
        reduced_opening_proof: &ReducedOpeningProof<F, PCS, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        #[cfg(test)]
        if let Some(prover_openings) = &self.prover_opening_accumulator {
            assert_eq!(prover_openings.len(), self.len());
        }

        let total_challenges_needed: usize = self
            .sumchecks
            .iter()
            .map(|sumcheck| {
                if sumcheck.polynomials.len() > 1 {
                    sumcheck.polynomials.len()
                } else {
                    1
                }
            })
            .sum();

        let all_gammas: Vec<F> = transcript.challenge_vector(total_challenges_needed);

        let mut gamma_offsets = vec![0];
        for sumcheck in self.sumchecks.iter() {
            let num_gammas = if sumcheck.polynomials.len() > 1 {
                sumcheck.polynomials.len()
            } else {
                1
            };
            gamma_offsets.push(gamma_offsets.last().unwrap() + num_gammas);
        }

        self.sumchecks
            .par_iter_mut()
            .zip(gamma_offsets.par_iter())
            .for_each(|(sumcheck, &offset)| {
                let num_gammas = if sumcheck.polynomials.len() > 1 {
                    sumcheck.polynomials.len()
                } else {
                    1
                };
                let gammas_slice = &all_gammas[offset..offset + num_gammas];
                sumcheck.prepare_sumcheck(None, gammas_slice);
            });

        let num_sumcheck_rounds = self
            .sumchecks
            .iter()
            .map(|opening| opening.opening_point.len())
            .max()
            .unwrap();

        self.sumchecks
            .iter_mut()
            .zip(reduced_opening_proof.sumcheck_claims.iter())
            .for_each(|(opening, claim)| opening.sumcheck_claim = Some(*claim));

        // Verify the sumcheck
        let r_sumcheck =
            self.verify_batch_opening_reduction(&reduced_opening_proof.sumcheck_proof, transcript)?;

        transcript.append_scalars(&reduced_opening_proof.sumcheck_claims);

        let gamma: F = transcript.challenge_scalar();
        let mut gamma_powers = vec![F::one()];
        for i in 1..self.sumchecks.len() {
            gamma_powers.push(gamma_powers[i - 1] * gamma);
        }

        // Compute the commitment for the reduced opening proof by homomorphically combining
        // the commitments of the individual polynomials.
        let joint_commitment = {
            let mut rlc_map = HashMap::new();
            for (gamma, sumcheck) in gamma_powers.iter().zip(self.sumchecks.iter()) {
                for (coeff, polynomial) in
                    sumcheck.rlc_coeffs.iter().zip(sumcheck.polynomials.iter())
                {
                    if let Some(value) = rlc_map.get_mut(&polynomial) {
                        *value += *coeff * gamma;
                    } else {
                        rlc_map.insert(polynomial, *coeff * gamma);
                    }
                }
            }

            let (coeffs, commitments): (Vec<F>, Vec<PCS::Commitment>) = rlc_map
                .into_iter()
                .map(|(k, v)| (v, commitment_map.remove(k).unwrap()))
                .unzip();
            debug_assert!(commitment_map.is_empty(), "Every commitment should be used");

            PCS::combine_commitments(&commitments, &coeffs)
        };

        #[cfg(test)]
        assert_eq!(
            joint_commitment, reduced_opening_proof.joint_commitment,
            "joint commitment mismatch"
        );

        // Compute joint claim = ∑ᵢ γⁱ⋅ claimᵢ
        let joint_claim: F = gamma_powers
            .iter()
            .zip(reduced_opening_proof.sumcheck_claims.iter())
            .zip(self.sumchecks.iter())
            .map(|((coeff, claim), opening)| {
                let r_slice = &r_sumcheck[..num_sumcheck_rounds - opening.opening_point.len()];
                let lagrange_eval: F = r_slice.iter().map(|r| F::one() - r).product();
                *coeff * claim * lagrange_eval
            })
            .sum();

        // Verify the reduced opening proof
        PCS::verify(
            &reduced_opening_proof.joint_opening_proof,
            pcs_setup,
            transcript,
            &r_sumcheck,
            &joint_claim,
            &joint_commitment,
        )
    }

    /// Verifies the sumcheck proven in `ProverOpeningAccumulator::prove_batch_opening_reduction`.
    fn verify_batch_opening_reduction<ProofTranscript: Transcript>(
        &self,
        sumcheck_proof: &SumcheckInstanceProof<F, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<Vec<F>, ProofVerifyError> {
        let instances: Vec<&dyn SumcheckInstance<F>> = self
            .sumchecks
            .iter()
            .map(|opening| {
                let instance: &dyn SumcheckInstance<F> = opening;
                instance
            })
            .collect();
        BatchedSumcheck::verify(sumcheck_proof, instances, None, transcript)
    }
}
