//! This is a port of the sumcheck-based batch opening proof protocol implemented
//! in Nova: https://github.com/microsoft/Nova/blob/2772826ba296b66f1cd5deecf7aca3fd1d10e1f4/src/spartan/snark.rs#L410-L424
//! and such code is Copyright (c) Microsoft Corporation.
//! For additively homomorphic commitment schemes (including Zeromorph, HyperKZG) we
//! can use a sumcheck to reduce multiple opening proofs (multiple polynomials, not
//! necessarily of the same size, each opened at a different point) into a single opening.

use crate::poly::rlc_polynomial::{RLCPolynomial, RLCStreamingData};
#[cfg(feature = "allocative")]
use crate::utils::profiling::write_flamegraph_svg;
use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use num_derive::FromPrimitive;
use rayon::prelude::*;
#[cfg(test)]
use std::cell::RefCell;
use std::{
    collections::{BTreeMap, HashMap},
    sync::{Arc, RwLock},
};
use tracer::LazyTraceIterator;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use itertools::Itertools;

use super::{
    commitment::commitment_scheme::CommitmentScheme,
    eq_poly::EqPolynomial,
    multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
};
#[cfg(feature = "allocative")]
use crate::utils::profiling::print_data_structure_heap_usage;
use crate::{
    field::JoltField,
    poly::{
        one_hot_polynomial::{EqAddressState, EqCycleState, OneHotPolynomialProverOpening},
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck::{BatchedSumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
    zkvm::witness::{CommittedPolynomial, VirtualPolynomial},
};

pub type Endianness = bool;
pub const BIG_ENDIAN: Endianness = false;
pub const LITTLE_ENDIAN: Endianness = true;

/// Degree of the sumcheck round polynomials in [`OpeningProofReductionSumcheckVerifier`].
const OPENING_SUMCHECK_DEGREE: usize = 2;

#[derive(Clone, Debug, PartialEq, Default, Allocative)]
pub struct OpeningPoint<const E: Endianness, F: JoltField> {
    pub r: Vec<F::Challenge>,
}

impl<const E: Endianness, F: JoltField> std::ops::Index<usize> for OpeningPoint<E, F> {
    type Output = F::Challenge;

    fn index(&self, index: usize) -> &Self::Output {
        &self.r[index]
    }
}

impl<const E: Endianness, F: JoltField> std::ops::Index<std::ops::RangeFull>
    for OpeningPoint<E, F>
{
    type Output = [F::Challenge];

    fn index(&self, _index: std::ops::RangeFull) -> &Self::Output {
        &self.r[..]
    }
}

impl<const E: Endianness, F: JoltField> OpeningPoint<E, F> {
    pub fn len(&self) -> usize {
        self.r.len()
    }

    pub fn split_at_r(&self, mid: usize) -> (&[F::Challenge], &[F::Challenge]) {
        self.r.split_at(mid)
    }

    pub fn split_at(&self, mid: usize) -> (Self, Self) {
        let (left, right) = self.r.split_at(mid);
        (Self::new(left.to_vec()), Self::new(right.to_vec()))
    }
}

impl<const E: Endianness, F: JoltField> OpeningPoint<E, F> {
    pub fn new(r: Vec<F::Challenge>) -> Self {
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

impl<F: JoltField> From<Vec<F::Challenge>> for OpeningPoint<LITTLE_ENDIAN, F> {
    fn from(r: Vec<F::Challenge>) -> Self {
        Self::new(r)
    }
}

impl<F: JoltField> From<Vec<F::Challenge>> for OpeningPoint<BIG_ENDIAN, F> {
    fn from(r: Vec<F::Challenge>) -> Self {
        Self::new(r)
    }
}

impl<const E: Endianness, F: JoltField> Into<Vec<F::Challenge>> for OpeningPoint<E, F> {
    fn into(self) -> Vec<F::Challenge> {
        self.r
    }
}

impl<const E: Endianness, F: JoltField> Into<Vec<F::Challenge>> for &OpeningPoint<E, F>
where
    F: Clone,
{
    fn into(self) -> Vec<F::Challenge> {
        self.r.clone()
    }
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, FromPrimitive, Allocative)]
#[repr(u8)]
pub enum SumcheckId {
    SpartanOuter,
    SpartanInner,
    SpartanShift,
    ProductVirtualization,
    InstructionInputVirtualization,
    InstructionBooleanity,
    InstructionHammingWeight,
    InstructionReadRaf,
    InstructionRaVirtualization,
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

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
pub enum OpeningId {
    Committed(CommittedPolynomial, SumcheckId),
    Virtual(VirtualPolynomial, SumcheckId),
    UntrustedAdvice,
    TrustedAdvice,
}

pub type Openings<F> = BTreeMap<OpeningId, (OpeningPoint<BIG_ENDIAN, F>, F)>;

#[derive(Clone, Debug, Allocative)]
pub struct SharedDensePolynomial<F: JoltField> {
    pub poly: MultilinearPolynomial<F>,
    /// The number of variables that have been bound during sumcheck so far
    pub num_variables_bound: usize,
}

impl<F: JoltField> SharedDensePolynomial<F> {
    fn new(poly: MultilinearPolynomial<F>) -> Self {
        Self {
            poly,
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
#[derive(Clone, Allocative)]
pub struct DensePolynomialProverOpening<F: JoltField> {
    /// The polynomial being opened. May be a random linear combination
    /// of multiple polynomials all being opened at the same point.
    pub polynomial: Option<Arc<RwLock<SharedDensePolynomial<F>>>>,
    /// The multilinear extension EQ(x, opening_point). This is typically
    /// an intermediate value used to compute `claim`, but is also used in
    /// the `ProverOpeningAccumulator::prove_batch_opening_reduction` sumcheck.
    pub eq_poly: Arc<RwLock<EqCycleState<F>>>,
}

impl<F: JoltField> DensePolynomialProverOpening<F> {
    #[tracing::instrument(skip_all, name = "DensePolynomialProverOpening::compute_message")]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let shared_eq = self.eq_poly.read().unwrap();
        let polynomial_ref = self.polynomial.as_ref().unwrap();
        let polynomial = &polynomial_ref.read().unwrap().poly;
        let gruen_eq = &shared_eq.D;

        // Compute q(0) = sum of polynomial(i) * eq(r, i) for i in [0, mle_half)
        let [q_0] = gruen_eq.par_fold_out_in_unreduced::<9, 1>(&|g| {
            // TODO(Quang): can special case on polynomial type
            // (if not bound, can have faster multiplication + avoid conversion to field)
            [polynomial.get_bound_coeff(2 * g)]
        });

        gruen_eq.gruen_poly_deg_2(q_0, previous_claim)
    }

    #[tracing::instrument(skip_all, name = "DensePolynomialProverOpening::bind")]
    fn bind(&mut self, r_j: F::Challenge, round: usize) {
        let mut shared_eq = self.eq_poly.write().unwrap();
        if shared_eq.num_variables_bound <= round {
            shared_eq.D.bind(r_j);
            shared_eq.num_variables_bound += 1;
        }

        let shared_poly_ref = self.polynomial.as_mut().unwrap();
        let mut shared_poly = shared_poly_ref.write().unwrap();
        if shared_poly.num_variables_bound <= round {
            shared_poly.poly.bind_parallel(r_j, BindingOrder::LowToHigh);
            shared_poly.num_variables_bound += 1;
        }
    }

    fn final_sumcheck_claim(&self) -> F {
        let poly_ref = self.polynomial.as_ref().unwrap();
        poly_ref.read().unwrap().poly.final_sumcheck_claim()
    }
}

#[derive(derive_more::From, Clone, Allocative)]
pub enum ProverOpening<F: JoltField> {
    Dense(DensePolynomialProverOpening<F>),
    OneHot(OneHotPolynomialProverOpening<F>),
}

#[derive(Clone, Allocative)]
pub struct OpeningProofReductionSumcheckProver<F>
where
    F: JoltField,
{
    prover_state: ProverOpening<F>,
    /// Represents the polynomial opened.
    polynomial: CommittedPolynomial,
    /// The ID of the sumcheck these openings originated from
    sumcheck_id: SumcheckId,
    input_claim: F,
    opening_point: Vec<F::Challenge>,
    sumcheck_claim: Option<F>,
    log_T: usize,
}

impl<F> OpeningProofReductionSumcheckProver<F>
where
    F: JoltField,
{
    fn new_dense(
        polynomial: CommittedPolynomial,
        sumcheck_id: SumcheckId,
        eq_poly: Arc<RwLock<EqCycleState<F>>>,
        opening_point: Vec<F::Challenge>,
        claim: F,
        log_T: usize,
    ) -> Self {
        let opening = DensePolynomialProverOpening {
            polynomial: None, // Defer initialization until opening proof reduction sumcheck
            eq_poly,
        };
        Self {
            polynomial,
            sumcheck_id,
            input_claim: claim,
            prover_state: opening.into(),
            opening_point,
            sumcheck_claim: None,
            log_T,
        }
    }

    fn new_one_hot(
        polynomial: CommittedPolynomial,
        sumcheck_id: SumcheckId,
        eq_address: Arc<RwLock<EqAddressState<F>>>,
        eq_cycle: Arc<RwLock<EqCycleState<F>>>,
        opening_point: Vec<F::Challenge>,
        claim: F,
        log_T: usize,
    ) -> Self {
        let opening = OneHotPolynomialProverOpening::new(eq_address, eq_cycle);
        Self {
            polynomial,
            sumcheck_id,
            input_claim: claim,
            prover_state: opening.into(),
            opening_point,
            sumcheck_claim: None,
            log_T,
        }
    }

    #[tracing::instrument(skip_all, name = "OpeningProofReductionSumcheck::prepare_sumcheck")]
    fn prepare_sumcheck(
        &mut self,
        polynomials_map: &HashMap<CommittedPolynomial, MultilinearPolynomial<F>>,
        shared_dense_polynomials: &HashMap<
            CommittedPolynomial,
            Arc<RwLock<SharedDensePolynomial<F>>>,
        >,
    ) {
        #[cfg(test)]
        {
            use crate::poly::multilinear_polynomial::PolynomialEvaluation;
            let poly = polynomials_map.get(&self.polynomial).unwrap();
            debug_assert_eq!(
                poly.evaluate(&self.opening_point),
                self.input_claim,
                "Evaluation mismatch for {:?} {:?}",
                self.sumcheck_id,
                self.polynomial,
            );
            let num_vars = poly.get_num_vars();
            let opening_point_len = self.opening_point.len();
            debug_assert_eq!(
                        num_vars,
                        opening_point_len,
                        "{:?} has {num_vars} variables but opening point from {:?} has length {opening_point_len}",
                        self.polynomial,
                        self.sumcheck_id,
                    );
        }

        match &mut self.prover_state {
            ProverOpening::Dense(opening) => {
                let poly = shared_dense_polynomials.get(&self.polynomial).unwrap();
                opening.polynomial = Some(poly.clone());
            }
            ProverOpening::OneHot(opening) => {
                let poly = polynomials_map.get(&self.polynomial).unwrap();
                if let MultilinearPolynomial::OneHot(one_hot) = poly {
                    opening.initialize(one_hot.clone());
                } else {
                    panic!("Unexpected non-one-hot polynomial")
                }
            }
        };
    }

    fn cache_sumcheck_claim(&mut self) {
        debug_assert!(self.sumcheck_claim.is_none());
        let claim = match &mut self.prover_state {
            ProverOpening::Dense(opening) => opening.final_sumcheck_claim(),
            ProverOpening::OneHot(opening) => opening.final_sumcheck_claim(),
        };
        self.sumcheck_claim = Some(claim);
    }
}

impl<F, T: Transcript> SumcheckInstanceProver<F, T> for OpeningProofReductionSumcheckProver<F>
where
    F: JoltField,
{
    fn degree(&self) -> usize {
        OPENING_SUMCHECK_DEGREE
    }

    fn num_rounds(&self) -> usize {
        self.opening_point.len()
    }

    fn input_claim(&self, _accumulator: &ProverOpeningAccumulator<F>) -> F {
        self.input_claim
    }

    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        match &mut self.prover_state {
            ProverOpening::Dense(opening) => opening.compute_message(round, previous_claim),
            ProverOpening::OneHot(opening) => opening.compute_message(round, previous_claim),
        }
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        match &mut self.prover_state {
            ProverOpening::Dense(opening) => opening.bind(r_j, round),
            ProverOpening::OneHot(opening) => opening.bind(r_j, round),
        }
    }

    fn cache_openings(
        &self,
        _accumulator: &mut ProverOpeningAccumulator<F>,
        _transcript: &mut T,
        _sumcheck_challenges: &[F::Challenge],
    ) {
        // Nothing to cache.
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

struct OpeningProofReductionSumcheckVerifier<F>
where
    F: JoltField,
{
    /// Represents the polynomial opened.
    polynomial: CommittedPolynomial,
    input_claim: F,
    opening_point: Vec<F::Challenge>,
    sumcheck_claim: Option<F>,
    log_T: usize,
}

impl<F: JoltField> OpeningProofReductionSumcheckVerifier<F> {
    pub fn new(
        polynomial: CommittedPolynomial,
        opening_point: Vec<F::Challenge>,
        input_claim: F,
        log_T: usize,
    ) -> Self {
        Self {
            polynomial,
            input_claim,
            opening_point,
            sumcheck_claim: None,
            log_T,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for OpeningProofReductionSumcheckVerifier<F>
{
    fn degree(&self) -> usize {
        OPENING_SUMCHECK_DEGREE
    }

    fn num_rounds(&self) -> usize {
        self.opening_point.len()
    }

    fn input_claim(&self, _accumulator: &VerifierOpeningAccumulator<F>) -> F {
        self.input_claim
    }

    fn expected_output_claim(
        &self,
        _accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let mut r = sumcheck_challenges.to_vec();
        match self.polynomial {
            CommittedPolynomial::RdInc | CommittedPolynomial::RamInc => r.reverse(),
            CommittedPolynomial::InstructionRa(_)
            | CommittedPolynomial::BytecodeRa(_)
            | CommittedPolynomial::RamRa(_) => {
                let log_K = r.len() - self.log_T;
                // reverse log_T rounds since they are bounded LowToHigh
                r[log_K..].reverse();
            }
        }
        let eq_eval = EqPolynomial::<F>::mle(&self.opening_point, &r);
        eq_eval * self.sumcheck_claim.unwrap()
    }

    fn cache_openings(
        &self,
        _accumulator: &mut VerifierOpeningAccumulator<F>,
        _transcript: &mut T,
        _sumcheck_challenges: &[F::Challenge],
    ) {
        // Nothing to cache.
    }
}

/// Accumulates openings computed by the prover over the course of Jolt,
/// so that they can all be reduced to a single opening proof using sumcheck.
#[derive(Clone, Allocative)]
pub struct ProverOpeningAccumulator<F>
where
    F: JoltField,
{
    pub sumchecks: Vec<OpeningProofReductionSumcheckProver<F>>,
    pub openings: Openings<F>,
    dense_polynomial_map: HashMap<CommittedPolynomial, Arc<RwLock<SharedDensePolynomial<F>>>>,
    eq_cycle_map: HashMap<Vec<F::Challenge>, Arc<RwLock<EqCycleState<F>>>>,
    #[cfg(test)]
    pub appended_virtual_openings: RefCell<Vec<OpeningId>>,
    log_T: usize,
}

/// Accumulates openings encountered by the verifier over the course of Jolt,
/// so that they can all be reduced to a single opening proof verification using sumcheck.
pub struct VerifierOpeningAccumulator<F>
where
    F: JoltField,
{
    sumchecks: Vec<OpeningProofReductionSumcheckVerifier<F>>,
    pub openings: Openings<F>,
    /// In testing, the Jolt verifier may be provided the prover's openings so that we
    /// can detect any places where the openings don't match up.
    #[cfg(test)]
    prover_opening_accumulator: Option<ProverOpeningAccumulator<F>>,
    log_T: usize,
}

pub trait OpeningAccumulator<F: JoltField> {
    fn get_virtual_polynomial_opening(
        &self,
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
    ) -> (OpeningPoint<BIG_ENDIAN, F>, F);

    fn get_committed_polynomial_opening(
        &self,
        polynomial: CommittedPolynomial,
        sumcheck: SumcheckId,
    ) -> (OpeningPoint<BIG_ENDIAN, F>, F);
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
        Self::new(0)
    }
}

impl<F: JoltField> OpeningAccumulator<F> for ProverOpeningAccumulator<F> {
    fn get_virtual_polynomial_opening(
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

    fn get_committed_polynomial_opening(
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
}

impl<F> ProverOpeningAccumulator<F>
where
    F: JoltField,
{
    pub fn new(log_T: usize) -> Self {
        Self {
            sumchecks: vec![],
            openings: BTreeMap::new(),
            eq_cycle_map: HashMap::new(),
            dense_polynomial_map: HashMap::new(),
            #[cfg(test)]
            appended_virtual_openings: std::cell::RefCell::new(vec![]),
            log_T,
            // #[cfg(test)]
            // joint_commitment: None,
        }
    }

    pub fn len(&self) -> usize {
        self.sumchecks.len()
    }

    pub fn evaluation_openings_mut(&mut self) -> &mut Openings<F> {
        &mut self.openings
    }

    /// Get the value of an opening by key
    pub fn get_opening(&self, key: OpeningId) -> F {
        self.openings.get(&key).unwrap().1
    }

    pub fn get_untrusted_advice_opening(&self) -> Option<(OpeningPoint<BIG_ENDIAN, F>, F)> {
        let (point, claim) = self.openings.get(&OpeningId::UntrustedAdvice)?;
        Some((point.clone(), *claim))
    }

    pub fn get_trusted_advice_opening(&self) -> Option<(OpeningPoint<BIG_ENDIAN, F>, F)> {
        let (point, claim) = self.openings.get(&OpeningId::TrustedAdvice)?;
        Some((point.clone(), *claim))
    }

    /// Adds an opening of a dense polynomial to the accumulator.
    /// The given `polynomial` is opened at `opening_point`, yielding the claimed
    /// evaluation `claim`.
    #[tracing::instrument(skip_all, name = "ProverOpeningAccumulator::append_dense")]
    pub fn append_dense<T: Transcript>(
        &mut self,
        transcript: &mut T,
        polynomial: CommittedPolynomial,
        sumcheck: SumcheckId,
        opening_point: Vec<F::Challenge>,
        claim: F,
    ) {
        transcript.append_scalar(&claim);

        let shared_eq = self
            .eq_cycle_map
            .entry(opening_point.clone())
            .or_insert_with(|| Arc::new(RwLock::new(EqCycleState::new(&opening_point))));

        // Add opening to map
        let key = OpeningId::Committed(polynomial, sumcheck);
        self.openings.insert(
            key,
            (
                OpeningPoint::<BIG_ENDIAN, F>::new(opening_point.clone()),
                claim,
            ),
        );

        let sumcheck = OpeningProofReductionSumcheckProver::new_dense(
            polynomial,
            sumcheck,
            shared_eq.clone(),
            opening_point,
            claim,
            self.log_T,
        );
        self.sumchecks.push(sumcheck);
    }

    #[tracing::instrument(skip_all, name = "ProverOpeningAccumulator::append_sparse")]
    pub fn append_sparse<T: Transcript>(
        &mut self,
        transcript: &mut T,
        polynomials: Vec<CommittedPolynomial>,
        sumcheck: SumcheckId,
        r_address: Vec<F::Challenge>,
        r_cycle: Vec<F::Challenge>,
        claims: Vec<F>,
    ) {
        claims.iter().for_each(|claim| {
            transcript.append_scalar(claim);
        });
        let r_concat = [r_address.as_slice(), r_cycle.as_slice()].concat();

        let shared_eq_address = Arc::new(RwLock::new(EqAddressState::new(&r_address)));
        let shared_eq_cycle = self
            .eq_cycle_map
            .entry(r_cycle.clone())
            .or_insert(Arc::new(RwLock::new(EqCycleState::new(&r_cycle))));

        // Add openings to map
        for (label, claim) in polynomials.iter().zip(claims.iter()) {
            let opening_point_struct = OpeningPoint::<BIG_ENDIAN, F>::new(r_concat.clone());
            let key = OpeningId::Committed(*label, sumcheck);
            self.openings
                .insert(key, (opening_point_struct.clone(), *claim));
        }

        for (label, claim) in polynomials.into_iter().zip(claims.into_iter()) {
            let sumcheck = OpeningProofReductionSumcheckProver::new_one_hot(
                label,
                sumcheck,
                shared_eq_address.clone(),
                shared_eq_cycle.clone(),
                r_concat.clone(),
                claim,
                self.log_T,
            );
            self.sumchecks.push(sumcheck);
        }
    }

    pub fn append_virtual<T: Transcript>(
        &mut self,
        transcript: &mut T,
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
        claim: F,
    ) {
        transcript.append_scalar(&claim);
        assert!(
            self.openings
                .insert(
                    OpeningId::Virtual(polynomial, sumcheck),
                    (opening_point, claim),
                )
                .is_none(),
            "Key ({polynomial:?}, {sumcheck:?}) is already in opening map"
        );
        #[cfg(test)]
        self.appended_virtual_openings
            .borrow_mut()
            .push(OpeningId::Virtual(polynomial, sumcheck));
    }

    pub fn append_untrusted_advice<T: Transcript>(
        &mut self,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
        claim: F,
    ) {
        transcript.append_scalar(&claim);
        self.openings
            .insert(OpeningId::UntrustedAdvice, (opening_point, claim));
    }

    pub fn append_trusted_advice<T: Transcript>(
        &mut self,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
        claim: F,
    ) {
        transcript.append_scalar(&claim);
        self.openings
            .insert(OpeningId::TrustedAdvice, (opening_point, claim));
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
        streaming_context: Option<(LazyTraceIterator, Arc<RLCStreamingData>)>,
    ) -> ReducedOpeningProof<F, PCS, ProofTranscript> {
        tracing::debug!(
            "{} sumcheck instances in batched opening proof reduction",
            self.sumchecks.len()
        );

        let prepare_span = tracing::span!(
            tracing::Level::INFO,
            "prepare_all_sumchecks",
            count = self.sumchecks.len()
        );
        let _enter = prepare_span.enter();

        // Populate dense_polynomial_map
        for sumcheck in self.sumchecks.iter() {
            if let ProverOpening::Dense(_) = &sumcheck.prover_state {
                // If not already in `dense_polynomial_map`, create shared polynomial
                // and insert it into the map.
                self.dense_polynomial_map
                    .entry(sumcheck.polynomial)
                    .or_insert_with(|| {
                        let poly = polynomials.get(&sumcheck.polynomial).unwrap().clone();
                        Arc::new(RwLock::new(SharedDensePolynomial::new(poly)))
                    });
            }
        }

        self.sumchecks.par_iter_mut().for_each(|sumcheck| {
            sumcheck.prepare_sumcheck(&polynomials, &self.dense_polynomial_map);
        });

        drop(_enter);

        // Use sumcheck reduce many openings to one
        let (sumcheck_proof, mut r_sumcheck, sumcheck_claims) =
            self.prove_batch_opening_reduction(transcript);
        let log_K = r_sumcheck.len() - self.log_T;
        // reverse log_T rounds since they are bounded LowToHigh
        r_sumcheck[log_K..].reverse();

        transcript.append_scalars(&sumcheck_claims);

        let gamma_powers: Vec<F> = transcript.challenge_scalar_powers(self.sumchecks.len());

        // Combines the individual polynomials into the RLC that will be used for the
        // batched opening proof.
        let (joint_poly, hint) = {
            let mut rlc_map = BTreeMap::new();
            for (gamma, sumcheck) in gamma_powers.iter().zip(self.sumchecks.iter()) {
                if let Some(value) = rlc_map.get_mut(&sumcheck.polynomial) {
                    *value += *gamma;
                } else {
                    rlc_map.insert(sumcheck.polynomial, *gamma);
                }
            }

            let (poly_ids, coeffs, polys): (
                Vec<CommittedPolynomial>,
                Vec<F>,
                Vec<MultilinearPolynomial<F>>,
            ) = rlc_map
                .iter()
                .map(|(k, v)| (*k, *v, polynomials.remove(k).unwrap()))
                .multiunzip();

            let poly_arcs: Vec<Arc<MultilinearPolynomial<F>>> =
                polys.into_iter().map(Arc::new).collect();

            let joint_poly = MultilinearPolynomial::RLC(RLCPolynomial::linear_combination(
                poly_ids.clone(),
                poly_arcs.clone(),
                &coeffs,
                streaming_context,
            ));

            let hints: Vec<PCS::OpeningProofHint> = rlc_map
                .into_keys()
                .map(|k| opening_hints.remove(&k).unwrap())
                .collect();
            debug_assert!(
                opening_hints.is_empty(),
                "Commitments to {:?} are not used",
                opening_hints.keys()
            );
            // Compute the opening proof hint for the reduced opening by homomorphically combining
            // the hints for the individual sumchecks.
            let hint = PCS::combine_hints(hints, &coeffs);

            #[cfg(test)]
            let joint_poly = (joint_poly, poly_ids, poly_arcs, coeffs);

            (joint_poly, hint)
        };

        #[cfg(test)]
        let joint_commitment = {
            let (joint_poly_ref, poly_ids, poly_arcs, coeffs) = &joint_poly;
            let materialized_poly = match joint_poly_ref {
                MultilinearPolynomial::RLC(rlc) => {
                    MultilinearPolynomial::RLC(rlc.materialize(poly_ids, poly_arcs, coeffs))
                }
                _ => joint_poly_ref.clone(),
            };
            PCS::commit(&materialized_poly, pcs_setup).0
        };

        #[cfg(test)]
        let joint_poly = joint_poly.0;

        #[cfg(not(test))]
        {
            let sumchecks = std::mem::take(&mut self.sumchecks);
            crate::utils::thread::drop_in_background_thread(sumchecks);
        }

        // Reduced opening proof
        let joint_opening_proof =
            PCS::prove(pcs_setup, &joint_poly, &r_sumcheck, Some(hint), transcript);

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
    ) -> (
        SumcheckInstanceProof<F, ProofTranscript>,
        Vec<F::Challenge>,
        Vec<F>,
    ) {
        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage("Opening accumulator", &(*self));
            let mut flamegraph = FlameGraphBuilder::default();
            flamegraph.visit_root(&(*self));
            write_flamegraph_svg(flamegraph, "stage7_start_flamechart.svg");
        }

        let instances = self
            .sumchecks
            .iter_mut()
            .map(|opening| opening as &mut _)
            .collect();

        let (sumcheck_proof, r_sumcheck) = BatchedSumcheck::prove(
            instances,
            &mut ProverOpeningAccumulator::new(self.log_T),
            transcript,
        );

        #[cfg(feature = "allocative")]
        {
            let mut flamegraph = FlameGraphBuilder::default();
            flamegraph.visit_root(&(*self));
            write_flamegraph_svg(flamegraph, "stage7_end_flamechart.svg");
        }

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
        Self::new(0)
    }
}

impl<F: JoltField> OpeningAccumulator<F> for VerifierOpeningAccumulator<F> {
    fn get_virtual_polynomial_opening(
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

    fn get_committed_polynomial_opening(
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
}

impl<F> VerifierOpeningAccumulator<F>
where
    F: JoltField,
{
    pub fn new(log_T: usize) -> Self {
        Self {
            sumchecks: vec![],
            openings: BTreeMap::new(),
            #[cfg(test)]
            prover_opening_accumulator: None,
            log_T,
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

    pub fn get_untrusted_advice_opening(&self) -> Option<(OpeningPoint<BIG_ENDIAN, F>, F)> {
        let (point, claim) = self.openings.get(&OpeningId::UntrustedAdvice)?;
        Some((point.clone(), *claim))
    }

    pub fn get_trusted_advice_opening(&self) -> Option<(OpeningPoint<BIG_ENDIAN, F>, F)> {
        let (point, claim) = self.openings.get(&OpeningId::TrustedAdvice)?;
        Some((point.clone(), *claim))
    }

    /// Adds an opening of a dense polynomial the accumulator.
    /// The given `polynomial` is opened at `opening_point`.
    pub fn append_dense<T: Transcript>(
        &mut self,
        transcript: &mut T,
        polynomial: CommittedPolynomial,
        sumcheck: SumcheckId,
        opening_point: Vec<F::Challenge>,
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
        }

        let claim = self
            .openings
            .get(&OpeningId::Committed(polynomial, sumcheck))
            .unwrap()
            .1;
        transcript.append_scalar(&claim);

        self.sumchecks
            .push(OpeningProofReductionSumcheckVerifier::new(
                polynomial,
                opening_point,
                claim,
                self.log_T,
            ));
    }

    /// Adds openings to the accumulator. The polynomials underlying the given
    /// `commitments` are opened at `opening_point`, yielding the claimed evaluations
    /// `claims`.
    /// Multiple sparse polynomials opened at a single point are NOT batched into
    /// a single polynomial opened at the same point.
    pub fn append_sparse<T: Transcript>(
        &mut self,
        transcript: &mut T,
        polynomials: Vec<CommittedPolynomial>,
        sumcheck: SumcheckId,
        opening_point: Vec<F::Challenge>,
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
                    (prover_opening.polynomial, prover_opening.sumcheck_id),
                    (label, sumcheck),
                    "Polynomial mismatch"
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
            transcript.append_scalar(&claim);

            self.sumchecks
                .push(OpeningProofReductionSumcheckVerifier::new(
                    label,
                    opening_point.clone(),
                    claim,
                    self.log_T,
                ));
        }
    }

    /// Populates the opening point for an existing claim in the evaluation_openings map.
    pub fn append_virtual<T: Transcript>(
        &mut self,
        transcript: &mut T,
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let key = OpeningId::Virtual(polynomial, sumcheck);
        if let Some((_, claim)) = self.openings.get(&key) {
            transcript.append_scalar(claim);
            let claim = *claim; // Copy the claim value
            self.openings.insert(key, (opening_point.clone(), claim));
        } else {
            panic!("Tried to populate opening point for non-existent key: {key:?}");
        }
    }

    pub fn append_untrusted_advice<T: Transcript>(
        &mut self,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        if let Some((_, claim)) = self.openings.get(&OpeningId::UntrustedAdvice) {
            transcript.append_scalar(claim);
            let claim = *claim;
            self.openings
                .insert(OpeningId::UntrustedAdvice, (opening_point.clone(), claim));
        } else {
            panic!(
                "Tried to populate opening point for non-existent key: {:?}",
                OpeningId::UntrustedAdvice
            );
        }
    }

    pub fn append_trusted_advice<T: Transcript>(
        &mut self,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        if let Some((_, claim)) = self.openings.get(&OpeningId::TrustedAdvice) {
            transcript.append_scalar(claim);
            let claim = *claim;
            self.openings
                .insert(OpeningId::TrustedAdvice, (opening_point.clone(), claim));
        } else {
            panic!(
                "Tried to populate opening point for non-existent key: {:?}",
                OpeningId::TrustedAdvice
            );
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
        let mut r_sumcheck =
            self.verify_batch_opening_reduction(&reduced_opening_proof.sumcheck_proof, transcript)?;
        let log_K = r_sumcheck.len() - self.log_T;
        r_sumcheck[log_K..].reverse();

        transcript.append_scalars(&reduced_opening_proof.sumcheck_claims);

        let gamma_powers: Vec<F> = transcript.challenge_scalar_powers(self.sumchecks.len());

        // Compute the commitment for the reduced opening proof by homomorphically combining
        // the commitments of the individual polynomials.
        let joint_commitment = {
            let mut rlc_map = HashMap::new();
            for (gamma, sumcheck) in gamma_powers.iter().zip(self.sumchecks.iter()) {
                if let Some(value) = rlc_map.get_mut(&sumcheck.polynomial) {
                    *value += *gamma;
                } else {
                    rlc_map.insert(sumcheck.polynomial, *gamma);
                }
            }

            let (coeffs, commitments): (Vec<F>, Vec<PCS::Commitment>) = rlc_map
                .into_iter()
                .map(|(k, v)| (v, commitment_map.remove(&k).unwrap()))
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
    ) -> Result<Vec<F::Challenge>, ProofVerifyError> {
        let instances: Vec<&dyn SumcheckInstanceVerifier<F, ProofTranscript>> = self
            .sumchecks
            .iter()
            .map(|opening| {
                let instance: &dyn SumcheckInstanceVerifier<F, ProofTranscript> = opening;
                instance
            })
            .collect();
        BatchedSumcheck::verify(
            sumcheck_proof,
            instances,
            &mut VerifierOpeningAccumulator::new(self.log_T),
            transcript,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::{dense_mlpoly::DensePolynomial, unipoly::UniPoly};
    use ark_bn254::Fr;
    use ark_std::{test_rng, Zero};
    use rand_core::RngCore;

    fn dense_polynomial_equivalence<const LOG_T: usize>() {
        let T: usize = 1 << LOG_T;

        let mut rng = test_rng();

        // Create a random dense polynomial
        let poly_coeffs: Vec<Fr> = (0..T).map(|_| Fr::from(rng.next_u64())).collect();
        let mut dense_poly = DensePolynomial::new(poly_coeffs);

        let r_cycle = std::iter::repeat_with(|| <Fr as JoltField>::Challenge::random(&mut rng))
            .take(LOG_T)
            .collect::<Vec<_>>();

        let eq_cycle_state = EqCycleState::new(&r_cycle);

        let mut dense_opening = DensePolynomialProverOpening {
            polynomial: Some(Arc::new(RwLock::new(SharedDensePolynomial {
                poly: MultilinearPolynomial::from(dense_poly.Z.clone()),
                num_variables_bound: 0,
            }))),
            eq_poly: Arc::new(RwLock::new(eq_cycle_state)),
        };

        let mut eq = DensePolynomial::new(EqPolynomial::<Fr>::evals(&r_cycle));

        // Compute the initial input claim
        let input_claim: Fr = (0..dense_poly.len()).map(|i| dense_poly[i] * eq[i]).sum();
        let mut previous_claim = input_claim;

        for round in 0..LOG_T {
            let dense_message = dense_opening.compute_message(round, previous_claim);
            let mut expected_message = vec![Fr::zero(), Fr::zero()];
            let mle_half = dense_poly.len() / 2;

            expected_message[0] = (0..mle_half).map(|i| dense_poly[2 * i] * eq[2 * i]).sum();
            expected_message[1] = (0..mle_half)
                .map(|i| {
                    let poly_bound_point =
                        dense_poly[2 * i + 1] + dense_poly[2 * i + 1] - dense_poly[2 * i];
                    let eq_bound_point = eq[2 * i + 1] + eq[2 * i + 1] - eq[2 * i];
                    poly_bound_point * eq_bound_point
                })
                .sum();

            assert_eq!(
                [
                    dense_message.eval_at_zero(),
                    dense_message.evaluate::<Fr>(&Fr::from(2))
                ],
                *expected_message,
                "round {round} prover message mismatch"
            );

            let r = <Fr as JoltField>::Challenge::random(&mut rng);

            // Update previous_claim by evaluating the univariate polynomial at r
            let eval_at_1 = previous_claim - expected_message[0];
            let univariate_evals = vec![expected_message[0], eval_at_1, expected_message[1]];
            let univariate_poly = UniPoly::from_evals(&univariate_evals);
            previous_claim = univariate_poly.evaluate(&r);

            dense_opening.bind(r, round);
            dense_poly.bind_parallel(r, BindingOrder::LowToHigh);
            eq.bind_parallel(r, BindingOrder::LowToHigh);
        }
        assert_eq!(
            dense_opening.final_sumcheck_claim(),
            dense_poly[0],
            "final sumcheck claim"
        );
    }

    #[test]
    fn dense_opening_correctness() {
        dense_polynomial_equivalence::<6>();
    }
}
