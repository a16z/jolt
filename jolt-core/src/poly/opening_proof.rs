//! This is a port of the sumcheck-based batch opening proof protocol implemented
//! in Nova: https://github.com/microsoft/Nova/blob/2772826ba296b66f1cd5deecf7aca3fd1d10e1f4/src/spartan/snark.rs#L410-L424
//! and such code is Copyright (c) Microsoft Corporation.
//! For additively homomorphic commitment schemes (including Zeromorph, HyperKZG) we
//! can use a sumcheck to reduce multiple opening proofs (multiple polynomials, not
//! necessarily of the same size, each opened at a different point) into a single opening.

#[cfg(feature = "allocative")]
use crate::utils::profiling::write_flamegraph_svg;
use crate::{
    poly::rlc_polynomial::{RLCPolynomial, RLCStreamingData},
    zkvm::config::OneHotParams,
};
use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use itertools::Itertools;
use num_derive::FromPrimitive;
use rayon::prelude::*;
#[cfg(test)]
use std::cell::RefCell;
use std::{
    collections::{BTreeMap, HashMap},
    sync::{Arc, RwLock},
};
use tracer::LazyTraceIterator;

use super::{
    commitment::commitment_scheme::CommitmentScheme, multilinear_polynomial::MultilinearPolynomial,
};
#[cfg(test)]
use super::{eq_poly::EqPolynomial, multilinear_polynomial::BindingOrder};
#[cfg(test)]
use crate::subprotocols::opening_reduction::DensePolynomialProverOpening;
#[cfg(feature = "allocative")]
use crate::utils::profiling::print_data_structure_heap_usage;
use crate::{
    field::JoltField,
    poly::one_hot_polynomial::{EqAddressState, EqCycleState},
    subprotocols::{
        opening_reduction::{
            OpeningProofReductionSumcheckProver, OpeningProofReductionSumcheckVerifier,
            ProverOpening, SharedDensePolynomial,
        },
        sumcheck::{BatchedSumcheck, SumcheckInstanceProof},
        sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
    zkvm::witness::{CommittedPolynomial, VirtualPolynomial},
};

pub type Endianness = bool;
pub const BIG_ENDIAN: Endianness = false;
pub const LITTLE_ENDIAN: Endianness = true;

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

#[derive(
    Hash,
    PartialEq,
    Eq,
    Copy,
    Clone,
    Debug,
    PartialOrd,
    Ord,
    FromPrimitive,
    Allocative,
    strum_macros::EnumCount,
)]
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
    InstructionClaimReduction,
    RamReadWriteChecking,
    RamRafEvaluation,
    RamHammingWeight,
    RamHammingBooleanity,
    RamBooleanity,
    RamRaReduction,
    RamRaVirtualization,
    RamOutputCheck,
    RamValEvaluation,
    RamValFinalEvaluation,
    RegistersReadWriteChecking,
    RegistersValEvaluation,
    RegistersClaimReduction,
    BytecodeReadRaf,
    BytecodeBooleanity,
    BytecodeHammingWeight,
    OpeningReduction,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
pub enum OpeningId {
    Committed(CommittedPolynomial, SumcheckId),
    Virtual(VirtualPolynomial, SumcheckId),
    /// Untrusted advice opened at r_address derived from the given sumcheck.
    /// - `RamReadWriteChecking`: opened at r_address from RamVal (used by ValEvaluation)
    /// - `RamOutputCheck`: opened at r_address from RamValFinal (used by ValFinal)
    UntrustedAdvice(SumcheckId),
    /// Trusted advice opened at r_address derived from the given sumcheck.
    /// - `RamReadWriteChecking`: opened at r_address from RamVal (used by ValEvaluation)
    /// - `RamOutputCheck`: opened at r_address from RamValFinal (used by ValFinal)
    TrustedAdvice(SumcheckId),
}

/// (point, claim)
pub type Opening<F> = (OpeningPoint<BIG_ENDIAN, F>, F);
pub type Openings<F> = BTreeMap<OpeningId, Opening<F>>;

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
    pub log_T: usize,
    pub opening_reduction_state: Option<OpeningReductionState<F>>,
    pub polynomials_for_opening: Option<HashMap<CommittedPolynomial, MultilinearPolynomial<F>>>,
    pub cached_opening_claims: Vec<(CommittedPolynomial, F)>,
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
    /// State from Stage 7 (batch opening sumcheck) for Stage 8 (Dory opening)
    pub opening_reduction_state: Option<OpeningReductionState<F>>,
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

/// Intermediate state between Stage 7 (batch opening reduction sumcheck) and Stage 8 (Dory opening).
/// Stored in prover/verifier state to bridge the two stages.
#[derive(Clone, Allocative)]
pub struct OpeningReductionState<F: JoltField> {
    pub r_sumcheck: Vec<F::Challenge>,
    pub gamma_powers: Vec<F>,
    pub sumcheck_claims: Vec<F>,
    pub polynomials: Vec<CommittedPolynomial>,
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
            opening_reduction_state: None,
            polynomials_for_opening: None,
            cached_opening_claims: vec![],
        }
    }

    /// Caches an opening claim from the opening reduction sumcheck.
    /// Called from `OpeningProofReductionSumcheckProver::cache_openings`.
    pub fn cache_opening_reduction_claim(&mut self, polynomial: CommittedPolynomial, claim: F) {
        self.cached_opening_claims.push((polynomial, claim));
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

    pub fn get_untrusted_advice_opening(
        &self,
        sumcheck_id: SumcheckId,
    ) -> Option<(OpeningPoint<BIG_ENDIAN, F>, F)> {
        let (point, claim) = self
            .openings
            .get(&OpeningId::UntrustedAdvice(sumcheck_id))?;
        Some((point.clone(), *claim))
    }

    pub fn get_trusted_advice_opening(
        &self,
        sumcheck_id: SumcheckId,
    ) -> Option<(OpeningPoint<BIG_ENDIAN, F>, F)> {
        let (point, claim) = self.openings.get(&OpeningId::TrustedAdvice(sumcheck_id))?;
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
        sumcheck_id: SumcheckId,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
        claim: F,
    ) {
        transcript.append_scalar(&claim);
        self.openings.insert(
            OpeningId::UntrustedAdvice(sumcheck_id),
            (opening_point, claim),
        );
    }

    pub fn append_trusted_advice<T: Transcript>(
        &mut self,
        transcript: &mut T,
        sumcheck_id: SumcheckId,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
        claim: F,
    ) {
        transcript.append_scalar(&claim);
        self.openings.insert(
            OpeningId::TrustedAdvice(sumcheck_id),
            (opening_point, claim),
        );
    }

    // ========== Stage 7: Batch Opening Reduction Sumcheck ==========

    /// Prepares sumcheck instances for the batch opening reduction.
    /// Must be called before `prove_batch_opening_sumcheck`.
    #[tracing::instrument(skip_all)]
    pub fn prepare_for_sumcheck(
        &mut self,
        polynomials: &HashMap<CommittedPolynomial, MultilinearPolynomial<F>>,
    ) {
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
                self.dense_polynomial_map
                    .entry(sumcheck.polynomial)
                    .or_insert_with(|| {
                        let poly = polynomials.get(&sumcheck.polynomial).unwrap().clone();
                        Arc::new(RwLock::new(SharedDensePolynomial::new(poly)))
                    });
            }
        }

        self.sumchecks.par_iter_mut().for_each(|sumcheck| {
            sumcheck.prepare_sumcheck(polynomials, &self.dense_polynomial_map);
        });
    }

    /// Proves the batch opening reduction sumcheck (Stage 7).
    /// Returns the sumcheck proof and challenges.
    #[tracing::instrument(skip_all)]
    pub fn prove_batch_opening_sumcheck<T: Transcript>(
        &mut self,
        transcript: &mut T,
    ) -> (SumcheckInstanceProof<F, T>, Vec<F::Challenge>) {
        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage("Opening accumulator", &(*self));
            let mut flamegraph = FlameGraphBuilder::default();
            flamegraph.visit_root(&(*self));
            write_flamegraph_svg(flamegraph, "stage7_start_flamechart.svg");
        }

        // Temporarily take sumchecks so we can pass self to BatchedSumcheck::prove
        let mut sumchecks = std::mem::take(&mut self.sumchecks);
        let instances = sumchecks
            .iter_mut()
            .map(|opening| opening as &mut _)
            .collect();

        let (sumcheck_proof, r_sumcheck) = BatchedSumcheck::prove(instances, self, transcript);

        // Restore sumchecks (with cached claims from cache_openings)
        self.sumchecks = sumchecks;

        #[cfg(feature = "allocative")]
        {
            let mut flamegraph = FlameGraphBuilder::default();
            flamegraph.visit_root(&(*self));
            write_flamegraph_svg(flamegraph, "stage7_end_flamechart.svg");
        }

        (sumcheck_proof, r_sumcheck)
    }

    /// Finalizes the batch opening reduction sumcheck.
    /// Uses cached claims from `cache_openings`, appends them to transcript, derives gamma powers,
    /// and cleans up sumcheck instances.
    /// Returns the state needed for Stage 8.
    #[tracing::instrument(
        skip_all,
        name = "ProverOpeningAccumulator::finalize_batch_opening_sumcheck"
    )]
    pub fn finalize_batch_opening_sumcheck<T: Transcript>(
        &mut self,
        r_sumcheck: Vec<F::Challenge>,
        transcript: &mut T,
    ) -> OpeningReductionState<F> {
        // Extract claims and polynomials from cached opening claims (populated by cache_openings)
        let (polynomials, sumcheck_claims): (Vec<CommittedPolynomial>, Vec<F>) =
            std::mem::take(&mut self.cached_opening_claims)
                .into_iter()
                .unzip();

        // Adjust r_sumcheck endianness
        let mut r_sumcheck = r_sumcheck;
        let log_K = r_sumcheck.len() - self.log_T;
        r_sumcheck[..log_K].reverse();
        r_sumcheck[log_K..].reverse();

        // Append claims and derive gamma powers
        transcript.append_scalars(&sumcheck_claims);
        let gamma_powers: Vec<F> = transcript.challenge_scalar_powers(sumcheck_claims.len());

        // Drop sumchecks in background - they're no longer needed
        #[cfg(not(test))]
        {
            let sumchecks = std::mem::take(&mut self.sumchecks);
            crate::utils::thread::drop_in_background_thread(sumchecks);
        }

        OpeningReductionState {
            r_sumcheck,
            gamma_powers,
            sumcheck_claims,
            polynomials,
        }
    }

    // ========== Stage 8: Dory Batch Opening Proof ==========

    /// Builds the RLC polynomial and combined hint for the batch opening proof.
    #[tracing::instrument(skip_all, name = "ProverOpeningAccumulator::build_rlc_polynomial")]
    pub fn build_rlc_polynomial<PCS: CommitmentScheme<Field = F>>(
        &self,
        mut polynomials: HashMap<CommittedPolynomial, MultilinearPolynomial<F>>,
        mut opening_hints: HashMap<CommittedPolynomial, PCS::OpeningProofHint>,
        state: &OpeningReductionState<F>,
        streaming_context: Option<(LazyTraceIterator, Arc<RLCStreamingData>, OneHotParams)>,
    ) -> (MultilinearPolynomial<F>, PCS::OpeningProofHint) {
        let mut rlc_map = BTreeMap::new();
        for (gamma, poly) in state.gamma_powers.iter().zip(state.polynomials.iter()) {
            if let Some(value) = rlc_map.get_mut(poly) {
                *value += *gamma;
            } else {
                rlc_map.insert(*poly, *gamma);
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
            poly_arcs,
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

        let hint = PCS::combine_hints(hints, &coeffs);

        (joint_poly, hint)
    }

    /// Computes the joint commitment for testing purposes.
    /// If streaming_context is provided, uses RLC streaming; otherwise uses homomorphic combination.
    #[cfg(test)]
    pub fn compute_joint_commitment_for_test<PCS: CommitmentScheme<Field = F>>(
        &self,
        polynomials: &HashMap<CommittedPolynomial, MultilinearPolynomial<F>>,
        state: &OpeningReductionState<F>,
        pcs_setup: &PCS::ProverSetup,
        streaming_context: Option<(LazyTraceIterator, Arc<RLCStreamingData>, OneHotParams)>,
    ) -> PCS::Commitment {
        let mut rlc_map = BTreeMap::new();
        for (gamma, poly) in state.gamma_powers.iter().zip(state.polynomials.iter()) {
            if let Some(value) = rlc_map.get_mut(poly) {
                *value += *gamma;
            } else {
                rlc_map.insert(*poly, *gamma);
            }
        }

        if streaming_context.is_some() {
            // Use RLC streaming with materialization
            let (poly_ids, coeffs, polys): (
                Vec<CommittedPolynomial>,
                Vec<F>,
                Vec<MultilinearPolynomial<F>>,
            ) = rlc_map
                .iter()
                .map(|(k, v)| (*k, *v, polynomials.get(k).unwrap().clone()))
                .multiunzip();

            let poly_arcs: Vec<Arc<MultilinearPolynomial<F>>> =
                polys.into_iter().map(Arc::new).collect();

            let rlc = RLCPolynomial::linear_combination(
                poly_ids.clone(),
                poly_arcs.clone(),
                &coeffs,
                streaming_context,
            );
            let materialized_rlc = rlc.materialize(&poly_ids, &poly_arcs, &coeffs);
            let joint_poly = MultilinearPolynomial::RLC(materialized_rlc);

            PCS::commit(&joint_poly, pcs_setup).0
        } else {
            // Use homomorphic combination of commitments
            let (coeffs, commitments): (Vec<F>, Vec<PCS::Commitment>) = rlc_map
                .iter()
                .map(|(k, v)| {
                    let poly = polynomials.get(k).unwrap();
                    let (commitment, _) = PCS::commit(poly, pcs_setup);
                    (*v, commitment)
                })
                .unzip();

            PCS::combine_commitments(&commitments, &coeffs)
        }
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
            opening_reduction_state: None,
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

    pub fn get_untrusted_advice_opening(
        &self,
        sumcheck_id: SumcheckId,
    ) -> Option<(OpeningPoint<BIG_ENDIAN, F>, F)> {
        let (point, claim) = self
            .openings
            .get(&OpeningId::UntrustedAdvice(sumcheck_id))?;
        Some((point.clone(), *claim))
    }

    pub fn get_trusted_advice_opening(
        &self,
        sumcheck_id: SumcheckId,
    ) -> Option<(OpeningPoint<BIG_ENDIAN, F>, F)> {
        let (point, claim) = self.openings.get(&OpeningId::TrustedAdvice(sumcheck_id))?;
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
                prover_opening.opening.0.r, opening_point,
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
                    prover_opening.opening.0.r, opening_point,
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
        sumcheck_id: SumcheckId,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let key = OpeningId::UntrustedAdvice(sumcheck_id);
        if let Some((_, claim)) = self.openings.get(&key) {
            transcript.append_scalar(claim);
            let claim = *claim;
            self.openings.insert(key, (opening_point.clone(), claim));
        } else {
            panic!("Tried to populate opening point for non-existent key: {key:?}");
        }
    }

    pub fn append_trusted_advice<T: Transcript>(
        &mut self,
        transcript: &mut T,
        sumcheck_id: SumcheckId,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let key = OpeningId::TrustedAdvice(sumcheck_id);
        if let Some((_, claim)) = self.openings.get(&key) {
            transcript.append_scalar(claim);
            let claim = *claim;
            self.openings.insert(key, (opening_point.clone(), claim));
        } else {
            panic!("Tried to populate opening point for non-existent key: {key:?}");
        }
    }

    // ========== Stage 7: Batch Opening Reduction Sumcheck ==========

    /// Prepares the verifier for the batch opening reduction sumcheck.
    /// Populates sumcheck claims from the proof.
    pub fn prepare_for_sumcheck(&mut self, sumcheck_claims: &[F]) {
        #[cfg(test)]
        if let Some(prover_openings) = &self.prover_opening_accumulator {
            assert_eq!(prover_openings.len(), self.len());
        }

        self.sumchecks
            .iter_mut()
            .zip(sumcheck_claims.iter())
            .for_each(|(opening, claim)| opening.sumcheck_claim = Some(*claim));
    }

    /// Verifies the batch opening reduction sumcheck (Stage 7).
    pub fn verify_batch_opening_sumcheck<T: Transcript>(
        &self,
        sumcheck_proof: &SumcheckInstanceProof<F, T>,
        transcript: &mut T,
    ) -> Result<Vec<F::Challenge>, ProofVerifyError> {
        let instances: Vec<&dyn SumcheckInstanceVerifier<F, T>> = self
            .sumchecks
            .iter()
            .map(|opening| {
                let instance: &dyn SumcheckInstanceVerifier<F, T> = opening;
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

    /// Finalizes the batch opening reduction sumcheck verification.
    /// Returns the state needed for Stage 8.
    pub fn finalize_batch_opening_sumcheck<T: Transcript>(
        &self,
        r_sumcheck: Vec<F::Challenge>,
        sumcheck_claims: &[F],
        transcript: &mut T,
    ) -> OpeningReductionState<F> {
        // Extract polynomial labels
        let polynomials: Vec<CommittedPolynomial> =
            self.sumchecks.iter().map(|s| s.polynomial).collect();

        // Adjust r_sumcheck endianness
        let mut r_sumcheck = r_sumcheck;
        let log_K = r_sumcheck.len() - self.log_T;
        r_sumcheck[..log_K].reverse();
        r_sumcheck[log_K..].reverse();

        // Append claims and derive gamma powers
        transcript.append_scalars(sumcheck_claims);
        let gamma_powers: Vec<F> = transcript.challenge_scalar_powers(self.sumchecks.len());

        OpeningReductionState {
            r_sumcheck,
            gamma_powers,
            sumcheck_claims: sumcheck_claims.to_vec(),
            polynomials,
        }
    }

    // ========== Stage 8: Dory Batch Opening Verification ==========

    /// Computes the joint commitment by homomorphically combining individual commitments.
    pub fn compute_joint_commitment<PCS: CommitmentScheme<Field = F>>(
        &self,
        commitment_map: &mut HashMap<CommittedPolynomial, PCS::Commitment>,
        state: &OpeningReductionState<F>,
    ) -> PCS::Commitment {
        let mut rlc_map = HashMap::new();
        for (gamma, poly) in state.gamma_powers.iter().zip(state.polynomials.iter()) {
            if let Some(value) = rlc_map.get_mut(poly) {
                *value += *gamma;
            } else {
                rlc_map.insert(*poly, *gamma);
            }
        }

        let (coeffs, commitments): (Vec<F>, Vec<PCS::Commitment>) = rlc_map
            .into_iter()
            .map(|(k, v)| (v, commitment_map.remove(&k).unwrap()))
            .unzip();
        debug_assert!(commitment_map.is_empty(), "Every commitment should be used");

        PCS::combine_commitments(&commitments, &coeffs)
    }

    /// Computes the joint claim for the batch opening verification.
    pub fn compute_joint_claim<T: Transcript>(&self, state: &OpeningReductionState<F>) -> F {
        let num_sumcheck_rounds = self
            .sumchecks
            .iter()
            .map(|opening| SumcheckInstanceVerifier::<F, T>::num_rounds(opening))
            .max()
            .unwrap();

        state
            .gamma_powers
            .iter()
            .zip(state.sumcheck_claims.iter())
            .zip(self.sumchecks.iter())
            .map(|((coeff, claim), opening)| {
                let r_slice = &state.r_sumcheck
                    [..num_sumcheck_rounds - SumcheckInstanceVerifier::<F, T>::num_rounds(opening)];
                let lagrange_eval: F = r_slice.iter().map(|r| F::one() - r).product();
                *coeff * claim * lagrange_eval
            })
            .sum()
    }

    /// Verifies the joint opening proof (Stage 8).
    pub fn verify_joint_opening<T: Transcript, PCS: CommitmentScheme<Field = F>>(
        &self,
        pcs_setup: &PCS::VerifierSetup,
        joint_opening_proof: &PCS::Proof,
        joint_commitment: &PCS::Commitment,
        state: &OpeningReductionState<F>,
        transcript: &mut T,
    ) -> Result<(), ProofVerifyError> {
        let joint_claim = self.compute_joint_claim::<T>(state);

        PCS::verify(
            joint_opening_proof,
            pcs_setup,
            transcript,
            &state.r_sumcheck,
            &joint_claim,
            joint_commitment,
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
