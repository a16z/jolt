//! This is a port of the sumcheck-based batch opening proof protocol implemented
//! in Nova: https://github.com/microsoft/Nova/blob/2772826ba296b66f1cd5deecf7aca3fd1d10e1f4/src/spartan/snark.rs#L410-L424
//! and such code is Copyright (c) Microsoft Corporation.
//! For additively homomorphic commitment schemes (including Zeromorph, HyperKZG) we
//! can use a sumcheck to reduce multiple opening proofs (multiple polynomials, not
//! necessarily of the same size, each opened at a different point) into a single opening.

use crate::{
    poly::rlc_polynomial::{RLCPolynomial, RLCStreamingData, TraceSource},
    zkvm::{claim_reductions::AdviceKind, config::OneHotParams},
};
use allocative::Allocative;
use num_derive::FromPrimitive;
#[cfg(test)]
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use super::{
    commitment::commitment_scheme::CommitmentScheme, multilinear_polynomial::MultilinearPolynomial,
};
use crate::{
    field::JoltField,
    transcripts::Transcript,
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
    SpartanProductVirtualization,
    SpartanShift,
    InstructionClaimReduction,
    InstructionInputVirtualization,
    InstructionReadRaf,
    InstructionRaVirtualization,
    RamReadWriteChecking,
    RamRafEvaluation,
    RamOutputCheck,
    RamValEvaluation,
    RamValFinalEvaluation,
    RamRaClaimReduction,
    RamHammingBooleanity,
    RamRaVirtualization,
    RegistersClaimReduction,
    RegistersReadWriteChecking,
    RegistersValEvaluation,
    BytecodeReadRaf,
    Booleanity,
    AdviceClaimReductionCyclePhase,
    AdviceClaimReduction,
    IncClaimReduction,
    HammingWeightClaimReduction,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
pub enum PolynomialId {
    Committed(CommittedPolynomial),
    Virtual(VirtualPolynomial),
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
pub enum OpeningId {
    Polynomial(PolynomialId, SumcheckId),
    /// Untrusted advice opened at r_address derived from the given sumcheck.
    /// - `RamReadWriteChecking`: opened at r_address from RamVal (used by ValEvaluation)
    /// - `RamOutputCheck`: opened at r_address from RamValFinal (used by ValFinal)
    UntrustedAdvice(SumcheckId),
    /// Trusted advice opened at r_address derived from the given sumcheck.
    /// - `RamReadWriteChecking`: opened at r_address from RamVal (used by ValEvaluation)
    /// - `RamOutputCheck`: opened at r_address from RamValFinal (used by ValFinal)
    TrustedAdvice(SumcheckId),
}

impl OpeningId {
    pub fn virt(poly: VirtualPolynomial, sc: SumcheckId) -> Self {
        Self::Polynomial(PolynomialId::Virtual(poly), sc)
    }

    pub fn committed(poly: CommittedPolynomial, sc: SumcheckId) -> Self {
        Self::Polynomial(PolynomialId::Committed(poly), sc)
    }
}

/// (point, claim)
pub type Opening<F> = (OpeningPoint<BIG_ENDIAN, F>, F);
pub type Openings<F> = BTreeMap<OpeningId, Opening<F>>;

/// ZK data collected during prove_zk for later use by BlindFold.
///
/// When ZK sumcheck is used, this stores the polynomial coefficients and
/// blinding factors that are needed to construct the BlindFold witness.
/// The commitments are stored as serialized bytes to keep the accumulator
/// curve-agnostic.
#[cfg(feature = "zk")]
#[derive(Clone, Debug)]
pub struct ZkStageData<F: JoltField> {
    /// Initial batched claim for this sumcheck stage
    pub initial_claim: F,
    /// Pedersen commitments to round polynomials (serialized G1 points)
    pub round_commitments: Vec<Vec<u8>>,
    /// Full polynomial coefficients for each round
    pub poly_coeffs: Vec<Vec<F>>,
    /// Blinding factors used for Pedersen commitments (one per round)
    pub blinding_factors: Vec<F>,
    /// Challenges derived during this sumcheck
    pub challenges: Vec<F::Challenge>,
    /// Batching coefficients for this stage (one per batched instance).
    /// Used in final output constraint: final_claim = Σⱼ αⱼ · yⱼ
    pub batching_coefficients: Vec<F>,
    /// Expected output evaluations for each batched instance.
    /// These are the polynomial evaluations at the random sumcheck point,
    /// proven correct via ZK-Dory externally.
    pub expected_evaluations: Vec<F>,
    pub output_constraints: Vec<Option<crate::subprotocols::blindfold::OutputClaimConstraint>>,
    pub constraint_challenge_values: Vec<Vec<F>>,
    pub input_constraints: Vec<crate::subprotocols::blindfold::InputClaimConstraint>,
    pub input_constraint_challenge_values: Vec<Vec<F>>,
    pub input_claim_scaling_exponents: Vec<usize>,
    pub output_claims_blinding: F,
    pub output_claims_commitment_bytes: Vec<u8>,
}

/// ZK data for uni-skip first round (Stages 1-2).
/// Unlike regular sumcheck, uni-skip uses full polynomial (not compressed).
#[cfg(feature = "zk")]
#[derive(Clone, Debug)]
pub struct UniSkipStageData<F: JoltField> {
    /// Initial claim for this uni-skip round
    pub input_claim: F,
    /// Full polynomial coefficients (not compressed)
    pub poly_coeffs: Vec<F>,
    /// Blinding factor for Pedersen commitment
    pub blinding_factor: F,
    /// Challenge derived after committing
    pub challenge: F::Challenge,
    /// Polynomial degree
    pub poly_degree: usize,
    /// Serialized commitment bytes
    pub commitment_bytes: Vec<u8>,
    pub input_constraint: crate::subprotocols::blindfold::InputClaimConstraint,
    pub input_constraint_challenge_values: Vec<F>,
    pub output_claims: Vec<F>,
    pub output_claims_blinding: F,
    pub output_claims_commitment_bytes: Vec<u8>,
}

/// Accumulates openings computed by the prover over the course of Jolt,
/// so that they can all be reduced to a single opening proof using sumcheck.
#[derive(Clone, Allocative)]
pub struct ProverOpeningAccumulator<F>
where
    F: JoltField,
{
    pub openings: Openings<F>,
    #[cfg(test)]
    pub appended_virtual_openings: RefCell<Vec<OpeningId>>,
    log_T: usize,
    #[cfg(feature = "zk")]
    #[allocative(skip)]
    zk_stage_data: Vec<ZkStageData<F>>,
    #[cfg(feature = "zk")]
    #[allocative(skip)]
    uniskip_stage_data: Vec<UniSkipStageData<F>>,
    /// In ZK mode, skip absorbing cleartext claims into the transcript.
    pub zk_mode: bool,
    #[allocative(skip)]
    pending_claims: Vec<F>,
}

/// Accumulates openings encountered by the verifier over the course of Jolt,
/// so that they can all be reduced to a single opening proof verification using sumcheck.
pub struct VerifierOpeningAccumulator<F>
where
    F: JoltField,
{
    pub openings: Openings<F>,
    /// In testing, the Jolt verifier may be provided the prover's openings so that we
    /// can detect any places where the openings don't match up.
    #[cfg(test)]
    prover_opening_accumulator: Option<ProverOpeningAccumulator<F>>,
    pub log_T: usize,
    /// In ZK mode, skip absorbing cleartext claims into the transcript.
    pub zk_mode: bool,
    pending_claims: Vec<F>,
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

    fn get_advice_opening(
        &self,
        kind: AdviceKind,
        sumcheck: SumcheckId,
    ) -> Option<(OpeningPoint<BIG_ENDIAN, F>, F)>;
}

/// State for Dory batch opening (Stage 8).
/// This is a generic interface for batch opening proofs.
#[derive(Clone, Allocative)]
pub struct DoryOpeningState<F: JoltField> {
    /// Unified opening point for all polynomials (length = log_k_chunk + log_T)
    pub opening_point: Vec<F::Challenge>,
    /// γ^i coefficients for the RLC polynomial
    pub gamma_powers: Vec<F>,
    /// (polynomial, claim) pairs at the opening point
    /// (with Lagrange factors already applied for shorter polys)
    pub polynomial_claims: Vec<(CommittedPolynomial, F)>,
}

impl<F: JoltField> DoryOpeningState<F> {
    /// Build streaming RLC polynomial from this state.
    /// Streams directly from trace - no witness regeneration needed.
    /// Advice polynomials are passed separately (not streamed from trace).
    #[tracing::instrument(skip_all)]
    pub fn build_streaming_rlc<PCS: CommitmentScheme<Field = F>>(
        &self,
        one_hot_params: OneHotParams,
        trace_source: TraceSource,
        rlc_streaming_data: Arc<RLCStreamingData>,
        mut opening_hints: HashMap<CommittedPolynomial, PCS::OpeningProofHint>,
        advice_polys: HashMap<CommittedPolynomial, MultilinearPolynomial<F>>,
    ) -> (MultilinearPolynomial<F>, PCS::OpeningProofHint) {
        // Accumulate gamma coefficients per polynomial
        let mut rlc_map = BTreeMap::new();
        for (gamma, (poly, _claim)) in self.gamma_powers.iter().zip(self.polynomial_claims.iter()) {
            *rlc_map.entry(*poly).or_insert(F::zero()) += *gamma;
        }

        let (poly_ids, coeffs): (Vec<CommittedPolynomial>, Vec<F>) =
            rlc_map.iter().map(|(k, v)| (*k, *v)).unzip();

        let joint_poly = MultilinearPolynomial::RLC(RLCPolynomial::new_streaming(
            one_hot_params,
            rlc_streaming_data,
            trace_source,
            poly_ids.clone(),
            &coeffs,
            advice_polys,
        ));

        let hints: Vec<PCS::OpeningProofHint> = rlc_map
            .into_keys()
            .map(|k| opening_hints.remove(&k).unwrap())
            .collect();

        let hint = PCS::combine_hints(hints, &coeffs);

        (joint_poly, hint)
    }
}

impl<F> Default for ProverOpeningAccumulator<F>
where
    F: JoltField,
{
    fn default() -> Self {
        Self::new(0, false)
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
            .get(&OpeningId::virt(polynomial, sumcheck))
            .unwrap_or_else(|| panic!("opening for {sumcheck:?} {polynomial:?} not found"));
        #[cfg(test)]
        {
            let mut virtual_openings = self.appended_virtual_openings.borrow_mut();
            if let Some(index) = virtual_openings
                .iter()
                .position(|id| id == &OpeningId::virt(polynomial, sumcheck))
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
            .get(&OpeningId::committed(polynomial, sumcheck))
            .unwrap_or_else(|| panic!("opening for {sumcheck:?} {polynomial:?} not found"));
        (point.clone(), *claim)
    }

    fn get_advice_opening(
        &self,
        kind: AdviceKind,
        sumcheck_id: SumcheckId,
    ) -> Option<(OpeningPoint<BIG_ENDIAN, F>, F)> {
        let opening_id = match kind {
            AdviceKind::Trusted => OpeningId::TrustedAdvice(sumcheck_id),
            AdviceKind::Untrusted => OpeningId::UntrustedAdvice(sumcheck_id),
        };
        let (point, claim) = self.openings.get(&opening_id)?;
        Some((point.clone(), *claim))
    }
}

impl<F> ProverOpeningAccumulator<F>
where
    F: JoltField,
{
    pub fn new(log_T: usize, zk_mode: bool) -> Self {
        Self {
            openings: BTreeMap::new(),
            #[cfg(test)]
            appended_virtual_openings: std::cell::RefCell::new(vec![]),
            log_T,
            #[cfg(feature = "zk")]
            zk_stage_data: Vec::new(),
            #[cfg(feature = "zk")]
            uniskip_stage_data: Vec::new(),
            zk_mode,
            pending_claims: Vec::new(),
        }
    }

    pub fn evaluation_openings_mut(&mut self) -> &mut Openings<F> {
        &mut self.openings
    }

    /// Get the value of an opening by key
    pub fn get_opening(&self, key: OpeningId) -> F {
        self.openings.get(&key).unwrap().1
    }

    /// Adds an opening of a dense polynomial to the accumulator.
    /// The given `polynomial` is opened at `opening_point`, yielding the claimed
    /// evaluation `claim`.
    #[tracing::instrument(skip_all, name = "ProverOpeningAccumulator::append_dense")]
    pub fn append_dense(
        &mut self,
        polynomial: CommittedPolynomial,
        sumcheck: SumcheckId,
        opening_point: Vec<F::Challenge>,
        claim: F,
    ) {
        let key = OpeningId::committed(polynomial, sumcheck);
        self.openings.insert(
            key,
            (
                OpeningPoint::<BIG_ENDIAN, F>::new(opening_point.clone()),
                claim,
            ),
        );
        self.pending_claims.push(claim);
    }

    #[tracing::instrument(skip_all, name = "ProverOpeningAccumulator::append_sparse")]
    pub fn append_sparse(
        &mut self,
        polynomials: Vec<CommittedPolynomial>,
        sumcheck: SumcheckId,
        r_address: Vec<F::Challenge>,
        r_cycle: Vec<F::Challenge>,
        claims: Vec<F>,
    ) {
        let r_concat = [r_address.as_slice(), r_cycle.as_slice()].concat();

        for (label, claim) in polynomials.iter().zip(claims.iter()) {
            let opening_point_struct = OpeningPoint::<BIG_ENDIAN, F>::new(r_concat.clone());
            let key = OpeningId::committed(*label, sumcheck);
            self.openings
                .insert(key, (opening_point_struct.clone(), *claim));
            self.pending_claims.push(*claim);
        }
    }

    pub fn append_virtual(
        &mut self,
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
        claim: F,
    ) {
        assert!(
            self.openings
                .insert(
                    OpeningId::virt(polynomial, sumcheck),
                    (opening_point, claim),
                )
                .is_none(),
            "Key ({polynomial:?}, {sumcheck:?}) is already in opening map"
        );
        #[cfg(test)]
        self.appended_virtual_openings
            .borrow_mut()
            .push(OpeningId::virt(polynomial, sumcheck));
        self.pending_claims.push(claim);
    }

    pub fn append_untrusted_advice(
        &mut self,
        sumcheck_id: SumcheckId,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
        claim: F,
    ) {
        self.openings.insert(
            OpeningId::UntrustedAdvice(sumcheck_id),
            (opening_point, claim),
        );
        self.pending_claims.push(claim);
    }

    pub fn append_trusted_advice(
        &mut self,
        sumcheck_id: SumcheckId,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
        claim: F,
    ) {
        self.openings.insert(
            OpeningId::TrustedAdvice(sumcheck_id),
            (opening_point, claim),
        );
        self.pending_claims.push(claim);
    }

    #[cfg(feature = "zk")]
    pub fn push_zk_stage_data(&mut self, data: ZkStageData<F>) {
        self.zk_stage_data.push(data);
    }

    #[cfg(feature = "zk")]
    pub fn take_zk_stage_data(&mut self) -> Vec<ZkStageData<F>> {
        std::mem::take(&mut self.zk_stage_data)
    }

    #[cfg(feature = "zk")]
    pub fn push_uniskip_stage_data(&mut self, data: UniSkipStageData<F>) {
        self.uniskip_stage_data.push(data);
    }

    #[cfg(feature = "zk")]
    pub fn take_uniskip_stage_data(&mut self) -> Vec<UniSkipStageData<F>> {
        std::mem::take(&mut self.uniskip_stage_data)
    }

    pub fn flush_to_transcript<T: Transcript>(&mut self, transcript: &mut T) {
        for claim in self.pending_claims.drain(..) {
            transcript.append_scalar(b"opening_claim", &claim);
        }
    }

    pub fn take_pending_claims(&mut self) -> Vec<F> {
        std::mem::take(&mut self.pending_claims)
    }
}

impl<F> Default for VerifierOpeningAccumulator<F>
where
    F: JoltField,
{
    fn default() -> Self {
        Self::new(0, false)
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
            .get(&OpeningId::virt(polynomial, sumcheck))
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
            .get(&OpeningId::committed(polynomial, sumcheck))
            .unwrap_or_else(|| panic!("No opening found for {sumcheck:?} {polynomial:?}"));
        (point.clone(), *claim)
    }

    fn get_advice_opening(
        &self,
        kind: AdviceKind,
        sumcheck_id: SumcheckId,
    ) -> Option<(OpeningPoint<BIG_ENDIAN, F>, F)> {
        let opening_id = match kind {
            AdviceKind::Trusted => OpeningId::TrustedAdvice(sumcheck_id),
            AdviceKind::Untrusted => OpeningId::UntrustedAdvice(sumcheck_id),
        };
        let (point, claim) = self.openings.get(&opening_id)?;
        Some((point.clone(), *claim))
    }
}

impl<F> VerifierOpeningAccumulator<F>
where
    F: JoltField,
{
    pub fn new(log_T: usize, zk_mode: bool) -> Self {
        Self {
            openings: BTreeMap::new(),
            #[cfg(test)]
            prover_opening_accumulator: None,
            log_T,
            zk_mode,
            pending_claims: Vec::new(),
        }
    }

    /// Compare this accumulator to the corresponding `ProverOpeningAccumulator` and panic
    /// if the openings appended differ from the prover's openings.
    #[cfg(test)]
    pub fn compare_to(&mut self, prover_openings: ProverOpeningAccumulator<F>) {
        self.prover_opening_accumulator = Some(prover_openings);
    }

    pub fn append_dense(
        &mut self,
        polynomial: CommittedPolynomial,
        sumcheck: SumcheckId,
        opening_point: Vec<F::Challenge>,
    ) {
        let key = OpeningId::committed(polynomial, sumcheck);
        let claim = self
            .openings
            .get(&key)
            .map(|(_, c)| *c)
            .unwrap_or(F::zero());
        self.openings.insert(
            key,
            (
                OpeningPoint::<BIG_ENDIAN, F>::new(opening_point.clone()),
                claim,
            ),
        );
        self.pending_claims.push(claim);
    }

    pub fn append_sparse(
        &mut self,
        polynomials: Vec<CommittedPolynomial>,
        sumcheck: SumcheckId,
        opening_point: Vec<F::Challenge>,
    ) {
        for label in polynomials.into_iter() {
            let key = OpeningId::committed(label, sumcheck);
            let claim = self
                .openings
                .get(&key)
                .map(|(_, c)| *c)
                .unwrap_or(F::zero());
            self.openings.insert(
                key,
                (
                    OpeningPoint::<BIG_ENDIAN, F>::new(opening_point.clone()),
                    claim,
                ),
            );
            self.pending_claims.push(claim);
        }
    }

    pub fn append_virtual(
        &mut self,
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let key = OpeningId::virt(polynomial, sumcheck);
        let claim = self
            .openings
            .get(&key)
            .map(|(_, c)| *c)
            .unwrap_or(F::zero());
        self.openings.insert(key, (opening_point.clone(), claim));
        self.pending_claims.push(claim);
    }

    pub fn append_untrusted_advice(
        &mut self,
        sumcheck_id: SumcheckId,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let key = OpeningId::UntrustedAdvice(sumcheck_id);
        let claim = self
            .openings
            .get(&key)
            .map(|(_, c)| *c)
            .unwrap_or(F::zero());
        self.openings.insert(key, (opening_point.clone(), claim));
        self.pending_claims.push(claim);
    }

    pub fn append_trusted_advice(
        &mut self,
        sumcheck_id: SumcheckId,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let key = OpeningId::TrustedAdvice(sumcheck_id);
        let claim = self
            .openings
            .get(&key)
            .map(|(_, c)| *c)
            .unwrap_or(F::zero());
        self.openings.insert(key, (opening_point.clone(), claim));
        self.pending_claims.push(claim);
    }

    pub fn flush_to_transcript<T: Transcript>(&mut self, transcript: &mut T) {
        for claim in self.pending_claims.drain(..) {
            transcript.append_scalar(b"opening_claim", &claim);
        }
    }

    pub fn take_pending_claims(&mut self) -> Vec<F> {
        std::mem::take(&mut self.pending_claims)
    }
}

/// Computes the Lagrange factor for embedding a smaller "advice" polynomial into the top-left
/// block of the main Dory matrix.
///
/// Advice polynomials have fewer variables than main polynomials. To batch them together,
/// we embed advice in the top-left corner of the larger matrix and multiply by a Lagrange
/// selector that is 1 on that block and 0 elsewhere:
///
/// ```text
/// Lagrange factor = ∏_{r ∈ opening_point, r ∉ advice_opening_point} (1 - r)
/// ```
///
/// # Arguments
/// - `opening_point`: The unified opening point for the Dory opening proof
/// - `advice_opening_point`: The opening point for the advice polynomial
///
/// # Returns
/// The Lagrange factor as a field element
pub fn compute_advice_lagrange_factor<F: JoltField>(
    opening_point: &[F::Challenge],
    advice_opening_point: &[F::Challenge],
) -> F {
    #[cfg(test)]
    {
        for r in advice_opening_point.iter() {
            assert!(opening_point.contains(r));
        }
    }
    opening_point
        .iter()
        .map(|r| {
            if advice_opening_point.contains(r) {
                F::one()
            } else {
                F::one() - r
            }
        })
        .product()
}
