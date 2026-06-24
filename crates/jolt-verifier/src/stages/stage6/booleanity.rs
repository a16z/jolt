//! The stage 6 booleanity sumcheck instances (address phase + cycle phase).
//!
//! Booleanity proves every one-hot `Ra` chunk (instruction, bytecode, RAM) is
//! boolean. It runs in two phases: the stage-6a address phase binds the
//! `log_k_chunk` address variables and stages the `BooleanityAddrClaim`
//! intermediate; the stage-6b cycle phase binds the `log_t` cycle variables and
//! opens the committed per-family `Ra` claims. The cycle phase's single public,
//! `EqAddressCycle`, ties the full two-phase sumcheck point to the reference
//! address/cycle drawn from the stage-5 instruction opening.

use core::marker::PhantomData;

use jolt_claims::protocols::jolt::{
    formulas::booleanity::{self, BooleanityDimensions},
    BooleanityChallenge, BooleanityPublic, JoltChallengeId, JoltOpeningId, JoltPublicId,
    JoltRelationClaims, JoltRelationId,
};
use jolt_field::Field;
use jolt_poly::try_eq_mle;
use jolt_verifier_derive::{InputClaims, OutputClaims};
use serde::{Deserialize, Serialize};

use crate::stages::relations::{GetPoint, OpeningClaim, ConcreteSumcheck};
use crate::VerifierError;

// ---------------------------------------------------------------------------
// Address phase (stage 6a)
// ---------------------------------------------------------------------------

/// The staged `BooleanityAddrClaim` intermediate produced by the address phase
/// and consumed by the cycle phase.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(Booleanity)]
pub struct BooleanityAddressPhaseOutputClaims<C> {
    #[opening(BooleanityAddrClaim)]
    pub intermediate: C,
}

/// The address phase consumes no openings (its input claim is the constant zero).
pub struct BooleanityAddressPhaseInputClaims<C> {
    _cell: PhantomData<C>,
}

impl<C> Default for BooleanityAddressPhaseInputClaims<C> {
    fn default() -> Self {
        Self { _cell: PhantomData }
    }
}

impl<F: Field> BooleanityAddressPhaseInputClaims<OpeningClaim<F>> {
    pub fn from_upstream() -> Self {
        Self::default()
    }
}

impl<F: Field> crate::stages::relations::InputClaims<F>
    for BooleanityAddressPhaseInputClaims<OpeningClaim<F>>
{
    fn resolve_input(&self, _id: &JoltOpeningId) -> Option<F> {
        None
    }
}

pub struct BooleanityAddressPhase<F: Field> {
    claims: JoltRelationClaims<F>,
}

impl<F: Field> BooleanityAddressPhase<F> {
    pub fn new(dimensions: BooleanityDimensions) -> Self {
        Self {
            claims: booleanity::booleanity_address_phase(dimensions),
        }
    }
}

impl<F: Field> ConcreteSumcheck<F> for BooleanityAddressPhase<F> {
    type Inputs<C> = BooleanityAddressPhaseInputClaims<C>;
    type Outputs<C> = BooleanityAddressPhaseOutputClaims<C>;

    fn sumcheck_relation(&self) -> &JoltRelationClaims<F> {
        &self.claims
    }

    fn derive_opening_points<C: GetPoint<F>>(
        &self,
        sumcheck_point: &[F],
        _inputs: &BooleanityAddressPhaseInputClaims<C>,
    ) -> Result<BooleanityAddressPhaseOutputClaims<Vec<F>>, VerifierError> {
        // The address opening point (`booleanity_r_address`) is the reversed
        // address sumcheck point; the cycle phase prepends it to its cycle point.
        Ok(BooleanityAddressPhaseOutputClaims {
            intermediate: sumcheck_point.iter().rev().copied().collect(),
        })
    }
}

// ---------------------------------------------------------------------------
// Cycle phase (stage 6b)
// ---------------------------------------------------------------------------

/// The committed per-family `Ra` openings produced by the cycle phase; every
/// opening shares the single booleanity opening point (`r_address ++ r_cycle`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(Booleanity)]
pub struct BooleanityOutputClaims<C> {
    #[opening(committed = InstructionRa)]
    pub instruction_ra: Vec<C>,
    #[opening(committed = BytecodeRa)]
    pub bytecode_ra: Vec<C>,
    #[opening(committed = RamRa)]
    pub ram_ra: Vec<C>,
}

/// The `BooleanityAddrClaim` intermediate consumed from the address phase.
#[derive(Clone, Debug, InputClaims)]
pub struct BooleanityInputClaims<C> {
    #[opening(BooleanityAddrClaim, from = Booleanity)]
    pub address_phase: C,
}

impl<F: Field> BooleanityInputClaims<OpeningClaim<F>> {
    pub fn from_upstream(address_phase: OpeningClaim<F>) -> Self {
        Self { address_phase }
    }
}

pub struct Booleanity<F: Field> {
    claims: JoltRelationClaims<F>,
    dimensions: BooleanityDimensions,
    gamma: F,
    /// The address opening prefix from the stage-6a phase.
    r_address: Vec<F>,
    /// The reference address/cycle the `EqAddressCycle` public compares against.
    reference_address: Vec<F>,
    reference_cycle: Vec<F>,
}

impl<F: Field> Booleanity<F> {
    pub fn new(
        dimensions: BooleanityDimensions,
        gamma: F,
        r_address: Vec<F>,
        reference_address: Vec<F>,
        reference_cycle: Vec<F>,
    ) -> Self {
        Self {
            claims: booleanity::booleanity_cycle_phase(dimensions),
            dimensions,
            gamma,
            r_address,
            reference_address,
            reference_cycle,
        }
    }
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::Booleanity,
        reason: reason.to_string(),
    }
}

impl<F: Field> ConcreteSumcheck<F> for Booleanity<F> {
    type Inputs<C> = BooleanityInputClaims<C>;
    type Outputs<C> = BooleanityOutputClaims<C>;

    fn sumcheck_relation(&self) -> &JoltRelationClaims<F> {
        &self.claims
    }

    fn derive_opening_points<C: GetPoint<F>>(
        &self,
        sumcheck_point: &[F],
        _inputs: &BooleanityInputClaims<C>,
    ) -> Result<BooleanityOutputClaims<Vec<F>>, VerifierError> {
        let r_cycle = sumcheck_point.iter().rev().copied().collect::<Vec<_>>();
        let opening_point = [self.r_address.as_slice(), r_cycle.as_slice()].concat();
        let layout = self.dimensions.layout;
        Ok(BooleanityOutputClaims {
            instruction_ra: vec![opening_point.clone(); layout.instruction()],
            bytecode_ra: vec![opening_point.clone(); layout.bytecode()],
            ram_ra: vec![opening_point; layout.ram()],
        })
    }

    fn resolve_challenge(&self, id: &JoltChallengeId) -> Result<F, VerifierError> {
        match id {
            JoltChallengeId::Booleanity(BooleanityChallenge::Gamma) => Ok(self.gamma),
            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        }
    }

    fn resolve_public<C: GetPoint<F>>(
        &self,
        id: &JoltPublicId,
        _inputs: &BooleanityInputClaims<C>,
        outputs: Option<&BooleanityOutputClaims<OpeningClaim<F>>>,
    ) -> Result<F, VerifierError> {
        let outputs =
            outputs.ok_or(VerifierError::MissingStageClaimPublic { id: *id })?;
        let JoltPublicId::Booleanity(BooleanityPublic::EqAddressCycle) = id else {
            return Err(VerifierError::MissingStageClaimPublic { id: *id });
        };
        // Recover the raw two-phase sumcheck point from a produced opening point
        // (`r_address ++ r_cycle`): each half is the reverse of its phase's
        // sumcheck sub-point, and `EqAddressCycle` compares `[6a ++ 6b]` against
        // `reversed(reference_address) ++ reversed(reference_cycle)`.
        let opening_point = outputs
            .instruction_ra
            .first()
            .or_else(|| outputs.bytecode_ra.first())
            .or_else(|| outputs.ram_ra.first())
            .map(GetPoint::point)
            .ok_or_else(|| public_input_failed("booleanity produced no openings"))?;
        let log_k_chunk = self.dimensions.log_k_chunk;
        let (r_address, r_cycle) = opening_point.split_at(log_k_chunk);
        let full_sumcheck_point = r_address
            .iter()
            .rev()
            .chain(r_cycle.iter().rev())
            .copied()
            .collect::<Vec<_>>();
        let reference_eq_point = self
            .reference_address
            .iter()
            .rev()
            .chain(self.reference_cycle.iter().rev())
            .copied()
            .collect::<Vec<_>>();
        try_eq_mle(&full_sumcheck_point, &reference_eq_point).map_err(public_input_failed)
    }
}
