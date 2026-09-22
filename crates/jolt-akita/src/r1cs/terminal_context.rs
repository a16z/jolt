//! Conditional native terminal suffix; earlier replay authentication is external.
//! See `specs/akita-wrapper/terminal-context.md` for the exact entry boundary.
use akita_types::{GrindingPlan, GrindingRun, GrindingSite, TerminalFoldParams};
use jolt_field::Fr;
use jolt_r1cs::fp128_bn254::{Fp128Error, Fp128Var};
use jolt_r1cs::R1csBuilder;
use jolt_transcript::r1cs::Blake2bR1csError;
use thiserror::Error;

use super::terminal::TerminalZ;
use super::terminal_relation::{
    TerminalChallengesVar, TerminalRelationError, TerminalRelationInputs, TerminalRelationProfile,
    TerminalRelationWitness,
};
use super::{
    AkitaTranscriptVar, D64AcceptedVar, D64FoldDrawShape, D64FoldDrawVar, D64RetryProfile,
    FoldDrawError, FoldResponseNonceVar,
};

#[derive(Debug, Error)]
pub enum TerminalContextError {
    #[error("terminal context does not match the native public plan or input shape")]
    Shape,
    #[error(transparent)]
    Field(#[from] Fp128Error),
    #[error(transparent)]
    Blake(#[from] Blake2bR1csError),
    #[error(transparent)]
    Draw(#[from] FoldDrawError),
    #[error(transparent)]
    Relation(#[from] TerminalRelationError),
}

/// Public continuation position, not evidence of prior constrained replay.
/// A whole-verifier caller must reach this position through its actual schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TerminalPlanPosition {
    pub query_count: u64,
    pub nonce_bits: usize,
}

/// The same incoming predecessor values feed transcript and terminal equations.
/// No independent terminal t/point/claim witness copies are accepted.
pub struct TerminalIncomingValues<'a> {
    pub t: &'a [Fp128Var],
    pub point: &'a [Fp128Var],
    pub claim: &'a Fp128Var,
}

/// Checked profile for one complete terminal group and its exact final plan runs.
/// Setup/plan authenticity and the inherited transcript prefix remain external.
pub struct TerminalContextProfile {
    relation: TerminalRelationProfile,
    draw: D64FoldDrawShape,
    capacity: D64RetryProfile,
    incoming: TerminalPlanPosition,
    outgoing: TerminalPlanPosition,
    level: u32,
}

impl TerminalContextProfile {
    pub fn new(
        params: &TerminalFoldParams,
        a: &[u128],
        plan: &GrindingPlan,
        capacity: D64RetryProfile,
    ) -> Result<Self, TerminalContextError> {
        let relation = TerminalRelationProfile::new(params, a)?;
        let [prefix @ .., response, coordinates] = plan.runs() else {
            return Err(TerminalContextError::Shape);
        };
        let GrindingSite::FoldResponse { level } = response.site() else {
            return Err(TerminalContextError::Shape);
        };
        if *response != GrindingRun::fold_response(level)
            || *coordinates
                != GrindingRun::fold_challenge_group(level, 0, 7)
                    .map_err(|_| TerminalContextError::Shape)?
        {
            return Err(TerminalContextError::Shape);
        }
        let mut incoming = TerminalPlanPosition {
            query_count: 0,
            nonce_bits: 0,
        };
        for run in prefix {
            incoming.query_count = incoming
                .query_count
                .checked_add(run.multiplicity())
                .ok_or(TerminalContextError::Shape)?;
            let bits = usize::try_from(run.multiplicity())
                .ok()
                .and_then(|n| n.checked_mul(usize::from(run.nonce_bits())))
                .ok_or(TerminalContextError::Shape)?;
            incoming.nonce_bits = incoming
                .nonce_bits
                .checked_add(bits)
                .ok_or(TerminalContextError::Shape)?;
        }
        Ok(Self {
            relation,
            draw: D64FoldDrawShape::new(0, 7, 1)?,
            capacity,
            incoming,
            outgoing: TerminalPlanPosition {
                query_count: plan.expanded_query_count(),
                nonce_bits: plan.total_nonce_bits(),
            },
            level,
        })
    }

    pub fn incoming_position(&self) -> TerminalPlanPosition {
        self.incoming
    }

    pub fn level(&self) -> u32 {
        self.level
    }

    /// Consume the native suffix and enforce every terminal row on shared handles.
    /// The transcript must be the actual prior constrained replay at `position`;
    /// this equality checks public scheduling, not authenticity of that prefix.
    /// No terminal z bytes are needed for existential typed proof acceptance.
    /// The final dead z absorb is omitted. The incoming transcript is consumed;
    /// no native post-proof state is returned or available for later challenges.
    pub fn enforce(
        &self,
        builder: &mut R1csBuilder<Fr>,
        mut transcript: AkitaTranscriptVar,
        position: TerminalPlanPosition,
        input: TerminalContextInputs<'_>,
        witness: TerminalRelationWitness<'_>,
    ) -> Result<TerminalContextVar, TerminalContextError> {
        if position != self.incoming
            || input.incoming.t.len() != 1344
            || input.incoming.point.len() != 17
            || input.e.len() != 448
        {
            return Err(TerminalContextError::Shape);
        }
        let mut t_bytes = Vec::with_capacity(1344 * 16);
        for field in input.incoming.t {
            t_bytes.extend(field.to_le_bytes(builder)?);
        }
        transcript.append_bytes(builder, &t_bytes)?;
        for field in input.incoming.point {
            let bytes = field.to_le_bytes(builder)?;
            transcript.append_bytes(builder, &bytes)?;
        }
        let claim_bytes = input.incoming.claim.to_le_bytes(builder)?;
        transcript.append_bytes(builder, &claim_bytes)?;
        let mut e_bytes = Vec::with_capacity(448 * 16);
        for field in input.e {
            e_bytes.extend(field.to_le_bytes(builder)?);
        }
        transcript.append_bytes(builder, &e_bytes)?;
        let draw = self.draw.draw(builder, &mut transcript, input.nonce)?;
        let mut accepted = Vec::with_capacity(7);
        for index in 0..7 {
            accepted.push(draw.sample_coordinate(builder, index, self.capacity)?);
        }
        let coefficients: Vec<_> = accepted.iter().map(|x| *x.coefficients()).collect();
        let challenges = TerminalChallengesVar::bind(builder, &coefficients)?;
        self.relation.enforce(
            builder,
            TerminalRelationInputs {
                z: input.z,
                e: input.e,
                t: input.incoming.t,
                point: input.incoming.point,
                claim: input.incoming.claim,
                challenges: &challenges,
            },
            witness,
        )?;
        Ok(TerminalContextVar {
            draw,
            accepted,
            completed: self.outgoing,
        })
    }
}

pub struct TerminalContextInputs<'a> {
    pub incoming: TerminalIncomingValues<'a>,
    pub e: &'a [Fp128Var],
    pub z: &'a TerminalZ,
    pub nonce: &'a FoldResponseNonceVar,
}

/// Conditional terminal acceptance and its shared derived challenges.
/// This does not authenticate the inherited prefix or public setup.
pub struct TerminalContextVar {
    pub draw: D64FoldDrawVar,
    pub accepted: Vec<D64AcceptedVar>,
    pub completed: TerminalPlanPosition,
}
