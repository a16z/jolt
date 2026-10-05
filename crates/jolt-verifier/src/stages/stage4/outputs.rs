//! Typed inputs consumed and outputs produced by stage 4 verification.

use jolt_field::JoltField;
use jolt_sumcheck::BatchedCommittedSumcheckConsistency;

use crate::stages::relations::{ClaimRoutes, SumcheckBatch};
use crate::stages::zk::outputs::CommittedOutputClaimOutput;
use crate::VerifierError;

#[cfg(feature = "field-inline")]
pub use super::field_registers_read_write_checking::{
    FieldRegistersReadWriteChecking, FieldRegistersReadWriteOutputClaims,
};
use super::ram_val_check::{
    RamValCheck, RamValCheckInitialEvaluation, RamValCheckOutputClaims, RamValCheckStagedOpenings,
};
use super::registers_read_write_checking::{
    RegistersReadWriteChecking, RegistersReadWriteOutputClaims,
};

/// Source-of-truth for stage 4's sumcheck batch, in Fiat-Shamir batch order (registers
/// read-write, the field-inline field-register read-write when composed, then RAM
/// value-check). `#[derive(SumcheckBatch)]` generates the `Stage4InputClaims<F>`,
/// `Stage4InputPoints<F>`, `Stage4OutputClaims<F>`, `Stage4OutputPoints<F>`, and
/// `Stage4Challenges<F>` aggregates — one field per instance, in this declaration order.
///
/// Besides its `ram_ra`/`ram_inc`, the RAM value-check member produces the
/// staged `Val_init` advice and program-image contribution openings
/// ([`RamValCheckStagedOpenings`]),
/// routed [`ClaimRoute::Staged`](crate::stages::relations::ClaimRoute::Staged):
/// a clear proof sends them before the batch, and a committed proof commits
/// them in declaration order with the rest of the stage's claims.
#[derive(SumcheckBatch)]
#[sumcheck_batch(routes, crate = "crate")]
pub struct Stage4Sumchecks<F: JoltField> {
    pub registers_read_write: RegistersReadWriteChecking<F>,
    /// The field-inline Twist read/write instance over `T * 2^log_k`. Declaration position
    /// (after the ordinary registers read-write, before the RAM value-check) is the spec's
    /// stage-4 batch order and gamma draw order (`specs/field-inline-protocol.md`, "Stage 4
    /// Composition").
    #[cfg(feature = "field-inline")]
    pub field_registers_read_write: FieldRegistersReadWriteChecking<F>,
    pub ram_val_check: RamValCheck<F>,
}

impl<F: JoltField> Stage4OutputClaims<F> {
    /// Construct the ordinary stage-4 claims. Producers without field-inline semantics use
    /// this regardless of the build's feature set — the field-inline read-write slot defaults
    /// to all-zero claims, inert because such producers' proofs never declare the field-inline
    /// axis.
    pub fn new(
        registers_read_write: RegistersReadWriteOutputClaims<F>,
        ram_val_check: RamValCheckOutputClaims<F>,
    ) -> Self {
        Self {
            registers_read_write,
            #[cfg(feature = "field-inline")]
            field_registers_read_write: Default::default(),
            ram_val_check,
        }
    }
}

/// The shared opening-point accessors over the point-only stage-4 aggregate.
impl<F: JoltField> Stage4OutputPoints<F> {
    /// The stage's claim routes: the RAM value check's staged contribution
    /// cells are [`ClaimRoute::Staged`](crate::stages::relations::ClaimRoute::Staged).
    pub fn claim_routes(&self) -> Result<ClaimRoutes, VerifierError> {
        Ok(RamValCheckStagedOpenings::from_claims(&self.ram_val_check).claim_routes())
    }

    /// The register read-write opening point (shared by all five register
    /// openings).
    pub fn registers_read_write_point(&self) -> &[F] {
        self.registers_read_write.registers_val()
    }

    /// The field-register read-write opening point (shared by all five field-inline openings).
    #[cfg(feature = "field-inline")]
    pub fn field_registers_read_write_point(&self) -> &[F] {
        self.field_registers_read_write.registers_val()
    }

    /// The RAM value-check opening point (shared by `ram_ra`/`ram_inc`).
    pub fn ram_val_check_point(&self) -> &[F] {
        self.ram_val_check.ram_ra()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
pub struct Stage4ClearOutput<F: JoltField> {
    /// The produced stage-4 opening *values* (wire form); read by later stages and
    /// the Fiat-Shamir opening-claim encoder.
    pub output_values: Stage4OutputClaims<F>,
    /// The produced stage-4 opening *points*, paired field-for-field with
    /// `output_values`.
    pub output_points: Stage4OutputPoints<F>,
    pub ram_val_check_init: RamValCheckInitialEvaluation<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4ZkOutput<F: JoltField, C> {
    pub challenges: Stage4Challenges<F>,
    pub batch_consistency: BatchedCommittedSumcheckConsistency<F, C>,
    pub batch_output_claims: CommittedOutputClaimOutput<C>,
    pub ram_val_check_public_eval: F,
    /// The produced opening points, the ZK counterpart of the clear path's
    /// `output_points`. Read through the same `*_point()` accessors.
    pub output_points: Stage4OutputPoints<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage4Output<F: JoltField, C> {
    Clear(Stage4ClearOutput<F>),
    Zk(Stage4ZkOutput<F, C>),
}

impl<F: JoltField, C> Stage4Output<F, C> {
    /// The produced opening points, available regardless of proving mode.
    pub fn output_points(&self) -> &Stage4OutputPoints<F> {
        match self {
            Self::Clear(output) => &output.output_points,
            Self::Zk(output) => &output.output_points,
        }
    }

    pub fn clear(&self) -> Result<&Stage4ClearOutput<F>, crate::VerifierError> {
        match self {
            Self::Clear(output) => Ok(output),
            Self::Zk(_) => Err(crate::VerifierError::ExpectedClearProof { field: "stage4" }),
        }
    }

    pub fn zk(&self) -> Result<&Stage4ZkOutput<F, C>, crate::VerifierError> {
        match self {
            Self::Zk(output) => Ok(output),
            Self::Clear(_) => Err(crate::VerifierError::ExpectedCommittedProof { field: "stage4" }),
        }
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::stages::relations::test_transcript::assert_same_draws;
    #[cfg(feature = "field-inline")]
    use jolt_claims::protocols::field_inline::FieldInlineConfig;
    use jolt_claims::protocols::jolt::geometry::dimensions::{
        ReadWriteDimensions, TraceDimensions, REGISTER_ADDRESS_BITS,
    };
    use jolt_claims::protocols::jolt::geometry::ram::RamValCheckInit;
    use jolt_claims::protocols::jolt::relations::ram::RamValCheckOutputClaims;
    use jolt_claims::protocols::jolt::relations::registers::RegistersReadWriteOutputClaims;
    use jolt_field::{Fr, Ring};
    use jolt_transcript::Channel;

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn registers_claims() -> RegistersReadWriteOutputClaims<Fr> {
        RegistersReadWriteOutputClaims {
            registers_val: fr(3),
            rs1_ra: fr(4),
            rs2_ra: fr(5),
            rd_wa: fr(6),
            rd_inc: fr(7),
        }
    }

    fn claims_with_advice(with_advice: bool) -> Stage4OutputClaims<Fr> {
        Stage4OutputClaims::<Fr> {
            registers_read_write: registers_claims(),
            #[cfg(feature = "field-inline")]
            field_registers_read_write: FieldRegistersReadWriteOutputClaims {
                registers_val: fr(21),
                rs1_ra: fr(22),
                rs2_ra: fr(23),
                rd_wa: fr(24),
                rd_inc: fr(25),
            },
            ram_val_check: RamValCheckOutputClaims {
                untrusted_advice: with_advice.then(|| fr(1)),
                trusted_advice: with_advice.then(|| fr(2)),
                program_image: with_advice.then(|| fr(10)),
                ram_ra: fr(8),
                ram_inc: fr(9),
            },
        }
    }

    /// Under `field-inline` the five field-register read-write openings splice between the
    /// register and RAM value-check openings — the spec's committed row order
    /// (`specs/field-inline-protocol.md`, "Stage 4 Composition").
    #[cfg(feature = "field-inline")]
    fn field_inline_splice() -> Vec<Fr> {
        (21..=25).map(fr).collect()
    }

    #[cfg(not(feature = "field-inline"))]
    fn field_inline_splice() -> Vec<Fr> {
        Vec::new()
    }

    /// The staged routes when every staged contribution is present.
    fn staged_routes() -> ClaimRoutes {
        RamValCheckStagedOpenings::<Vec<Fr>> {
            untrusted_advice: Some(Vec::new()),
            trusted_advice: Some(Vec::new()),
            program_image: Some(Vec::new()),
        }
        .claim_routes()
    }

    /// The clear post-round claims, pinned with distinct sentinels: the five
    /// register openings, under `field-inline` the five field-register openings,
    /// then `ram_ra`/`ram_inc`; the staged contributions travel before the batch.
    #[test]
    fn wire_claims_omit_staged_openings() {
        let expected: Vec<Fr> = (3..=7)
            .map(fr)
            .chain(field_inline_splice())
            .chain([fr(8), fr(9)])
            .collect();
        assert_eq!(
            Stage4Sumchecks::wire_claim_values(&claims_with_advice(true), &staged_routes()),
            expected
        );
    }

    /// The committed rows are declaration order: the registers, under
    /// `field-inline` the field registers, then the RAM value check's fields
    /// (untrusted, trusted, program image, `ram_ra`, `ram_inc`).
    #[test]
    fn committed_claims_follow_declaration_order() {
        let expected: Vec<Fr> = (3..=7)
            .map(fr)
            .chain(field_inline_splice())
            .chain([fr(1), fr(2), fr(10), fr(8), fr(9)])
            .collect();
        assert_eq!(
            Stage4Sumchecks::committed_claim_values(&claims_with_advice(true), &staged_routes()),
            expected
        );
    }

    fn sumchecks() -> Stage4Sumchecks<Fr> {
        let log_t = 4usize;
        let ram_log_k = 3usize;
        Stage4Sumchecks::<Fr> {
            registers_read_write: RegistersReadWriteChecking::new(ReadWriteDimensions::new(
                log_t,
                REGISTER_ADDRESS_BITS,
                2,
                1,
            )),
            #[cfg(feature = "field-inline")]
            field_registers_read_write: FieldRegistersReadWriteChecking::new(
                FieldInlineConfig::enabled().read_write_dimensions(log_t),
            ),
            ram_val_check: RamValCheck::new(
                TraceDimensions::new(log_t),
                ram_log_k,
                RamValCheckInit::from(fr(0)),
                RamValCheckStagedOpenings::default(),
            ),
        }
    }

    /// The batch draws one uniform gamma per member in declaration order: the
    /// registers gamma, under `field-inline` the field-register read-write gamma
    /// (the spec's draw slot: after the registers gamma, before the RAM
    /// value-check gamma), then the RAM value-check gamma.
    #[test]
    fn draw_challenges_follow_member_order() {
        let sumchecks = sumchecks();
        #[cfg(not(feature = "field-inline"))]
        let gamma_draws = 2usize;
        #[cfg(feature = "field-inline")]
        let gamma_draws = 3usize;
        let (challenges, gammas) = assert_same_draws(
            |t| sumchecks.draw_challenges(t).unwrap(),
            |t| (0..gamma_draws).map(|_| t.challenge()).collect::<Vec<Fr>>(),
        );

        #[cfg(not(feature = "field-inline"))]
        let drawn_gammas = vec![
            challenges.registers_read_write.gamma,
            challenges.ram_val_check.gamma,
        ];
        #[cfg(feature = "field-inline")]
        let drawn_gammas = vec![
            challenges.registers_read_write.gamma,
            challenges.field_registers_read_write.gamma,
            challenges.ram_val_check.gamma,
        ];
        assert_eq!(drawn_gammas, gammas);
    }
}
