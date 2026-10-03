use jolt_field::JoltField;
use jolt_sumcheck::BatchedCommittedSumcheckConsistency;
use jolt_transcript::Transcript;

use crate::stages::relations::{OutputClaims, SumcheckBatch};
use crate::stages::zk::outputs::CommittedOutputClaimOutput;

#[cfg(feature = "field-inline")]
pub use super::field_registers_read_write_checking::{
    FieldRegistersReadWriteChecking, FieldRegistersReadWriteOutputClaims,
};
use super::ram_val_check::{RamValCheck, RamValCheckInitialEvaluation, RamValCheckOutputClaims};
use super::registers_read_write_checking::{
    RegistersReadWriteChecking, RegistersReadWriteOutputClaims,
};

#[derive(SumcheckBatch)]
#[sumcheck_batch(no_opening_values, crate = "crate")]
pub struct Stage4Sumchecks<F: JoltField> {
    pub registers_read_write: RegistersReadWriteChecking<F>,
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

impl<F: JoltField> Stage4Sumchecks<F> {
    pub fn opening_values(&self, claims: &Stage4OutputClaims<F>) -> Vec<F> {
        claims.opening_values()
    }
}

impl<F: JoltField> Stage4OutputClaims<F> {
    pub fn opening_values(&self) -> Vec<F> {
        let ram = &self.ram_val_check;
        let mut values: Vec<F> = ram
            .untrusted_advice
            .into_iter()
            .chain(ram.trusted_advice)
            .chain(ram.program_image)
            .chain(self.registers_read_write.opening_values())
            .collect();
        #[cfg(feature = "field-inline")]
        values.extend(self.field_registers_read_write.opening_values());
        values.extend([ram.ram_ra, ram.ram_inc]);
        values
    }

    pub fn append_to_transcript<T: Transcript<Challenge = F>>(&self, transcript: &mut T) {
        for value in self.opening_values() {
            transcript.append_labeled(b"opening_claim", &value);
        }
    }
}

impl<F: JoltField> Stage4OutputPoints<F> {
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
    pub output_values: Stage4OutputClaims<F>,
    /// The produced stage-4 opening *points*, paired field-for-field with
    /// `output_values` for the register and RAM value-check leaves. The advice /
    /// program-image opening points are carried on `ram_val_check_init` (they sit at
    /// the staged RAM address sub-point, not the batch sumcheck point), so they are
    /// left absent here.
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
    /// `output_points`. Read through the same `*_point()` accessors. The advice /
    /// program-image leaves are absent in ZK (BlindFold carries those openings), so
    /// only the register and RAM value-check points are populated.
    pub output_points: Stage4OutputPoints<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage4Output<F: JoltField, C> {
    Clear(Stage4ClearOutput<F>),
    Zk(Stage4ZkOutput<F, C>),
}

impl<F: JoltField, C> Stage4Output<F, C> {
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
    use crate::stages::relations::draw_recording::{record, DrawEvent};
    #[cfg(feature = "field-inline")]
    use jolt_claims::protocols::field_inline::FieldInlineConfig;
    use jolt_claims::protocols::jolt::geometry::dimensions::{
        ReadWriteDimensions, TraceDimensions, REGISTER_ADDRESS_BITS,
    };
    use jolt_claims::protocols::jolt::geometry::ram::RamValCheckInit;
    use jolt_claims::protocols::jolt::relations::ram::RamValCheckOutputClaims;
    use jolt_claims::protocols::jolt::relations::registers::RegistersReadWriteOutputClaims;
    use jolt_field::{Fr, Ring};
    use jolt_transcript::Transcript;

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

    #[cfg(feature = "field-inline")]
    fn field_inline_splice() -> Vec<Fr> {
        (21..=25).map(fr).collect()
    }

    #[cfg(not(feature = "field-inline"))]
    fn field_inline_splice() -> Vec<Fr> {
        Vec::new()
    }

    #[test]
    fn opening_values_follow_canonical_order_without_advice() {
        let expected: Vec<Fr> = (3..=7)
            .map(fr)
            .chain(field_inline_splice())
            .chain([fr(8), fr(9)])
            .collect();
        assert_eq!(claims_with_advice(false).opening_values(), expected);
    }

    #[test]
    fn opening_values_interleave_advice_then_registers_then_ram() {
        let expected: Vec<Fr> = [fr(1), fr(2), fr(10)]
            .into_iter()
            .chain((3..=7).map(fr))
            .chain(field_inline_splice())
            .chain([fr(8), fr(9)])
            .collect();
        assert_eq!(claims_with_advice(true).opening_values(), expected);
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
            ),
        }
    }

    #[test]
    fn draw_challenges_matches_inline_draw_sequence() {
        use crate::stages::relations::ConcreteSumcheck as _;

        let sumchecks = sumchecks();
        #[cfg(not(feature = "field-inline"))]
        let leading_gamma_draws = 1usize;
        #[cfg(feature = "field-inline")]
        let leading_gamma_draws = 2usize;
        let (inline_events, (inline_gammas, inline_ram_gamma)) = record(|t| {
            let gammas = (0..leading_gamma_draws)
                .map(|_| t.challenge_scalar())
                .collect::<Vec<Fr>>();
            let ram = sumchecks.ram_val_check.draw_challenges(t).unwrap();
            (gammas, ram.gamma)
        });
        let (draw_events, challenges) = record(|t| sumchecks.draw_challenges(t).unwrap());

        assert_eq!(draw_events, inline_events);
        assert!(matches!(draw_events.first(), Some(DrawEvent::Squeeze(1))));
        assert!(draw_events
            .iter()
            .any(|event| matches!(event, DrawEvent::Append(_))));
        #[cfg(not(feature = "field-inline"))]
        let drawn_gammas = vec![challenges.registers_read_write.gamma];
        #[cfg(feature = "field-inline")]
        let drawn_gammas = vec![
            challenges.registers_read_write.gamma,
            challenges.field_registers_read_write.gamma,
        ];
        assert_eq!(drawn_gammas, inline_gammas);
        assert_eq!(challenges.ram_val_check.gamma, inline_ram_gamma);
    }

    #[test]
    fn output_claim_count_matches_absorbed_openings() {
        let sumchecks = sumchecks();
        #[cfg(not(feature = "field-inline"))]
        assert_eq!(sumchecks.output_claim_count(), 7);
        #[cfg(feature = "field-inline")]
        assert_eq!(sumchecks.output_claim_count(), 12);
        assert_eq!(
            claims_with_advice(false).opening_values().len(),
            sumchecks.output_claim_count(),
        );
    }
}
