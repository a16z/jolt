//! Typed inputs consumed and outputs produced by stage 1 verification.

use std::collections::{btree_map::Entry, BTreeMap};

use jolt_claims::protocols::jolt::{JoltRelationId, JoltVirtualPolynomial};
use jolt_field::Field;
use jolt_riscv::CircuitFlags;
use jolt_sumcheck::{BatchedCommittedSumcheckConsistency, CommittedSumcheckConsistency};
use serde::{Deserialize, Serialize};

use super::outer_remainder::{OuterRemainder, OuterRemainderOutputClaims};
use crate::stages::relations::SumcheckBatch;
use crate::stages::zk::outputs::CommittedOutputClaimOutput;
use crate::VerifierError;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Stage1OutputClaims<F: Field> {
    pub uniskip_output_claim: F,
    pub outer: Stage1BatchOutputClaims<F>,
}

/// Source-of-truth for stage 1's singleton sumcheck batch: the Spartan outer
/// *remainder* sumcheck (the companion uni-skip first round is a separate
/// sub-sumcheck, not a batch member — see [`OuterUniskip`](super::OuterUniskip)).
/// `#[derive(SumcheckBatch)]` generates the `Stage1BatchInputClaims<F>` /
/// `Stage1BatchInputPoints<F>`, `Stage1BatchOutputClaims<F>` /
/// `Stage1BatchOutputPoints<F>`, and `Stage1BatchChallenges<F>` aggregates — one
/// field per instance, in this declaration order. With a single instance and no
/// cross-relation aliasing there is no `custom_opening_values` opt-out: the
/// generated `opening_values` / `append_to_transcript` delegates to
/// `OuterRemainderOutputClaims` in `dimensions.variables()` order (the canonical 35
/// R1CS-input order), byte-identical to the previous explicit append loop.
#[derive(SumcheckBatch)]
pub struct Stage1BatchSumchecks<F: Field> {
    pub outer_remainder: OuterRemainder<F>,
}

/// The shared opening-point accessor over the point-only stage-1 aggregate.
impl<F: Field> Stage1BatchOutputPoints<F> {
    /// The Spartan outer remainder *opening* point (shared by all 35 openings): the
    /// bound remainder sumcheck point reversed, as `derive_opening_points` produces.
    /// The raw (un-reversed) reduction point that downstream stages slice is exposed
    /// by [`Stage1Output::remainder_point`].
    pub fn remainder_opening_point(&self) -> &[F] {
        self.outer_remainder.left_instruction_input()
    }
}

/// Assemble the stage-1 produced openings (the verifier-only wire form,
/// `OuterRemainderOutputClaims<F>`) from a prover-supplied `(variable, value)`
/// iterator, preserving the canonical `SPARTAN_OUTER_R1CS_INPUTS` field order. The
/// BTreeMap dedup/missing/extra checks guard the prover-supplied data (it has no
/// generated equivalent): every R1CS input must appear exactly once.
pub fn outer_remainder_outputs_from_r1cs_inputs<F: Field>(
    claims: impl IntoIterator<Item = (JoltVirtualPolynomial, F)>,
) -> Result<OuterRemainderOutputClaims<F>, VerifierError> {
    let mut values = collect_r1cs_inputs(claims)?;
    let outer = OuterRemainderOutputClaims {
        left_instruction_input: take_r1cs_input(
            &mut values,
            JoltVirtualPolynomial::LeftInstructionInput,
        )?,
        right_instruction_input: take_r1cs_input(
            &mut values,
            JoltVirtualPolynomial::RightInstructionInput,
        )?,
        product: take_r1cs_input(&mut values, JoltVirtualPolynomial::Product)?,
        should_branch: take_r1cs_input(&mut values, JoltVirtualPolynomial::ShouldBranch)?,
        pc: take_r1cs_input(&mut values, JoltVirtualPolynomial::PC)?,
        unexpanded_pc: take_r1cs_input(&mut values, JoltVirtualPolynomial::UnexpandedPC)?,
        imm: take_r1cs_input(&mut values, JoltVirtualPolynomial::Imm)?,
        ram_address: take_r1cs_input(&mut values, JoltVirtualPolynomial::RamAddress)?,
        rs1_value: take_r1cs_input(&mut values, JoltVirtualPolynomial::Rs1Value)?,
        rs2_value: take_r1cs_input(&mut values, JoltVirtualPolynomial::Rs2Value)?,
        rd_write_value: take_r1cs_input(&mut values, JoltVirtualPolynomial::RdWriteValue)?,
        ram_read_value: take_r1cs_input(&mut values, JoltVirtualPolynomial::RamReadValue)?,
        ram_write_value: take_r1cs_input(&mut values, JoltVirtualPolynomial::RamWriteValue)?,
        left_lookup_operand: take_r1cs_input(
            &mut values,
            JoltVirtualPolynomial::LeftLookupOperand,
        )?,
        right_lookup_operand: take_r1cs_input(
            &mut values,
            JoltVirtualPolynomial::RightLookupOperand,
        )?,
        next_unexpanded_pc: take_r1cs_input(&mut values, JoltVirtualPolynomial::NextUnexpandedPC)?,
        next_pc: take_r1cs_input(&mut values, JoltVirtualPolynomial::NextPC)?,
        next_is_virtual: take_r1cs_input(&mut values, JoltVirtualPolynomial::NextIsVirtual)?,
        next_is_first_in_sequence: take_r1cs_input(
            &mut values,
            JoltVirtualPolynomial::NextIsFirstInSequence,
        )?,
        lookup_output: take_r1cs_input(&mut values, JoltVirtualPolynomial::LookupOutput)?,
        should_jump: take_r1cs_input(&mut values, JoltVirtualPolynomial::ShouldJump)?,
        add_operands: take_flag(&mut values, CircuitFlags::AddOperands)?,
        subtract_operands: take_flag(&mut values, CircuitFlags::SubtractOperands)?,
        multiply_operands: take_flag(&mut values, CircuitFlags::MultiplyOperands)?,
        load: take_flag(&mut values, CircuitFlags::Load)?,
        store: take_flag(&mut values, CircuitFlags::Store)?,
        jump: take_flag(&mut values, CircuitFlags::Jump)?,
        write_lookup_output_to_rd: take_flag(&mut values, CircuitFlags::WriteLookupOutputToRD)?,
        virtual_instruction: take_flag(&mut values, CircuitFlags::VirtualInstruction)?,
        assert: take_flag(&mut values, CircuitFlags::Assert)?,
        do_not_update_unexpanded_pc: take_flag(&mut values, CircuitFlags::DoNotUpdateUnexpandedPC)?,
        advice: take_flag(&mut values, CircuitFlags::Advice)?,
        is_compressed: take_flag(&mut values, CircuitFlags::IsCompressed)?,
        is_first_in_sequence: take_flag(&mut values, CircuitFlags::IsFirstInSequence)?,
        is_last_in_sequence: take_flag(&mut values, CircuitFlags::IsLastInSequence)?,
    };
    reject_extra_r1cs_inputs(&values)?;
    Ok(outer)
}

fn collect_r1cs_inputs<F: Field>(
    claims: impl IntoIterator<Item = (JoltVirtualPolynomial, F)>,
) -> Result<BTreeMap<JoltVirtualPolynomial, F>, VerifierError> {
    let mut values = BTreeMap::new();
    for (variable, value) in claims {
        match values.entry(variable) {
            Entry::Vacant(entry) => {
                let _ = entry.insert(value);
            }
            Entry::Occupied(_) => {
                return Err(stage1_public_input_failed(format!(
                    "duplicate Stage 1 R1CS input {variable:?}"
                )));
            }
        }
    }
    Ok(values)
}

fn take_r1cs_input<F: Field>(
    values: &mut BTreeMap<JoltVirtualPolynomial, F>,
    variable: JoltVirtualPolynomial,
) -> Result<F, VerifierError> {
    values.remove(&variable).ok_or_else(|| {
        stage1_public_input_failed(format!("missing Stage 1 R1CS input {variable:?}"))
    })
}

fn take_flag<F: Field>(
    values: &mut BTreeMap<JoltVirtualPolynomial, F>,
    flag: CircuitFlags,
) -> Result<F, VerifierError> {
    take_r1cs_input(values, JoltVirtualPolynomial::OpFlags(flag))
}

fn reject_extra_r1cs_inputs<F: Field>(
    values: &BTreeMap<JoltVirtualPolynomial, F>,
) -> Result<(), VerifierError> {
    if let Some(variable) = values.keys().next() {
        return Err(stage1_public_input_failed(format!(
            "unexpected Stage 1 R1CS input {variable:?}"
        )));
    }
    Ok(())
}

fn stage1_public_input_failed(reason: String) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::SpartanOuter,
        reason,
    }
}

/// The Fiat-Shamir values the verifier draws during stage 1: the irreducible
/// Spartan outer `tau` point and the uni-skip reduction challenge. Drawn
/// path-agnostically before the ZK/clear branch; carried in [`Stage1ZkOutput`]
/// so BlindFold can source `tau`/`uniskip` from `challenges.<field>` (matching
/// the `input.stageN.challenges.<field>` idiom used by the sibling stages). The
/// remainder sumcheck point is opening-derived, so it lives on the produced
/// reduction (clear: `output_points.remainder_opening_point()`; ZK:
/// `remainder_consistency`) rather than here; the singleton remainder batching
/// coefficient is likewise
/// read from `remainder_consistency` on the ZK path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1Challenges<F: Field> {
    pub tau: Vec<F>,
    pub uniskip_challenge: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1ClearOutput<F: Field> {
    /// The produced remainder opening *values* (wire form). The opening point is
    /// derived from the remainder's sumcheck point; later stages read values through
    /// `.outer_remainder.<field>`.
    pub output_values: Stage1BatchOutputClaims<F>,
    /// The produced remainder opening *points*, paired field-for-field with
    /// `output_values`. All 35 openings share the single remainder point, exposed
    /// through
    /// [`remainder_opening_point`](Stage1BatchOutputPoints::remainder_opening_point).
    pub output_points: Stage1BatchOutputPoints<F>,
    /// The Spartan outer uni-skip's reduced opening (consumed as the remainder's
    /// input claim; absorbed into the transcript before the remainder RLC squeeze).
    pub uniskip_output_claim: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1ZkOutput<F: Field, C> {
    pub challenges: Stage1Challenges<F>,
    pub uniskip_consistency: CommittedSumcheckConsistency<F, C>,
    pub uniskip_output_claims: CommittedOutputClaimOutput<C>,
    pub remainder_consistency: BatchedCommittedSumcheckConsistency<F, C>,
    pub remainder_output_claims: CommittedOutputClaimOutput<C>,
}

// The clear variant carries the located opening claims (point + value) read on the
// hot path by later stages; the ZK variant carries committed consistency. Boxing
// the common clear variant to shrink the rarer ZK one would add indirection to
// every clear-path access.
#[expect(
    clippy::large_enum_variant,
    reason = "clear variant holds the located opening claims read on the hot path; boxing it would penalize the common case"
)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage1Output<F: Field, C> {
    Clear(Stage1ClearOutput<F>),
    Zk(Stage1ZkOutput<F, C>),
}

impl<F: Field, C> Stage1Output<F, C> {
    /// The raw (un-reversed) Spartan outer remainder sumcheck reduction point,
    /// available regardless of proving mode. The remainder is a singleton batch, so
    /// the clear-path bound point and the ZK committed round challenges are the same
    /// vector. Downstream consumers (stage 2's `tau_low`, BlindFold's stage-1 cycle
    /// bindings) slice and reverse this point themselves, so it must NOT be the
    /// already-reversed opening point: the clear path stores the openings at the
    /// reversed point (`derive_opening_points`), so we reverse it back here to
    /// recover the raw reduction point the ZK `challenges()` returns directly.
    pub fn remainder_point(&self) -> Vec<F> {
        match self {
            Self::Clear(output) => output
                .output_points
                .remainder_opening_point()
                .iter()
                .rev()
                .copied()
                .collect(),
            Self::Zk(output) => output.remainder_consistency.challenges(),
        }
    }

    pub fn clear(&self) -> Result<&Stage1ClearOutput<F>, VerifierError> {
        match self {
            Self::Clear(output) => Ok(output),
            Self::Zk(_) => Err(VerifierError::ExpectedClearProof { field: "stage1" }),
        }
    }

    pub fn zk(&self) -> Result<&Stage1ZkOutput<F, C>, VerifierError> {
        match self {
            Self::Zk(output) => Ok(output),
            Self::Clear(_) => Err(VerifierError::ExpectedCommittedProof { field: "stage1" }),
        }
    }
}
