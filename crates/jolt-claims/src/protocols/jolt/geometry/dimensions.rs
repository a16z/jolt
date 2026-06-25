use jolt_field::Field;
use serde::{Deserialize, Serialize};

pub use super::error::{JoltFormulaDimensionsError, JoltFormulaPointError};

use super::{
    bytecode::BytecodeReadRafDimensions,
    instruction::{InstructionRaVirtualizationDimensions, InstructionReadRafDimensions},
    ra::JoltRaPolynomialLayout,
    ram::RamRaVirtualizationDimensions,
};

pub const REGISTER_ADDRESS_BITS: usize = 7;
pub const OUTER_UNISKIP_DOMAIN_SIZE: usize = 10;
pub const OUTER_UNISKIP_FIRST_ROUND_DEGREE: usize = 27;
pub const PRODUCT_UNISKIP_DOMAIN_SIZE: usize = 3;
pub const PRODUCT_UNISKIP_FIRST_ROUND_DEGREE: usize = 6;

// The sumcheck spec/domain are now protocol-agnostic crate-root types; these
// aliases keep the `Jolt*` spelling that the rest of the codebase uses.
pub use crate::{SumcheckDomain as JoltSumcheckDomain, SumcheckSpec as JoltSumcheckSpec};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum TracePolynomialOrder {
    #[default]
    CycleMajor,
    AddressMajor,
}

impl TracePolynomialOrder {
    pub const fn transcript_scalar(self) -> u64 {
        match self {
            Self::CycleMajor => 0,
            Self::AddressMajor => 1,
        }
    }

    pub const fn address_cycle_to_index(
        self,
        address: usize,
        cycle: usize,
        num_addresses: usize,
        num_cycles: usize,
    ) -> usize {
        match self {
            Self::CycleMajor => address * num_cycles + cycle,
            Self::AddressMajor => cycle * num_addresses + address,
        }
    }

    pub const fn index_to_address_cycle(
        self,
        index: usize,
        num_addresses: usize,
        num_cycles: usize,
    ) -> (usize, usize) {
        match self {
            Self::CycleMajor => (index / num_cycles, index % num_cycles),
            Self::AddressMajor => (index % num_addresses, index / num_addresses),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TraceDimensions {
    log_t: usize,
}

impl TraceDimensions {
    pub const fn new(log_t: usize) -> Self {
        Self { log_t }
    }

    pub const fn log_t(self) -> usize {
        self.log_t
    }

    pub const fn sumcheck(self, degree: usize) -> JoltSumcheckSpec {
        JoltSumcheckSpec::boolean(self.log_t, degree)
    }

    pub fn cycle_opening_point<F: Field>(
        self,
        challenges: &[F],
    ) -> Result<Vec<F>, JoltFormulaPointError> {
        if challenges.len() != self.log_t {
            return Err(JoltFormulaPointError::ChallengeLengthMismatch {
                expected: self.log_t,
                got: challenges.len(),
            });
        }

        Ok(challenges.iter().rev().copied().collect())
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ReadWriteDimensions {
    log_t: usize,
    log_k: usize,
    phase1_num_rounds: usize,
    phase2_num_rounds: usize,
}

impl ReadWriteDimensions {
    pub const fn new(
        log_t: usize,
        log_k: usize,
        phase1_num_rounds: usize,
        phase2_num_rounds: usize,
    ) -> Self {
        Self {
            log_t,
            log_k,
            phase1_num_rounds,
            phase2_num_rounds,
        }
    }

    pub const fn log_t(self) -> usize {
        self.log_t
    }

    pub const fn log_k(self) -> usize {
        self.log_k
    }

    pub const fn phase1_num_rounds(self) -> usize {
        self.phase1_num_rounds
    }

    pub const fn phase2_num_rounds(self) -> usize {
        self.phase2_num_rounds
    }

    pub const fn phase3_cycle_rounds(self) -> usize {
        self.log_t - self.phase1_num_rounds
    }

    pub const fn read_write_sumcheck(self) -> JoltSumcheckSpec {
        JoltSumcheckSpec::boolean(self.log_t + self.log_k, 3)
    }

    pub const fn raf_evaluation_sumcheck(self) -> JoltSumcheckSpec {
        JoltSumcheckSpec::boolean(self.log_t + self.log_k - self.phase1_num_rounds, 2)
    }

    pub const fn output_check_sumcheck(self) -> JoltSumcheckSpec {
        JoltSumcheckSpec::boolean(self.log_t + self.log_k - self.phase1_num_rounds, 3)
    }

    pub fn read_write_opening_point<F: Field>(
        self,
        challenges: &[F],
    ) -> Result<ReadWriteOpeningPoint<F>, JoltFormulaPointError> {
        self.validate_phase_split()?;
        let expected = self.log_t + self.log_k;
        if challenges.len() != expected {
            return Err(JoltFormulaPointError::ChallengeLengthMismatch {
                expected,
                got: challenges.len(),
            });
        }

        let (phase1, rest) = challenges.split_at(self.phase1_num_rounds);
        let (phase2, rest) = rest.split_at(self.phase2_num_rounds);
        let (phase3_cycle, phase3_address) = rest.split_at(self.log_t - self.phase1_num_rounds);

        let r_cycle = phase3_cycle
            .iter()
            .rev()
            .copied()
            .chain(phase1.iter().rev().copied())
            .collect::<Vec<_>>();
        let r_address = phase3_address
            .iter()
            .rev()
            .copied()
            .chain(phase2.iter().rev().copied())
            .collect::<Vec<_>>();
        let opening_point = [r_address.as_slice(), r_cycle.as_slice()].concat();

        Ok(ReadWriteOpeningPoint {
            r_address,
            r_cycle,
            opening_point,
        })
    }

    pub fn address_opening_point<F: Field>(
        self,
        challenges: &[F],
    ) -> Result<Vec<F>, JoltFormulaPointError> {
        self.validate_phase_split()?;
        let cycle_gap_rounds = self.phase3_cycle_rounds();
        let expected = self.log_k + cycle_gap_rounds;
        if challenges.len() != expected {
            return Err(JoltFormulaPointError::ChallengeLengthMismatch {
                expected,
                got: challenges.len(),
            });
        }

        let phase3_address_start = self.phase2_num_rounds + cycle_gap_rounds;
        let mut address = Vec::with_capacity(self.log_k);
        address.extend_from_slice(&challenges[..self.phase2_num_rounds]);
        address.extend_from_slice(&challenges[phase3_address_start..]);
        address.reverse();
        Ok(address)
    }

    const fn validate_phase_split(self) -> Result<(), JoltFormulaPointError> {
        if self.phase1_num_rounds > self.log_t || self.phase2_num_rounds > self.log_k {
            return Err(JoltFormulaPointError::InvalidReadWritePhaseSplit {
                phase1_num_rounds: self.phase1_num_rounds,
                log_t: self.log_t,
                phase2_num_rounds: self.phase2_num_rounds,
                log_k: self.log_k,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReadWriteOpeningPoint<F: Field> {
    pub r_address: Vec<F>,
    pub r_cycle: Vec<F>,
    pub opening_point: Vec<F>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JoltReadWriteConfig {
    pub ram_rw_phase1_num_rounds: u8,
    pub ram_rw_phase2_num_rounds: u8,
    pub registers_rw_phase1_num_rounds: u8,
    pub registers_rw_phase2_num_rounds: u8,
}

impl JoltReadWriteConfig {
    pub const fn ram_dimensions(self, log_t: usize, ram_log_k: usize) -> ReadWriteDimensions {
        ReadWriteDimensions::new(
            log_t,
            ram_log_k,
            self.ram_rw_phase1_num_rounds as usize,
            self.ram_rw_phase2_num_rounds as usize,
        )
    }

    pub const fn register_dimensions(
        self,
        log_t: usize,
        register_log_k: usize,
    ) -> ReadWriteDimensions {
        ReadWriteDimensions::new(
            log_t,
            register_log_k,
            self.registers_rw_phase1_num_rounds as usize,
            self.registers_rw_phase2_num_rounds as usize,
        )
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CommitmentMatrixShape {
    column_vars: usize,
    row_vars: usize,
}

impl CommitmentMatrixShape {
    pub const fn new(column_vars: usize, row_vars: usize) -> Self {
        Self {
            column_vars,
            row_vars,
        }
    }

    pub const fn column_vars(self) -> usize {
        self.column_vars
    }

    pub const fn row_vars(self) -> usize {
        self.row_vars
    }

    pub const fn total_vars(self) -> usize {
        self.column_vars + self.row_vars
    }

    pub const fn balanced(total_vars: usize) -> Self {
        let column_vars = total_vars.div_ceil(2);
        Self {
            column_vars,
            row_vars: total_vars - column_vars,
        }
    }

    pub fn advice_from_max_bytes(max_advice_size_bytes: usize) -> Self {
        let words = max_advice_size_bytes / 8;
        let len = words.next_power_of_two().max(1);
        Self::balanced(log2_power_of_two(len))
    }
}

pub(crate) fn log2_power_of_two(value: usize) -> usize {
    assert!(
        value.is_power_of_two(),
        "expected a power-of-two dimension, got {value}"
    );
    value.trailing_zeros() as usize
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct JoltOneHotDimensions {
    pub log_t: usize,
    pub instruction_address_bits: usize,
    pub bytecode_k: usize,
    pub ram_k: usize,
    pub committed_chunk_bits: usize,
    pub lookup_virtual_chunk_bits: usize,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JoltOneHotConfig {
    pub log_k_chunk: u8,
    pub lookups_ra_virtual_log_k_chunk: u8,
}

impl JoltOneHotConfig {
    pub const fn committed_chunk_bits(self) -> usize {
        self.log_k_chunk as usize
    }

    pub const fn lookup_virtual_chunk_bits(self) -> usize {
        self.lookups_ra_virtual_log_k_chunk as usize
    }

    pub fn committed_address_chunks<F: Field>(self, r_address: &[F]) -> Vec<Vec<F>> {
        committed_address_chunks(r_address, self.committed_chunk_bits())
    }

    pub const fn dimensions(
        self,
        log_t: usize,
        instruction_address_bits: usize,
        bytecode_k: usize,
        ram_k: usize,
    ) -> JoltOneHotDimensions {
        JoltOneHotDimensions {
            log_t,
            instruction_address_bits,
            bytecode_k,
            ram_k,
            committed_chunk_bits: self.committed_chunk_bits(),
            lookup_virtual_chunk_bits: self.lookup_virtual_chunk_bits(),
        }
    }
}

pub fn committed_address_chunks<F: Field>(r_address: &[F], chunk_bits: usize) -> Vec<Vec<F>> {
    if chunk_bits == 0 {
        return Vec::new();
    }

    let padding = r_address
        .len()
        .next_multiple_of(chunk_bits)
        .saturating_sub(r_address.len());
    let mut padded = Vec::with_capacity(r_address.len() + padding);
    padded.extend((0..padding).map(|_| F::zero()));
    padded.extend_from_slice(r_address);
    padded
        .chunks(chunk_bits)
        .map(<[F]>::to_vec)
        .collect::<Vec<_>>()
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct JoltFormulaDimensions {
    pub trace: TraceDimensions,
    pub ra_layout: JoltRaPolynomialLayout,
    pub instruction_read_raf: InstructionReadRafDimensions,
    pub instruction_ra_virtualization: InstructionRaVirtualizationDimensions,
    pub bytecode_read_raf: BytecodeReadRafDimensions,
    pub ram_ra_virtualization: RamRaVirtualizationDimensions,
}

impl TryFrom<JoltOneHotDimensions> for JoltFormulaDimensions {
    type Error = JoltFormulaDimensionsError;

    fn try_from(dimensions: JoltOneHotDimensions) -> Result<Self, Self::Error> {
        require_nonzero(
            dimensions.instruction_address_bits,
            "instruction_address_bits",
        )?;
        require_nonzero(dimensions.bytecode_k, "bytecode_k")?;
        require_nonzero(dimensions.ram_k, "ram_k")?;
        require_nonzero(dimensions.committed_chunk_bits, "committed_chunk_bits")?;
        require_nonzero(
            dimensions.lookup_virtual_chunk_bits,
            "lookup_virtual_chunk_bits",
        )?;

        if dimensions.lookup_virtual_chunk_bits < dimensions.committed_chunk_bits {
            return Err(JoltFormulaDimensionsError::InvalidChunkOrder {
                committed_chunk_bits: dimensions.committed_chunk_bits,
                lookup_virtual_chunk_bits: dimensions.lookup_virtual_chunk_bits,
            });
        }

        require_divisible(
            "lookup_virtual_chunk_bits",
            dimensions.lookup_virtual_chunk_bits,
            "committed_chunk_bits",
            dimensions.committed_chunk_bits,
        )?;
        require_divisible(
            "instruction_address_bits",
            dimensions.instruction_address_bits,
            "lookup_virtual_chunk_bits",
            dimensions.lookup_virtual_chunk_bits,
        )?;

        let instruction_address_bits = dimensions.instruction_address_bits;
        let bytecode_log_k = ceil_log_2(dimensions.bytecode_k);
        let ram_log_k = ceil_log_2(dimensions.ram_k);
        let instruction_d = instruction_address_bits.div_ceil(dimensions.committed_chunk_bits);
        let bytecode_d = bytecode_log_k.div_ceil(dimensions.committed_chunk_bits);
        let ram_d = ram_log_k.div_ceil(dimensions.committed_chunk_bits);
        let virtual_instruction_ra_polys =
            instruction_address_bits / dimensions.lookup_virtual_chunk_bits;
        let committed_per_virtual =
            dimensions.lookup_virtual_chunk_bits / dimensions.committed_chunk_bits;

        Ok(Self {
            trace: TraceDimensions::new(dimensions.log_t),
            ra_layout: JoltRaPolynomialLayout::try_from((instruction_d, bytecode_d, ram_d))?,
            instruction_read_raf: InstructionReadRafDimensions::try_from((
                dimensions.log_t,
                instruction_address_bits,
                virtual_instruction_ra_polys,
            ))?,
            instruction_ra_virtualization: InstructionRaVirtualizationDimensions::try_from((
                dimensions.log_t,
                virtual_instruction_ra_polys,
                committed_per_virtual,
            ))?,
            bytecode_read_raf: BytecodeReadRafDimensions::new(
                dimensions.log_t,
                bytecode_log_k,
                bytecode_d,
            ),
            ram_ra_virtualization: RamRaVirtualizationDimensions::new(dimensions.log_t, ram_d),
        })
    }
}

fn require_nonzero(value: usize, name: &'static str) -> Result<(), JoltFormulaDimensionsError> {
    if value == 0 {
        Err(JoltFormulaDimensionsError::Zero { name })
    } else {
        Ok(())
    }
}

fn require_divisible(
    value_name: &'static str,
    value: usize,
    divisor_name: &'static str,
    divisor: usize,
) -> Result<(), JoltFormulaDimensionsError> {
    if value.is_multiple_of(divisor) {
        Ok(())
    } else {
        Err(JoltFormulaDimensionsError::NotDivisible {
            value_name,
            value,
            divisor_name,
            divisor,
        })
    }
}

fn ceil_log_2(value: usize) -> usize {
    if value <= 1 {
        0
    } else {
        usize::BITS as usize - (value - 1).leading_zeros() as usize
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::panic, reason = "tests fail loudly on unexpected errors")]

    use super::super::claim_reductions::advice::AdviceClaimReductionLayout;
    use super::super::claim_reductions::precommitted::{
        PrecommittedClaimReduction, PrecommittedReductionLayout,
    };
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt, Invertible};
    use jolt_poly::EqPolynomial;

    fn dimensions() -> JoltOneHotDimensions {
        JoltOneHotDimensions {
            log_t: 20,
            instruction_address_bits: 128,
            bytecode_k: 1024,
            ram_k: 4096,
            committed_chunk_bits: 8,
            lookup_virtual_chunk_bits: 32,
        }
    }

    #[test]
    fn derives_all_runtime_formula_dimensions() -> Result<(), JoltFormulaDimensionsError> {
        let dimensions = JoltFormulaDimensions::try_from(dimensions())?;

        assert_eq!(dimensions.ra_layout.instruction(), 16);
        assert_eq!(dimensions.trace.log_t(), 20);
        assert_eq!(dimensions.ra_layout.bytecode(), 2);
        assert_eq!(dimensions.ra_layout.ram(), 2);
        assert_eq!(
            dimensions.instruction_read_raf.sumcheck(),
            JoltSumcheckSpec::boolean(148, 6)
        );
        assert_eq!(dimensions.instruction_read_raf.num_virtual_ra_polys(), 4);
        assert_eq!(
            dimensions
                .instruction_ra_virtualization
                .num_committed_per_virtual(),
            4
        );
        assert_eq!(
            dimensions
                .instruction_ra_virtualization
                .num_committed_ra_polys(),
            16
        );
        assert_eq!(dimensions.bytecode_read_raf.num_committed_ra_polys(), 2);
        assert_eq!(dimensions.ram_ra_virtualization.num_committed_ra_polys(), 2);
        Ok(())
    }

    #[test]
    fn supports_zero_bytecode_and_ram_d() -> Result<(), JoltFormulaDimensionsError> {
        let dimensions = JoltFormulaDimensions::try_from(JoltOneHotDimensions {
            bytecode_k: 1,
            ram_k: 1,
            ..dimensions()
        })?;

        assert_eq!(dimensions.ra_layout.instruction(), 16);
        assert_eq!(dimensions.ra_layout.bytecode(), 0);
        assert_eq!(dimensions.ra_layout.ram(), 0);
        assert_eq!(dimensions.ra_layout.total(), 16);
        assert_eq!(dimensions.bytecode_read_raf.num_committed_ra_polys(), 0);
        assert_eq!(dimensions.ram_ra_virtualization.num_committed_ra_polys(), 0);
        Ok(())
    }

    #[test]
    fn trace_dimensions_normalize_cycle_opening_point() {
        let challenges = [Fr::from_u64(3), Fr::from_u64(5), Fr::from_u64(7)];

        assert_eq!(
            TraceDimensions::new(3)
                .cycle_opening_point(&challenges)
                .unwrap_or_else(|err| panic!("cycle point should normalize: {err}")),
            vec![Fr::from_u64(7), Fr::from_u64(5), Fr::from_u64(3)]
        );
    }

    #[test]
    fn trace_polynomial_order_indexes_match_protocol_order() {
        assert_eq!(
            TracePolynomialOrder::CycleMajor.address_cycle_to_index(3, 4, 10, 20),
            64
        );
        assert_eq!(
            TracePolynomialOrder::AddressMajor.address_cycle_to_index(3, 4, 10, 20),
            43
        );
        assert_eq!(
            TracePolynomialOrder::CycleMajor.index_to_address_cycle(64, 10, 20),
            (3, 4)
        );
        assert_eq!(
            TracePolynomialOrder::AddressMajor.index_to_address_cycle(43, 10, 20),
            (3, 4)
        );
        assert_eq!(TracePolynomialOrder::CycleMajor.transcript_scalar(), 0);
        assert_eq!(TracePolynomialOrder::AddressMajor.transcript_scalar(), 1);
    }

    #[test]
    fn commitment_matrix_shapes_follow_balanced_policy() {
        let shape = CommitmentMatrixShape::balanced(13);
        assert_eq!(shape.column_vars(), 7);
        assert_eq!(shape.row_vars(), 6);
        assert_eq!(shape.total_vars(), 13);

        assert_eq!(
            CommitmentMatrixShape::advice_from_max_bytes(64),
            CommitmentMatrixShape::new(2, 1)
        );
        assert_eq!(
            CommitmentMatrixShape::advice_from_max_bytes(0),
            CommitmentMatrixShape::new(0, 0)
        );
    }

    fn advice_layout(
        trace_order: TracePolynomialOrder,
        log_t: usize,
        log_k_chunk: usize,
        max_advice_size_bytes: usize,
    ) -> AdviceClaimReductionLayout {
        let advice_vars =
            CommitmentMatrixShape::advice_from_max_bytes(max_advice_size_bytes).total_vars();
        let scheduling_reference = PrecommittedClaimReduction::scheduling_reference(
            log_t + log_k_chunk,
            &[advice_vars],
            log_k_chunk,
        );
        AdviceClaimReductionLayout::balanced(
            trace_order,
            log_t,
            scheduling_reference,
            max_advice_size_bytes,
        )
        .unwrap_or_else(|error| panic!("advice layout should build: {error}"))
    }

    #[test]
    fn advice_layout_extracts_cycle_phase_variables_without_dory_globals() {
        let layout = advice_layout(TracePolynomialOrder::CycleMajor, 8, 4, 64);
        let challenges = (1..=8).map(Fr::from_u64).collect::<Vec<_>>();

        assert_eq!(layout.advice_shape(), CommitmentMatrixShape::new(2, 1));
        assert_eq!(layout.precommitted().cycle_phase_rounds(), &[0, 1, 6]);
        assert_eq!(layout.precommitted().num_address_phase_rounds(), 0);
        assert_eq!(layout.dimensions().cycle_phase_total_rounds(), 8);
        assert_eq!(layout.dimensions().address_phase_total_rounds(), 4);
        assert!(!layout.dimensions().has_address_phase());
        assert_eq!(
            layout
                .cycle_phase_variable_challenges(&challenges)
                .unwrap_or_else(|error| panic!("cycle phase variables should extract: {error}")),
            vec![Fr::from_u64(1), Fr::from_u64(2), Fr::from_u64(7)]
        );
        assert_eq!(
            layout
                .cycle_phase_opening_point(&challenges)
                .unwrap_or_else(|error| panic!("cycle phase point should normalize: {error}")),
            vec![Fr::from_u64(7), Fr::from_u64(2), Fr::from_u64(1)]
        );
    }

    #[test]
    fn advice_layout_extracts_address_phase_point_without_dory_globals() {
        // 2048 bytes = 256 words: an 8-variable advice polynomial with shape (4, 4).
        let layout = advice_layout(TracePolynomialOrder::CycleMajor, 8, 4, 2048);
        let cycle_challenges = (1..=8).map(Fr::from_u64).collect::<Vec<_>>();
        let cycle_vars = layout
            .cycle_phase_variable_challenges(&cycle_challenges)
            .unwrap_or_else(|error| panic!("cycle variables should extract: {error}"));
        let address_challenges = [
            Fr::from_u64(101),
            Fr::from_u64(102),
            Fr::from_u64(103),
            Fr::from_u64(104),
        ];

        assert_eq!(layout.advice_shape(), CommitmentMatrixShape::new(4, 4));
        assert_eq!(
            layout.precommitted().cycle_phase_rounds(),
            &[0, 1, 2, 3, 6, 7]
        );
        assert_eq!(layout.precommitted().address_phase_rounds(), &[0, 1]);
        assert!(layout.dimensions().has_address_phase());
        assert_eq!(
            layout
                .address_phase_opening_point(&cycle_vars, &address_challenges)
                .unwrap_or_else(|error| panic!("address phase point should normalize: {error}")),
            vec![
                Fr::from_u64(102),
                Fr::from_u64(101),
                Fr::from_u64(8),
                Fr::from_u64(7),
                Fr::from_u64(4),
                Fr::from_u64(3),
                Fr::from_u64(2),
                Fr::from_u64(1),
            ]
        );
    }

    #[test]
    fn advice_layout_tracks_address_major_cycle_gap() {
        let layout = advice_layout(TracePolynomialOrder::AddressMajor, 8, 4, 64);
        let challenges = (1..=8).map(Fr::from_u64).collect::<Vec<_>>();
        let cycle_vars = layout
            .cycle_phase_variable_challenges(&challenges)
            .unwrap_or_else(|error| panic!("cycle variables should extract: {error}"));
        let address_challenges = [
            Fr::from_u64(101),
            Fr::from_u64(102),
            Fr::from_u64(103),
            Fr::from_u64(104),
        ];

        assert_eq!(layout.precommitted().cycle_phase_rounds(), &[2]);
        assert_eq!(layout.precommitted().address_phase_rounds(), &[0, 1]);
        assert_eq!(layout.dimensions().cycle_phase_total_rounds(), 8);
        assert_eq!(layout.dimensions().address_phase_total_rounds(), 4);
        assert!(layout.dimensions().has_address_phase());
        assert_eq!(cycle_vars, vec![Fr::from_u64(3)]);
        assert_eq!(
            layout
                .address_phase_opening_point(&cycle_vars, &address_challenges)
                .unwrap_or_else(|error| panic!("address phase point should normalize: {error}")),
            vec![Fr::from_u64(3), Fr::from_u64(102), Fr::from_u64(101)]
        );
    }

    #[test]
    fn advice_final_output_scale_includes_cycle_phase_skip_rounds() {
        let layout = advice_layout(TracePolynomialOrder::CycleMajor, 8, 4, 64);
        let challenges = (1..=8).map(Fr::from_u64).collect::<Vec<_>>();
        // Permutation order for the active cycle rounds is [6, 1, 0].
        let permuted_point = [Fr::from_u64(7), Fr::from_u64(2), Fr::from_u64(1)];
        let reference_point = [Fr::from_u64(101), Fr::from_u64(102), Fr::from_u64(103)];
        let two_inv = Fr::from_u64(2).inv_or_zero();
        let skip_scale = (0..5).fold(Fr::from_u64(1), |scale, _| scale * two_inv);
        let expected = EqPolynomial::<Fr>::mle(&permuted_point, &reference_point) * skip_scale;

        assert_eq!(
            layout
                .cycle_phase_final_output_scale(&reference_point, &challenges)
                .unwrap_or_else(|error| panic!("final output scale should compute: {error}")),
            expected
        );
    }

    #[test]
    fn rejects_zero_dimensions() {
        assert_eq!(
            JoltFormulaDimensions::try_from(JoltOneHotDimensions {
                instruction_address_bits: 0,
                ..dimensions()
            }),
            Err(JoltFormulaDimensionsError::Zero {
                name: "instruction_address_bits"
            })
        );
        assert_eq!(
            JoltFormulaDimensions::try_from(JoltOneHotDimensions {
                bytecode_k: 0,
                ..dimensions()
            }),
            Err(JoltFormulaDimensionsError::Zero { name: "bytecode_k" })
        );
    }

    #[test]
    fn rejects_incompatible_chunks() {
        assert_eq!(
            JoltFormulaDimensions::try_from(JoltOneHotDimensions {
                committed_chunk_bits: 16,
                lookup_virtual_chunk_bits: 8,
                ..dimensions()
            }),
            Err(JoltFormulaDimensionsError::InvalidChunkOrder {
                committed_chunk_bits: 16,
                lookup_virtual_chunk_bits: 8,
            })
        );
        assert_eq!(
            JoltFormulaDimensions::try_from(JoltOneHotDimensions {
                lookup_virtual_chunk_bits: 20,
                ..dimensions()
            }),
            Err(JoltFormulaDimensionsError::NotDivisible {
                value_name: "lookup_virtual_chunk_bits",
                value: 20,
                divisor_name: "committed_chunk_bits",
                divisor: 8,
            })
        );
        assert_eq!(
            JoltFormulaDimensions::try_from(JoltOneHotDimensions {
                lookup_virtual_chunk_bits: 48,
                ..dimensions()
            }),
            Err(JoltFormulaDimensionsError::NotDivisible {
                value_name: "instruction_address_bits",
                value: 128,
                divisor_name: "lookup_virtual_chunk_bits",
                divisor: 48,
            })
        );
    }

    #[test]
    fn read_write_dimensions_normalize_full_opening_point() {
        let dimensions = ReadWriteDimensions::new(4, 3, 1, 2);
        let challenges = (1..=7).map(Fr::from_u64).collect::<Vec<_>>();

        let point = dimensions
            .read_write_opening_point(&challenges)
            .unwrap_or_else(|error| panic!("read-write opening point should evaluate: {error}"));

        assert_eq!(
            point.r_cycle,
            vec![
                Fr::from_u64(6),
                Fr::from_u64(5),
                Fr::from_u64(4),
                Fr::from_u64(1)
            ]
        );
        assert_eq!(
            point.r_address,
            vec![Fr::from_u64(7), Fr::from_u64(3), Fr::from_u64(2)]
        );
        assert_eq!(
            point.opening_point,
            vec![
                Fr::from_u64(7),
                Fr::from_u64(3),
                Fr::from_u64(2),
                Fr::from_u64(6),
                Fr::from_u64(5),
                Fr::from_u64(4),
                Fr::from_u64(1),
            ]
        );
    }

    #[test]
    fn read_write_dimensions_extract_address_opening_point() {
        let dimensions = ReadWriteDimensions::new(4, 3, 1, 2);
        let challenges = (10..=15).map(Fr::from_u64).collect::<Vec<_>>();

        assert_eq!(
            dimensions
                .address_opening_point(&challenges)
                .unwrap_or_else(|error| panic!("address opening point should evaluate: {error}")),
            vec![Fr::from_u64(15), Fr::from_u64(11), Fr::from_u64(10)]
        );
    }

    #[test]
    fn read_write_point_helpers_reject_bad_shapes() {
        let dimensions = ReadWriteDimensions::new(4, 3, 5, 2);
        assert_eq!(
            dimensions.read_write_opening_point::<Fr>(&[]),
            Err(JoltFormulaPointError::InvalidReadWritePhaseSplit {
                phase1_num_rounds: 5,
                log_t: 4,
                phase2_num_rounds: 2,
                log_k: 3,
            })
        );

        let dimensions = ReadWriteDimensions::new(4, 3, 1, 2);
        assert_eq!(
            dimensions.address_opening_point::<Fr>(&[Fr::from_u64(0)]),
            Err(JoltFormulaPointError::ChallengeLengthMismatch {
                expected: 6,
                got: 1,
            })
        );
    }
}
