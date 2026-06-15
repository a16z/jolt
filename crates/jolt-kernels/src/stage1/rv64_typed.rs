use std::cmp::Ordering;

use jolt_field::signed::{S128, S160, S192, S64};
use jolt_field::{Field, Fr, Limbs};
use jolt_poly::lagrange::{lagrange_evals, lagrange_kernel_eval};
use jolt_poly::{EqPolynomial, UnivariatePoly};
use jolt_r1cs::R1csKey;
use rayon::prelude::*;

use super::{
    boolean_index, DenseOuterState, Stage1KernelError, Stage1OuterR1csData,
    Stage1OuterRemainingContext, Stage1OuterRemainingEvaluator, Stage1RemainingRoundProof,
    OUTER_SECOND_GROUP_ROWS, OUTER_UNISKIP_BASE_START, OUTER_UNISKIP_DEGREE,
    OUTER_UNISKIP_DOMAIN_SIZE, OUTER_UNISKIP_TARGET_COEFFS,
};

const RV64_NUM_CIRCUIT_FLAGS: usize = 14;
const FLAG_ADD_OPERANDS: usize = 0;
const FLAG_SUBTRACT_OPERANDS: usize = 1;
const FLAG_MULTIPLY_OPERANDS: usize = 2;
const FLAG_LOAD: usize = 3;
const FLAG_STORE: usize = 4;
const FLAG_JUMP: usize = 5;
const FLAG_WRITE_LOOKUP_OUTPUT_TO_RD: usize = 6;
const FLAG_VIRTUAL_INSTRUCTION: usize = 7;
const FLAG_ASSERT: usize = 8;
const FLAG_DO_NOT_UPDATE_UNEXPANDED_PC: usize = 9;
const FLAG_ADVICE: usize = 10;
const FLAG_IS_COMPRESSED: usize = 11;
const FLAG_IS_FIRST_IN_SEQUENCE: usize = 12;
const FLAG_IS_LAST_IN_SEQUENCE: usize = 13;

#[derive(Clone, Copy, Debug)]
pub struct Stage1Rv64Cycle {
    pub left_input: u64,
    pub right_input: S64,
    pub product: S128,
    pub left_lookup: u64,
    pub right_lookup: u128,
    pub lookup_output: u64,
    pub rs1_read_value: u64,
    pub rs2_read_value: u64,
    pub rd_write_value: u64,
    pub ram_addr: u64,
    pub ram_read_value: u64,
    pub ram_write_value: u64,
    pub pc: u64,
    pub next_pc: u64,
    pub unexpanded_pc: u64,
    pub next_unexpanded_pc: u64,
    pub imm: S64,
    pub flags: [bool; RV64_NUM_CIRCUIT_FLAGS],
    pub should_jump: bool,
    pub should_branch: bool,
    pub next_is_virtual: bool,
    pub next_is_first_in_sequence: bool,
}

impl Stage1Rv64Cycle {
    pub fn padding() -> Self {
        let mut flags = [false; RV64_NUM_CIRCUIT_FLAGS];
        flags[FLAG_DO_NOT_UPDATE_UNEXPANDED_PC] = true;
        Self {
            left_input: 0,
            right_input: S64::from_u64(0),
            product: S128::from_u64(0),
            left_lookup: 0,
            right_lookup: 0,
            lookup_output: 0,
            rs1_read_value: 0,
            rs2_read_value: 0,
            rd_write_value: 0,
            ram_addr: 0,
            ram_read_value: 0,
            ram_write_value: 0,
            pc: 0,
            next_pc: 0,
            unexpanded_pc: 0,
            next_unexpanded_pc: 0,
            imm: S64::from_u64(0),
            flags,
            should_jump: false,
            should_branch: false,
            next_is_virtual: false,
            next_is_first_in_sequence: false,
        }
    }
}

#[derive(Debug)]
pub struct Stage1OuterRv64Data<'a> {
    pub(crate) field_data: Stage1OuterR1csData<'a, Fr>,
    cycles: &'a [Stage1Rv64Cycle],
}

impl<'a> Stage1OuterRv64Data<'a> {
    pub fn new(
        key: &'a R1csKey<Fr>,
        witness: &'a [Fr],
        cycles: &'a [Stage1Rv64Cycle],
    ) -> Result<Self, Stage1KernelError> {
        if cycles.len() != key.num_cycles {
            return Err(Stage1KernelError::InvalidInputLength {
                input: "rv64_cycles",
                expected: key.num_cycles,
                actual: cycles.len(),
            });
        }
        Ok(Self {
            field_data: Stage1OuterR1csData::new(key, witness)?,
            cycles,
        })
    }

    #[tracing::instrument(skip_all, name = "Stage1OuterRv64Data::dense_outer_state")]
    fn dense_outer_state(
        &self,
        context: Stage1OuterRemainingContext<'_, Fr>,
        num_rounds: usize,
        batching_coeff: Fr,
    ) -> DenseOuterState<Fr> {
        let tau_high = context.tau[context.tau.len() - 1];
        let tau_low = &context.tau[..context.tau.len() - 1];
        let lagrange_tau_r0 = lagrange_kernel_eval(
            OUTER_UNISKIP_BASE_START,
            OUTER_UNISKIP_DOMAIN_SIZE,
            tau_high,
            context.r0,
        );
        let weights = lagrange_evals(
            OUTER_UNISKIP_BASE_START,
            OUTER_UNISKIP_DOMAIN_SIZE,
            context.r0,
        );
        let len = 1usize << num_rounds;
        let scale = lagrange_tau_r0 * batching_coeff;
        let eq_evals = EqPolynomial::new(tau_low.to_vec()).evaluations();
        let mut eq = vec![Fr::from_u64(0); len];
        let mut az = vec![Fr::from_u64(0); len];
        let mut bz = vec![Fr::from_u64(0); len];
        eq.par_chunks_mut(2)
            .zip(az.par_chunks_mut(2))
            .zip(bz.par_chunks_mut(2))
            .enumerate()
            .for_each(|(cycle, ((eq_pair, az_pair), bz_pair))| {
                let index = cycle << 1;
                let eval = Stage1Rv64Eval::new(&self.cycles[cycle]);
                let (az_g0, bz_g0) = eval.first_group_linear(&weights);
                let (az_g1, bz_g1) = eval.second_group_linear(&weights);
                eq_pair[0] = eq_evals[index] * scale;
                az_pair[0] = az_g0;
                bz_pair[0] = bz_g0;
                eq_pair[1] = eq_evals[index + 1] * scale;
                az_pair[1] = az_g1;
                bz_pair[1] = bz_g1;
            });
        DenseOuterState {
            eq,
            az,
            bz,
            eq_scratch: Vec::with_capacity(len / 2),
            az_scratch: Vec::with_capacity(len / 2),
            bz_scratch: Vec::with_capacity(len / 2),
        }
    }
}

impl Stage1OuterRemainingEvaluator<Fr> for Stage1OuterRv64Data<'_> {
    fn evaluate(&self, context: Stage1OuterRemainingContext<'_, Fr>, point: &[Fr]) -> Fr {
        self.field_data.evaluate(context, point)
    }

    #[tracing::instrument(skip_all, name = "Stage1OuterRv64Data::uniskip_extended_evals")]
    fn uniskip_extended_evals(&self, tau: &[Fr]) -> Option<Vec<Fr>> {
        if tau.len() != self.field_data.key.num_cycle_vars() + 2 {
            return None;
        }
        let tau_low = &tau[..tau.len() - 1];
        let num_rounds = self.field_data.key.num_cycle_vars() + 1;
        let eq_evals = EqPolynomial::new(tau_low.to_vec()).evaluations();
        let num_cycles = 1usize << (num_rounds - 1);
        if self.cycles.len() != num_cycles {
            return None;
        }
        let accumulators = (0..num_cycles)
            .into_par_iter()
            .fold(
                || [FrSignedProductAccumulator::zero(); OUTER_UNISKIP_DEGREE],
                |mut local, cycle| {
                    let eval = Stage1Rv64Eval::new(&self.cycles[cycle]);
                    let (first_products, second_products) = eval.uniskip_products();
                    let index = cycle << 1;
                    for (target, accumulator) in local.iter_mut().enumerate() {
                        accumulator.fmadd_s192(eq_evals[index], first_products[target]);
                        accumulator.fmadd_s192(eq_evals[index + 1], second_products[target]);
                    }
                    local
                },
            )
            .reduce(
                || [FrSignedProductAccumulator::zero(); OUTER_UNISKIP_DEGREE],
                |mut left, right| {
                    for (left, right) in left.iter_mut().zip(right) {
                        left.merge(right);
                    }
                    left
                },
            );
        Some(
            accumulators
                .into_iter()
                .map(FrSignedProductAccumulator::reduce)
                .collect(),
        )
    }

    fn evaluate_virtual_oracle(
        &self,
        context: Stage1OuterRemainingContext<'_, Fr>,
        oracle: &str,
        point: &[Fr],
    ) -> Option<Fr> {
        self.evaluate_virtual_oracles(context, &[oracle], point)
            .and_then(|values| values.into_iter().next())
    }

    #[tracing::instrument(skip_all, name = "Stage1OuterRv64Data::evaluate_virtual_oracles")]
    fn evaluate_virtual_oracles(
        &self,
        _context: Stage1OuterRemainingContext<'_, Fr>,
        oracles: &[&str],
        point: &[Fr],
    ) -> Option<Vec<Fr>> {
        if point.len() != self.field_data.key.num_cycle_vars() + 1 {
            return None;
        }
        let rv64_oracles = oracles
            .iter()
            .map(|oracle| Stage1Rv64Oracle::from_name(oracle))
            .collect::<Option<Vec<_>>>()?;
        let cycle_point = Stage1OuterR1csData::<Fr>::remaining_cycle_point(point);
        if let Some(cycle) = boolean_index(&cycle_point) {
            let row = self.cycles.get(cycle)?;
            return Some(
                rv64_oracles
                    .iter()
                    .map(|oracle| oracle.field_value(row))
                    .collect(),
            );
        }

        let eq = EqPolynomial::new(cycle_point).evaluations();
        let accumulators = eq
            .par_iter()
            .take(self.cycles.len())
            .enumerate()
            .fold(
                || vec![FrSignedProductAccumulator::zero(); rv64_oracles.len()],
                |mut local, (cycle, &weight)| {
                    let row = &self.cycles[cycle];
                    for (accumulator, oracle) in local.iter_mut().zip(&rv64_oracles) {
                        accumulator.fmadd_rv64_scalar(weight, oracle.scalar(row));
                    }
                    local
                },
            )
            .reduce(
                || vec![FrSignedProductAccumulator::zero(); rv64_oracles.len()],
                |mut left, right| {
                    for (left, right) in left.iter_mut().zip(right) {
                        left.merge(right);
                    }
                    left
                },
            );
        Some(
            accumulators
                .into_iter()
                .map(FrSignedProductAccumulator::reduce)
                .collect(),
        )
    }

    fn prove_remaining_rounds(
        &self,
        context: Stage1OuterRemainingContext<'_, Fr>,
        num_rounds: usize,
        batching_coeff: Fr,
        initial_claim: Fr,
        observe_round: &mut dyn FnMut(&UnivariatePoly<Fr>) -> Fr,
    ) -> Option<Stage1RemainingRoundProof<Fr>> {
        #[cfg(feature = "cuda")]
        if context.backend == "cuda" {
            if let Some(result) = crate::cuda_stage1::prove_remaining_rounds_cuda(
                &self.field_data,
                context,
                num_rounds,
                batching_coeff,
                initial_claim,
                observe_round,
            ) {
                return Some(result);
            }
        }

        let mut state = self.dense_outer_state(context, num_rounds, batching_coeff);
        let mut running_sum = initial_claim * batching_coeff;
        let mut point = Vec::with_capacity(num_rounds);
        let mut round_polynomials = Vec::with_capacity(num_rounds);

        for _round in 0..num_rounds {
            let poly = state.round_poly();
            if poly.evaluate(Fr::from_u64(0)) + poly.evaluate(Fr::from_u64(1)) != running_sum {
                return Some(Err(Stage1KernelError::InvalidProof {
                    driver: "stage1.outer.remaining",
                    reason: "dense outer remaining claim mismatch",
                }));
            }
            let challenge = observe_round(&poly);
            running_sum = poly.evaluate(challenge);
            state.bind(challenge);
            point.push(challenge);
            round_polynomials.push(poly);
        }
        Some(Ok((point, round_polynomials)))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage1Rv64Oracle {
    LeftInstructionInput,
    RightInstructionInput,
    Product,
    ShouldBranch,
    Pc,
    UnexpandedPc,
    Imm,
    RamAddress,
    Rs1Value,
    Rs2Value,
    RdWriteValue,
    RamReadValue,
    RamWriteValue,
    LeftLookupOperand,
    RightLookupOperand,
    NextUnexpandedPc,
    NextPc,
    NextIsVirtual,
    NextIsFirstInSequence,
    LookupOutput,
    ShouldJump,
    OpFlagAddOperands,
    OpFlagSubtractOperands,
    OpFlagMultiplyOperands,
    OpFlagLoad,
    OpFlagStore,
    OpFlagJump,
    OpFlagWriteLookupOutputToRd,
    OpFlagVirtualInstruction,
    OpFlagAssert,
    OpFlagDoNotUpdateUnexpandedPc,
    OpFlagAdvice,
    OpFlagIsCompressed,
    OpFlagIsFirstInSequence,
    OpFlagIsLastInSequence,
}

impl Stage1Rv64Oracle {
    fn from_name(name: &str) -> Option<Self> {
        match name {
            "LeftInstructionInput" => Some(Self::LeftInstructionInput),
            "RightInstructionInput" => Some(Self::RightInstructionInput),
            "Product" => Some(Self::Product),
            "ShouldBranch" => Some(Self::ShouldBranch),
            "PC" => Some(Self::Pc),
            "UnexpandedPC" => Some(Self::UnexpandedPc),
            "Imm" => Some(Self::Imm),
            "RamAddress" => Some(Self::RamAddress),
            "Rs1Value" => Some(Self::Rs1Value),
            "Rs2Value" => Some(Self::Rs2Value),
            "RdWriteValue" => Some(Self::RdWriteValue),
            "RamReadValue" => Some(Self::RamReadValue),
            "RamWriteValue" => Some(Self::RamWriteValue),
            "LeftLookupOperand" => Some(Self::LeftLookupOperand),
            "RightLookupOperand" => Some(Self::RightLookupOperand),
            "NextUnexpandedPC" => Some(Self::NextUnexpandedPc),
            "NextPC" => Some(Self::NextPc),
            "NextIsVirtual" => Some(Self::NextIsVirtual),
            "NextIsFirstInSequence" => Some(Self::NextIsFirstInSequence),
            "LookupOutput" => Some(Self::LookupOutput),
            "ShouldJump" => Some(Self::ShouldJump),
            "OpFlagAddOperands" => Some(Self::OpFlagAddOperands),
            "OpFlagSubtractOperands" => Some(Self::OpFlagSubtractOperands),
            "OpFlagMultiplyOperands" => Some(Self::OpFlagMultiplyOperands),
            "OpFlagLoad" => Some(Self::OpFlagLoad),
            "OpFlagStore" => Some(Self::OpFlagStore),
            "OpFlagJump" => Some(Self::OpFlagJump),
            "OpFlagWriteLookupOutputToRD" => Some(Self::OpFlagWriteLookupOutputToRd),
            "OpFlagVirtualInstruction" => Some(Self::OpFlagVirtualInstruction),
            "OpFlagAssert" => Some(Self::OpFlagAssert),
            "OpFlagDoNotUpdateUnexpandedPC" => Some(Self::OpFlagDoNotUpdateUnexpandedPc),
            "OpFlagAdvice" => Some(Self::OpFlagAdvice),
            "OpFlagIsCompressed" => Some(Self::OpFlagIsCompressed),
            "OpFlagIsFirstInSequence" => Some(Self::OpFlagIsFirstInSequence),
            "OpFlagIsLastInSequence" => Some(Self::OpFlagIsLastInSequence),
            _ => None,
        }
    }

    #[inline]
    fn scalar(self, row: &Stage1Rv64Cycle) -> Stage1Rv64Scalar {
        match self {
            Self::LeftInstructionInput => Stage1Rv64Scalar::U64(row.left_input),
            Self::RightInstructionInput => Stage1Rv64Scalar::S64(row.right_input),
            Self::Product => Stage1Rv64Scalar::S128(row.product),
            Self::ShouldBranch => Stage1Rv64Scalar::Bool(row.should_branch),
            Self::Pc => Stage1Rv64Scalar::U64(row.pc),
            Self::UnexpandedPc => Stage1Rv64Scalar::U64(row.unexpanded_pc),
            Self::Imm => Stage1Rv64Scalar::S64(row.imm),
            Self::RamAddress => Stage1Rv64Scalar::U64(row.ram_addr),
            Self::Rs1Value => Stage1Rv64Scalar::U64(row.rs1_read_value),
            Self::Rs2Value => Stage1Rv64Scalar::U64(row.rs2_read_value),
            Self::RdWriteValue => Stage1Rv64Scalar::U64(row.rd_write_value),
            Self::RamReadValue => Stage1Rv64Scalar::U64(row.ram_read_value),
            Self::RamWriteValue => Stage1Rv64Scalar::U64(row.ram_write_value),
            Self::LeftLookupOperand => Stage1Rv64Scalar::U64(row.left_lookup),
            Self::RightLookupOperand => Stage1Rv64Scalar::U128(row.right_lookup),
            Self::NextUnexpandedPc => Stage1Rv64Scalar::U64(row.next_unexpanded_pc),
            Self::NextPc => Stage1Rv64Scalar::U64(row.next_pc),
            Self::NextIsVirtual => Stage1Rv64Scalar::Bool(row.next_is_virtual),
            Self::NextIsFirstInSequence => Stage1Rv64Scalar::Bool(row.next_is_first_in_sequence),
            Self::LookupOutput => Stage1Rv64Scalar::U64(row.lookup_output),
            Self::ShouldJump => Stage1Rv64Scalar::Bool(row.should_jump),
            Self::OpFlagAddOperands => Stage1Rv64Scalar::Bool(row.flags[FLAG_ADD_OPERANDS]),
            Self::OpFlagSubtractOperands => {
                Stage1Rv64Scalar::Bool(row.flags[FLAG_SUBTRACT_OPERANDS])
            }
            Self::OpFlagMultiplyOperands => {
                Stage1Rv64Scalar::Bool(row.flags[FLAG_MULTIPLY_OPERANDS])
            }
            Self::OpFlagLoad => Stage1Rv64Scalar::Bool(row.flags[FLAG_LOAD]),
            Self::OpFlagStore => Stage1Rv64Scalar::Bool(row.flags[FLAG_STORE]),
            Self::OpFlagJump => Stage1Rv64Scalar::Bool(row.flags[FLAG_JUMP]),
            Self::OpFlagWriteLookupOutputToRd => {
                Stage1Rv64Scalar::Bool(row.flags[FLAG_WRITE_LOOKUP_OUTPUT_TO_RD])
            }
            Self::OpFlagVirtualInstruction => {
                Stage1Rv64Scalar::Bool(row.flags[FLAG_VIRTUAL_INSTRUCTION])
            }
            Self::OpFlagAssert => Stage1Rv64Scalar::Bool(row.flags[FLAG_ASSERT]),
            Self::OpFlagDoNotUpdateUnexpandedPc => {
                Stage1Rv64Scalar::Bool(row.flags[FLAG_DO_NOT_UPDATE_UNEXPANDED_PC])
            }
            Self::OpFlagAdvice => Stage1Rv64Scalar::Bool(row.flags[FLAG_ADVICE]),
            Self::OpFlagIsCompressed => Stage1Rv64Scalar::Bool(row.flags[FLAG_IS_COMPRESSED]),
            Self::OpFlagIsFirstInSequence => {
                Stage1Rv64Scalar::Bool(row.flags[FLAG_IS_FIRST_IN_SEQUENCE])
            }
            Self::OpFlagIsLastInSequence => {
                Stage1Rv64Scalar::Bool(row.flags[FLAG_IS_LAST_IN_SEQUENCE])
            }
        }
    }

    #[inline]
    fn field_value(self, row: &Stage1Rv64Cycle) -> Fr {
        let mut accumulator = FrSignedProductAccumulator::zero();
        accumulator.fmadd_rv64_scalar(Fr::from_u64(1), self.scalar(row));
        accumulator.reduce()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage1Rv64Scalar {
    Bool(bool),
    U64(u64),
    U128(u128),
    S64(S64),
    S128(S128),
}

struct Stage1Rv64Eval<'a> {
    row: &'a Stage1Rv64Cycle,
}

impl<'a> Stage1Rv64Eval<'a> {
    fn new(row: &'a Stage1Rv64Cycle) -> Self {
        Self { row }
    }

    #[inline]
    fn first_group_linear(&self, weights: &[Fr]) -> (Fr, Fr) {
        let mut az = Fr::from_u64(0);
        let mut bz = FrSignedProductAccumulator::zero();

        Self::accumulate_first_linear(
            &mut az,
            &mut bz,
            weights[0],
            self.not_load_store(),
            S128::from_u64(self.row.ram_addr),
        );
        Self::accumulate_first_linear(
            &mut az,
            &mut bz,
            weights[1],
            self.row.flags[FLAG_LOAD],
            S128::zero_extend_from(&S64::from_diff_u64s(
                self.row.ram_read_value,
                self.row.ram_write_value,
            )),
        );
        Self::accumulate_first_linear(
            &mut az,
            &mut bz,
            weights[2],
            self.row.flags[FLAG_LOAD],
            S128::zero_extend_from(&S64::from_diff_u64s(
                self.row.ram_read_value,
                self.row.rd_write_value,
            )),
        );
        Self::accumulate_first_linear(
            &mut az,
            &mut bz,
            weights[3],
            self.row.flags[FLAG_STORE],
            S128::zero_extend_from(&S64::from_diff_u64s(
                self.row.rs2_read_value,
                self.row.ram_write_value,
            )),
        );
        Self::accumulate_first_linear(
            &mut az,
            &mut bz,
            weights[4],
            self.add_sub_mul(),
            S128::from_u64(self.row.left_lookup),
        );
        Self::accumulate_first_linear(
            &mut az,
            &mut bz,
            weights[5],
            !self.add_sub_mul(),
            S128::zero_extend_from(&S64::from_diff_u64s(
                self.row.left_lookup,
                self.row.left_input,
            )),
        );
        Self::accumulate_first_linear(
            &mut az,
            &mut bz,
            weights[6],
            self.row.flags[FLAG_ASSERT],
            S128::zero_extend_from(&S64::from_diff_u64s(self.row.lookup_output, 1)),
        );
        Self::accumulate_first_linear(
            &mut az,
            &mut bz,
            weights[7],
            self.row.should_jump,
            S128::zero_extend_from(&S64::from_diff_u64s(
                self.row.next_unexpanded_pc,
                self.row.lookup_output,
            )),
        );
        Self::accumulate_first_linear(
            &mut az,
            &mut bz,
            weights[8],
            self.row.flags[FLAG_VIRTUAL_INSTRUCTION] && !self.row.flags[FLAG_IS_LAST_IN_SEQUENCE],
            S128::zero_extend_from(&S64::from_diff_u64s(
                self.row.next_pc,
                self.row.pc.wrapping_add(1),
            )),
        );
        Self::accumulate_first_linear(
            &mut az,
            &mut bz,
            weights[9],
            self.row.next_is_virtual && !self.row.next_is_first_in_sequence,
            S128::from_u64(u64::from(!self.row.flags[FLAG_DO_NOT_UPDATE_UNEXPANDED_PC])),
        );

        (az, bz.reduce())
    }

    #[inline]
    fn second_group_linear(&self, weights: &[Fr]) -> (Fr, Fr) {
        let mut az = Fr::from_u64(0);
        let mut bz = FrSignedProductAccumulator::zero();

        Self::accumulate_second_linear(
            &mut az,
            &mut bz,
            weights[0],
            self.load_or_store(),
            S192::from_i128(self.ram_addr_minus_rs1_plus_imm()),
        );
        Self::accumulate_second_linear(
            &mut az,
            &mut bz,
            weights[1],
            self.row.flags[FLAG_ADD_OPERANDS],
            (S160::from(self.row.right_lookup) - S160::from(self.right_add_expected()))
                .to_signed_bigint_nplus1::<3>(),
        );
        Self::accumulate_second_linear(
            &mut az,
            &mut bz,
            weights[2],
            self.row.flags[FLAG_SUBTRACT_OPERANDS],
            (S160::from(self.row.right_lookup) - S160::from(self.right_sub_expected()))
                .to_signed_bigint_nplus1::<3>(),
        );
        Self::accumulate_second_linear(
            &mut az,
            &mut bz,
            weights[3],
            self.row.flags[FLAG_MULTIPLY_OPERANDS],
            (S160::from(self.row.right_lookup) - S160::from(self.row.product))
                .to_signed_bigint_nplus1::<3>(),
        );
        Self::accumulate_second_linear(
            &mut az,
            &mut bz,
            weights[4],
            !self.add_sub_mul_advice(),
            (S160::from(self.row.right_lookup)
                - S160::from(S128::zero_extend_from(&self.row.right_input)))
            .to_signed_bigint_nplus1::<3>(),
        );
        Self::accumulate_second_linear(
            &mut az,
            &mut bz,
            weights[5],
            self.row.flags[FLAG_WRITE_LOOKUP_OUTPUT_TO_RD],
            S192::zero_extend_from(&S64::from_diff_u64s(
                self.row.rd_write_value,
                self.row.lookup_output,
            )),
        );
        Self::accumulate_second_linear(
            &mut az,
            &mut bz,
            weights[6],
            self.row.flags[FLAG_JUMP],
            S192::zero_extend_from(&S64::from_diff_u64s(
                self.row.rd_write_value,
                self.expected_pc_plus_const(),
            )),
        );
        Self::accumulate_second_linear(
            &mut az,
            &mut bz,
            weights[7],
            self.row.should_branch,
            S192::from_i128(self.next_unexpanded_pc_minus_pc_plus_imm()),
        );
        Self::accumulate_second_linear(
            &mut az,
            &mut bz,
            weights[8],
            !self.row.flags[FLAG_JUMP] && !self.row.should_branch,
            S192::zero_extend_from(&S64::from_diff_u64s(
                self.row.next_unexpanded_pc,
                self.expected_next_unexpanded_pc(),
            )),
        );

        (az, bz.reduce())
    }

    #[inline]
    fn uniskip_products(&self) -> ([S192; OUTER_UNISKIP_DEGREE], [S192; OUTER_UNISKIP_DEGREE]) {
        let (first_guards, first_terms) = self.first_group_terms();
        let (second_guards, second_terms) = self.second_group_terms();
        (
            core::array::from_fn(|target| {
                Self::first_group_product_from_terms(target, &first_guards, &first_terms)
            }),
            core::array::from_fn(|target| {
                Self::second_group_product_from_terms(target, &second_guards, &second_terms)
            }),
        )
    }

    #[inline]
    fn first_group_terms(
        &self,
    ) -> (
        [bool; OUTER_UNISKIP_DOMAIN_SIZE],
        [S128; OUTER_UNISKIP_DOMAIN_SIZE],
    ) {
        (
            [
                self.not_load_store(),
                self.row.flags[FLAG_LOAD],
                self.row.flags[FLAG_LOAD],
                self.row.flags[FLAG_STORE],
                self.add_sub_mul(),
                !self.add_sub_mul(),
                self.row.flags[FLAG_ASSERT],
                self.row.should_jump,
                self.row.flags[FLAG_VIRTUAL_INSTRUCTION]
                    && !self.row.flags[FLAG_IS_LAST_IN_SEQUENCE],
                self.row.next_is_virtual && !self.row.next_is_first_in_sequence,
            ],
            [
                S128::from_u64(self.row.ram_addr),
                S128::zero_extend_from(&S64::from_diff_u64s(
                    self.row.ram_read_value,
                    self.row.ram_write_value,
                )),
                S128::zero_extend_from(&S64::from_diff_u64s(
                    self.row.ram_read_value,
                    self.row.rd_write_value,
                )),
                S128::zero_extend_from(&S64::from_diff_u64s(
                    self.row.rs2_read_value,
                    self.row.ram_write_value,
                )),
                S128::from_u64(self.row.left_lookup),
                S128::zero_extend_from(&S64::from_diff_u64s(
                    self.row.left_lookup,
                    self.row.left_input,
                )),
                S128::zero_extend_from(&S64::from_diff_u64s(self.row.lookup_output, 1)),
                S128::zero_extend_from(&S64::from_diff_u64s(
                    self.row.next_unexpanded_pc,
                    self.row.lookup_output,
                )),
                S128::zero_extend_from(&S64::from_diff_u64s(
                    self.row.next_pc,
                    self.row.pc.wrapping_add(1),
                )),
                S128::from_u64(u64::from(!self.row.flags[FLAG_DO_NOT_UPDATE_UNEXPANDED_PC])),
            ],
        )
    }

    #[inline]
    fn second_group_terms(
        &self,
    ) -> (
        [bool; OUTER_SECOND_GROUP_ROWS.len()],
        [S192; OUTER_SECOND_GROUP_ROWS.len()],
    ) {
        (
            [
                self.load_or_store(),
                self.row.flags[FLAG_ADD_OPERANDS],
                self.row.flags[FLAG_SUBTRACT_OPERANDS],
                self.row.flags[FLAG_MULTIPLY_OPERANDS],
                !self.add_sub_mul_advice(),
                self.row.flags[FLAG_WRITE_LOOKUP_OUTPUT_TO_RD],
                self.row.flags[FLAG_JUMP],
                self.row.should_branch,
                !self.row.flags[FLAG_JUMP] && !self.row.should_branch,
            ],
            [
                S192::from_i128(self.ram_addr_minus_rs1_plus_imm()),
                (S160::from(self.row.right_lookup) - S160::from(self.right_add_expected()))
                    .to_signed_bigint_nplus1::<3>(),
                (S160::from(self.row.right_lookup) - S160::from(self.right_sub_expected()))
                    .to_signed_bigint_nplus1::<3>(),
                (S160::from(self.row.right_lookup) - S160::from(self.row.product))
                    .to_signed_bigint_nplus1::<3>(),
                (S160::from(self.row.right_lookup)
                    - S160::from(S128::zero_extend_from(&self.row.right_input)))
                .to_signed_bigint_nplus1::<3>(),
                S192::zero_extend_from(&S64::from_diff_u64s(
                    self.row.rd_write_value,
                    self.row.lookup_output,
                )),
                S192::zero_extend_from(&S64::from_diff_u64s(
                    self.row.rd_write_value,
                    self.expected_pc_plus_const(),
                )),
                S192::from_i128(self.next_unexpanded_pc_minus_pc_plus_imm()),
                S192::zero_extend_from(&S64::from_diff_u64s(
                    self.row.next_unexpanded_pc,
                    self.expected_next_unexpanded_pc(),
                )),
            ],
        )
    }

    #[inline]
    fn first_group_product_from_terms(
        target: usize,
        guards: &[bool; OUTER_UNISKIP_DOMAIN_SIZE],
        terms: &[S128; OUTER_UNISKIP_DOMAIN_SIZE],
    ) -> S192 {
        let coefficients = OUTER_UNISKIP_TARGET_COEFFS[target];
        let mut az = 0i32;
        let mut bz = S128::zero();
        for ((&guard, &term), &coefficient) in guards.iter().zip(terms).zip(&coefficients) {
            Self::accumulate_first(&mut az, &mut bz, coefficient as i32, guard, term);
        }
        S64::from_i64(az as i64).mul_trunc::<2, 3>(&bz)
    }

    #[inline]
    fn second_group_product_from_terms(
        target: usize,
        guards: &[bool; OUTER_SECOND_GROUP_ROWS.len()],
        terms: &[S192; OUTER_SECOND_GROUP_ROWS.len()],
    ) -> S192 {
        let coefficients = OUTER_UNISKIP_TARGET_COEFFS[target];
        let mut az = 0i32;
        let mut bz = S192::zero();
        for ((&guard, &term), &coefficient) in guards.iter().zip(terms).zip(&coefficients) {
            Self::accumulate_second(&mut az, &mut bz, coefficient as i32, guard, term);
        }
        S64::from_i64(az as i64).mul_trunc::<3, 3>(&bz)
    }

    #[inline]
    fn accumulate_first_linear(
        az: &mut Fr,
        bz: &mut FrSignedProductAccumulator,
        weight: Fr,
        guard: bool,
        term: S128,
    ) {
        if guard {
            *az += weight;
        } else {
            bz.fmadd_s128(weight, term);
        }
    }

    #[inline]
    fn accumulate_second_linear(
        az: &mut Fr,
        bz: &mut FrSignedProductAccumulator,
        weight: Fr,
        guard: bool,
        term: S192,
    ) {
        if guard {
            *az += weight;
        } else {
            bz.fmadd_s192(weight, term);
        }
    }

    #[inline]
    fn accumulate_first(az: &mut i32, bz: &mut S128, coefficient: i32, guard: bool, term: S128) {
        if guard {
            *az += coefficient;
        } else {
            fmadd_i32_s128(bz, coefficient, term);
        }
    }

    #[inline]
    fn accumulate_second(az: &mut i32, bz: &mut S192, coefficient: i32, guard: bool, term: S192) {
        if guard {
            *az += coefficient;
        } else {
            fmadd_i32_s192(bz, coefficient, term);
        }
    }

    #[inline]
    fn not_load_store(&self) -> bool {
        !self.load_or_store()
    }

    #[inline]
    fn load_or_store(&self) -> bool {
        self.row.flags[FLAG_LOAD] || self.row.flags[FLAG_STORE]
    }

    #[inline]
    fn add_sub_mul(&self) -> bool {
        self.row.flags[FLAG_ADD_OPERANDS]
            || self.row.flags[FLAG_SUBTRACT_OPERANDS]
            || self.row.flags[FLAG_MULTIPLY_OPERANDS]
    }

    #[inline]
    fn add_sub_mul_advice(&self) -> bool {
        self.add_sub_mul() || self.row.flags[FLAG_ADVICE]
    }

    #[inline]
    fn ram_addr_minus_rs1_plus_imm(&self) -> i128 {
        let expected = if self.row.imm.is_positive {
            self.row.rs1_read_value as i128 + self.row.imm.magnitude_as_u64() as i128
        } else {
            self.row.rs1_read_value as i128 - self.row.imm.magnitude_as_u64() as i128
        };
        self.row.ram_addr as i128 - expected
    }

    #[inline]
    fn right_add_expected(&self) -> i128 {
        self.row.left_input as i128 + self.row.right_input.to_i128()
    }

    #[inline]
    fn right_sub_expected(&self) -> i128 {
        self.row.left_input as i128 - self.row.right_input.to_i128() + (1i128 << 64)
    }

    #[inline]
    fn expected_pc_plus_const(&self) -> u64 {
        let const_term = 4 - if self.row.flags[FLAG_IS_COMPRESSED] {
            2
        } else {
            0
        };
        self.row.unexpanded_pc.wrapping_add(const_term)
    }

    #[inline]
    fn next_unexpanded_pc_minus_pc_plus_imm(&self) -> i128 {
        self.row.next_unexpanded_pc as i128
            - (self.row.unexpanded_pc as i128 + self.row.imm.to_i128())
    }

    #[inline]
    fn expected_next_unexpanded_pc(&self) -> u64 {
        let const_term =
            4 - if self.row.flags[FLAG_DO_NOT_UPDATE_UNEXPANDED_PC] {
                4
            } else {
                0
            } - if self.row.flags[FLAG_IS_COMPRESSED] {
                2
            } else {
                0
            };
        self.row.unexpanded_pc.wrapping_add(const_term)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FrSignedProductAccumulator {
    positive: Limbs<9>,
    negative: Limbs<9>,
}

impl FrSignedProductAccumulator {
    #[inline]
    fn zero() -> Self {
        Self {
            positive: Limbs::zero(),
            negative: Limbs::zero(),
        }
    }

    #[inline]
    fn fmadd_s192(&mut self, field: Fr, scalar: S192) {
        if scalar.magnitude_limbs() == [0u64; 3] {
            return;
        }
        self.fmadd_limbs(field, &scalar.magnitude, scalar.is_positive);
    }

    #[inline]
    fn fmadd_rv64_scalar(&mut self, field: Fr, scalar: Stage1Rv64Scalar) {
        match scalar {
            Stage1Rv64Scalar::Bool(value) => {
                if value {
                    self.add_positive_field(field);
                }
            }
            Stage1Rv64Scalar::U64(value) => self.fmadd_u64(field, value),
            Stage1Rv64Scalar::U128(value) => self.fmadd_u128(field, value),
            Stage1Rv64Scalar::S64(value) => self.fmadd_s64(field, value),
            Stage1Rv64Scalar::S128(value) => self.fmadd_s128(field, value),
        }
    }

    #[inline]
    fn fmadd_u64(&mut self, field: Fr, scalar: u64) {
        if scalar == 0 {
            return;
        }
        if scalar == 1 {
            self.add_positive_field(field);
            return;
        }
        self.fmadd_limbs(field, &Limbs::<1>::from_u64(scalar), true);
    }

    #[inline]
    fn fmadd_u128(&mut self, field: Fr, scalar: u128) {
        if scalar == 0 {
            return;
        }
        if scalar <= u64::MAX as u128 {
            self.fmadd_u64(field, scalar as u64);
            return;
        }
        self.fmadd_limbs(
            field,
            &Limbs::<2>::new([scalar as u64, (scalar >> 64) as u64]),
            true,
        );
    }

    #[inline]
    fn fmadd_s64(&mut self, field: Fr, scalar: S64) {
        if scalar.magnitude_limbs() == [0u64; 1] {
            return;
        }
        if scalar.is_positive {
            self.fmadd_u64(field, scalar.magnitude_as_u64());
            return;
        }
        self.fmadd_limbs(field, scalar.as_magnitude(), false);
    }

    #[inline]
    fn fmadd_s128(&mut self, field: Fr, scalar: S128) {
        if scalar.magnitude_limbs() == [0u64; 2] {
            return;
        }
        self.fmadd_limbs(field, scalar.as_magnitude(), scalar.is_positive);
    }

    #[inline]
    fn add_positive_field(&mut self, field: Fr) {
        self.positive
            .add_assign_trunc::<9>(&Limbs::<9>::zero_extend_from::<4>(&field.inner_limbs()));
    }

    #[inline]
    fn fmadd_limbs<const L: usize>(&mut self, field: Fr, scalar: &Limbs<L>, is_positive: bool) {
        let mut product = Limbs::<9>::zero();
        product.fmadd::<4, L>(&field.inner_limbs(), scalar);
        if is_positive {
            self.positive.add_assign_trunc::<9>(&product);
        } else {
            self.negative.add_assign_trunc::<9>(&product);
        }
    }

    #[inline]
    fn merge(&mut self, other: Self) {
        self.positive.add_assign_trunc::<9>(&other.positive);
        self.negative.add_assign_trunc::<9>(&other.negative);
    }

    #[inline]
    fn reduce(self) -> Fr {
        match self.positive.cmp(&self.negative) {
            Ordering::Greater | Ordering::Equal => {
                let difference = self.positive.sub_trunc::<9, 9>(&self.negative);
                Fr::from_barrett_reduced_limbs(difference)
            }
            Ordering::Less => {
                let difference = self.negative.sub_trunc::<9, 9>(&self.positive);
                -Fr::from_barrett_reduced_limbs(difference)
            }
        }
    }
}

#[inline]
fn fmadd_i32_s128(sum: &mut S128, coefficient: i32, term: S128) {
    if coefficient == 0 || term.magnitude_as_u128() == 0 {
        return;
    }
    let coefficient_s64 = S64::from_i64(coefficient as i64);
    *sum += coefficient_s64.mul_trunc::<2, 2>(&term);
}

#[inline]
fn fmadd_i32_s192(sum: &mut S192, coefficient: i32, term: S192) {
    if coefficient == 0 || term.magnitude_limbs() == [0u64; 3] {
        return;
    }
    let coefficient_s64 = S64::from_i64(coefficient as i64);
    *sum += coefficient_s64.mul_trunc::<3, 3>(&term);
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests use explicit panic messages")]
mod tests {
    use jolt_field::{Field, Fr};
    use jolt_r1cs::{constraints::rv64, R1csKey};

    use super::*;

    static RV64_ORACLE_NAMES: &[&str] = &[
        "LeftInstructionInput",
        "RightInstructionInput",
        "Product",
        "ShouldBranch",
        "PC",
        "UnexpandedPC",
        "Imm",
        "RamAddress",
        "Rs1Value",
        "Rs2Value",
        "RdWriteValue",
        "RamReadValue",
        "RamWriteValue",
        "LeftLookupOperand",
        "RightLookupOperand",
        "NextUnexpandedPC",
        "NextPC",
        "NextIsVirtual",
        "NextIsFirstInSequence",
        "LookupOutput",
        "ShouldJump",
        "OpFlagAddOperands",
        "OpFlagSubtractOperands",
        "OpFlagMultiplyOperands",
        "OpFlagLoad",
        "OpFlagStore",
        "OpFlagJump",
        "OpFlagWriteLookupOutputToRD",
        "OpFlagVirtualInstruction",
        "OpFlagAssert",
        "OpFlagDoNotUpdateUnexpandedPC",
        "OpFlagAdvice",
        "OpFlagIsCompressed",
        "OpFlagIsFirstInSequence",
        "OpFlagIsLastInSequence",
    ];

    fn rv64_eval_test_cycles() -> Vec<Stage1Rv64Cycle> {
        let mut first = Stage1Rv64Cycle::padding();
        first.left_input = 7;
        first.right_input = S64::from_i64(-5);
        first.product = S128::from_i128(-35);
        first.left_lookup = 11;
        first.right_lookup = u128::MAX - 4;
        first.lookup_output = 1;
        first.rs1_read_value = 13;
        first.rs2_read_value = 17;
        first.rd_write_value = 19;
        first.ram_addr = 23;
        first.ram_read_value = 29;
        first.ram_write_value = 31;
        first.pc = 37;
        first.next_pc = 41;
        first.unexpanded_pc = 43;
        first.next_unexpanded_pc = 47;
        first.imm = S64::from_i64(-53);
        first.flags[FLAG_ADD_OPERANDS] = true;
        first.flags[FLAG_LOAD] = true;
        first.flags[FLAG_IS_FIRST_IN_SEQUENCE] = true;
        first.should_branch = true;
        first.next_is_virtual = true;

        let mut second = Stage1Rv64Cycle::padding();
        second.left_input = 59;
        second.right_input = S64::from_u64(61);
        second.product = S128::from_u128(3599);
        second.left_lookup = 67;
        second.right_lookup = 71;
        second.lookup_output = 73;
        second.rs1_read_value = 79;
        second.rs2_read_value = 83;
        second.rd_write_value = 89;
        second.ram_addr = 97;
        second.ram_read_value = 101;
        second.ram_write_value = 103;
        second.pc = 107;
        second.next_pc = 109;
        second.unexpanded_pc = 113;
        second.next_unexpanded_pc = 127;
        second.imm = S64::from_u64(131);
        second.flags[FLAG_MULTIPLY_OPERANDS] = true;
        second.flags[FLAG_STORE] = true;
        second.flags[FLAG_JUMP] = true;
        second.flags[FLAG_IS_LAST_IN_SEQUENCE] = true;
        second.should_jump = true;
        second.next_is_first_in_sequence = true;

        vec![first, second]
    }

    fn rv64_eval_test_witness(key: &R1csKey<Fr>, cycles: &[Stage1Rv64Cycle]) -> Vec<Fr> {
        let mut witness = vec![Fr::from_u64(0); key.num_cycles * key.num_vars_padded];
        for (cycle, row) in cycles.iter().enumerate() {
            let base = cycle * key.num_vars_padded;
            witness[base + rv64::V_CONST] = Fr::from_u64(1);
            for name in RV64_ORACLE_NAMES {
                let oracle = Stage1Rv64Oracle::from_name(name).expect("known RV64 oracle");
                let variable =
                    super::super::r1cs_oracle_variable(name).expect("known R1CS variable");
                witness[base + variable] = oracle.field_value(row);
            }
        }
        witness
    }

    #[test]
    fn typed_virtual_oracle_evals_match_r1cs_columns() {
        let cycles = rv64_eval_test_cycles();
        let key = R1csKey::new(rv64::rv64_constraints::<Fr>(), cycles.len());
        let witness = rv64_eval_test_witness(&key, &cycles);
        let r1cs_data = Stage1OuterR1csData::new(&key, &witness).expect("valid witness shape");
        let rv64_data =
            Stage1OuterRv64Data::new(&key, &witness, &cycles).expect("valid RV64 shape");
        let tau = [Fr::from_u64(0); 3];
        let context = Stage1OuterRemainingContext {
            tau: &tau,
            r0: Fr::from_u64(0),
            backend: "cpu",
        };

        for point in [
            vec![Fr::from_u64(3), Fr::from_u64(5)],
            vec![Fr::from_u64(3), Fr::from_u64(1)],
        ] {
            assert_eq!(
                rv64_data.evaluate_virtual_oracles(context, RV64_ORACLE_NAMES, &point),
                r1cs_data.evaluate_virtual_oracles(context, RV64_ORACLE_NAMES, &point)
            );
        }
    }
}
