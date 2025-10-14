use std::{cell::RefCell, iter::once, rc::Rc, sync::Arc};

use num_traits::Zero;

use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        identity_poly::IdentityPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
            BIG_ENDIAN,
        },
        ra_poly::RaPolynomial,
    },
    subprotocols::sumcheck::SumcheckInstance,
    transcripts::Transcript,
    utils::{
        expanding_table::ExpandingTable, math::Math, small_scalar::SmallScalar,
        thread::unsafe_allocate_zero_vec,
    },
    zkvm::{
        dag::state_manager::StateManager,
        instruction::{
            CircuitFlags, InstructionFlags, InstructionLookup, InterleavedBitsMarker,
            NUM_CIRCUIT_FLAGS,
        },
        lookup_table::{LookupTables, NUM_LOOKUP_TABLES},
        witness::{CommittedPolynomial, VirtualPolynomial},
    },
};
use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use common::constants::{REGISTER_COUNT, XLEN};
use rayon::prelude::*;
use strum::{EnumCount, IntoEnumIterator};
use tracer::instruction::NormalizedInstruction;

// Bytecode Read+RAF sumcheck
//
// Proves an address-and-cycle routed aggregation equality of the form
//   Σ_{a ∈ {0,1}^{log K}} Σ_{c ∈ {0,1}^{log T}}
//     EQ(r_address, a) · PC_routing(a, c) ·
//       [ (Val_1(a) + γ^3·Int(a)) · EQ(r_cycle_1, c)         // Spartan outer: instructions & immediates
//       + γ · Val_2(a) · EQ(r_cycle_2, c)                    // Register read-write checking
//       + γ^2 · (Val_3(a) + γ^2·Int(a)) · EQ(r_cycle_3, c) ] // Register val eval & instruction lookups
//   = rv_claim,
// where PC_routing(a, c) := Π_{i=0}^{d-1} 1[chunk_i(PC(c)) = chunk_i(a)] routes cycle c to address PC(c).
//
// Expanded statement:
// Let K = 2^{log K}, T = 2^{log T}. For address a = (a_0..a_{logK-1}) ∈ {0,1}^{logK} and cycle c = (c_0..c_{logT-1}) ∈ {0,1}^{logT}:
//
//   Σ_{a} Σ_{c}  EQ_addr(r_addr, a) · [Π_{i=0}^{d-1} 1[chunk_i(PC(c)) = chunk_i(a)]] ·
//                 ( (Val1(a) + γ^3·Int(a)) · EQ_cyc1(r_cyc1, c)
//                   + γ · Val2(a) · EQ_cyc2(r_cyc2, c)
//                   + γ^2 · (Val3(a) + γ^2·Int(a)) · EQ_cyc3(r_cyc3, c) )
// =  Σ_{c} EQ_cyc1(r_cyc1, c) · Val1(PC(c))
//   + γ · Σ_{c} EQ_cyc2(r_cyc2, c) · Val2(PC(c))
//   + γ^2 · Σ_{c} EQ_cyc3(r_cyc3, c) · Val3(PC(c))
//   + γ^3 · Σ_{c} EQ_cyc1(r_cyc1, c) · Int(PC(c))
//   + γ^4 · Σ_{c} EQ_shift(r_shift, c) · Int(PC(c+1))
//
// where:
//   EQ_addr(r_addr, a)   = Π_{j=0}^{logK-1} (1 - r_addr[j] - a_j + 2·r_addr[j]·a_j)
//   EQ_cycℓ(r_cycℓ, c)   = Π_{t=0}^{logT-1} (1 - r_cycℓ[t] - c_t + 2·r_cycℓ[t]·c_t),  for ℓ ∈ {1,2,3}
//   EQ_shift(r_shift, c) = Π_{t=0}^{logT-1} (1 - r_shift[t] - c_t + 2·r_shift[t]·c_t)  (the Spartan-Shift cycle EQ)
//   chunk_i(a)           = integer formed by the i-th address chunk bits of a (K^{1/d}-ary digit)
//   Int(a)               = Σ_{j=0}^{logK-1} 2^j · a_j
//   Val1(a)              = unexpanded_pc(a) + γ·imm(a) + γ^2·rd(a) + Σ_{j} γ^{3+j} · flag_j(a)
//   Val2(a)              = rd(a, r_reg) + γ·rs1(a, r_reg) + γ^2·rs2(a, r_reg)
//   Val3(a)              = rd(a, r_reg) + γ·unexpanded_pc(a) + γ^2·is_noop(a) + γ^3·is_not_interleaved(a)
//                          + Σ_{t=0}^{L-1} γ^{4+t} · table_flag_t(a)

/// Number of batched read-checking sumchecks bespokely
const STAGES: usize = 3;

#[derive(Allocative)]
struct ReadCheckingProverState<F: JoltField> {
    /// F_s[k] := Σ_j EQ(r_cycle_s, j) · 1[PC(j) = k]. Pre-aggregated routing mass over
    /// addresses for stage s (phase 1).
    F: [MultilinearPolynomial<F>; STAGES],
    /// ra_i(k,j) := 1[chunk_i(PC(j)) = chunk_i(k)]. Per-chunk routing indicators (RaPolynomial)
    /// constructed at the phase switch; Π_i ra_i(k,j) = 1[PC(j) = k].
    ra: Vec<RaPolynomial<u8, F>>,
    /// v_i[u] := prefix-weight for chunk i at index u (ExpandingTable). Used to accumulate routing
    /// mass during phase 1 address binding.
    v: Vec<ExpandingTable<F>>,
    /// EQ_s(j) := eq(r_cycle_s, j). Cycle EQ MLEs for stages s = 1..3 (phase 2).
    eq_polys: [MultilinearPolynomial<F>; STAGES],
    /// val_gamma[s] := (Val_s(r_address) + RAF_aug_s(r_address)) · γ^{s-1}. Per-stage scalars after
    /// phase 1 (used in phase 2).
    val_gamma: Option<[F; STAGES]>,
    /// program counter index per cycle j, used to form routing polynomials.
    pc: Vec<usize>,
}

#[derive(Allocative)]
pub struct ReadRafSumcheck<F: JoltField> {
    gamma: [F; STAGES],
    gamma_cub: F,
    gamma_sqr: F,
    rv_claim: F,
    log_K_chunk: usize,
    K_chunk: usize,
    log_K: usize,
    log_T: usize,
    d: usize,
    prover_state: Option<ReadCheckingProverState<F>>,
    val_polys: [MultilinearPolynomial<F>; STAGES],
    int_poly: IdentityPolynomial<F>,
}

#[derive(Debug, Clone, Copy)]
enum ReadCheckingValType {
    /// Spartan outer sumcheck
    Stage1,
    /// Registers read-write sumcheck
    Stage2,
    /// Registers val sumcheck wa, PCSumcheck, Instruction Lookups
    Stage3,
}

impl<F: JoltField> ReadRafSumcheck<F> {
    #[tracing::instrument(skip_all, name = "BytecodeReadRafSumcheck::new_prover")]
    pub fn new_prover(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Self {
        let K = sm.get_prover_data().0.shared.bytecode.code_size;
        let log_K = K.log_2();
        let d = sm.get_prover_data().0.shared.bytecode.d;
        let log_T = sm.get_prover_data().1.len().log_2();
        let log_K_chunk = log_K.div_ceil(d);
        let K_chunk = 1 << log_K_chunk;
        let gamma: F = sm.transcript.borrow_mut().challenge_scalar();
        let gamma_sqr = gamma.square();
        let gamma_cub = gamma_sqr * gamma;
        let gamma_four = gamma_sqr.square();
        let (val_1, rv_claim_1) = Self::compute_val_rv(sm, ReadCheckingValType::Stage1);
        let (val_2, rv_claim_2) = Self::compute_val_rv(sm, ReadCheckingValType::Stage2);
        let (val_3, rv_claim_3) = Self::compute_val_rv(sm, ReadCheckingValType::Stage3);
        let r_cycles = Self::get_r_cycle(&sm.get_prover_accumulator());
        let (_, raf_claim) =
            sm.get_virtual_polynomial_opening(VirtualPolynomial::PC, SumcheckId::SpartanOuter);
        let (_, raf_shift_claim) =
            sm.get_virtual_polynomial_opening(VirtualPolynomial::PC, SumcheckId::SpartanShift);
        let int_poly = IdentityPolynomial::<F>::new(log_K);
        let (preprocessing, trace, _, _) = sm.get_prover_data();

        let rv_claim = rv_claim_1
            + gamma * rv_claim_2
            + gamma_sqr * rv_claim_3
            + gamma_cub * raf_claim
            + gamma_four * raf_shift_claim;

        let eq_evals = [
            EqPolynomial::evals(&r_cycles[0]),
            EqPolynomial::evals(&r_cycles[1]),
            EqPolynomial::evals(&r_cycles[2]),
        ];

        let T = trace.len();
        let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
        let chunk_size = (T / num_chunks).max(1);

        let F = trace
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_index, trace_chunk)| {
                let mut result_1: Vec<F> = unsafe_allocate_zero_vec(K);
                let mut result_2: Vec<F> = unsafe_allocate_zero_vec(K);
                let mut result_3: Vec<F> = unsafe_allocate_zero_vec(K);
                let mut j = chunk_index * chunk_size;
                for cycle in trace_chunk {
                    let pc = preprocessing.shared.bytecode.get_pc(cycle);
                    result_1[pc] += eq_evals[0][j];
                    result_2[pc] += eq_evals[1][j];
                    result_3[pc] += eq_evals[2][j];
                    j += 1;
                }
                (result_1, result_2, result_3)
            })
            .reduce(
                || {
                    (
                        unsafe_allocate_zero_vec(K),
                        unsafe_allocate_zero_vec(K),
                        unsafe_allocate_zero_vec(K),
                    )
                },
                |(mut running_1, mut running_2, mut running_3), (new_1, new_2, new_3)| {
                    running_1
                        .par_iter_mut()
                        .zip(new_1.into_par_iter())
                        .for_each(|(x, y)| *x += y);
                    running_2
                        .par_iter_mut()
                        .zip(new_2.into_par_iter())
                        .for_each(|(x, y)| *x += y);
                    running_3
                        .par_iter_mut()
                        .zip(new_3.into_par_iter())
                        .for_each(|(x, y)| *x += y);
                    (running_1, running_2, running_3)
                },
            );

        let eq_polys = eq_evals
            .into_iter()
            .map(MultilinearPolynomial::from)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let F = [
            MultilinearPolynomial::from(F.0),
            MultilinearPolynomial::from(F.1),
            MultilinearPolynomial::from(F.2),
        ];

        let mut v = (0..d)
            .map(|_| ExpandingTable::new(K_chunk))
            .collect::<Vec<_>>();
        v.par_iter_mut().for_each(|v| v.reset(F::one()));

        let pc = trace
            .par_iter()
            .map(|cycle| preprocessing.shared.bytecode.get_pc(cycle))
            .collect();

        Self {
            rv_claim,
            log_K,
            log_K_chunk,
            K_chunk,
            d,
            log_T,
            prover_state: Some(ReadCheckingProverState {
                F,
                ra: Vec::with_capacity(d),
                v,
                eq_polys,
                val_gamma: None,
                pc,
            }),
            val_polys: [
                MultilinearPolynomial::from(val_1),
                MultilinearPolynomial::from(val_2),
                MultilinearPolynomial::from(val_3),
            ],
            int_poly,
            // r_cycles: [r_cycle_1, r_cycle_2, r_cycle_3],
            gamma: [F::one(), gamma, gamma_sqr],
            gamma_sqr,
            gamma_cub,
        }
    }

    pub fn new_verifier(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Self {
        let K = sm.get_verifier_data().0.shared.bytecode.code_size;
        let log_K = K.log_2();
        let d = sm.get_verifier_data().0.shared.bytecode.d;
        let log_T = sm.get_verifier_data().2.log_2();
        let log_K_chunk = log_K.div_ceil(d);
        let gamma: F = sm.transcript.borrow_mut().challenge_scalar();
        let gamma_sqr = gamma.square();
        let gamma_cub = gamma_sqr * gamma;
        let gamma_four = gamma_sqr.square();
        let (val_1, rv_claim_1) = Self::compute_val_rv(sm, ReadCheckingValType::Stage1);
        let (val_2, rv_claim_2) = Self::compute_val_rv(sm, ReadCheckingValType::Stage2);
        let (val_3, rv_claim_3) = Self::compute_val_rv(sm, ReadCheckingValType::Stage3);
        let int_poly = IdentityPolynomial::new(log_K);

        let val_polys = [
            MultilinearPolynomial::from(val_1),
            MultilinearPolynomial::from(val_2),
            MultilinearPolynomial::from(val_3),
        ];

        let (_, raf_claim) =
            sm.get_virtual_polynomial_opening(VirtualPolynomial::PC, SumcheckId::SpartanOuter);
        let (_, raf_shift_claim) =
            sm.get_virtual_polynomial_opening(VirtualPolynomial::PC, SumcheckId::SpartanShift);
        let rv_claim = rv_claim_1
            + gamma * rv_claim_2
            + gamma_sqr * rv_claim_3
            + gamma_cub * raf_claim
            + gamma_four * raf_shift_claim;

        Self {
            gamma: [F::one(), gamma, gamma_sqr],
            gamma_sqr,
            gamma_cub,
            rv_claim,
            log_K,
            log_K_chunk,
            K_chunk: 1 << log_K_chunk,
            d,
            log_T,
            prover_state: None,
            val_polys,
            int_poly,
            // r_cycles: [r_cycle_1, r_cycle_2, r_cycle_3],
        }
    }

    fn compute_val_rv(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        val_type: ReadCheckingValType,
    ) -> (Vec<F>, F) {
        match val_type {
            ReadCheckingValType::Stage1 => {
                let gamma_powers = get_gamma_powers(
                    &mut *sm.get_transcript().borrow_mut(),
                    3 + NUM_CIRCUIT_FLAGS,
                );
                (
                    Self::compute_val_1(sm, &gamma_powers),
                    Self::compute_rv_claim_1(sm, &gamma_powers),
                )
            }
            ReadCheckingValType::Stage2 => {
                let gamma_powers = get_gamma_powers(&mut *sm.get_transcript().borrow_mut(), 3);
                (
                    Self::compute_val_2(sm, &gamma_powers),
                    Self::compute_rv_claim_2(sm, &gamma_powers),
                )
            }
            ReadCheckingValType::Stage3 => {
                let gamma_powers = get_gamma_powers(
                    &mut *sm.get_transcript().borrow_mut(),
                    4 + NUM_LOOKUP_TABLES,
                );
                (
                    Self::compute_val_3(sm, &gamma_powers),
                    Self::compute_rv_claim_3(sm, &gamma_powers),
                )
            }
        }
    }

    fn get_r_cycle(acc: &Rc<RefCell<ProverOpeningAccumulator<F>>>) -> [Vec<F::Challenge>; STAGES] {
        let (r_cycle_1, _) = acc
            .borrow()
            .get_virtual_polynomial_opening(VirtualPolynomial::Imm, SumcheckId::SpartanOuter);
        let (r, _) = acc.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::Rs1Ra,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, r_cycle_2) = r.split_at((REGISTER_COUNT as usize).log_2());
        let (r, _) = acc.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersValEvaluation,
        );
        let (_, r_cycle_3) = r.split_at((REGISTER_COUNT as usize).log_2());
        [r_cycle_1.r, r_cycle_2.r, r_cycle_3.r]
    }

    fn get_r_cycle_verif(
        acc: &Rc<RefCell<VerifierOpeningAccumulator<F>>>,
    ) -> [Vec<F::Challenge>; STAGES] {
        let (r_cycle_1, _) = acc
            .borrow()
            .get_virtual_polynomial_opening(VirtualPolynomial::Imm, SumcheckId::SpartanOuter);
        let (r, _) = acc.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::Rs1Ra,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, r_cycle_2) = r.split_at((REGISTER_COUNT as usize).log_2());
        let (r, _) = acc.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersValEvaluation,
        );
        let (_, r_cycle_3) = r.split_at((REGISTER_COUNT as usize).log_2());
        [r_cycle_1.r, r_cycle_2.r, r_cycle_3.r]
    }

    /// Returns a vec of evaluations:
    ///    Val(k) = unexpanded_pc(k) + gamma * imm(k)
    ///             + gamma^2 * circuit_flags[0](k) + gamma^3 * circuit_flags[1](k) + ...
    /// This particular Val virtualizes claims output by Spartan's "outer" sumcheck
    fn compute_val_1(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        gamma_powers: &[F],
    ) -> Vec<F> {
        sm.get_bytecode()
            .par_iter()
            .map(|instruction| {
                let NormalizedInstruction {
                    address: unexpanded_pc,
                    operands,
                    ..
                } = instruction.normalize();

                let mut linear_combination = F::zero();
                linear_combination += F::from_u64(unexpanded_pc as u64);
                linear_combination += operands.imm.field_mul(gamma_powers[1]);
                linear_combination += (operands.rd as u64).field_mul(gamma_powers[2]);
                let flags = instruction.circuit_flags();
                // sanity check
                assert!(
                    !flags[CircuitFlags::IsCompressed]
                        || !flags[CircuitFlags::DoNotUpdateUnexpandedPC]
                );
                for (flag, gamma_power) in flags.iter().zip(gamma_powers[3..].iter()) {
                    if *flag {
                        linear_combination += *gamma_power;
                    }
                }

                linear_combination
            })
            .collect()
    }

    fn compute_rv_claim_1(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        gamma_powers: &[F],
    ) -> F {
        let (_, unexpanded_pc_claim) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanOuter,
        );
        let (_, imm_claim) =
            sm.get_virtual_polynomial_opening(VirtualPolynomial::Imm, SumcheckId::SpartanOuter);
        let (_, rd_claim) =
            sm.get_virtual_polynomial_opening(VirtualPolynomial::Rd, SumcheckId::SpartanOuter);
        once(unexpanded_pc_claim)
            .chain(once(imm_claim))
            .chain(once(rd_claim))
            .chain(CircuitFlags::iter().map(|flag| {
                sm.get_virtual_polynomial_opening(
                    VirtualPolynomial::OpFlags(flag),
                    SumcheckId::SpartanOuter,
                )
                .1
            }))
            .zip(gamma_powers)
            .map(|(claim, gamma)| claim * gamma)
            .sum()
    }

    /// Returns a vec of evaluations:
    ///    Val(k) = rd(k, r_register) + gamma * rs1(k, r_register) + gamma^2 * rs2(k, r_register)
    /// where rd, rs1, rs2 are MLEs of the corresponding indicator functions (1 on matching {0,1}-points),
    /// e.g., rd(k,k') = 1 if the k'th instruction has rd=k' on the Boolean hypercube
    /// and analogously for rs1(k, k') and rs2(k, k').
    /// This particular Val virtualizes claims output by the registers read/write checking sumcheck.
    fn compute_val_2(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        gamma_powers: &[F],
    ) -> Vec<F> {
        let r_register = sm
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RdWa,
                SumcheckId::RegistersReadWriteChecking,
            )
            .0
            .r;
        let r_register = &r_register[..(REGISTER_COUNT as usize).log_2()];
        let eq_r_register = EqPolynomial::<F>::evals(r_register);
        debug_assert_eq!(eq_r_register.len(), REGISTER_COUNT as usize);
        sm.get_bytecode()
            .par_iter()
            .map(|instruction| {
                let instr = instruction.normalize();

                std::iter::empty()
                    .chain(once(instr.operands.rd))
                    .chain(once(instr.operands.rs1))
                    .chain(once(instr.operands.rs2))
                    .map(|r| eq_r_register[r as usize])
                    .zip(gamma_powers)
                    .map(|(claim, gamma)| claim * gamma)
                    .sum::<F>()
            })
            .collect()
    }

    fn compute_rv_claim_2(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        gamma_powers: &[F],
    ) -> F {
        std::iter::empty()
            .chain(once(VirtualPolynomial::RdWa))
            .chain(once(VirtualPolynomial::Rs1Ra))
            .chain(once(VirtualPolynomial::Rs2Ra))
            .map(|vp| {
                sm.get_virtual_polynomial_opening(vp, SumcheckId::RegistersReadWriteChecking)
                    .1
            })
            .zip(gamma_powers)
            .map(|(claim, gamma)| claim * gamma)
            .sum()
    }

    /// Returns a vec of evaluations:
    ///    Val(k) = rd(k, r_register) + gamma * unexpanded_pc(k) + gamma^2 * instr_raf_flag(k)
    ///             + gamma^3 * lookup_table_flag[0](k)
    ///             + gamma^4 * lookup_table_flag[1](k) + ...
    /// where rd(k, k') = 1 if the k'th instruction in the bytecode has rd = k'
    /// This particular Val virtualizes claims output by the PCSumcheck,
    /// the registers val-evaluation sumcheck, and the instruction lookups sumcheck.
    fn compute_val_3(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        gamma_powers: &[F],
    ) -> Vec<F> {
        let r_register = sm
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RdWa,
                SumcheckId::RegistersValEvaluation,
            )
            .0
            .r;
        let r_register: Vec<_> = r_register[..(REGISTER_COUNT as usize).log_2()].to_vec();
        let eq_r_register = EqPolynomial::evals(&r_register);
        debug_assert_eq!(eq_r_register.len(), REGISTER_COUNT as usize);
        sm.get_bytecode()
            .par_iter()
            .map(|instruction| {
                let instr = instruction.normalize();
                let flags = instruction.circuit_flags();
                let unexpanded_pc = instr.address;

                let mut linear_combination: F = F::zero();

                linear_combination += eq_r_register[instr.operands.rd as usize];
                linear_combination += gamma_powers[1].mul_u64(unexpanded_pc as u64);
                if flags[CircuitFlags::IsNoop] {
                    linear_combination += gamma_powers[2];
                }
                if !flags.is_interleaved_operands() {
                    linear_combination += gamma_powers[3];
                }

                if let Some(table) = instruction.lookup_table() {
                    let table_index = LookupTables::enum_index(&table);
                    linear_combination += gamma_powers[4 + table_index];
                }

                linear_combination
            })
            .collect()
    }

    fn compute_rv_claim_3(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        gamma_powers: &[F],
    ) -> F {
        let (_, rd_wa_claim) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersValEvaluation,
        );
        let (_, unexpanded_pc_claim) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanShift,
        );
        let (_, is_noop_claim) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::IsNoop),
            SumcheckId::SpartanShift,
        );
        let (_, raf_flag_claim) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionRafFlag,
            SumcheckId::InstructionReadRaf,
        );
        std::iter::empty()
            .chain(once(rd_wa_claim))
            .chain(once(unexpanded_pc_claim))
            .chain(once(is_noop_claim))
            .chain(once(raf_flag_claim))
            .chain((0..LookupTables::<XLEN>::COUNT).map(|i| {
                sm.get_virtual_polynomial_opening(
                    VirtualPolynomial::LookupTableFlag(i),
                    SumcheckId::InstructionReadRaf,
                )
                .1
            }))
            .zip(gamma_powers)
            .map(|(claim, gamma)| claim * gamma)
            .sum()
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstance<F, T> for ReadRafSumcheck<F> {
    fn degree(&self) -> usize {
        self.d + 1
    }

    fn num_rounds(&self) -> usize {
        self.log_K + self.log_T
    }

    fn input_claim(&self) -> F {
        self.rv_claim
    }

    #[tracing::instrument(skip_all, name = "BytecodeReadRafSumcheck::compute_prover_message")]
    fn compute_prover_message(&mut self, round: usize, _previous_claim: F) -> Vec<F> {
        let ps = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        if round < self.log_K {
            const DEGREE: usize = 2;

            (0..self.val_polys[0].len() / 2)
                .into_par_iter()
                .map(|i| {
                    let ra_evals = ps.F.iter().map(|poly| {
                        poly.sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow)
                    });
                    let int_evals =
                        self.int_poly
                            .sumcheck_evals(i, DEGREE, BindingOrder::HighToLow);
                    // We have a separate Val polynomial for each stage
                    // Additionally, for stages 1 and 3 we have an Int polynomial for RAF
                    // So we would have:
                    // Stage 1: gamma^0 * (Val_1 + gamma^3 * Int)
                    // Stage 2: gamma^1 * (Val_2)
                    // Stage 3: gamma^2 * (Val_3 + gamma^2 * Int)
                    // Which matches with the input claim:
                    // rv_1 + gamma * rv_2 + gamma^2 * rv_3 + gamma^3 * raf_1 + gamma^4 * raf_3
                    let val_evals = self
                        .val_polys
                        .iter()
                        // Val polynomials
                        .map(|val| val.sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow))
                        // Here are the RAF polynomials and their powers
                        .zip([Some(&int_evals), None, Some(&int_evals)])
                        .zip([Some(self.gamma_cub), None, Some(self.gamma_sqr)])
                        .map(|((val_evals, int_evals), gamma)| {
                            std::array::from_fn::<F, DEGREE, _>(|j| {
                                val_evals[j]
                                    + int_evals.map_or(F::zero(), |int_evals| {
                                        int_evals[j] * gamma.unwrap()
                                    })
                            })
                        });

                    // Compute ra * val * gamma, and sum together
                    ra_evals
                        .zip(val_evals)
                        .zip(self.gamma.iter())
                        .map(|((ra_evals, val_evals), gamma)| {
                            std::array::from_fn::<F::Unreduced<9>, DEGREE, _>(|j| {
                                let val_gamma = val_evals[j] * gamma;
                                ra_evals[j].mul_unreduced::<9>(val_gamma)
                            })
                        })
                        .fold([F::Unreduced::zero(); DEGREE], |mut running, new| {
                            for i in 0..DEGREE {
                                running[i] += new[i];
                            }
                            running
                        })
                })
                .reduce(
                    || [F::Unreduced::zero(); DEGREE],
                    |mut running, new| {
                        for i in 0..DEGREE {
                            running[i] += new[i];
                        }
                        running
                    },
                )
                .into_iter()
                .map(F::from_montgomery_reduce)
                .collect()
        } else {
            let degree = <Self as SumcheckInstance<F, T>>::degree(self);
            (0..ps.ra[0].len() / 2)
                .into_par_iter()
                .map(|i| {
                    let eq_evals = ps
                        .eq_polys
                        .iter()
                        .map(|eq| eq.sumcheck_evals(i, degree, BindingOrder::LowToHigh));
                    let ra_evals = ps
                        .ra
                        .iter()
                        .map(|ra| ra.sumcheck_evals(i, degree, BindingOrder::LowToHigh));
                    let eq_times_val = eq_evals
                        .zip(ps.val_gamma.as_ref().unwrap().iter())
                        .map(|(eq_evals, val_evals)| {
                            eq_evals
                                .into_iter()
                                .map(|eq_eval| eq_eval * val_evals)
                                .collect()
                        })
                        .fold(
                            vec![F::zero(); degree],
                            |mut running: Vec<F>, new: Vec<F>| {
                                for i in 0..degree {
                                    running[i] += new[i];
                                }
                                running
                            },
                        );

                    let ra_evals = ra_evals.fold(vec![F::one(); degree], |mut running, new| {
                        for i in 0..degree {
                            running[i] *= new[i];
                        }
                        running
                    });

                    ra_evals
                        .into_iter()
                        .zip(eq_times_val)
                        .map(|(ra, eq)| ra.mul_unreduced::<9>(eq))
                        .collect::<Vec<_>>()
                })
                .reduce(
                    || vec![F::Unreduced::zero(); degree],
                    |mut running, new| {
                        for i in 0..degree {
                            running[i] += new[i];
                        }
                        running
                    },
                )
                .into_iter()
                .map(F::from_montgomery_reduce)
                .collect()
        }
    }

    #[tracing::instrument(skip_all, name = "BytecodeReadRafSumcheck::bind")]
    fn bind(&mut self, r_j: F::Challenge, round: usize) {
        let ps = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");

        if round < self.log_K {
            rayon::scope(|s| {
                s.spawn(|_| {
                    self.val_polys
                        .par_iter_mut()
                        .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow))
                });
                s.spawn(|_| {
                    self.int_poly.bind_parallel(r_j, BindingOrder::HighToLow);
                });
                s.spawn(|_| {
                    ps.F.par_iter_mut()
                        .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow));
                });
                s.spawn(|_| {
                    ps.v[round / self.log_K_chunk].update(r_j);
                });
            });
            if round == self.log_K - 1 {
                self.init_log_t_rounds();
            }
        } else {
            ps.ra
                .par_iter_mut()
                .for_each(|ra| ra.bind_parallel(r_j, BindingOrder::LowToHigh));
            ps.eq_polys
                .par_iter_mut()
                .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::LowToHigh));
        }
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F::Challenge],
    ) -> F {
        let accumulator = accumulator.as_ref().unwrap();
        let (r_address_prime, r_cycle_prime) = r.split_at(self.log_K);
        // r_cycle is bound LowToHigh, so reverse
        let r_cycle_prime = r_cycle_prime
            .iter()
            .rev()
            .copied()
            .collect::<Vec<F::Challenge>>();

        let int_poly = self.int_poly.evaluate(r_address_prime);

        let ra_claims = (0..self.d).map(|i| {
            accumulator
                .borrow()
                .get_committed_polynomial_opening(
                    CommittedPolynomial::BytecodeRa(i),
                    SumcheckId::BytecodeReadRaf,
                )
                .1
        });
        let r_cycles = Self::get_r_cycle_verif(accumulator);

        // We have a separate Val polynomial for each stage
        // Additionally, for stages 1 and 3 we have an Int polynomial for RAF
        // So we would have:
        // Stage 1: gamma^0 * (Val_1 + gamma^3 * Int)
        // Stage 2: gamma^1 * (Val_2)
        // Stage 3: gamma^2 * (Val_3 + gamma^2 * Int)
        // Which matches with the input claim:
        // rv_1 + gamma * rv_2 + gamma^2 * rv_3 + gamma^3 * raf_1 + gamma^4 * raf_3
        let val = self
            .val_polys
            .iter()
            .zip(r_cycles.iter())
            .zip(self.gamma.iter())
            .zip([
                int_poly * self.gamma_cub, // RAF for Stage1
                F::zero(),                 // There's no raf for Stage2
                int_poly * self.gamma_sqr, // RAF for Stage3
            ])
            .map(|(((val, r_cycle), gamma), int_poly)| {
                (val.evaluate(r_address_prime) + int_poly)
                    * EqPolynomial::<F>::mle(r_cycle, &r_cycle_prime)
                    * gamma
            })
            .sum::<F>();

        ra_claims.fold(val, |running, ra_claim| running * ra_claim)
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        let mut r = opening_point.to_vec();
        r[self.log_K..].reverse();
        OpeningPoint::new(r)
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let ps = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        let (r_address, r_cycle) = opening_point.clone().split_at(self.log_K);

        for i in 0..self.d {
            let r_address = &r_address.r[self.log_K_chunk * i..self.log_K_chunk * (i + 1)];
            accumulator.borrow_mut().append_sparse(
                transcript,
                vec![CommittedPolynomial::BytecodeRa(i)],
                SumcheckId::BytecodeReadRaf,
                r_address.to_vec(),
                r_cycle.clone().into(),
                vec![ps.ra[i].final_sumcheck_claim()],
            );
        }
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let (r_address, r_cycle) = opening_point.split_at(self.log_K);
        (0..self.d).for_each(|i| {
            let r_address = &r_address.r[self.log_K_chunk * i..self.log_K_chunk * (i + 1)];
            accumulator.borrow_mut().append_sparse(
                transcript,
                vec![CommittedPolynomial::BytecodeRa(i)],
                SumcheckId::BytecodeReadRaf,
                [r_address, &r_cycle.r].concat(),
            );
        });
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

impl<F: JoltField> ReadRafSumcheck<F> {
    fn init_log_t_rounds(&mut self) {
        let ps = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");
        let int_poly = self.int_poly.final_sumcheck_claim();

        // We have a separate Val polynomial for each stage
        // Additionally, for stages 1 and 3 we have an Int polynomial for RAF
        // So we would have:
        // Stage 1: gamma^0 * (Val_1 + gamma^3 * Int)
        // Stage 2: gamma^1 * (Val_2)
        // Stage 3: gamma^2 * (Val_3 + gamma^2 * Int)
        // Which matches with the input claim:
        // rv_1 + gamma * rv_2 + gamma^2 * rv_3 + gamma^3 * raf_1 + gamma^4 * raf_3
        ps.val_gamma = Some(
            self.val_polys
                .iter()
                .zip(self.gamma.iter())
                .zip([
                    int_poly * self.gamma_cub,
                    F::zero(),
                    int_poly * self.gamma_sqr,
                ])
                .map(|((poly, gamma), int_poly)| (poly.final_sumcheck_claim() + int_poly) * gamma)
                .collect::<Vec<F>>()
                .try_into()
                .unwrap(),
        );

        ps.v.par_iter()
            .enumerate()
            .map(|(i, v)| {
                let ra_i: Vec<Option<u8>> = ps
                    .pc
                    .par_iter()
                    .map(|k| {
                        let k = (k >> (self.log_K_chunk * (self.d - i - 1))) % self.K_chunk;
                        Some(k as u8)
                    })
                    .collect();
                RaPolynomial::new(Arc::new(ra_i), v.clone_values())
            })
            .collect::<Vec<_>>()
            .into_iter()
            .for_each(|ra| {
                ps.ra.push(ra);
            });
    }
}

fn get_gamma_powers<F: JoltField>(transcript: &mut impl Transcript, amount: usize) -> Vec<F> {
    let mut gamma_powers = vec![F::one()];
    let gamma: F = transcript.challenge_scalar();
    for _ in 1..amount {
        gamma_powers.push(gamma * gamma_powers.last().unwrap());
    }
    gamma_powers
}
