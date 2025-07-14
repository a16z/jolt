#[cfg(feature = "prover")]
mod prover;
#[cfg(feature = "prover")]
pub use prover::*;

use crate::{
    field::JoltField,
    optimal_iter,
    poly::{
        compact_polynomial::SmallScalar,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    utils::{errors::ProofVerifyError, math::Math, transcript::Transcript},
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::constants::{BYTES_PER_INSTRUCTION, RAM_START_ADDRESS};
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use std::collections::BTreeMap;
#[cfg(not(feature = "parallel"))]
use std::iter::once;
use tracer::instruction::{NormalizedInstruction, RV32IMCycle, RV32IMInstruction};

#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct BytecodePreprocessing {
    bytecode: Vec<RV32IMInstruction>,
    /// Maps the memory address of each instruction in the bytecode to its "virtual" address.
    /// See Section 6.1 of the Jolt paper, "Reflecting the program counter". The virtual address
    /// is the one used to keep track of the next (potentially virtual) instruction to execute.
    /// Key: (ELF address, virtual sequence index or 0)
    virtual_address_map: BTreeMap<(usize, usize), usize>,
}

impl BytecodePreprocessing {
    #[tracing::instrument(skip_all, name = "BytecodePreprocessing::preprocess")]
    pub fn preprocess(mut bytecode: Vec<RV32IMInstruction>) -> Self {
        let mut virtual_address_map = BTreeMap::new();
        let mut virtual_address = 1; // Account for no-op instruction prepended to bytecode
        for instruction in bytecode.iter() {
            let instr = instruction.normalize();
            debug_assert!(instr.address >= RAM_START_ADDRESS as usize);
            debug_assert!(instr.address % BYTES_PER_INSTRUCTION == 0);
            assert_eq!(
                virtual_address_map.insert(
                    (instr.address, instr.virtual_sequence_remaining.unwrap_or(0)),
                    virtual_address
                ),
                None
            );
            virtual_address += 1;
        }

        // Bytecode: Prepend a single no-op instruction
        bytecode.insert(0, RV32IMInstruction::NoOp(0));
        assert_eq!(virtual_address_map.insert((0, 0), 0), None);

        // Bytecode: Pad to nearest power of 2
        // Get last address
        let last_address = bytecode.last().unwrap().normalize().address;
        let code_size = bytecode.len().next_power_of_two();
        let padding = code_size - bytecode.len();
        bytecode.extend((0..padding).map(|i| RV32IMInstruction::NoOp(last_address + 4 * (i + 1))));

        Self {
            bytecode,
            virtual_address_map,
        }
    }

    pub fn get_pc(&self, cycle: &RV32IMCycle, is_last: bool) -> usize {
        let instr = cycle.instruction().normalize();
        if matches!(cycle, tracer::instruction::RV32IMCycle::NoOp(_)) || is_last {
            return 0;
        }
        *self
            .virtual_address_map
            .get(&(instr.address, instr.virtual_sequence_remaining.unwrap_or(0)))
            .unwrap()
    }

    #[cfg(feature = "parallel")]
    pub fn map_trace_to_pc<'a, 'b>(
        &'b self,
        trace: &'a [RV32IMCycle],
    ) -> impl ParallelIterator<Item = u64> + use<'a, 'b> {
        let (_, init) = trace.split_last().unwrap();
        init.par_iter()
            .map(|cycle| self.get_pc(cycle, false) as u64)
            .chain(rayon::iter::once(0))
    }

    #[cfg(not(feature = "parallel"))]
    pub fn map_trace_to_pc<'a, 'b>(
        &'b self,
        trace: &'a [RV32IMCycle],
    ) -> impl Iterator<Item = u64> + use<'a, 'b> {
        let (_, init) = trace.split_last().unwrap();
        init.iter()
            .map(|cycle| self.get_pc(cycle, false) as u64)
            .chain(once(0))
    }
}

#[tracing::instrument(skip_all)]
fn bytecode_to_val<F: JoltField>(bytecode: &[RV32IMInstruction], gamma: F) -> Vec<F> {
    let mut gamma_powers = vec![F::one()];
    for _ in 0..5 {
        gamma_powers.push(gamma * gamma_powers.last().unwrap());
    }

    optimal_iter!(bytecode)
        .map(|instruction| {
            let NormalizedInstruction {
                address,
                operands,
                virtual_sequence_remaining: _,
            } = instruction.normalize();
            let mut linear_combination = F::zero();
            linear_combination += (address as u64).field_mul(gamma_powers[0]);
            linear_combination += (operands.rd as u64).field_mul(gamma_powers[1]);
            linear_combination += (operands.rs1 as u64).field_mul(gamma_powers[2]);
            linear_combination += (operands.rs2 as u64).field_mul(gamma_powers[3]);
            linear_combination += operands.imm.field_mul(gamma_powers[4]);
            // TODO(moodlezoup): Circuit and lookup flags
            linear_combination
        })
        .collect()
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct BytecodeShoutProof<F: JoltField, ProofTranscript: Transcript> {
    core_piop_sumcheck: SumcheckInstanceProof<F, ProofTranscript>,
    booleanity_sumcheck: SumcheckInstanceProof<F, ProofTranscript>,
    ra_claim: F,
    ra_claim_prime: F,
    rv_claim: F,
}

impl<F: JoltField, ProofTranscript: Transcript> BytecodeShoutProof<F, ProofTranscript> {
    pub fn verify(
        &self,
        preprocessing: &BytecodePreprocessing,
        T: usize,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        let K = preprocessing.bytecode.len();
        let r_cycle: Vec<F> = transcript.challenge_vector(T.log_2());
        let z: F = transcript.challenge_scalar();
        let gamma: F = transcript.challenge_scalar();

        let (sumcheck_claim, mut r_address) =
            self.core_piop_sumcheck
                .verify(self.rv_claim + z, K.log_2(), 2, transcript)?;
        r_address = r_address.into_iter().rev().collect();

        // Used to combine the various fields in each instruction into a single
        // field element.
        let val: Vec<F> = bytecode_to_val(&preprocessing.bytecode, gamma);
        let val = MultilinearPolynomial::from(val);

        assert_eq!(
            self.ra_claim * (z + val.evaluate(&r_address)),
            sumcheck_claim,
            "Core PIOP + Hamming weight sumcheck failed"
        );

        let (sumcheck_claim, r_booleanity) =
            self.booleanity_sumcheck
                .verify(F::zero(), K.log_2() + T.log_2(), 3, transcript)?;
        let (r_address_prime, r_cycle_prime) = r_booleanity.split_at(K.log_2());
        let eq_eval_address = EqPolynomial::new(r_address).evaluate(r_address_prime);
        let r_cycle: Vec<_> = r_cycle.iter().copied().rev().collect();
        let eq_eval_cycle = EqPolynomial::new(r_cycle).evaluate(r_cycle_prime);

        assert_eq!(
            eq_eval_address * eq_eval_cycle * (self.ra_claim_prime.square() - self.ra_claim_prime),
            sumcheck_claim,
            "Booleanity sumcheck failed"
        );

        // TODO: Reduce 2 ra claims to 1 (Section 4.5.2 of Proofs, Arguments, and Zero-Knowledge)
        // TODO: Append to opening proof accumulator

        Ok(())
    }
}
