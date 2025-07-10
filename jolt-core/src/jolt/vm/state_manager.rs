use std::collections::HashMap;
use std::ops::Index;
use std::sync::{Arc, Mutex};

use tracer::instruction::RV32IMCycle;

use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator};
use crate::r1cs::inputs::JoltR1CSInputs;
use crate::utils::transcript::Transcript;

pub type Openings<F> = HashMap<OpeningsKeys, (Vec<F>, F)>;

impl<F> Index<JoltR1CSInputs> for Openings<F> {
    type Output = F;

    fn index(&self, index: JoltR1CSInputs) -> &Self::Output {
        &self[&OpeningsKeys::SpartanZ(index)].1
    }
}

pub struct StateManager<'a, F: JoltField, PCS: CommitmentScheme<T, Field = F>, T: Transcript> {
    pub transcript: &'a mut T,
    pub prover_accumulator: Option<Arc<Mutex<&'a mut ProverOpeningAccumulator<F, PCS, T>>>>,
    pub verifier_accumulator: Option<Arc<Mutex<&'a mut VerifierOpeningAccumulator<F, PCS, T>>>>,
    pub openings: Arc<Mutex<Openings<F>>>,
    pub r_address: Vec<F>,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug)]
pub enum OpeningsKeys {
    SpartanZ(JoltR1CSInputs),
    InstructionTypeFlag(usize),
    InstructionRa(usize),
    // the claim of AddOperand + MultiplyOperands + SubtractOperands + Advice at r_cycle_prime
    InstructionRafFlag,
}

impl<'a, F: JoltField, PCS: CommitmentScheme<T, Field = F>, T: Transcript>
    StateManager<'a, F, PCS, T>
{
    pub fn prove() {
        todo!()
    }

    pub fn verify() {
        todo!()
    }

    pub fn z(&self, idx: JoltR1CSInputs) -> F {
        self.openings(OpeningsKeys::SpartanZ(idx))
    }

    pub fn r_cycle(&self) -> Vec<F> {
        self.openings_point(OpeningsKeys::SpartanZ(JoltR1CSInputs::Imm))
    }

    pub fn r_address(&self) -> Vec<F> {
        self.r_address.clone()
    }

    pub fn openings(&self, idx: OpeningsKeys) -> F {
        self.openings
            .lock()
            .unwrap()
            .get(&idx)
            .unwrap_or_else(|| panic!("No openings for {idx:?}"))
            .1
    }

    pub fn openings_point(&self, idx: OpeningsKeys) -> Vec<F> {
        self.openings
            .lock()
            .unwrap()
            .get(&idx)
            .unwrap_or_else(|| panic!("No openings for {idx:?}"))
            .0
            .clone()
    }

    pub fn temp_populate_openings(&mut self, trace: &[RV32IMCycle], r_cycle: Vec<F>) {
        use crate::jolt::instruction::LookupQuery;
        use crate::jolt::vm::instruction_lookups::WORD_SIZE;
        use crate::poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation};
        use rayon::prelude::*;
        self.openings.lock().unwrap().insert(
            OpeningsKeys::SpartanZ(JoltR1CSInputs::Imm),
            (r_cycle.clone(), F::zero()),
        );
        // TODO: Get this from spartan, right now calculated here
        let (left_operand_evals, right_operand_evals): (Vec<u64>, Vec<u64>) = trace
            .par_iter()
            .map(LookupQuery::<WORD_SIZE>::to_lookup_operands)
            .collect();
        let right_operand_claim =
            MultilinearPolynomial::from(right_operand_evals).evaluate(&r_cycle);
        let left_operand_claim = MultilinearPolynomial::from(left_operand_evals).evaluate(&r_cycle);
        let lookup_output_evals: Vec<u64> = trace
            .par_iter()
            .map(LookupQuery::<WORD_SIZE>::to_lookup_output)
            .collect();
        let lookup_output_claim =
            MultilinearPolynomial::from(lookup_output_evals).evaluate(&r_cycle);
        self.openings.lock().unwrap().insert(
            OpeningsKeys::SpartanZ(JoltR1CSInputs::RightLookupOperand),
            (r_cycle.clone(), right_operand_claim),
        );
        self.openings.lock().unwrap().insert(
            OpeningsKeys::SpartanZ(JoltR1CSInputs::LeftLookupOperand),
            (r_cycle.clone(), left_operand_claim),
        );
        self.openings.lock().unwrap().insert(
            OpeningsKeys::SpartanZ(JoltR1CSInputs::LookupOutput),
            (r_cycle.clone(), lookup_output_claim),
        );
    }
}
