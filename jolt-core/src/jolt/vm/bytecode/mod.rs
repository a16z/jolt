#[cfg(feature = "prover")]
mod prover;

use crate::jolt::vm::JoltCommitments;
use crate::jolt::witness::CommittedPolynomials;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::identity_poly::IdentityPolynomial;
use crate::poly::multilinear_polynomial::PolynomialEvaluation;
use crate::poly::opening_proof::VerifierOpeningAccumulator;
use crate::subprotocols::sumcheck::BatchableSumcheckVerifierInstance;
use crate::{
    field::JoltField,
    poly::{compact_polynomial::SmallScalar, multilinear_polynomial::MultilinearPolynomial},
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
use std::sync::Arc;
use tracer::instruction::{NormalizedInstruction, RV32IMCycle, RV32IMInstruction};

#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct BytecodePreprocessing {
    pub code_size: usize,
    bytecode: Vec<RV32IMInstruction>,
    /// Maps the memory address of each instruction in the bytecode to its "virtual" address.
    /// See Section 6.1 of the Jolt paper, "Reflecting the program counter". The virtual address
    /// is the one used to keep track of the next (potentially virtual) instruction to execute.
    /// Key: (ELF address, virtual sequence index or 0)
    pub virtual_address_map: BTreeMap<(usize, usize), usize>,
}
impl BytecodePreprocessing {
    #[tracing::instrument(skip_all, name = "BytecodePreprocessing::preprocess")]
    pub fn preprocess(mut bytecode: Vec<RV32IMInstruction>) -> Self {
        let mut virtual_address_map = BTreeMap::new();
        let mut virtual_address = 1; // Account for no-op instruction prepended to bytecode
        for instruction in bytecode.iter() {
            if instruction.normalize().address == 0 {
                // ignore unimplemented instructions
                continue;
            }
            let instr = instruction.normalize();
            debug_assert!(instr.address >= RAM_START_ADDRESS as usize);
            debug_assert!(instr.address.is_multiple_of(BYTES_PER_INSTRUCTION));
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
            code_size,
            bytecode,
            virtual_address_map,
        }
    }

    pub fn get_pc(&self, cycle: &RV32IMCycle, is_last: bool) -> usize {
        let instr = cycle.instruction().normalize();
        if matches!(cycle, RV32IMCycle::NoOp(_)) || is_last {
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

    bytecode
        .par_iter()
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
    core_piop_hamming: CorePIOPHammingProof<F, ProofTranscript>,
    booleanity: BooleanityProof<F, ProofTranscript>,
    raf_sumcheck: RafEvaluationProof<F, ProofTranscript>,
}

impl<F: JoltField, ProofTranscript: Transcript> BytecodeShoutProof<F, ProofTranscript> {
    pub fn verify<PCS: CommitmentScheme<ProofTranscript, Field = F>>(
        &self,
        preprocessing: &BytecodePreprocessing,
        commitments: &JoltCommitments<F, PCS, ProofTranscript>,
        T: usize,
        transcript: &mut ProofTranscript,
        opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS, ProofTranscript>,
    ) -> Result<(), ProofVerifyError> {
        let K = preprocessing.bytecode.len();
        // TODO: this should come from Spartan
        let r_cycle: Vec<F> = transcript.challenge_vector(T.log_2());
        let _r_shift: Vec<F> = transcript.challenge_vector(T.log_2());
        let z: F = transcript.challenge_scalar();
        let gamma: F = transcript.challenge_scalar();

        // Used to combine the various fields in each instruction into a single
        // field element.
        let val: Vec<F> = bytecode_to_val(&preprocessing.bytecode, gamma);

        // Verify core PIOP and Hamming weight sumcheck
        let r_address = self.core_piop_hamming.verify(&val, z, K, transcript)?;

        let r_address_rev: Vec<_> = r_address.iter().copied().rev().collect();
        let r_cycle_rev: Vec<_> = r_cycle.iter().copied().rev().collect();

        let r_concat = [r_address_rev.as_slice(), r_cycle.as_slice()].concat();
        let ra_commitment = &commitments.commitments[CommittedPolynomials::BytecodeRa.to_index()];
        opening_accumulator.append(
            &[ra_commitment],
            r_concat,
            &[self.core_piop_hamming.ra_claim],
            transcript,
        );

        // Verify booleanity sumcheck
        let (r_booleanity, ra_claim_prime) =
            self.booleanity
                .verify(&r_address_rev, &r_cycle_rev, K, T, transcript)?;

        let (r_address_prime, r_cycle_prime) = r_booleanity.split_at(K.log_2());
        let r_address_prime = r_address_prime.iter().copied().rev().collect::<Vec<_>>();
        let r_cycle_prime = r_cycle_prime.iter().rev().copied().collect::<Vec<_>>();
        let r_concat = [r_address_prime.as_slice(), r_cycle_prime.as_slice()].concat();

        opening_accumulator.append(&[ra_commitment], r_concat, &[ra_claim_prime], transcript);

        let challenge: F = transcript.challenge_scalar();
        let _ = self.raf_sumcheck.verify(K, challenge, transcript)?;

        Ok(())
    }
}

struct CorePIOPHammingProverState<F: JoltField> {
    ra_poly: MultilinearPolynomial<F>,
    val_poly: MultilinearPolynomial<F>,
}

pub struct CorePIOPHammingSumcheck<F: JoltField> {
    /// Input claim: rv_claim + z
    input_claim: F,
    /// z value shared by prover and verifier
    z: F,
    /// K value shared by prover and verifier
    K: usize,
    /// Prover state
    prover_state: Option<CorePIOPHammingProverState<F>>,
    /// Cached ra claim after sumcheck completes
    ra_claim: Option<F>,
    /// Cached val evaluation after sumcheck completes
    val_eval: Option<F>,
}

impl<F: JoltField> CorePIOPHammingSumcheck<F> {
    pub fn new_verifier(input_claim: F, z: F, K: usize) -> Self {
        Self {
            input_claim,
            z,
            K,
            prover_state: None,
            ra_claim: None,
            val_eval: None,
        }
    }
}

impl<F: JoltField, ProofTranscript: Transcript> BatchableSumcheckVerifierInstance<F, ProofTranscript>
for CorePIOPHammingSumcheck<F>
{
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        self.K.log_2()
    }

    fn input_claim(&self) -> F {
        self.input_claim
    }
    fn expected_output_claim(&self, _r: &[F]) -> F {
        let ra_claim = self.ra_claim.as_ref().expect("ra_claim not set");
        let val_eval = self.val_eval.as_ref().expect("val_eval not set");

        // Verify sumcheck_claim = ra_claim * (z + val_eval)
        *ra_claim * (self.z + *val_eval)
    }
}

impl<F: JoltField, ProofTranscript: Transcript> CorePIOPHammingProof<F, ProofTranscript> {
    pub fn verify(
        &self,
        _val: &[F],
        z: F,
        K: usize,
        transcript: &mut ProofTranscript,
    ) -> Result<Vec<F>, ProofVerifyError> {
        let input_claim = self.rv_claim + z;
        let mut core_piop_sumcheck = CorePIOPHammingSumcheck::new_verifier(input_claim, z, K);

        core_piop_sumcheck.ra_claim = Some(self.ra_claim);
        core_piop_sumcheck.val_eval = Some(self.val_eval);

        let r_address = core_piop_sumcheck.verify_single(&self.sumcheck_proof, transcript)?;

        Ok(r_address)
    }
}


struct BooleanityProverState<F: JoltField> {
    B: MultilinearPolynomial<F>,
    D: MultilinearPolynomial<F>,
    H: MultilinearPolynomial<F>,
    G: Vec<F>,
    F: Vec<F>,
    eq_r_r: F,
    // Precomputed arrays for phase 1
    eq_km_c: [[F; 3]; 2],
    eq_km_c_squared: [[F; 3]; 2],
}

struct BooleanityVerifierState<F: JoltField> {
    r_address: Option<Vec<F>>,
    r_cycle: Option<Vec<F>>,
}

pub struct BooleanitySumcheck<F: JoltField> {
    /// Input claim: always F::zero() for booleanity
    input_claim: F,
    /// K value shared by prover and verifier
    K: usize,
    /// T value shared by prover and verifier
    T: usize,
    /// Prover state
    prover_state: Option<BooleanityProverState<F>>,
    /// Verifier state
    verifier_state: Option<BooleanityVerifierState<F>>,
    /// Cached ra claim after sumcheck completes
    ra_claim_prime: Option<F>,
    /// Current round
    current_round: usize,
    /// Store preprocessing and trace for phase transition
    preprocessing: Option<Arc<BytecodePreprocessing>>,
    trace: Option<Arc<[RV32IMCycle]>>,
}

impl<F: JoltField> BooleanitySumcheck<F> {
    pub fn new_verifier(
        K: usize,
        T: usize,
        r_address: Vec<F>,
        r_cycle: Vec<F>,
        ra_claim_prime: F,
    ) -> Self {
        Self {
            input_claim: F::zero(),
            K,
            T,
            prover_state: None,
            verifier_state: Some(BooleanityVerifierState::<F> {
                r_address: Some(r_address),
                r_cycle: Some(r_cycle),
            }),
            ra_claim_prime: Some(ra_claim_prime),
            current_round: 0,
            preprocessing: None,
            trace: None,
        }
    }
}

impl<F: JoltField, ProofTranscript: Transcript> BatchableSumcheckVerifierInstance<F, ProofTranscript>
for BooleanitySumcheck<F>
{
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        self.K.log_2() + self.T.log_2()
    }

    fn input_claim(&self) -> F {
        self.input_claim
    }
    fn expected_output_claim(&self, r: &[F]) -> F {
        let ra_claim_prime = self.ra_claim_prime.expect("ra_claim_prime not set");
        let verifier_state = self
            .verifier_state
            .as_ref()
            .expect("Verifier state not initialized");

        // Split r into r_address_prime and r_cycle_prime
        let (r_address_prime, r_cycle_prime) = r.split_at(self.K.log_2());

        let r_address = verifier_state
            .r_address
            .as_ref()
            .expect("r_address not set");
        let r_cycle = verifier_state.r_cycle.as_ref().expect("r_cycle not set");

        let eq_eval_address = EqPolynomial::mle(r_address, r_address_prime);
        let eq_eval_cycle = EqPolynomial::mle(r_cycle, r_cycle_prime);

        eq_eval_address * eq_eval_cycle * (ra_claim_prime.square() - ra_claim_prime)
    }
}

struct RafBytecodeProverState<F: JoltField> {
    ra_poly: MultilinearPolynomial<F>,
    ra_poly_shift: MultilinearPolynomial<F>,
    int_poly: IdentityPolynomial<F>,
}

pub struct RafBytecode<F: JoltField> {
    /// Input claim: raf_claim + challenge * raf_claim_shift
    input_claim: F,
    /// Challenge value shared by prover and verifier
    challenge: F,
    /// K value shared by prover and verifier
    K: usize,
    /// Prover state
    prover_state: Option<RafBytecodeProverState<F>>,
    /// Cached ra claims after sumcheck completes
    ra_claims: Option<(F, F)>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct CorePIOPHammingProof<F: JoltField, ProofTranscript: Transcript> {
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    ra_claim: F,
    rv_claim: F,
    val_eval: F,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct BooleanityProof<F: JoltField, ProofTranscript: Transcript> {
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    ra_claim_prime: F,
}

impl<F: JoltField, ProofTranscript: Transcript> BooleanityProof<F, ProofTranscript> {
    pub fn verify(
        &self,
        r_address: &[F],
        r_cycle: &[F],
        K: usize,
        T: usize,
        transcript: &mut ProofTranscript,
    ) -> Result<(Vec<F>, F), ProofVerifyError> {
        let booleanity_sumcheck = BooleanitySumcheck::new_verifier(
            K,
            T,
            r_address.to_vec(),
            r_cycle.to_vec(),
            self.ra_claim_prime,
        );

        let r_combined = booleanity_sumcheck.verify_single(&self.sumcheck_proof, transcript)?;

        Ok((r_combined, self.ra_claim_prime))
    }
}

impl<F: JoltField> RafBytecode<F> {
    pub fn new_verifier(input_claim: F, challenge: F, K: usize) -> Self {
        Self {
            input_claim,
            challenge,
            K,
            prover_state: None,
            ra_claims: None,
        }
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct RafEvaluationProof<F: JoltField, ProofTranscript: Transcript> {
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    ra_claim: F,
    ra_claim_shift: F,
    raf_claim: F,
    raf_claim_shift: F,
}

impl<F: JoltField, ProofTranscript: Transcript> RafEvaluationProof<F, ProofTranscript> {
    pub fn verify(
        &self,
        K: usize,
        challenge: F,
        transcript: &mut ProofTranscript,
    ) -> Result<Vec<F>, ProofVerifyError> {
        let input_claim = self.raf_claim + challenge * self.raf_claim_shift;

        let mut raf_sumcheck = RafBytecode::new_verifier(input_claim, challenge, K);

        raf_sumcheck.ra_claims = Some((self.ra_claim, self.ra_claim_shift));

        let r_raf_sumcheck = raf_sumcheck.verify_single(&self.sumcheck_proof, transcript)?;

        Ok(r_raf_sumcheck)
    }
}

impl<F: JoltField, ProofTranscript: Transcript> BatchableSumcheckVerifierInstance<F, ProofTranscript>
for RafBytecode<F>
{
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        self.K.log_2()
    }

    fn input_claim(&self) -> F {
        self.input_claim
    }
    fn expected_output_claim(&self, r: &[F]) -> F {
        let (ra_claim, ra_claim_shift) = self.ra_claims.as_ref().expect("ra_claims not set");

        let int_eval = IdentityPolynomial::new(self.K.log_2()).evaluate(r);

        // Verify sumcheck_claim = int(r) * (ra_claim + challenge * ra_claim_shift)
        int_eval * (*ra_claim + self.challenge * *ra_claim_shift)
    }
}
