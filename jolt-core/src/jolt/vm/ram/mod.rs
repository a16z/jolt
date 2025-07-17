#[cfg(feature = "prover")]
mod prover;
#[cfg(feature = "prover")]
pub use prover::*;
mod verifier;

use crate::field::JoltField;
use crate::jolt::vm::output_check::OutputProof;
use crate::jolt::vm::ram_read_write_checking::RamReadWriteCheckingProof;
use crate::poly::identity_poly::UnmapRamAddressPolynomial;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::subprotocols::ra_virtual::RAProof;
use crate::subprotocols::sumcheck::SumcheckInstanceProof;
use crate::utils::transcript::Transcript;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::constants::BYTES_PER_INSTRUCTION;
use common::jolt_device::MemoryLayout;
use tracer::instruction::RV32IMCycle;

#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct RAMPreprocessing {
    min_bytecode_address: u64,
    bytecode_words: Vec<u32>,
}

impl RAMPreprocessing {
    pub fn preprocess(memory_init: Vec<(u64, u8)>) -> Self {
        let min_bytecode_address = memory_init
            .iter()
            .map(|(address, _)| *address)
            .min()
            .unwrap_or(0);

        let max_bytecode_address = memory_init
            .iter()
            .map(|(address, _)| *address)
            .max()
            .unwrap_or(0)
            + (BYTES_PER_INSTRUCTION as u64 - 1); // For RV32IM, instructions occupy 4 bytes, so the max bytecode address is the max instruction address + 3

        let num_words = max_bytecode_address.next_multiple_of(4) / 4 - min_bytecode_address / 4 + 1;
        let mut bytecode_words = vec![0u32; num_words as usize];
        // Convert bytes into words and populate `bytecode_words`
        for chunk in
            memory_init.chunk_by(|(address_a, _), (address_b, _)| address_a / 4 == address_b / 4)
        {
            let mut word = [0u8; 4];
            for (address, byte) in chunk {
                word[(address % 4) as usize] = *byte;
            }
            let word = u32::from_le_bytes(word);
            let remapped_index = (chunk[0].0 / 4 - min_bytecode_address / 4) as usize;
            bytecode_words[remapped_index] = word;
        }

        Self {
            min_bytecode_address,
            bytecode_words,
        }
    }
}

#[derive(Clone)]
struct ValEvaluationSumcheckClaims<F: JoltField> {
    /// Inc(r_cycle')
    inc_claim: F,
    /// wa(r_address, r_cycle')
    wa_claim: F,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct RAMTwistProof<F: JoltField, ProofTranscript: Transcript> {
    pub(crate) K: usize,
    /// Proof for the read-checking and write-checking sumchecks
    /// (steps 3 and 4 of Figure 9).
    read_write_checking_proof: RamReadWriteCheckingProof<F, ProofTranscript>,
    /// Proof of the Val-evaluation sumcheck (step 6 of Figure 9).
    val_evaluation_proof: ValEvaluationProof<F, ProofTranscript>,

    booleanity_proof: BooleanityProof<F, ProofTranscript>,
    ra_proof: RAProof<F, ProofTranscript>,
    hamming_weight_proof: HammingWeightProof<F, ProofTranscript>,
    raf_evaluation_proof: RafEvaluationProof<F, ProofTranscript>,
    output_proof: OutputProof<F, ProofTranscript>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct ReadWriteCheckingProof<F: JoltField, ProofTranscript: Transcript> {
    /// Joint sumcheck proof for the read-checking and write-checking sumchecks
    /// (steps 3 and 4 of Figure 9).
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    /// The claimed evaluation ra(r_address, r_cycle) output by the read/write-
    /// checking sumcheck.
    ra_claim: F,
    /// The claimed evaluation rv(r') proven by the read-checking sumcheck.
    rv_claim: F,
    /// The claimed evaluation wv(r_address, r_cycle) output by the read/write-
    /// checking sumcheck.
    wv_claim: F,
    /// The claimed evaluation val(r_address, r_cycle) output by the read/write-
    /// checking sumcheck.
    val_claim: F,
    /// The claimed evaluation Inc(r, r') proven by the write-checking sumcheck.
    inc_claim: F,
    /// The sumcheck round index at which we switch from binding cycle variables
    /// to binding address variables.
    sumcheck_switch_index: usize,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct ValEvaluationProof<F: JoltField, ProofTranscript: Transcript> {
    /// Sumcheck proof for the Val-evaluation sumcheck (steps 6 of Figure 9).
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    /// Inc(r_cycle')
    inc_claim: F,
    /// wa(r_address, r_cycle')
    wa_claim: F,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct BooleanityProof<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    ra_claims: Vec<F>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct HammingWeightProof<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    ra_claims: Vec<F>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct RafEvaluationProof<F: JoltField, ProofTranscript: Transcript> {
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    ra_claim: F,
    raf_claim: F,
}

struct RafEvaluationProverState<F: JoltField> {
    /// The ra polynomial
    ra: MultilinearPolynomial<F>,
    /// The unmap polynomial
    unmap: UnmapRamAddressPolynomial<F>,
}

struct RafEvaluationVerifierState {
    /// log K (number of rounds)
    log_K: usize,
    /// Start address for unmap polynomial
    start_address: u64,
}

struct RafEvaluationSumcheck<F: JoltField> {
    /// The initial claim (raf_claim)
    input_claim: F,
    /// Prover state (only present for prover)
    prover_state: Option<RafEvaluationProverState<F>>,
    /// Verifier state (only present for verifier)
    verifier_state: Option<RafEvaluationVerifierState>,
    /// Cached ra_claim after sumcheck completion
    cached_claim: Option<F>,
}

pub(crate) fn remap_address(address: u64, memory_layout: &MemoryLayout) -> u64 {
    if address == 0 {
        return 0; // [JOLT-135]: Better handling for no-ops
    }
    if address >= memory_layout.input_start {
        (address - memory_layout.input_start) / 4 + 1
    } else {
        panic!("Unexpected address {address}")
    }
}

struct ValEvaluationProverState<F: JoltField> {
    /// Inc polynomial
    inc: MultilinearPolynomial<F>,
    /// wa polynomial
    wa: MultilinearPolynomial<F>,
    /// LT polynomial
    lt: MultilinearPolynomial<F>,
}

struct ValEvaluationVerifierState<F: JoltField> {
    /// log T
    num_rounds: usize,
    /// used to compute LT evaluation
    r_address: Vec<F>,
    /// used to compute LT evaluation
    r_cycle: Vec<F>,
}

/// Val-evaluation sumcheck for RAM
struct ValEvaluationSumcheck<F: JoltField> {
    /// Initial claim value
    claimed_evaluation: F,
    /// Initial evaluation to subtract (for RAM)
    init_eval: F,
    /// Prover state
    prover_state: Option<ValEvaluationProverState<F>>,
    /// Verifier state
    verifier_state: Option<ValEvaluationVerifierState<F>>,
    /// Claims
    claims: Option<ValEvaluationSumcheckClaims<F>>,
}

struct BooleanityProverState<F: JoltField> {
    /// B polynomial (EqPolynomial)
    B: MultilinearPolynomial<F>,
    /// F array for phase 1
    F: Vec<F>,
    /// G arrays (precomputed) - one for each decomposed part
    G: Vec<Vec<F>>,
    /// D polynomial for phase 2
    D: MultilinearPolynomial<F>,
    /// H polynomials for phase 2 - one for each decomposed part
    H: Option<Vec<MultilinearPolynomial<F>>>,
    /// eq(r, r) value computed at end of phase 1
    eq_r_r: F,
    /// z powers
    z_powers: Vec<F>,
    /// D parameter as in Twist and Shout paper
    d: usize,
    /// Chunk sizes for variable-sized d-way decomposition
    chunk_sizes: Vec<usize>,
}

struct BooleanityVerifierState<F: JoltField> {
    /// Size of address space
    K: usize,
    /// Number of cycles
    T: usize,
    /// D parameter as in Twist and Shout paper
    d: usize,
    /// r_address challenge
    r_address: Vec<F>,
    /// r_prime (r_cycle) challenge
    r_prime: Vec<F>,
    /// z powers
    z_powers: Vec<F>,
}

struct BooleanitySumcheck<F: JoltField> {
    /// Size of address space
    K: usize,
    /// Number of trace steps
    T: usize,
    /// D parameter as in Twist and Shout paper
    d: usize,
    /// Prover state (if prover)
    prover_state: Option<BooleanityProverState<F>>,
    /// Verifier state (if verifier)
    verifier_state: Option<BooleanityVerifierState<F>>,
    /// Cached ra claims
    ra_claims: Option<Vec<F>>,
    /// Current round
    current_round: usize,
    /// Store trace and memory layout for phase transition
    trace: Option<Vec<RV32IMCycle>>,
    memory_layout: Option<MemoryLayout>,
}

struct HammingWeightProverState<F: JoltField> {
    /// The ra polynomials - one for each decomposed part
    ra: Vec<MultilinearPolynomial<F>>,
    /// z powers for batching
    z_powers: Vec<F>,
    /// D parameter as in Twist and Shout paper
    d: usize,
}

struct HammingWeightVerifierState<F: JoltField> {
    /// log K (number of rounds)
    log_K: usize,
    /// D parameter as in Twist and Shout paper
    d: usize,
    /// z powers for verification
    z_powers: Vec<F>,
}

struct HammingWeightSumcheck<F: JoltField> {
    /// The initial claim (sum of z powers for hamming weight)
    input_claim: F,
    /// Prover state
    prover_state: Option<HammingWeightProverState<F>>,
    /// Verifier state
    verifier_state: Option<HammingWeightVerifierState<F>>,
    /// Cached claims for all d polynomials
    cached_claims: Option<Vec<F>>,
    /// D parameter
    d: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::math::Math;
    use crate::utils::transcript::KeccakTranscript;
    use ark_bn254::Fr;

    #[test]
    fn test_raf_evaluation_no_ops() {
        const K: usize = 1 << 16;
        const T: usize = 1 << 8;

        let memory_layout = MemoryLayout {
            max_input_size: 256,
            max_output_size: 256,
            input_start: 0x80000000,
            input_end: 0x80000100,
            output_start: 0x80001000,
            output_end: 0x80001100,
            stack_size: 1024,
            stack_end: 0x7FFFFF00,
            memory_size: 0x10000,
            memory_end: 0x80010000,
            panic: 0x80002000,
            termination: 0x80002001,
            io_end: 0x80002002,
        };

        // Create trace with only no-ops (address = 0)
        let mut trace = Vec::new();
        for i in 0..T {
            trace.push(RV32IMCycle::NoOp(i));
        }

        let mut prover_transcript = KeccakTranscript::new(b"test_no_ops");
        let r_cycle: Vec<Fr> = prover_transcript.challenge_vector(T.log_2());

        // Prove
        let proof =
            RafEvaluationProof::prove(&trace, &memory_layout, r_cycle, K, &mut prover_transcript);

        // Verify
        let mut verifier_transcript = KeccakTranscript::new(b"test_no_ops");
        let _r_cycle: Vec<Fr> = verifier_transcript.challenge_vector(T.log_2());

        let r_address_result = proof.verify(K, &mut verifier_transcript, &memory_layout);

        assert!(
            r_address_result.is_ok(),
            "No-op RAF evaluation verification failed"
        );
    }
}
