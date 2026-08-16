use std::time::{Duration, Instant};

use jolt_claims::protocols::jolt::PrecommittedReductionLayout;
use jolt_prover_legacy::ark_bn254::Fr as LegacyFr;
use jolt_prover_legacy::field::JoltField as LegacyJoltField;
use jolt_prover_legacy::poly::commitment::dory::DoryGlobals;
use jolt_prover_legacy::poly::eq_poly::EqPolynomial;
use jolt_prover_legacy::poly::multilinear_polynomial::MultilinearPolynomial as LegacyPoly;
use jolt_prover_legacy::poly::opening_proof::{OpeningPoint, BIG_ENDIAN};
use jolt_prover_legacy::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use jolt_prover_legacy::transcripts::{Blake2bTranscript, Transcript};
use jolt_prover_legacy::zkvm::bytecode::chunks::build_committed_bytecode_chunk_coeffs;
use jolt_prover_legacy::zkvm::claim_reductions::{
    AdviceClaimReductionParams, AdviceClaimReductionProver, AdviceKind,
    BytecodeClaimReductionParams, BytecodeClaimReductionProver, PrecommittedClaimReduction,
    PrecommittedSchedulingReference, ProgramImageClaimReductionParams,
    ProgramImageClaimReductionProver,
};
use jolt_verifier::stages::PrecommittedSchedule;

type LegacyChallenge = <LegacyFr as LegacyJoltField>::Challenge;

pub struct LegacyPrecommittedBaseline {
    pub prepare: Duration,
    pub rounds: Duration,
}

pub struct LegacyPrecommittedInputs<'a> {
    pub log_t: usize,
    pub log_k_chunk: usize,
    pub trace_length: usize,
    pub ram_k: usize,
    pub bytecode: &'a [jolt_riscv::JoltInstructionRow],
    pub bytecode_chunk_count: usize,
    pub program_image_words: &'a [u64],
    pub program_image_start_index: usize,
    pub max_trusted_advice_size: usize,
    pub max_untrusted_advice_size: usize,
}

fn transcript() -> Blake2bTranscript {
    Blake2bTranscript::new(b"vertical-baseline")
}

fn challenges(count: usize) -> Vec<LegacyChallenge> {
    transcript().challenge_vector_optimized::<LegacyFr>(count)
}

fn scalars(count: usize) -> Vec<LegacyFr> {
    transcript().challenge_vector::<LegacyFr>(count)
}

fn scheduling_reference(
    inputs: &LegacyPrecommittedInputs<'_>,
    candidates: &[usize],
) -> PrecommittedSchedulingReference {
    PrecommittedClaimReduction::<LegacyFr>::scheduling_reference(
        inputs.log_t + inputs.log_k_chunk,
        candidates,
    )
}

fn drive_rounds<P>(prover: &mut P, rounds: usize) -> Duration
where
    P: SumcheckInstanceProver<LegacyFr, Blake2bTranscript>,
{
    let round_challenges = challenges(rounds + 1);
    let mut claim = LegacyFr::from(0u64);
    let mut bind: Option<LegacyChallenge> = None;
    let mut elapsed = Duration::ZERO;
    for (round, &challenge) in round_challenges.iter().enumerate().take(rounds) {
        let start = Instant::now();
        if let Some(previous) = bind {
            prover.ingest_challenge(previous, round - 1);
        }
        let message = prover.compute_message(round, claim);
        elapsed += start.elapsed();
        claim = message.evaluate(&challenge);
        bind = Some(challenge);
    }
    if let Some(challenge) = bind {
        let start = Instant::now();
        prover.ingest_challenge(challenge, rounds - 1);
        elapsed += start.elapsed();
    }
    elapsed
}

fn advice_params(
    inputs: &LegacyPrecommittedInputs<'_>,
    candidates: &[usize],
    kind: AdviceKind,
    advice_size_bytes: usize,
) -> AdviceClaimReductionParams<LegacyFr> {
    let reference = scheduling_reference(inputs, candidates);
    let (advice_col_vars, advice_row_vars) =
        DoryGlobals::advice_sigma_nu_from_max_bytes(advice_size_bytes);
    let precommitted = PrecommittedClaimReduction::new(advice_row_vars, advice_col_vars, reference);
    let r_val =
        OpeningPoint::<BIG_ENDIAN, LegacyFr>::new(challenges(advice_row_vars + advice_col_vars));
    AdviceClaimReductionParams {
        kind,
        precommitted,
        advice_col_vars,
        advice_row_vars,
        r_val,
    }
}

fn bytecode_params(
    inputs: &LegacyPrecommittedInputs<'_>,
    candidates: &[usize],
) -> BytecodeClaimReductionParams<LegacyFr> {
    let reference = scheduling_reference(inputs, candidates);
    let bytecode_len = inputs.bytecode.len();
    let chunk_count = inputs.bytecode_chunk_count;
    let log_bytecode_chunk_size = (bytecode_len / chunk_count).trailing_zeros() as usize;
    let lane_count =
        jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode::COMMITTED_BYTECODE_LANE_CAPACITY;
    let total_vars = lane_count.trailing_zeros() as usize + log_bytecode_chunk_size;
    let (bytecode_col_vars, bytecode_row_vars) = DoryGlobals::balanced_sigma_nu(total_vars);
    let precommitted =
        PrecommittedClaimReduction::new(bytecode_row_vars, bytecode_col_vars, reference);
    BytecodeClaimReductionParams {
        precommitted,
        eta: scalars(1)[0],
        eta_powers: scalars(5),
        chunk_rbc_weights: scalars(chunk_count),
        log_bytecode_chunk_size,
        bytecode_chunk_count: chunk_count,
        bytecode_col_vars,
        bytecode_row_vars,
        r_bc: OpeningPoint::<BIG_ENDIAN, LegacyFr>::new(challenges(log_bytecode_chunk_size)),
        lane_weights: scalars(lane_count),
    }
}

fn shifted_program_image_eq_slice(
    r_addr: &[LegacyChallenge],
    start_index: usize,
    padded_len_words: usize,
) -> Vec<LegacyFr> {
    let mut eq_slice = Vec::with_capacity(padded_len_words);
    let mut index = start_index;
    let mut remaining = padded_len_words;
    while remaining > 0 {
        let (block_size, block_evals) =
            EqPolynomial::<LegacyFr>::evals_for_max_aligned_block(r_addr, index, remaining);
        eq_slice.extend(block_evals);
        index += block_size;
        remaining -= block_size;
    }
    eq_slice
}

fn program_image_params(
    inputs: &LegacyPrecommittedInputs<'_>,
    candidates: &[usize],
) -> ProgramImageClaimReductionParams<LegacyFr> {
    let reference = scheduling_reference(inputs, candidates);
    let padded_len_words = inputs.program_image_words.len();
    let m = padded_len_words.trailing_zeros() as usize;
    let (prog_col_vars, prog_row_vars) = DoryGlobals::balanced_sigma_nu(m);
    let precommitted = PrecommittedClaimReduction::new(prog_row_vars, prog_col_vars, reference);
    let ram_num_vars = inputs.ram_k.trailing_zeros() as usize;
    let r_addr_rw = challenges(ram_num_vars);
    let shifted_eq_coeffs = shifted_program_image_eq_slice(
        &r_addr_rw,
        inputs.program_image_start_index,
        padded_len_words,
    );
    ProgramImageClaimReductionParams {
        precommitted,
        prog_col_vars,
        prog_row_vars,
        ram_num_vars,
        start_index: inputs.program_image_start_index,
        padded_len_words,
        m,
        r_addr_rw,
        shifted_eq_coeffs,
    }
}

pub fn candidate_total_vars(inputs: &LegacyPrecommittedInputs<'_>) -> Vec<usize> {
    let lane_vars = (jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode::COMMITTED_BYTECODE_LANE_CAPACITY
        .trailing_zeros()) as usize;
    let chunk_vars =
        (inputs.bytecode.len() / inputs.bytecode_chunk_count).trailing_zeros() as usize;
    vec![
        advice_candidate(inputs.max_trusted_advice_size),
        advice_candidate(inputs.max_untrusted_advice_size),
        lane_vars + chunk_vars,
        inputs.program_image_words.len().trailing_zeros() as usize,
    ]
}

fn advice_candidate(max_advice_size_bytes: usize) -> usize {
    ((max_advice_size_bytes / 8).next_power_of_two().max(1)).trailing_zeros() as usize
}

fn assert_schedule_matches(
    label: &str,
    legacy: &PrecommittedClaimReduction<LegacyFr>,
    modular: &jolt_claims::protocols::jolt::PrecommittedClaimReduction,
) {
    let legacy_permutation = legacy.poly_opening_round_permutation_be();
    let modular_permutation = modular.poly_opening_round_permutation_be();
    assert_eq!(
        legacy_permutation, modular_permutation,
        "{label}: the legacy and modular opening-round permutations disagree, so the two \
         baselines are not measuring the same schedule (check the DoryGlobals context)"
    );
    assert_eq!(
        legacy.cycle_phase_rounds(),
        modular.cycle_phase_rounds(),
        "{label}: legacy and modular active cycle rounds disagree"
    );
    assert_eq!(
        legacy.address_phase_rounds(),
        modular.address_phase_rounds(),
        "{label}: legacy and modular active address rounds disagree"
    );
}

pub fn advice_baseline(
    inputs: &LegacyPrecommittedInputs<'_>,
    schedule: &PrecommittedSchedule,
    kind: AdviceKind,
    address_phase: bool,
) -> LegacyPrecommittedBaseline {
    let _guard = DoryGlobals::initialize_main_with_log_embedding(
        1 << inputs.log_k_chunk,
        inputs.trace_length,
        main_total_vars(inputs),
        None,
    );
    let candidates = candidate_total_vars(inputs);
    let advice_size = match kind {
        AdviceKind::Trusted => inputs.max_trusted_advice_size,
        AdviceKind::Untrusted => inputs.max_untrusted_advice_size,
    };
    let params = advice_params(inputs, &candidates, kind, advice_size);
    let modular_kind = match kind {
        AdviceKind::Trusted => jolt_claims::protocols::jolt::JoltAdviceKind::Trusted,
        AdviceKind::Untrusted => jolt_claims::protocols::jolt::JoltAdviceKind::Untrusted,
    };
    if let Some(layout) = schedule.advice(modular_kind) {
        assert_schedule_matches("advice", &params.precommitted, layout.precommitted());
    }
    let cycle_rounds = params.precommitted.cycle_alignment_rounds();
    let address_rounds = params.precommitted.address_alignment_rounds();
    let words = vec![0u64; advice_size / 8];

    let start = Instant::now();
    let mut prover = AdviceClaimReductionProver::initialize(params, LegacyPoly::from(words));
    let prepare = start.elapsed();

    if !address_phase {
        return LegacyPrecommittedBaseline {
            prepare,
            rounds: drive_rounds(&mut prover, cycle_rounds),
        };
    }
    let _ = drive_rounds(&mut prover, cycle_rounds);
    prover.transition_to_address_phase();
    LegacyPrecommittedBaseline {
        prepare: Duration::ZERO,
        rounds: drive_rounds(&mut prover, address_rounds),
    }
}

pub fn bytecode_baseline(
    inputs: &LegacyPrecommittedInputs<'_>,
    schedule: &PrecommittedSchedule,
    address_phase: bool,
) -> LegacyPrecommittedBaseline {
    let _guard = DoryGlobals::initialize_main_with_log_embedding(
        1 << inputs.log_k_chunk,
        inputs.trace_length,
        main_total_vars(inputs),
        None,
    );
    let candidates = candidate_total_vars(inputs);
    let params = bytecode_params(inputs, &candidates);
    if let Some(layout) = schedule.bytecode.as_ref() {
        assert_schedule_matches("bytecode", &params.precommitted, layout.precommitted());
    }
    let cycle_rounds = params.precommitted.cycle_alignment_rounds();
    let address_rounds = params.precommitted.address_alignment_rounds();

    let start = Instant::now();
    let chunk_coeffs = build_committed_bytecode_chunk_coeffs::<LegacyFr>(
        inputs.bytecode,
        inputs.bytecode_chunk_count,
    );
    let mut prover = BytecodeClaimReductionProver::initialize(params, &chunk_coeffs);
    let prepare = start.elapsed();

    if !address_phase {
        return LegacyPrecommittedBaseline {
            prepare,
            rounds: drive_rounds(&mut prover, cycle_rounds),
        };
    }
    let _ = drive_rounds(&mut prover, cycle_rounds);
    prover.transition_to_address_phase();
    LegacyPrecommittedBaseline {
        prepare: Duration::ZERO,
        rounds: drive_rounds(&mut prover, address_rounds),
    }
}

pub fn program_image_baseline(
    inputs: &LegacyPrecommittedInputs<'_>,
    schedule: &PrecommittedSchedule,
    address_phase: bool,
) -> LegacyPrecommittedBaseline {
    let _guard = DoryGlobals::initialize_main_with_log_embedding(
        1 << inputs.log_k_chunk,
        inputs.trace_length,
        main_total_vars(inputs),
        None,
    );
    let candidates = candidate_total_vars(inputs);
    let words = inputs.program_image_words.to_vec();

    let start = Instant::now();
    let params = program_image_params(inputs, &candidates);
    let schedule_check = params.precommitted.clone();
    let cycle_rounds = schedule_check.cycle_alignment_rounds();
    let address_rounds = schedule_check.address_alignment_rounds();
    let mut prover = ProgramImageClaimReductionProver::initialize(params, words);
    let prepare = start.elapsed();

    if let Some(layout) = schedule.program_image.as_ref() {
        assert_schedule_matches("program image", &schedule_check, layout.precommitted());
    }

    if !address_phase {
        return LegacyPrecommittedBaseline {
            prepare,
            rounds: drive_rounds(&mut prover, cycle_rounds),
        };
    }
    let _ = drive_rounds(&mut prover, cycle_rounds);
    prover.transition_to_address_phase();
    LegacyPrecommittedBaseline {
        prepare: Duration::ZERO,
        rounds: drive_rounds(&mut prover, address_rounds),
    }
}

fn main_total_vars(inputs: &LegacyPrecommittedInputs<'_>) -> usize {
    let main = inputs.log_t + inputs.log_k_chunk;
    candidate_total_vars(inputs)
        .into_iter()
        .fold(main, usize::max)
}
