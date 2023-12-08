use crate::jolt::instruction::JoltInstruction;
use crate::jolt::vm::instruction_lookups::InstructionLookupsProof;
use crate::jolt::vm::rv32i_vm::{RV32IJoltVM, RV32I};
use crate::jolt::vm::Jolt;
use crate::lasso::surge::Surge;
use crate::poly::dense_mlpoly::bench::{
    init_commit_bench, init_commit_bench_ones, init_commit_small, run_commit_bench,
};
use crate::poly::dense_mlpoly::CommitHint;
use crate::utils::math::Math;
use crate::utils::random::RandomTape;
use crate::{jolt::instruction::xor::XORInstruction, utils::gen_random_point};
use ark_curve25519::{EdwardsProjective, Fr};
use ark_std::test_rng;
use criterion::black_box;
use merlin::Transcript;
use rand_chacha::rand_core::RngCore;
use rand_core::SeedableRng;

#[derive(Debug, Clone, clap::ValueEnum)]
pub enum BenchType {
    JoltDemo,
    Halo2Comparison,
    RV32,
    Poly,
}

#[allow(unreachable_patterns)] // good errors on new BenchTypes
pub fn benchmarks(bench_type: BenchType) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    match bench_type {
        BenchType::JoltDemo => jolt_demo_benchmarks(),
        BenchType::Halo2Comparison => halo2_comparison_benchmarks(),
        BenchType::RV32 => rv32i_lookup_benchmarks(),
        BenchType::Poly => dense_ml_poly(),
        _ => panic!("BenchType does not have a mapping"),
    }
}

fn jolt_demo_benchmarks() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    const C: usize = 4;
    const M: usize = 1 << 16;
    vec![
        (
            tracing::info_span!("XOR(2^10)"),
            random_surge_test::<C, M>(/* num_ops */ 1 << 10),
        ),
        (
            tracing::info_span!("XOR(2^12)"),
            random_surge_test::<C, M>(/* num_ops */ 1 << 12),
        ),
        (
            tracing::info_span!("XOR(2^14)"),
            random_surge_test::<C, M>(/* num_ops */ 1 << 14),
        ),
        (
            tracing::info_span!("XOR(2^16)"),
            random_surge_test::<C, M>(/* num_ops */ 1 << 16),
        ),
        (
            tracing::info_span!("XOR(2^18)"),
            random_surge_test::<C, M>(/* num_ops */ 1 << 18),
        ),
        (
            tracing::info_span!("XOR(2^20)"),
            random_surge_test::<C, M>(/* num_ops */ 1 << 20),
        ),
        (
            tracing::info_span!("XOR(2^22)"),
            random_surge_test::<C, M>(/* num_ops */ 1 << 22),
        ),
    ]
}

fn halo2_comparison_benchmarks() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    const C: usize = 1;
    const M: usize = 1 << 16;
    vec![
        (
            tracing::info_span!("XOR(2^10)"),
            random_surge_test::<C, M>(/* num_ops */ 1 << 10),
        ),
        (
            tracing::info_span!("XOR(2^12)"),
            random_surge_test::<C, M>(/* num_ops */ 1 << 12),
        ),
        (
            tracing::info_span!("XOR(2^14)"),
            random_surge_test::<C, M>(/* num_ops */ 1 << 14),
        ),
        (
            tracing::info_span!("XOR(2^16)"),
            random_surge_test::<C, M>(/* num_ops */ 1 << 16),
        ),
        (
            tracing::info_span!("XOR(2^18)"),
            random_surge_test::<C, M>(/* num_ops */ 1 << 18),
        ),
        (
            tracing::info_span!("XOR(2^20)"),
            random_surge_test::<C, M>(/* num_ops */ 1 << 20),
        ),
        (
            tracing::info_span!("XOR(2^22)"),
            random_surge_test::<C, M>(/* num_ops */ 1 << 22),
        ),
        (
            tracing::info_span!("XOR(2^24)"),
            random_surge_test::<C, M>(/* num_ops */ 1 << 24),
        ),
    ]
}

fn rv32i_lookup_benchmarks() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(1234567890);

    const NUM_CYCLES: usize = 64_000;
    let ops: Vec<RV32I> = vec![RV32I::random_instruction(&mut rng); NUM_CYCLES];
    println!("Running {:?}", ops.len());

    let work = Box::new(|| {
        let mut prover_transcript = Transcript::new(b"example");
        let mut random_tape = RandomTape::new(b"test_tape");
        let proof: InstructionLookupsProof<Fr, EdwardsProjective> =
            RV32IJoltVM::prove_instruction_lookups(ops, &mut prover_transcript, &mut random_tape);
        let mut verifier_transcript = Transcript::new(b"example");
        assert!(RV32IJoltVM::verify_instruction_lookups(proof, &mut verifier_transcript).is_ok());
    });
    vec![(tracing::info_span!("RV32IM"), work)]
}

fn e2e_except_r1cs() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    
    vec![]
}

fn random_surge_test<const C: usize, const M: usize>(num_ops: usize) -> Box<dyn FnOnce()> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(1234567890);
    let ops: Vec<XORInstruction> = vec![XORInstruction::default().random(&mut rng); num_ops];

    let func = move || {
        let mut prover_transcript = Transcript::new(b"test_transcript");
        let surge = <Surge<Fr, EdwardsProjective, XORInstruction, C, M>>::new(ops.clone());
        let proof = surge.prove(&mut prover_transcript);

        let mut verifier_transcript = Transcript::new(b"test_transcript");
        <Surge<Fr, EdwardsProjective, XORInstruction, C, M>>::verify(
            proof,
            &mut verifier_transcript,
        )
        .expect("should work");
    };

    Box::new(func)
}

fn dense_ml_poly() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    let log_sizes = [20];
    let mut tasks = Vec::new();

    // Normal benchmark
    for &log_size in &log_sizes {
        let (gens, poly) = init_commit_bench(log_size);
        let task = move || {
            black_box(run_commit_bench(gens, poly));
        };
        tasks.push((
            tracing::info_span!("DensePoly::commit", log_size = log_size),
            Box::new(task) as Box<dyn FnOnce()>,
        ));
    }

    // Commit only 0 / 1
    for &log_size in &log_sizes {
        let (gens, poly) = init_commit_bench_ones(log_size, 0.3);
        let task = move || {
            black_box(poly.commit_with_hint(&gens, CommitHint::Normal));
        };
        tasks.push((
            tracing::info_span!("DensePoly::commit(0/1)", log_size = log_size),
            Box::new(task) as Box<dyn FnOnce()>,
        ));

        let (gens, poly) = init_commit_bench_ones(log_size, 0.3);
        let task = move || {
            black_box(poly.commit_with_hint(&gens, CommitHint::Flags));
        };
        tasks.push((
            tracing::info_span!("DensePoly::commit_with_hint(0/1)", log_size = log_size),
            Box::new(task) as Box<dyn FnOnce()>,
        ));
    }

    // Commit only small field elements (as if counts / indices)
    for &log_size in &log_sizes {
        let (gens, poly) = init_commit_small(log_size, 1 << 16);
        let task = move || {
            black_box(poly.commit_with_hint(&gens, CommitHint::Normal));
        };
        tasks.push((
            tracing::info_span!("DensePoly::commit(small)", log_size = log_size),
            Box::new(task) as Box<dyn FnOnce()>,
        ));

        let (gens, poly) = init_commit_small(log_size, 1 << 16);
        let task = move || {
            black_box(poly.commit_with_hint(&gens, CommitHint::Small));
        };
        tasks.push((
            tracing::info_span!("DensePoly::commit_with_hint(small)", log_size = log_size),
            Box::new(task) as Box<dyn FnOnce()>,
        ));
    }

    tasks
}
