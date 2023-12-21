use crate::jolt::instruction::add::ADDInstruction;
use crate::jolt::instruction::and::ANDInstruction;
use crate::jolt::instruction::beq::BEQInstruction;
use crate::jolt::instruction::bge::BGEInstruction;
use crate::jolt::instruction::bgeu::BGEUInstruction;
use crate::jolt::instruction::blt::BLTInstruction;
use crate::jolt::instruction::bltu::BLTUInstruction;
use crate::jolt::instruction::bne::BNEInstruction;
use crate::jolt::instruction::jal::JALInstruction;
use crate::jolt::instruction::jalr::JALRInstruction;
use crate::jolt::instruction::or::ORInstruction;
use crate::jolt::instruction::sll::SLLInstruction;
use crate::jolt::instruction::sra::SRAInstruction;
use crate::jolt::instruction::srl::SRLInstruction;
use crate::jolt::instruction::sub::SUBInstruction;
use crate::jolt::instruction::JoltInstruction;
use crate::jolt::vm::bytecode::{random_bytecode_trace, ELFRow};
use crate::jolt::vm::instruction_lookups::InstructionLookupsProof;
use crate::jolt::vm::read_write_memory::{random_memory_trace, RandomInstruction};
use crate::jolt::vm::rv32i_vm::{RV32IJoltVM, RV32I};
use crate::jolt::vm::Jolt;
use crate::lasso::surge::Surge;
use crate::poly::dense_mlpoly::bench::{
    init_commit_bench, init_commit_bench_ones, init_commit_small, run_commit_bench,
};
use crate::poly::dense_mlpoly::{CommitHint, DensePolynomial};
use crate::poly::eq_poly::EqPolynomial;
use crate::subprotocols::sumcheck::{CubicSumcheckParams, SumcheckInstanceProof};
use crate::utils::math::Math;
use crate::utils::{index_to_field_bitvector, random::RandomTape};
use crate::{jolt::instruction::xor::XORInstruction, utils::gen_random_point};
use ark_curve25519::{EdwardsProjective, Fr};
use ark_std::{test_rng, UniformRand, Zero};
use common::ELFInstruction;
use criterion::black_box;
use merlin::Transcript;
use rand_chacha::rand_core::RngCore;
use rand_core::SeedableRng;

#[derive(Debug, Copy, Clone, clap::ValueEnum)]
pub enum BenchType {
    Poly,
    EverythingExceptR1CS,
    Bytecode,
    ReadWriteMemory,
    InstructionLookups,
    Sumcheck,
}

#[allow(unreachable_patterns)] // good errors on new BenchTypes
pub fn benchmarks(
    bench_type: BenchType,
    num_cycles: Option<usize>,
    memory_size: Option<usize>,
    bytecode_size: Option<usize>,
) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    match bench_type {
        BenchType::Poly => dense_ml_poly(),
        BenchType::EverythingExceptR1CS => {
            prove_e2e_except_r1cs(num_cycles, memory_size, bytecode_size)
        }
        BenchType::Bytecode => prove_bytecode(num_cycles, bytecode_size),
        BenchType::ReadWriteMemory => prove_memory(num_cycles, memory_size, bytecode_size),
        BenchType::InstructionLookups => prove_instruction_lookups(num_cycles),
        BenchType::Sumcheck => bench_sumcheck(),
        _ => panic!("BenchType does not have a mapping"),
    }
}

fn bench_sumcheck() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    let mut rng = test_rng();

    let num_vars = 22;
    let batch_size = 2;

    let mut tasks = vec![];

    for _ in 0..10 {
        let r_eq = std::iter::repeat_with(|| Fr::rand(&mut rng))
            .take(num_vars)
            .collect();
        let eq = DensePolynomial::new(EqPolynomial::new(r_eq).evals());
        let mut poly_as = Vec::with_capacity(batch_size);
        let mut poly_bs = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let ra = std::iter::repeat_with(|| Fr::rand(&mut rng))
                .take(num_vars)
                .collect();
            let rb = std::iter::repeat_with(|| Fr::rand(&mut rng))
                .take(num_vars)
                .collect();
            let a = DensePolynomial::new(EqPolynomial::new(ra).evals());
            let b = DensePolynomial::new(EqPolynomial::new(rb).evals());
            poly_as.push(a);
            poly_bs.push(b);
        }
        let params =
            CubicSumcheckParams::new_prod(poly_as.clone(), poly_bs.clone(), eq.clone(), num_vars);
        let coeffs: Vec<Fr> = std::iter::repeat_with(|| Fr::rand(&mut rng))
            .take(batch_size)
            .collect();

        let mut joint_claim = Fr::zero();
        for batch_i in 0..batch_size {
            let mut claim = Fr::zero();
            for var_i in 0..num_vars {
                let eval_a = poly_as[batch_i].evaluate(&index_to_field_bitvector(var_i, num_vars));
                let eval_b = poly_bs[batch_i].evaluate(&index_to_field_bitvector(var_i, num_vars));
                let eval_eq = eq.evaluate(&index_to_field_bitvector(var_i, num_vars));

                claim += eval_a * eval_b * eval_eq;
            }
            joint_claim += coeffs[batch_i] * claim;
        }

        let work = Box::new(move || {
            let mut transcript = Transcript::new(b"example");
            black_box(SumcheckInstanceProof::prove_cubic_batched::<
                EdwardsProjective,
            >(&joint_claim, params, &coeffs, &mut transcript));
        });
        tasks.push((
            tracing::info_span!("batched cubic sumcheck"),
            work as Box<dyn FnOnce()>,
        ));
    }

    tasks
}

fn prove_e2e_except_r1cs(
    num_cycles: Option<usize>,
    memory_size: Option<usize>,
    bytecode_size: Option<usize>,
) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(1234567890);

    let memory_size = memory_size.unwrap_or(1 << 22); // 4,194,304 = 4 MB
    let bytecode_size = bytecode_size.unwrap_or(1 << 16); // 65,536 = 64 kB
    let num_cycles = num_cycles.unwrap_or(1 << 16); // 65,536

    let ops: Vec<RV32I> = std::iter::repeat_with(|| RV32I::random_instruction(&mut rng))
        .take(num_cycles)
        .collect();

    let bytecode: Vec<ELFInstruction> = (0..bytecode_size)
        .map(|i| ELFInstruction::random(i, &mut rng))
        .collect();
    // 7 memory ops per instruction, rounded up to still be a power of 2
    let memory_trace = random_memory_trace(&bytecode, memory_size, 8 * num_cycles, &mut rng);
    let mut bytecode_rows: Vec<ELFRow> = (0..bytecode_size)
        .map(|i| ELFRow::random(i, &mut rng))
        .collect();
    let bytecode_trace = random_bytecode_trace(&bytecode_rows, num_cycles, &mut rng);

    let work = Box::new(|| {
        let mut transcript = Transcript::new(b"example");
        let mut random_tape = RandomTape::new(b"test_tape");
        let _ = RV32IJoltVM::prove_bytecode(
            bytecode_rows,
            bytecode_trace,
            &mut transcript,
            &mut random_tape,
        );
        let _ =
            RV32IJoltVM::prove_memory(bytecode, memory_trace, &mut transcript, &mut random_tape);
        let _: InstructionLookupsProof<Fr, EdwardsProjective> =
            RV32IJoltVM::prove_instruction_lookups(ops, &mut transcript, &mut random_tape);
    });
    vec![(
        tracing::info_span!("prove_bytecode + prove_memory + prove_instruction_lookups"),
        work,
    )]
}

fn prove_bytecode(
    num_cycles: Option<usize>,
    bytecode_size: Option<usize>,
) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(1234567890);

    let bytecode_size = bytecode_size.unwrap_or(1 << 16); // 65,536 = 64 kB
    let num_cycles = num_cycles.unwrap_or(1 << 16); // 65,536

    let bytecode: Vec<ELFInstruction> = (0..bytecode_size)
        .map(|i| ELFInstruction::random(i, &mut rng))
        .collect();
    let mut bytecode_rows: Vec<ELFRow> = (0..bytecode_size)
        .map(|i| ELFRow::random(i, &mut rng))
        .collect();
    let bytecode_trace = random_bytecode_trace(&bytecode_rows, num_cycles, &mut rng);

    let work = Box::new(|| {
        let mut transcript = Transcript::new(b"example");
        let mut random_tape: RandomTape<EdwardsProjective> = RandomTape::new(b"test_tape");
        let _ = RV32IJoltVM::prove_bytecode(
            bytecode_rows,
            bytecode_trace,
            &mut transcript,
            &mut random_tape,
        );
    });
    vec![(tracing::info_span!("prove_bytecode"), work)]
}

fn prove_memory(
    num_cycles: Option<usize>,
    memory_size: Option<usize>,
    bytecode_size: Option<usize>,
) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(1234567890);

    let memory_size = memory_size.unwrap_or(1 << 22); // 4,194,304 = 4 MB
    let bytecode_size = bytecode_size.unwrap_or(1 << 16); // 65,536 = 64 kB
    let num_cycles = num_cycles.unwrap_or(1 << 16); // 65,536

    let bytecode: Vec<ELFInstruction> = (0..bytecode_size)
        .map(|i| ELFInstruction::random(i, &mut rng))
        .collect();
    // 7 memory ops per instruction, rounded up to still be a power of 2
    let memory_trace = random_memory_trace(&bytecode, memory_size, 8 * num_cycles, &mut rng);

    let work = Box::new(|| {
        let mut transcript = Transcript::new(b"example");
        let mut random_tape: RandomTape<EdwardsProjective> = RandomTape::new(b"test_tape");
        let _ =
            RV32IJoltVM::prove_memory(bytecode, memory_trace, &mut transcript, &mut random_tape);
    });
    vec![(tracing::info_span!("prove_memory"), work)]
}

fn prove_instruction_lookups(num_cycles: Option<usize>) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(1234567890);

    let num_cycles = num_cycles.unwrap_or(1 << 16); // 65,536
    let ops: Vec<RV32I> = std::iter::repeat_with(|| RV32I::random_instruction(&mut rng))
        .take(num_cycles)
        .collect();

    let work = Box::new(|| {
        let mut transcript = Transcript::new(b"example");
        let mut random_tape: RandomTape<EdwardsProjective> = RandomTape::new(b"test_tape");
        RV32IJoltVM::prove_instruction_lookups(ops, &mut transcript, &mut random_tape);
    });
    vec![(tracing::info_span!("prove_instruction_lookups"), work)]
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
