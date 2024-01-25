use crate::jolt::instruction::add::ADDInstruction;
use crate::jolt::trace::rv::RVTraceRow;
use crate::jolt::trace::JoltProvableTrace;
use crate::jolt::vm::bytecode::{random_bytecode_trace, BytecodePolynomials, ELFRow};
use crate::jolt::vm::instruction_lookups::InstructionPolynomials;
use crate::jolt::vm::read_write_memory::{
    random_memory_trace, MemoryOp, RandomInstruction, ReadWriteMemory,
};
use crate::jolt::vm::rv32i_vm::{RV32IJoltVM, C, M, RV32I};
use crate::jolt::vm::Jolt;
use crate::poly::dense_mlpoly::bench::{init_commit_bench, run_commit_bench};
use crate::subprotocols::sparse;
use crate::subprotocols::sumcheck::bench::run_sparse;
use crate::utils::math::Math;
use ark_curve25519::{EdwardsProjective, Fr};
use ark_std::{test_rng, UniformRand};
use common::{constants::MEMORY_OPS_PER_INSTRUCTION, ELFInstruction};
use criterion::black_box;
use itertools::Itertools;
use merlin::Transcript;
use rand_core::SeedableRng;

#[derive(Debug, Copy, Clone, clap::ValueEnum)]
pub enum BenchType {
    Poly,
    SparsePolyBind,
    SparseSumcheck,
    EverythingExceptR1CS,
    Bytecode,
    ReadWriteMemory,
    InstructionLookups,
    Fibonacci,
    Hash,
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
        BenchType::SparsePolyBind => sparse_ml_poly_bind(),
        BenchType::SparseSumcheck => sparse_sumcheck(),
        BenchType::EverythingExceptR1CS => {
            prove_e2e_except_r1cs(num_cycles, memory_size, bytecode_size)
        }
        BenchType::Bytecode => prove_bytecode(num_cycles, bytecode_size),
        BenchType::ReadWriteMemory => prove_memory(num_cycles, memory_size, bytecode_size),
        BenchType::InstructionLookups => prove_instruction_lookups(num_cycles),
        BenchType::Hash => hash(),
        BenchType::Fibonacci => fibonacci(),
        _ => panic!("BenchType does not have a mapping"),
    }
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
    let memory_trace = random_memory_trace(&bytecode, memory_size, num_cycles, &mut rng);
    let bytecode_rows: Vec<ELFRow> = (0..bytecode_size)
        .map(|i| ELFRow::random(i, &mut rng))
        .collect();
    let bytecode_trace = random_bytecode_trace(&bytecode_rows, num_cycles, &mut rng);

    let work = Box::new(|| {
        let mut transcript = Transcript::new(b"example");
        let _: (_, BytecodePolynomials<Fr, EdwardsProjective>, _) =
            RV32IJoltVM::prove_bytecode(bytecode_rows, bytecode_trace, &mut transcript);
        let _: (_, ReadWriteMemory<Fr, EdwardsProjective>, _) =
            RV32IJoltVM::prove_memory(bytecode, memory_trace, &mut transcript);
        let _: (_, InstructionPolynomials<Fr, EdwardsProjective>, _) =
            RV32IJoltVM::prove_instruction_lookups(ops, &mut transcript);
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

    let bytecode_rows: Vec<ELFRow> = (0..bytecode_size)
        .map(|i| ELFRow::random(i, &mut rng))
        .collect();
    let bytecode_trace = random_bytecode_trace(&bytecode_rows, num_cycles, &mut rng);

    let work = Box::new(|| {
        let mut transcript = Transcript::new(b"example");
        let _: (_, BytecodePolynomials<Fr, EdwardsProjective>, _) =
            RV32IJoltVM::prove_bytecode(bytecode_rows, bytecode_trace, &mut transcript);
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
    let memory_trace = random_memory_trace(&bytecode, memory_size, num_cycles, &mut rng);

    let work = Box::new(|| {
        let mut transcript = Transcript::new(b"example");
        let _: (_, ReadWriteMemory<Fr, EdwardsProjective>, _) =
            RV32IJoltVM::prove_memory(bytecode, memory_trace, &mut transcript);
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
        let _: (_, InstructionPolynomials<Fr, EdwardsProjective>, _) =
            RV32IJoltVM::prove_instruction_lookups(ops, &mut transcript);
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
    tasks
}

fn hash() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    let mut tasks = Vec::new();
    use common::{path::JoltPaths, serializable::Serializable};
    compiler::cached_compile_example("hash");

    let task = move || {
        let trace_location = JoltPaths::trace_path("hash");
        let loaded_trace: Vec<common::RVTraceRow> =
            Vec::<common::RVTraceRow>::deserialize_from_file(&trace_location)
                .expect("deserialization failed");
        let bytecode_location = JoltPaths::bytecode_path("hash");
        let bytecode = Vec::<ELFInstruction>::deserialize_from_file(&bytecode_location)
            .expect("deserialization failed");
        let bytecode_rows: Vec<ELFRow> = bytecode.clone().iter().map(ELFRow::from).collect();

        let converted_trace: Vec<RVTraceRow> = loaded_trace
            .into_iter()
            .map(|common| RVTraceRow::from_common(common))
            .collect();

        let bytecode_trace: Vec<ELFRow> = converted_trace
            .iter()
            .map(|row| row.to_bytecode_trace())
            .collect();

        let instructions_r1cs: Vec<RV32I> = converted_trace
            .iter()
            .flat_map(|row| {
                let instructions = row.to_jolt_instructions();
                if instructions.is_empty() {
                    vec![ADDInstruction::<32>(0_u64, 0_u64).into()]
                } else {
                    instructions
                }
            })
            .collect();

        let memory_trace_r1cs = converted_trace
            .iter()
            .flat_map(|row| row.to_ram_ops())
            .collect_vec();

        let circuit_flags = converted_trace
            .iter()
            .flat_map(|row| row.to_circuit_flags::<Fr>())
            .collect::<Vec<_>>();

        let mut transcript = Transcript::new(b"Jolt transcript");
        // TODO(sragss): Swap this to &Vec<Instructions> to avoid clone
        <RV32IJoltVM as Jolt<'_, Fr, EdwardsProjective, C, M>>::prove_r1cs(
            instructions_r1cs.clone(),
            bytecode_rows,
            bytecode_trace.clone(),
            bytecode,
            memory_trace_r1cs,
            circuit_flags,
            &mut transcript,
        );

        let bytecode_location = JoltPaths::bytecode_path("hash");
        let bytecode = Vec::<ELFInstruction>::deserialize_from_file(&bytecode_location)
            .expect("deserialization failed");
        let bytecode_rows = bytecode.iter().map(ELFRow::from).collect();

        // // TODO(JOLT-89): Encapsulate this logic elsewhere.
        // // Emulator sets register 0xb to 0x1020 upon initialization for some reason,
        // // something about Linux boot requiring it...
        // let mut memory_trace: Vec<MemoryOp> = vec![MemoryOp::Write(11, 4128)];
        let mut memory_trace: Vec<[MemoryOp; MEMORY_OPS_PER_INSTRUCTION]> = converted_trace
            .into_iter()
            .map(|row| row.to_ram_ops().try_into().unwrap())
            .collect();
        memory_trace.resize(
            memory_trace.len().next_power_of_two(),
            std::array::from_fn(|_| MemoryOp::no_op()),
        );

        let mut transcript = Transcript::new(b"Jolt transcript");
        let (bytecode_proof, _, bytecode_commitment) =
            <RV32IJoltVM as Jolt<'_, _, EdwardsProjective, C, M>>::prove_bytecode(
                bytecode_rows,
                bytecode_trace,
                &mut transcript,
            );
        let (memory_proof, _, memory_commitment) =
            <RV32IJoltVM as Jolt<'_, _, EdwardsProjective, C, M>>::prove_memory(
                bytecode,
                memory_trace,
                &mut transcript,
            );
        let (instruction_lookups_proof, _, instruction_lookups_commitment) =
            <RV32IJoltVM as Jolt<'_, _, EdwardsProjective, C, M>>::prove_instruction_lookups(
                instructions_r1cs,
                &mut transcript,
            );

        let mut transcript = Transcript::new(b"Jolt transcript");
        assert!(
            RV32IJoltVM::verify_bytecode(bytecode_proof, bytecode_commitment, &mut transcript)
                .is_ok()
        );
        assert!(
            RV32IJoltVM::verify_memory(memory_proof, memory_commitment, &mut transcript).is_ok()
        );
        assert!(RV32IJoltVM::verify_instruction_lookups(
            instruction_lookups_proof,
            instruction_lookups_commitment,
            &mut transcript
        )
        .is_ok());
    };
    tasks.push((
        tracing::info_span!("HashR1CS"),
        Box::new(task) as Box<dyn FnOnce()>,
    ));

    tasks
}

fn fibonacci() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    let mut tasks = Vec::new();
    let task = || {
        use common::{path::JoltPaths, serializable::Serializable};
        compiler::cached_compile_example("fibonacci");

        let trace_location = JoltPaths::trace_path("fibonacci");
        let loaded_trace: Vec<common::RVTraceRow> =
            Vec::<common::RVTraceRow>::deserialize_from_file(&trace_location)
                .expect("deserialization failed");
        let bytecode_location = JoltPaths::bytecode_path("fibonacci");
        let bytecode = Vec::<ELFInstruction>::deserialize_from_file(&bytecode_location)
            .expect("deserialization failed");
        let bytecode_rows: Vec<ELFRow> = bytecode.clone().iter().map(ELFRow::from).collect();

        let converted_trace: Vec<RVTraceRow> = loaded_trace
            .into_iter()
            .map(|common| RVTraceRow::from_common(common))
            .collect();

        let bytecode_trace: Vec<ELFRow> = converted_trace
            .iter()
            .map(|row| row.to_bytecode_trace())
            .collect();

        let instructions_r1cs: Vec<RV32I> = converted_trace
            .clone()
            .into_iter()
            .flat_map(|row| {
                let instructions = row.to_jolt_instructions();
                if instructions.is_empty() {
                    vec![ADDInstruction::<32>(0_u64, 0_u64).into()]
                } else {
                    instructions
                }
            })
            .collect();

        let memory_trace_r1cs = converted_trace
            .clone()
            .into_iter()
            .flat_map(|row| row.to_ram_ops())
            .collect_vec();

        let circuit_flags = converted_trace
            .clone()
            .iter()
            .flat_map(|row| {
                let flags: Vec<Fr> = row.to_circuit_flags();
                // flags.reverse();
                flags.into_iter()
            })
            .collect::<Vec<_>>();

        let mut transcript = Transcript::new(b"Jolt transcript");
        <RV32IJoltVM as Jolt<'_, _, EdwardsProjective, C, M>>::prove_r1cs(
            instructions_r1cs,
            bytecode_rows,
            bytecode_trace,
            bytecode,
            memory_trace_r1cs,
            circuit_flags,
            &mut transcript,
        );

        // use common::{path::JoltPaths, serializable::Serializable, ELFInstruction};
        compiler::cached_compile_example("fibonacci");

        let trace_location = JoltPaths::trace_path("fibonacci");
        let loaded_trace: Vec<common::RVTraceRow> =
            Vec::<common::RVTraceRow>::deserialize_from_file(&trace_location)
                .expect("deserialization failed");
        let bytecode_location = JoltPaths::bytecode_path("fibonacci");
        let bytecode = Vec::<ELFInstruction>::deserialize_from_file(&bytecode_location)
            .expect("deserialization failed");
        let bytecode_rows = bytecode.iter().map(ELFRow::from).collect();

        let converted_trace: Vec<RVTraceRow> = loaded_trace
            .into_iter()
            .map(|common| RVTraceRow::from_common(common))
            .collect();

        let bytecode_trace: Vec<ELFRow> = converted_trace
            .iter()
            .map(|row| row.to_bytecode_trace())
            .collect();

        let instructions: Vec<RV32I> = converted_trace
            .clone()
            .into_iter()
            .flat_map(|row| row.to_jolt_instructions())
            .collect();

        // // TODO(JOLT-89): Encapsulate this logic elsewhere.
        // // Emulator sets register 0xb to 0x1020 upon initialization for some reason,
        // // something about Linux boot requiring it...
        // let mut memory_trace: Vec<MemoryOp> = vec![MemoryOp::Write(11, 4128)];
        let mut memory_trace: Vec<[MemoryOp; MEMORY_OPS_PER_INSTRUCTION]> = converted_trace
            .into_iter()
            .map(|row| row.to_ram_ops().try_into().unwrap())
            .collect();
        memory_trace.resize(
            memory_trace.len().next_power_of_two(),
            std::array::from_fn(|_| MemoryOp::no_op()),
        );

        let mut transcript = Transcript::new(b"Jolt transcript");
        let (bytecode_proof, _, bytecode_commitment) =
            <RV32IJoltVM as Jolt<'_, _, EdwardsProjective, C, M>>::prove_bytecode(
                bytecode_rows,
                bytecode_trace,
                &mut transcript,
            );
        let (memory_proof, _, memory_commitment) =
            <RV32IJoltVM as Jolt<'_, _, EdwardsProjective, C, M>>::prove_memory(
                bytecode,
                memory_trace,
                &mut transcript,
            );
        let (instruction_lookups_proof, _, instruction_lookups_commitment) =
            <RV32IJoltVM as Jolt<'_, _, EdwardsProjective, C, M>>::prove_instruction_lookups(
                instructions,
                &mut transcript,
            );

        let mut transcript = Transcript::new(b"Jolt transcript");
        assert!(
            RV32IJoltVM::verify_bytecode(bytecode_proof, bytecode_commitment, &mut transcript)
                .is_ok()
        );
        assert!(
            RV32IJoltVM::verify_memory(memory_proof, memory_commitment, &mut transcript).is_ok()
        );
        assert!(RV32IJoltVM::verify_instruction_lookups(
            instruction_lookups_proof,
            instruction_lookups_commitment,
            &mut transcript
        )
        .is_ok());
    };
    tasks.push((
        tracing::info_span!("FibonacciR1CS"),
        Box::new(task) as Box<dyn FnOnce()>,
    ));

    tasks
}

fn sparse_ml_poly_bind() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    let mut tasks = Vec::new();

    let log_size = 24;
    let mut sparse_poly = crate::subprotocols::sparse::bench::init_bind_bench::<Fr>(log_size, 0.93);
    let mut dense_poly = sparse_poly.clone().to_dense();

    let mut rng = test_rng();
    let r = Fr::rand(&mut rng);
    let task = move || {
        sparse_poly.bound_poly_var_top(&r);
    };

    tasks.push((
        tracing::info_span!("SparsePoly::bound_poly_var_top(24)"),
        Box::new(task) as Box<dyn FnOnce()>
    ));

    let task = move || {
        dense_poly.bound_poly_var_top_many_ones(&r);
    };

    tasks.push((
        tracing::info_span!("DensePoly::bound_poly_var_top_many_ones(24)"),
        Box::new(task) as Box<dyn FnOnce()>
    ));

    tasks
}

fn sparse_sumcheck() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    vec![(tracing::info_span!("BigSpan"), Box::new(move || run_sparse()))]
}
