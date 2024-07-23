use crate::field::binius::BiniusField;
use crate::field::JoltField;
use crate::host;
use crate::jolt::vm::rv32i_vm::{RV32IJoltVM, C, M};
use crate::jolt::vm::Jolt;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::commitment::hyperkzg::HyperKZG;
use crate::poly::commitment::hyrax::HyraxScheme;
use crate::poly::commitment::mock::MockCommitScheme;
use crate::poly::commitment::zeromorph::Zeromorph;
use crate::r1cs::spartan;
use crate::subprotocols::sumcheck::{BiniusSpartanSumcheckBackend, CurveSpartanSumcheckBackend};
use ark_bn254::{Bn254, Fr, G1Projective};
use ark_std::rand::Rng;
use ark_std::test_rng;
use serde::Serialize;

#[derive(Debug, Copy, Clone, clap::ValueEnum)]
pub enum PCSType {
    Hyrax,
    Zeromorph,
    HyperKZG,
    Binius,
}

#[derive(Debug, Copy, Clone, clap::ValueEnum)]
pub enum BenchType {
    Fibonacci,
    Sha2,
    Sha3,
    Sha2Chain,
    Spartan
}

#[allow(unreachable_patterns)] // good errors on new BenchTypes
pub fn benchmarks(
    pcs_type: PCSType,
    bench_type: BenchType,
    _num_cycles: Option<usize>,
    _memory_size: Option<usize>,
    _bytecode_size: Option<usize>,
) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    match pcs_type {
        PCSType::Hyrax => match bench_type {
            BenchType::Sha2 => sha2::<Fr, HyraxScheme<G1Projective>>(),
            BenchType::Sha3 => sha3::<Fr, HyraxScheme<G1Projective>>(),
            BenchType::Sha2Chain => sha2chain::<Fr, HyraxScheme<G1Projective>>(),
            BenchType::Fibonacci => fibonacci::<Fr, HyraxScheme<G1Projective>>(),
            BenchType::Spartan => spartan(false),
            _ => panic!("BenchType does not have a mapping"),
        },
        PCSType::Zeromorph => match bench_type {
            BenchType::Sha2 => sha2::<Fr, Zeromorph<Bn254>>(),
            BenchType::Sha3 => sha3::<Fr, Zeromorph<Bn254>>(),
            BenchType::Sha2Chain => sha2chain::<Fr, Zeromorph<Bn254>>(),
            BenchType::Fibonacci => fibonacci::<Fr, Zeromorph<Bn254>>(),
            _ => panic!("BenchType does not have a mapping"),
        },
        PCSType::HyperKZG => match bench_type {
            BenchType::Sha2 => sha2::<Fr, HyperKZG<Bn254>>(),
            BenchType::Sha3 => sha3::<Fr, HyperKZG<Bn254>>(),
            BenchType::Sha2Chain => sha2chain::<Fr, HyperKZG<Bn254>>(),
            BenchType::Fibonacci => fibonacci::<Fr, HyperKZG<Bn254>>(),
            _ => panic!("BenchType does not have a mapping"),
        },
        PCSType::Binius => match bench_type {
            BenchType::Spartan => spartan(true),
            _ => panic!("BenchType does not have a mapping"),
        }
        _ => panic!("PCS Type does not have a mapping"),
    }
}

fn fibonacci<F, PCS>() -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    prove_example::<u32, PCS, F>("fibonacci-guest", &9u32)
}

fn sha2<F, PCS>() -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    prove_example::<Vec<u8>, PCS, F>("sha2-guest", &vec![5u8; 2048])
}

fn sha3<F, PCS>() -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    prove_example::<Vec<u8>, PCS, F>("sha3-guest", &vec![5u8; 2048])
}


fn spartan(binius: bool) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    const LOG_LEN: usize = 28;
    const LEN: usize = 1 << LOG_LEN;
    const SPARSITY: f64 = 0.35; // pct non-zeros

    let bz: Vec<u64> = vec![1; LEN];
    let mut az: Vec<u64> = Vec::new();
    let mut rng = test_rng();
    for i in 0..LEN {
        let prob: f64 = rng.gen();
        if prob < SPARSITY {
            az.push(i as u64);
        } else {
            az.push(0);
        }
    }
    let cz = az.clone();

    if binius {
        use binius_field::BinaryField128bPolyval as BF;
        type Field = BiniusField<BF>;

        let task = move || {
            spartan::bench::cubic_sumcheck::<Field, MockCommitScheme<Field>, BiniusSpartanSumcheckBackend>(az, bz, cz);
        };

        vec![(tracing::info_span!("Binius"), Box::new(task))]
    } else {
        type Field = ark_bn254::Fr;

        let task = move || {
            spartan::bench::cubic_sumcheck::<Field, MockCommitScheme<Field>, CurveSpartanSumcheckBackend>(az, bz, cz);
            // spartan::bench::cubic_sumcheck::<Field, MockCommitScheme<Field>, BiniusSpartanSumcheckBackend>(az, bz, cz);
        };

        vec![(tracing::info_span!("Curve"), Box::new(task))]
    }
}

#[allow(dead_code)]
fn serialize_and_print_size(name: &str, item: &impl ark_serialize::CanonicalSerialize) {
    use std::fs::File;
    let mut file = File::create("temp_file").unwrap();
    item.serialize_compressed(&mut file).unwrap();
    let file_size_bytes = file.metadata().unwrap().len();
    let file_size_kb = file_size_bytes as f64 / 1024.0;
    let file_size_mb = file_size_kb / 1024.0;
    println!("{:<30} : {:.3} MB", name, file_size_mb);
}

fn prove_example<T: Serialize, PCS, F>(
    example_name: &str,
    input: &T,
) -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    let mut tasks = Vec::new();
    let mut program = host::Program::new(example_name);
    program.set_input(input);

    let task = move || {
        let (bytecode, memory_init) = program.decode();
        let (io_device, trace, circuit_flags) = program.trace();

        let preprocessing: crate::jolt::vm::JoltPreprocessing<F, PCS> =
            RV32IJoltVM::preprocess(bytecode.clone(), memory_init, 1 << 20, 1 << 20, 1 << 22);

        let (jolt_proof, jolt_commitments) = <RV32IJoltVM as Jolt<_, PCS, C, M>>::prove(
            io_device,
            trace,
            circuit_flags,
            preprocessing.clone(),
        );

        println!("Proof sizing:");
        serialize_and_print_size("jolt_commitments", &jolt_commitments);
        serialize_and_print_size("jolt_proof", &jolt_proof);
        serialize_and_print_size(" jolt_proof.r1cs", &jolt_proof.r1cs);
        serialize_and_print_size(" jolt_proof.bytecode", &jolt_proof.bytecode);
        serialize_and_print_size(
            " jolt_proof.read_write_memory",
            &jolt_proof.read_write_memory,
        );
        serialize_and_print_size(
            " jolt_proof.instruction_lookups",
            &jolt_proof.instruction_lookups,
        );

        let verification_result = RV32IJoltVM::verify(preprocessing, jolt_proof, jolt_commitments);
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    };

    tasks.push((
        tracing::info_span!("Example_E2E"),
        Box::new(task) as Box<dyn FnOnce()>,
    ));

    tasks
}

fn sha2chain<F, PCS>() -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    let mut tasks = Vec::new();
    let mut program = host::Program::new("sha2-chain-guest");
    program.set_input(&[5u8; 32]);
    program.set_input(&1024u32);

    let task = move || {
        let (bytecode, memory_init) = program.decode();
        let (io_device, trace, circuit_flags) = program.trace();

        let preprocessing: crate::jolt::vm::JoltPreprocessing<F, PCS> =
            RV32IJoltVM::preprocess(bytecode.clone(), memory_init, 1 << 20, 1 << 20, 1 << 22);

        let (jolt_proof, jolt_commitments) = <RV32IJoltVM as Jolt<_, PCS, C, M>>::prove(
            io_device,
            trace,
            circuit_flags,
            preprocessing.clone(),
        );
        let verification_result = RV32IJoltVM::verify(preprocessing, jolt_proof, jolt_commitments);
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    };

    tasks.push((
        tracing::info_span!("Example_E2E"),
        Box::new(task) as Box<dyn FnOnce()>,
    ));

    tasks
}
