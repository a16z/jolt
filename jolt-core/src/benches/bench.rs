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
use crate::jolt::instruction::slt::SLTInstruction;
use crate::jolt::instruction::sra::SRAInstruction;
use crate::jolt::instruction::srl::SRLInstruction;
use crate::jolt::instruction::sub::SUBInstruction;
use crate::jolt::vm::instruction_lookups::InstructionLookupsProof;
use crate::jolt::vm::rv32i_vm::{RV32IJoltVM, RV32I};
use crate::jolt::vm::Jolt;
use crate::lasso::surge::Surge;
use crate::utils::math::Math;
use crate::{jolt::instruction::xor::XORInstruction, utils::gen_random_point};
use ark_curve25519::{EdwardsProjective, Fr};
use ark_std::{log2, test_rng};
use merlin::Transcript;
use rand_chacha::rand_core::RngCore;

#[derive(Debug, Clone, clap::ValueEnum)]
pub enum BenchType {
  JoltDemo,
  Halo2Comparison,
  RV32,
}

#[allow(unreachable_patterns)] // good errors on new BenchTypes
pub fn benchmarks(bench_type: BenchType) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
  match bench_type {
    BenchType::JoltDemo => jolt_demo_benchmarks(),
    BenchType::Halo2Comparison => halo2_comparison_benchmarks(),
    BenchType::RV32 => rv32i_lookup_benchmarks(),
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
  let mut rng = test_rng();

  let mut ops: Vec<RV32I> = vec![
    RV32I::ADD(ADDInstruction(rng.next_u32() as u64, rng.next_u32() as u64)),
    RV32I::AND(ANDInstruction(rng.next_u32() as u64, rng.next_u32() as u64)),
    RV32I::BEQ(BEQInstruction(rng.next_u32() as u64, rng.next_u32() as u64)),
    RV32I::BGE(BGEInstruction(rng.next_u32() as i64, rng.next_u32() as i64)),
    RV32I::BGEU(BGEUInstruction(
      rng.next_u32() as u64,
      rng.next_u32() as u64,
    )),
    RV32I::BLT(BLTInstruction(rng.next_u32() as i64, rng.next_u32() as i64)),
    RV32I::BLTU(BLTUInstruction(
      rng.next_u32() as u64,
      rng.next_u32() as u64,
    )),
    RV32I::BNE(BNEInstruction(rng.next_u32() as u64, rng.next_u32() as u64)),
    RV32I::JAL(JALInstruction(rng.next_u32() as u64, rng.next_u32() as u64)),
    RV32I::JALR(JALRInstruction(
      rng.next_u32() as u64,
      rng.next_u32() as u64,
    )),
    RV32I::OR(ORInstruction(rng.next_u32() as u64, rng.next_u32() as u64)),
    RV32I::SLL(SLLInstruction(rng.next_u32() as u64, rng.next_u32() as u64)),
    RV32I::SRA(SRAInstruction(rng.next_u32() as u64, rng.next_u32() as u64)),
    RV32I::SRL(SRLInstruction(rng.next_u32() as u64, rng.next_u32() as u64)),
    RV32I::SUB(SUBInstruction(rng.next_u32() as u64, rng.next_u32() as u64)),
    RV32I::XOR(XORInstruction(rng.next_u32() as u64, rng.next_u32() as u64)),
  ];
  for _ in 0..10 {
    ops.extend(ops.clone());
  }

  let r: Vec<Fr> = gen_random_point::<Fr>(ops.len().log_2());
  let work = Box::new(|| {
    let r: Vec<Fr> = gen_random_point::<Fr>(ops.len().log_2());
    let mut prover_transcript = Transcript::new(b"example");
    let proof: InstructionLookupsProof<Fr, EdwardsProjective> =
      RV32IJoltVM::prove_instruction_lookups(ops, r.clone(), &mut prover_transcript);
    let mut verifier_transcript = Transcript::new(b"example");
    assert!(RV32IJoltVM::verify_instruction_lookups(proof, r, &mut verifier_transcript).is_ok());
  });
  vec![(tracing::info_span!("RV32IM"), work)]
}

fn random_surge_test<const C: usize, const M: usize>(num_ops: usize) -> Box<dyn FnOnce()> {
  let mut rng = test_rng();

  let mut ops: Vec<XORInstruction> = Vec::with_capacity(num_ops);
  for _ in 0..num_ops {
    let a = rng.next_u32();
    let b = rng.next_u32();
    ops.push(XORInstruction(a as u64, b as u64));
  }

  let func = move || {
    let mut prover_transcript = Transcript::new(b"test_transcript");
    let surge = <Surge<Fr, EdwardsProjective, XORInstruction, C, M>>::new(ops.clone());
    let proof = surge.prove(&mut prover_transcript);

    let mut verifier_transcript = Transcript::new(b"test_transcript");
    <Surge<Fr, EdwardsProjective, XORInstruction, C, M>>::verify(proof, &mut verifier_transcript)
      .expect("should work");
  };

  Box::new(func)
}
