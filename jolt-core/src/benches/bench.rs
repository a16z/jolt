use crate::jolt::instruction::xor::XORInstruction;
use crate::lasso::surge::Surge;
use ark_curve25519::{EdwardsProjective, Fr};
use ark_std::{log2, test_rng};
use merlin::Transcript;
use rand_chacha::rand_core::RngCore;

#[derive(Debug, Clone, clap::ValueEnum)]
pub enum BenchType {
  JoltDemo,
  Halo2Comparison,
}

#[allow(unreachable_patterns)] // good errors on new BenchTypes
pub fn benchmarks(bench_type: BenchType) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
  match bench_type {
    BenchType::JoltDemo => jolt_demo_benchmarks(),
    BenchType::Halo2Comparison => halo2_comparison_benchmarks(),
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
