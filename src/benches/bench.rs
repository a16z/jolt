use crate::jolt::instruction::xor::XORInstruction;
use crate::lasso::surge::SurgeProof;
use ark_curve25519::EdwardsProjective;
use ark_std::{test_rng, log2};
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
  vec![
    (tracing::info_span!("EQ(2^10)"), random_surge_test(/* num_ops */ 1 << 10, /* C= */ 8, /* M= */ 1 << 16)),
    (tracing::info_span!("EQ(2^12)"), random_surge_test(/* num_ops */ 1 << 12, /* C= */ 8, /* M= */ 1 << 16)),
    (tracing::info_span!("EQ(2^14)"), random_surge_test(/* num_ops */ 1 << 14, /* C= */ 8, /* M= */ 1 << 16)),
    (tracing::info_span!("EQ(2^16)"), random_surge_test(/* num_ops */ 1 << 16, /* C= */ 8, /* M= */ 1 << 16)),
    (tracing::info_span!("EQ(2^18)"), random_surge_test(/* num_ops */ 1 << 18, /* C= */ 8, /* M= */ 1 << 16)),
    (tracing::info_span!("EQ(2^20)"), random_surge_test(/* num_ops */ 1 << 20, /* C= */ 8, /* M= */ 1 << 16)),
    (tracing::info_span!("EQ(2^22)"), random_surge_test(/* num_ops */ 1 << 22, /* C= */ 8, /* M= */ 1 << 16)),
  ]
}

fn halo2_comparison_benchmarks() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
  vec![
    (tracing::info_span!("EQ(2^10)"), random_surge_test(/* num_ops */ 1 << 10, /* C= */ 1, /* M= */ 1 << 16)),
    (tracing::info_span!("EQ(2^12)"), random_surge_test(/* num_ops */ 1 << 12, /* C= */ 1, /* M= */ 1 << 16)),
    (tracing::info_span!("EQ(2^14)"), random_surge_test(/* num_ops */ 1 << 14, /* C= */ 1, /* M= */ 1 << 16)),
    (tracing::info_span!("EQ(2^16)"), random_surge_test(/* num_ops */ 1 << 16, /* C= */ 1, /* M= */ 1 << 16)),
    (tracing::info_span!("EQ(2^18)"), random_surge_test(/* num_ops */ 1 << 18, /* C= */ 1, /* M= */ 1 << 16)),
    (tracing::info_span!("EQ(2^20)"), random_surge_test(/* num_ops */ 1 << 20, /* C= */ 1, /* M= */ 1 << 16)),
    (tracing::info_span!("EQ(2^22)"), random_surge_test(/* num_ops */ 1 << 22, /* C= */ 1, /* M= */ 1 << 16)),
    (tracing::info_span!("EQ(2^24)"), random_surge_test(/* num_ops */ 1 << 24, /* C= */ 1, /* M= */ 1 << 16)),
  ]
}

fn random_surge_test(num_ops: usize, C: usize, M: usize) -> Box<dyn FnOnce()> {
  let mut rng = test_rng();

  let mut ops: Vec<XORInstruction> = Vec::with_capacity(num_ops);
  let operand_max: u64 = (1 << ((log2(M) as usize) * C - 1)).try_into().unwrap();
  for _ in 0..num_ops {
    let a = rng.next_u64() % operand_max;
    let b = rng.next_u64() % operand_max;
    ops.push(XORInstruction(a, b));
  }

  let func = move || {
    let mut prover_transcript = Transcript::new(b"test_transcript");
    let proof: SurgeProof<EdwardsProjective, _> = SurgeProof::prove(ops.clone(), C, M, &mut prover_transcript);

    let mut verifier_transcript = Transcript::new(b"test_transcript");
    proof.verify(&mut verifier_transcript).expect("should work");
  };

  Box::new(func)
}
