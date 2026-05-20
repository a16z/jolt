mod support;

use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use support::*;

#[test]
fn r1cs_pipeline_satisfies_three_stage_claim_chain() {
    let mut prover = SumcheckTestProver::new(ChaCha20Rng::from_seed([5; 32]));
    let (stage1, stage2, stage3, values) = generated_deep_triple(&mut prover);

    assert!(build_deep_relation(&stage1, &stage2, &stage3, &values).is_ok());
}

#[test]
fn r1cs_pipeline_rejects_late_stage_mismatch() {
    let mut prover = SumcheckTestProver::new(ChaCha20Rng::from_seed([6; 32]));
    let (stage1, stage2, stage3, values) = generated_deep_triple(&mut prover);
    let mut tampered_stage3 = stage3.clone();
    tampered_stage3.proof = stage2.proof.clone();

    assert!(build_deep_relation(&stage1, &stage2, &tampered_stage3, &values).is_err());
}
