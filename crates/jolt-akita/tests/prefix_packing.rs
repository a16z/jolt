#![expect(
    clippy::unwrap_used,
    reason = "tests exercise successful PCS operations"
)]

use jolt_akita::{AkitaField, AkitaScheme, AkitaSetupParams};
use jolt_openings::{CommitmentScheme, PrefixPackedClaims, PrefixPackedLayout};
use jolt_poly::Polynomial;
use jolt_transcript::{Blake2bTranscript, Transcript};

fn f(value: u64) -> AkitaField {
    AkitaField::from_u64(value)
}

fn polynomial(num_vars: usize, offset: u64) -> Polynomial<AkitaField> {
    Polynomial::new(
        (0..(1usize << num_vars))
            .map(|index| f(offset + index as u64))
            .collect(),
    )
}

#[test]
fn fixed_prefix_claim_opens_the_materialized_akita_polynomial() {
    let digest = [0x51; 32];
    let logical_num_vars = 12;
    let layout = PrefixPackedLayout::new(logical_num_vars, 4, [2_u8, 0, 1]).unwrap();
    let logical = [
        polynomial(logical_num_vars, 1),
        polynomial(logical_num_vars, 10_001),
        polynomial(logical_num_vars, 20_001),
    ];
    let mut physical = vec![f(0); 1 << layout.packed_num_vars()];
    for (id, polynomial) in layout.ids().iter().zip(&logical) {
        for (index, value) in polynomial.evaluations().iter().enumerate() {
            physical[layout.packed_index(id, index).unwrap()] = *value;
        }
    }
    let physical = Polynomial::new(physical);
    let logical_point = (0..logical_num_vars)
        .map(|index| f(37 + 4 * index as u64))
        .collect::<Vec<_>>();
    let evaluations = logical
        .iter()
        .map(|polynomial| polynomial.evaluate(&logical_point))
        .collect::<Vec<_>>();
    let claims = PrefixPackedClaims::new(digest, logical_point, evaluations);

    let (prover_setup, verifier_setup) = AkitaScheme::setup(AkitaSetupParams::dense_only(
        layout.packed_num_vars(),
        1,
        digest,
    ))
    .unwrap();
    let (commitment, hint) = AkitaScheme::commit(&physical, &prover_setup).unwrap();
    let mut prover_transcript = Blake2bTranscript::new(b"akita/fixed-prefix");
    let physical_claim = layout
        .reduce_claims(&claims, &mut prover_transcript)
        .unwrap();
    let proof = AkitaScheme::open(
        &physical,
        physical_claim.point.as_slice(),
        physical_claim.value,
        &prover_setup,
        Some(hint),
        &mut prover_transcript,
    )
    .unwrap();

    let mut verifier_transcript = Blake2bTranscript::new(b"akita/fixed-prefix");
    let verifier_claim = layout
        .reduce_claims(&claims, &mut verifier_transcript)
        .unwrap();
    AkitaScheme::verify(
        &commitment,
        verifier_claim.point.as_slice(),
        verifier_claim.value,
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
    )
    .unwrap();
}

#[test]
fn changed_fixed_prefix_statement_rejects_the_original_proof() {
    let digest = [0x52; 32];
    let logical_num_vars = 13;
    let layout = PrefixPackedLayout::new(logical_num_vars, 2, [0_u8, 1]).unwrap();
    let logical = [
        polynomial(logical_num_vars, 1),
        polynomial(logical_num_vars, 10_001),
    ];
    let physical = Polynomial::new(
        logical
            .iter()
            .flat_map(|polynomial| polynomial.evaluations().iter().copied())
            .collect(),
    );
    let logical_point = (0..logical_num_vars)
        .map(|index| f(7 + 4 * index as u64))
        .collect::<Vec<_>>();
    let mut evaluations = logical
        .iter()
        .map(|polynomial| polynomial.evaluate(&logical_point))
        .collect::<Vec<_>>();
    let claims = PrefixPackedClaims::new(digest, logical_point.clone(), evaluations.clone());
    let (prover_setup, verifier_setup) = AkitaScheme::setup(AkitaSetupParams::dense_only(
        layout.packed_num_vars(),
        1,
        digest,
    ))
    .unwrap();
    let (commitment, hint) = AkitaScheme::commit(&physical, &prover_setup).unwrap();
    let mut prover_transcript = Blake2bTranscript::new(b"akita/fixed-prefix-tamper");
    let physical_claim = layout
        .reduce_claims(&claims, &mut prover_transcript)
        .unwrap();
    let proof = AkitaScheme::open(
        &physical,
        physical_claim.point.as_slice(),
        physical_claim.value,
        &prover_setup,
        Some(hint),
        &mut prover_transcript,
    )
    .unwrap();

    let mut honest_verifier_transcript = Blake2bTranscript::new(b"akita/fixed-prefix-tamper");
    let honest_claim = layout
        .reduce_claims(&claims, &mut honest_verifier_transcript)
        .unwrap();
    AkitaScheme::verify(
        &commitment,
        honest_claim.point.as_slice(),
        honest_claim.value,
        &proof,
        &verifier_setup,
        &mut honest_verifier_transcript,
    )
    .unwrap();

    evaluations[1] += f(1);
    let changed = PrefixPackedClaims::new(digest, logical_point, evaluations);
    let mut verifier_transcript = Blake2bTranscript::new(b"akita/fixed-prefix-tamper");
    let changed_claim = layout
        .reduce_claims(&changed, &mut verifier_transcript)
        .unwrap();
    assert!(AkitaScheme::verify(
        &commitment,
        changed_claim.point.as_slice(),
        changed_claim.value,
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
    )
    .is_err());
}
