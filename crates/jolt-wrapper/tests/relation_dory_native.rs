//! The relation's named Dory scalars against the native deferred check: the
//! final pairing equation of `dory::verify` is re-evaluated with every scalar
//! taken from the witness by `DoryScalar` name and applied to the real setup
//! constants, commitments and proof elements. Only the native pairing of
//! scalar to base makes it hold, so a mis-indexed link is a hard failure.
//!
//! `cargo nextest run -p jolt-wrapper --features prover-fixtures --cargo-quiet --test relation_dory_native --no-capture`

#![cfg(feature = "prover-fixtures")]
#![expect(clippy::expect_used)]

use std::collections::HashMap;
use std::path::PathBuf;

use ark_bn254::{Bn254, Fq12, Fr as ArkFrInner};
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_ff::PrimeField;
use common::jolt_device::JoltDevice;
use dory::backends::arkworks::{ArkFr, ArkG1, ArkG2, ArkGT};
use dory::primitives::arithmetic::{Field, Group};
use jolt_claims::protocols::jolt::geometry::committed_openings::final_opening_polynomial_order;
use jolt_claims::protocols::jolt::JoltCommittedPolynomial;
use jolt_crypto::{Bn254G1, Pedersen};
use jolt_dory::DoryScheme;
use jolt_field::{CanonicalBytes, Fr};
use jolt_verifier::proof::JoltProof;
use jolt_verifier::stages::formula_dimensions_from_parts;
use jolt_verifier::JoltVerifierPreprocessing;
use jolt_wrapper::profile::WrapperProfile;
use jolt_wrapper::relation::{generate_witness, DoryLinks, DoryScalar, Witness};

type Pcs = DoryScheme;
type Vc = Pedersen<Bn254G1>;
type Proof = JoltProof<Pcs, Vc>;
type VerifierPreprocessing = JoltVerifierPreprocessing<Pcs, Vc>;

const CACHE: &str = "/Volumes/Dev/scratch/wrapper-fixtures/fibonacci_2_18_blake3.bin";

fn fixture() -> (VerifierPreprocessing, JoltDevice, Proof) {
    let bytes = std::fs::read(PathBuf::from(CACHE))
        .expect("cached fibonacci 2^18 fixture (run relation_fixture first)");
    let (fixture, _): ((VerifierPreprocessing, JoltDevice, Proof), usize) =
        bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
            .expect("decode cached fixture");
    fixture
}

fn ark(value: Fr) -> ArkFr {
    let mut bytes = [0u8; 32];
    value.to_bytes_le(&mut bytes);
    ArkFr(ArkFrInner::from_le_bytes_mod_order(&bytes))
}

fn gt(value: &jolt_dory::DoryCommitment) -> ArkGT {
    ArkGT(PairingOutput(Fq12::from(value.0)))
}

/// The named scalars of the witness, as arkworks field elements.
struct Scalars(HashMap<DoryScalar, ArkFr>);

impl Scalars {
    fn new(links: &DoryLinks, witness: &Witness) -> Self {
        Self(
            links
                .scalars
                .iter()
                .map(|(scalar, variable)| (scalar.clone(), ark(witness.values[variable.index()])))
                .collect(),
        )
    }

    fn get(&self, scalar: DoryScalar) -> ArkFr {
        assert!(
            self.0.contains_key(&scalar),
            "missing Dory scalar {scalar:?}"
        );
        self.0[&scalar]
    }
}

/// `Σ scalar · base` over GT.
fn gt_sum(terms: impl IntoIterator<Item = (ArkFr, ArkGT)>) -> ArkGT {
    terms
        .into_iter()
        .fold(ArkGT::identity(), |acc, (scalar, base)| {
            acc + base.scale(&scalar)
        })
}

/// The deferred right-hand side with the setup index `delta_index(k)` applied
/// to the `Delta1R(k)`/`Delta2R(k)` links (identity for the real equation).
fn deferred_rhs(
    scalars: &Scalars,
    proof: &dory::backends::arkworks::ArkDoryProof,
    setup: &dory::setup::VerifierSetup<dory::backends::arkworks::BN254>,
    commitments: &[ArkGT],
    delta_index: impl Fn(usize) -> usize,
) -> ArkGT {
    let sigma = proof.sigma;
    let mut terms = vec![(ArkFr::one(), proof.vmv_message.c)];
    for (index, commitment) in commitments.iter().enumerate() {
        terms.push((
            scalars.get(DoryScalar::CommitmentWeight(index)),
            *commitment,
        ));
    }
    terms.push((scalars.get(DoryScalar::D2Init), proof.vmv_message.d2));
    for j in 0..sigma {
        let first = &proof.first_messages[j];
        let second = &proof.second_messages[j];
        terms.extend([
            (scalars.get(DoryScalar::Alpha(j)), second.c_plus),
            (scalars.get(DoryScalar::AlphaInv(j)), second.c_minus),
            (scalars.get(DoryScalar::UAlpha(j)), first.d1_left),
            (scalars.get(DoryScalar::U(j)), first.d1_right),
            (scalars.get(DoryScalar::VAlphaInv(j)), first.d2_left),
            (scalars.get(DoryScalar::V(j)), first.d2_right),
        ]);
    }
    for k in 0..=sigma {
        terms.push((scalars.get(DoryScalar::Chi(k)), setup.chi[k]));
    }
    for k in 1..=sigma {
        terms.push((
            scalars.get(DoryScalar::Delta1R(k)),
            setup.delta_1r[delta_index(k)],
        ));
        terms.push((
            scalars.get(DoryScalar::Delta2R(k)),
            setup.delta_2r[delta_index(k)],
        ));
    }
    terms.push((scalars.get(DoryScalar::Ht), setup.ht));
    gt_sum(terms)
}

/// The four-pairing left-hand side with every group scalar taken from the
/// named links.
fn pairing_lhs(
    scalars: &Scalars,
    proof: &dory::backends::arkworks::ArkDoryProof,
    setup: &dory::setup::VerifierSetup<dory::backends::arkworks::BN254>,
) -> ArkGT {
    let sigma = proof.sigma;
    let mut e1_acc = proof.vmv_message.e1;
    let mut e2_acc = setup.g2_0.scale(&scalars.get(DoryScalar::Evaluation));
    for j in 0..sigma {
        let first = &proof.first_messages[j];
        let second = &proof.second_messages[j];
        e1_acc = e1_acc
            + first.e1_beta.scale(&scalars.get(DoryScalar::Beta(j)))
            + second.e1_plus.scale(&scalars.get(DoryScalar::Alpha(j)))
            + second.e1_minus.scale(&scalars.get(DoryScalar::AlphaInv(j)));
        e2_acc = e2_acc
            + first.e2_beta.scale(&scalars.get(DoryScalar::BetaInv(j)))
            + second.e2_plus.scale(&scalars.get(DoryScalar::Alpha(j)))
            + second.e2_minus.scale(&scalars.get(DoryScalar::AlphaInv(j)));
    }
    let final_message = proof
        .final_message
        .as_ref()
        .expect("transparent final message");
    let neg = |scalar: ArkFr| ArkFr(-scalar.0);
    let a: ArkG1 = final_message.e1 + setup.g1_0.scale(&scalars.get(DoryScalar::D));
    let b: ArkG2 = final_message.e2 + setup.g2_0.scale(&scalars.get(DoryScalar::DInv));
    let b_prime = e2_acc.scale(&neg(scalars.get(DoryScalar::Gamma)))
        + setup
            .g2_0
            .scale(&scalars.get(DoryScalar::PairingG2ZeroScalar));
    let a_prime = e1_acc.scale(&neg(scalars.get(DoryScalar::GammaInv)))
        + setup
            .g1_0
            .scale(&scalars.get(DoryScalar::PairingG1ZeroScalar));
    let a_double = proof
        .vmv_message
        .e1
        .scale(&scalars.get(DoryScalar::DSquared));
    ArkGT(Bn254::multi_pairing(
        [a.0, setup.h1.0, a_prime.0, a_double.0],
        [b.0, b_prime.0, setup.h2.0, setup.g2_0.0],
    ))
}

#[test]
fn named_dory_scalars_satisfy_the_native_deferred_check() {
    let (preprocessing, public_io, proof) = fixture();
    let profile = WrapperProfile::new(&preprocessing, &proof).expect("profile");
    let witness = generate_witness(&profile, &preprocessing, &public_io, &proof).expect("witness");
    let relation = jolt_wrapper::relation::build_relation(&profile).expect("relation");
    let scalars = Scalars::new(&relation.link.dory, &witness);
    let dory_proof = &proof.joint_opening_proof.0;
    let setup = &preprocessing.pcs_setup.0 .0;
    assert_eq!(relation.link.dory.sigma, dory_proof.sigma);

    let formula = formula_dimensions_from_parts(
        profile.one_hot_config,
        profile.log_t,
        profile.bytecode_len(),
        profile.ram_k(),
        jolt_claims::protocols::jolt::JoltRelationId::InstructionReadRaf,
    )
    .expect("formula dimensions");
    let commitments: Vec<ArkGT> =
        final_opening_polynomial_order(formula.ra_layout, false, false, None)
            .into_iter()
            .map(|polynomial| {
                let commitment = match polynomial {
                    JoltCommittedPolynomial::RamInc => Some(&proof.commitments.ram_inc),
                    JoltCommittedPolynomial::RdInc => Some(&proof.commitments.rd_inc),
                    JoltCommittedPolynomial::InstructionRa(i) => {
                        Some(&proof.commitments.instruction_ra[i])
                    }
                    JoltCommittedPolynomial::BytecodeRa(i) => {
                        Some(&proof.commitments.bytecode_ra[i])
                    }
                    JoltCommittedPolynomial::RamRa(i) => Some(&proof.commitments.ram_ra[i]),
                    _ => None,
                };
                gt(commitment.expect("trace commitment"))
            })
            .collect();

    let lhs = pairing_lhs(&scalars, dory_proof, setup);
    let rhs = deferred_rhs(&scalars, dory_proof, setup, &commitments, |k| k);
    assert_eq!(
        lhs, rhs,
        "named scalars × native bases must reproduce the accepting equation"
    );

    // Negative control: pairing the Delta links with the neighbouring setup
    // constant (the `σ − 1 − j` indexing) breaks the equation.
    let shifted = deferred_rhs(&scalars, dory_proof, setup, &commitments, |k| k - 1);
    assert_ne!(lhs, shifted, "a mis-indexed Delta link must be rejected");
    // Negative control: the Delta1R/Delta2R scalars are not interchangeable.
    let swapped_terms = {
        let sigma = dory_proof.sigma;
        let base = deferred_rhs(&scalars, dory_proof, setup, &commitments, |k| k);
        let mut fix = ArkGT::identity();
        for k in 1..=sigma {
            let d1 = scalars.get(DoryScalar::Delta1R(k));
            let d2 = scalars.get(DoryScalar::Delta2R(k));
            fix = fix
                + setup.delta_1r[k].scale(&ArkFr(d2.0 - d1.0))
                + setup.delta_2r[k].scale(&ArkFr(d1.0 - d2.0));
        }
        base + fix
    };
    assert_ne!(lhs, swapped_terms);
}
