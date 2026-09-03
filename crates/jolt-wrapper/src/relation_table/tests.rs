#![expect(
    clippy::expect_used,
    reason = "fixture tests fail on invalid setup or proof data"
)]

use std::path::Path;

use bincode::config::standard;
use bincode::serde::{decode_from_slice, encode_to_vec};
use common::jolt_device::JoltDevice;
use jolt_crypto::{Bn254, Bn254G1, Pedersen};
use jolt_dory::DoryScheme;
use jolt_field::{Fr, One, Ring, Zero};
use jolt_hyperkzg::{HyperKZGScheme, HyperKZGVerifierSetup};
use jolt_sumcheck::prover::ProveRounds;
use jolt_verifier::{JoltProof, JoltVerifierPreprocessing};

use super::*;
use crate::profile::WrapperProfile;
use crate::relation::{build_relation, generate_witness, ScheduleEntry};
use crate::stream::VerifierCost;

type Pcs = DoryScheme;
type Vc = Pedersen<Bn254G1>;
type Proof = JoltProof<Pcs, Vc>;
type Preprocessing = JoltVerifierPreprocessing<Pcs, Vc>;

const FIXTURE: &str = "/Volumes/Dev/scratch/wrapper-fixtures/fibonacci_2_18_blake3.bin";

fn fixture() -> (Preprocessing, JoltDevice, Proof) {
    let bytes = std::fs::read(Path::new(FIXTURE)).expect("cached fibonacci fixture");
    decode_from_slice(&bytes, standard())
        .expect("decode cached fibonacci fixture")
        .0
}

#[test]
#[expect(clippy::indexing_slicing, reason = "fixed real-fixture dimensions")]
fn fibonacci_relation_table_exactness_stream_and_tampers() {
    let (preprocessing, public_io, proof) = fixture();
    let profile = WrapperProfile::new(&preprocessing, &proof).expect("supported profile");
    let relation = build_relation(&profile).expect("build relation");
    let relation_witness =
        generate_witness(&profile, &preprocessing, &public_io, &proof).expect("relation witness");
    let rows = 1 << 18;
    let mut table = RelationTable::from_relation(&relation, rows).expect("lower R1CS");
    assert_eq!(table.gate_rows(), 38_981);
    assert_eq!(
        table.cell_layout(),
        RelationCellLayout {
            absorbed_word_base: 38_981,
            absorbed_words: 1_222,
            challenge_base: 40_203,
            challenges: 376,
            dory_scalar_base: 40_704,
            dory_scalars: 175,
            dory_scalar_capacity: 256,
        }
    );
    let beta = Fr::from_u64(0x1234_5678);
    let gamma = Fr::from_u64(0x9abc_def0);
    let table_witness = table
        .witness(&relation_witness.values, beta, gamma)
        .expect("lowered witness");
    assert!(relation
        .matrices
        .check_witness(&relation_witness.values)
        .is_ok());
    table
        .check_witness(&table_witness, beta, gamma)
        .expect("gate and copy checks");

    let challenge_variable = relation
        .link
        .schedule
        .iter()
        .find_map(|entry| match entry {
            ScheduleEntry::Squeeze { var, .. } => Some(*var),
            _ => None,
        })
        .expect("challenge wire");
    let mut bad_assignment = relation_witness.values.clone();
    bad_assignment[challenge_variable.index()] += Fr::one();
    assert!(relation.matrices.check_witness(&bad_assignment).is_err());
    assert!(table.wire_witness(&bad_assignment).is_err());

    let mut bad_wire = table
        .witness(&relation_witness.values, beta, gamma)
        .expect("lowered witness");
    bad_wire.columns[0][0] += Fr::one();
    assert!(table.check_witness(&bad_wire, beta, gamma).is_err());

    let old_sigma = table.fixed[SIGMA_A][0];
    table.fixed[SIGMA_A][0] += Fr::one();
    assert!(table.check_witness(&table_witness, beta, gamma).is_err());
    table.fixed[SIGMA_A][0] = old_sigma;

    let old_constant = table.fixed[Q_C][0];
    table.fixed[Q_C][0] += Fr::one();
    assert!(table.check_witness(&table_witness, beta, gamma).is_err());
    table.fixed[Q_C][0] = old_constant;

    let rho = Fr::from_u64(41);
    let scalar_link = DoryScalarLink::new(rows, table.cell_layout(), rho);
    let mut scalar_prover = scalar_link.prover(&table_witness);
    let mut direct = Fr::zero();
    let mut power = Fr::one();
    for (_, variable) in &relation.link.dory.scalars {
        direct += power * relation_witness.values[variable.index()];
        power *= rho;
    }
    assert_eq!(scalar_prover.input_claim(), direct);
    let mut scalar_claim = direct;
    let mut scalar_point = Vec::with_capacity(18);
    let mut bind = None;
    for round in 0..18 {
        let polynomial = scalar_prover
            .prove_round(bind, round, scalar_claim)
            .expect("scalar-link round");
        let challenge = Fr::from_u64((round + 43) as u64);
        scalar_claim = polynomial.evaluate(challenge);
        scalar_point.push(challenge);
        bind = Some(challenge);
    }
    scalar_prover
        .finish_rounds(bind.expect("scalar-link challenge"))
        .expect("finish scalar-link rounds");
    assert_eq!(
        scalar_claim,
        scalar_link.final_claim(&scalar_point, scalar_prover.wire_claim())
    );
    let mut scalar_cost = VerifierCost::default();
    assert_eq!(
        scalar_claim,
        scalar_link.final_claim_observed(
            &scalar_point,
            scalar_prover.wire_claim(),
            &mut scalar_cost,
        )
    );
    assert_eq!(scalar_cost.fr_mul, 34);

    let pcs_setup = HyperKZGScheme::<Bn254>::setup_from_secret(
        Fr::from_u64(0x5eed),
        rows * 16,
        Bn254::g1_generator(),
        Bn254::g2_generator(),
    );
    let verifier_setup = HyperKZGVerifierSetup::from(&pcs_setup);
    let (prover_key, verifier_key) = setup(
        table,
        profile.digest().expect("profile digest"),
        16,
        &pcs_setup,
    )
    .expect("relation table key");
    let table_proof =
        prove(&prover_key, &relation_witness.values, &pcs_setup).expect("prove relation table");
    let cost = verify(&verifier_key, &table_proof, &verifier_setup).expect("verify relation table");
    let bincode_bytes = encode_to_vec(&table_proof, standard())
        .expect("serialize relation table proof")
        .len();
    assert_eq!(table_proof.payload_bytes(), 4_896);
    assert_eq!(bincode_bytes, 4_959);
    assert_eq!(
        cost,
        VerifierCost {
            ec_mul: 87,
            ec_add: 86,
            pairing_pairs: 8,
            fr_mul: 7_364,
            fr_inv: 58,
            keccak: 326,
        }
    );

    let mut claim_tamper = table_proof.clone();
    claim_tamper.column_claims[WIRE_A] += Fr::one();
    assert!(verify(&verifier_key, &claim_tamper, &verifier_setup).is_err());
    let mut copy_tamper = table_proof.clone();
    copy_tamper.column_claims[SIGMA_A] += Fr::one();
    assert!(verify(&verifier_key, &copy_tamper, &verifier_setup).is_err());
    let mut challenge_tamper = table_proof.clone();
    challenge_tamper.wire_commitments[0] = table_proof.helper_commitments[0];
    assert!(verify(&verifier_key, &challenge_tamper, &verifier_setup).is_err());

    assert_eq!(verifier_key.layout.group_count, 3);
}

#[test]
fn real_shape_copy_link_binds_challenge_value() {
    let rows = 1 << 18;
    let mut selectors = std::array::from_fn(|_| vec![Fr::zero(); rows]);
    let mut ids = std::array::from_fn(|_| vec![Fr::zero(); rows]);
    for row in 0..175 {
        selectors[0][row] = Fr::one();
        ids[0][row] = Fr::from_u64(row as u64);
    }
    let left = CopyLinkSide::new(selectors.clone(), ids.clone()).expect("left side");
    let right = CopyLinkSide::new(selectors, ids).expect("right side");
    let link = CopyLink::new(left, right).expect("copy link");
    let mut left_values = std::array::from_fn(|_| vec![Fr::zero(); rows]);
    for row in 0..175 {
        left_values[0][row] = Fr::from_u64((3 * row + 99) as u64);
    }
    let right_values = left_values.clone();
    let beta = Fr::from_u64(17);
    let gamma = Fr::from_u64(31);
    let witness = link
        .witness(left_values.clone(), right_values.clone(), beta, gamma)
        .expect("copy witness");
    link.check(&witness, beta, gamma).expect("copy link check");
    let mut right_bad = right_values;
    right_bad[0][37] += Fr::one();
    let bad = link
        .witness(left_values, right_bad, beta, gamma)
        .expect("bad copy witness");
    assert!(link.check(&bad, beta, gamma).is_err());
}
