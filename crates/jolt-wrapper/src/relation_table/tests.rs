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
use jolt_poly::MultilinearPoly;
use jolt_sumcheck::prover::ProveRounds;
use jolt_verifier::{JoltProof, JoltVerifierPreprocessing};

use super::*;
use crate::limb_table::adapter::from_jolt;
use crate::limb_table::lookup::link_weights;
use crate::limb_table::schedule::build as build_limb_layout;
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
    assert_eq!(table.gate_rows(), 38_977);
    assert_eq!(
        table.cell_layout(),
        RelationCellLayout {
            absorbed_word_base: 38_977,
            absorbed_words: 1_222,
            public_input_base: 40_199,
            public_inputs: 7,
            challenge_base: 40_206,
            challenges: 376,
            dory_scalar_base: 40_704,
            dory_scalars: 172,
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

    let term_point = (0..18)
        .map(|index| Fr::from_u64((index + 5) as u64))
        .collect::<Vec<_>>();
    let term_tau = (0..18)
        .map(|index| Fr::from_u64((index + 29) as u64))
        .collect::<Vec<_>>();
    let relation_weights = [Fr::from_u64(53), Fr::from_u64(59), Fr::from_u64(61)];
    let relation_stage_coefficient = Fr::from_u64(67);
    let column_claims = (0..TOTAL_COLUMNS)
        .map(|column| {
            if column < FIXED_COLUMNS {
                table.fixed[column].as_slice().evaluate(&term_point)
            } else {
                table_witness.columns[column - FIXED_COLUMNS]
                    .as_slice()
                    .evaluate(&term_point)
            }
        })
        .collect::<Vec<_>>();
    let native_final = RelationTable::final_value(
        rows,
        &term_tau,
        beta,
        gamma,
        relation_weights,
        &term_point,
        &column_claims,
    )
    .expect("native final relation");
    let term_context = RelationTermsContext {
        columns: std::array::from_fn(|slot| ColumnId { group: 0, slot }),
        tau: &term_tau,
        point: &term_point,
        beta,
        gamma,
        relation_weights,
        stage_coefficient: relation_stage_coefficient,
    };
    let exporter = RelationTermExporter {
        rows,
        columns: term_context.columns,
        tau: &term_tau,
        beta,
        gamma,
        relation_weights,
        member_index: 0,
    };
    let export_context = TermContext {
        row_point: &term_point,
        batching_coefficients: &[relation_stage_coefficient],
        challenges: &[],
    };
    let mut relation_term_cost = VerifierCost::default();
    let terms = exporter.terms_observed(&export_context, &mut relation_term_cost);
    assert_eq!(exporter.terms(&export_context), terms);
    assert_eq!(table.terms(&term_context).expect("relation terms"), terms);
    assert_eq!(relation_term_cost.fr_mul, 79);
    assert_eq!(terms.len(), RELATION_TERM_COUNT);
    assert_eq!(
        terms.iter().map(|term| term.factors.len()).max(),
        Some(MAX_FACTORS)
    );
    assert_eq!(
        relation_stage_coefficient * native_final,
        evaluate_terms_observed(
            &terms,
            &|column| {
                if column.group != 0 {
                    return Err(RelationTableError::Claims);
                }
                column_claims
                    .get(column.slot)
                    .copied()
                    .ok_or(RelationTableError::Claims)
            },
            &mut relation_term_cost,
        )
        .expect("evaluate relation terms")
    );
    assert_eq!(relation_term_cost.fr_mul, 126);

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
    let t2_inputs = from_jolt(
        &preprocessing.pcs_setup,
        &proof.commitments,
        &proof.joint_opening_proof,
        &relation.link.dory,
        &relation_witness.values,
        Fr::from_u64(43),
    )
    .expect("T2 inputs");
    let t2_layout = build_limb_layout(
        &t2_inputs.check,
        &t2_inputs.values,
        &t2_inputs.setup,
        &t2_inputs.wire_order,
    );
    let scalar_link = DoryScalarLink::new(rows, table.cell_layout(), &t2_layout, rho);
    let mut scalar_prover = scalar_link.prover(&table_witness);
    let mut direct = Fr::zero();
    let weights = link_weights(&t2_layout, rho);
    for ((_, variable), weight) in relation.link.dory.scalars.iter().zip(weights) {
        direct += weight * relation_witness.values[variable.index()];
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
    assert_eq!(scalar_cost.fr_mul, 494);
    let scalar_term_context = DoryScalarTermsContext {
        wire: ColumnId { group: 0, slot: 0 },
        point: &scalar_point,
        stage_coefficient: Fr::from_u64(67),
    };
    let scalar_exporter = DoryScalarTermExporter {
        link: &scalar_link,
        wire: scalar_term_context.wire,
        member_index: 0,
    };
    let scalar_export_context = TermContext {
        row_point: &scalar_point,
        batching_coefficients: &[scalar_term_context.stage_coefficient],
        challenges: &[],
    };
    let mut scalar_term_cost = VerifierCost::default();
    let scalar_terms =
        scalar_exporter.terms_observed(&scalar_export_context, &mut scalar_term_cost);
    assert_eq!(scalar_exporter.terms(&scalar_export_context), scalar_terms);
    assert_eq!(
        scalar_link
            .terms(&scalar_term_context)
            .expect("scalar terms"),
        scalar_terms
    );
    assert_eq!(scalar_term_cost.fr_mul, 494);
    assert_eq!(scalar_terms.len(), DORY_SCALAR_TERM_COUNT);
    assert_eq!(scalar_terms[0].factors.len(), 1);
    assert_eq!(
        scalar_term_context.stage_coefficient * scalar_claim,
        evaluate_terms_observed(
            &scalar_terms,
            &|column| {
                if column == scalar_term_context.wire {
                    Ok(scalar_prover.wire_claim())
                } else {
                    Err(RelationTableError::Claims)
                }
            },
            &mut scalar_term_cost,
        )
        .expect("evaluate scalar terms")
    );
    assert_eq!(scalar_term_cost.fr_mul, 495);
    assert_eq!(
        RELATION_TERM_COUNT + COPY_LINK_TERM_COUNT + DORY_SCALAR_TERM_COUNT,
        26
    );

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
    assert_eq!(table_proof.payload_bytes(), 4_352);
    assert_eq!(bincode_bytes, 4_416);
    assert_eq!(
        cost,
        VerifierCost {
            ec_mul: 108,
            ec_add: 107,
            pairing_pairs: 8,
            fr_mul: 1_225,
            fr_inv: 6,
            keccak: 310,
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

    let column = |slot| ColumnId { group: 0, slot };
    let form_columns = |base| {
        std::array::from_fn(|wire| AffineForm {
            constant: Fr::zero(),
            weights: vec![(column(base + wire), Fr::one())],
        })
    };
    let tau = (0..18)
        .map(|index| Fr::from_u64((index + 7) as u64))
        .collect::<Vec<_>>();
    let point = (0..18)
        .map(|index| Fr::from_u64((index + 37) as u64))
        .collect::<Vec<_>>();
    let exporter = CopyLinkTermExporter {
        link: &link,
        left: CopyLinkTermSide {
            selectors: [column(0), column(1), column(2)],
            ids: form_columns(3),
            values: form_columns(6),
            helper: column(18),
        },
        right: CopyLinkTermSide {
            selectors: [column(9), column(10), column(11)],
            ids: form_columns(12),
            values: form_columns(15),
            helper: column(19),
        },
        tau: &tau,
        beta,
        gamma,
        relation_weights: [Fr::from_u64(43), Fr::from_u64(47), Fr::from_u64(53)],
        member_index: 0,
    };
    let mut term_cost = VerifierCost::default();
    let terms = exporter.terms_observed(
        &TermContext {
            row_point: &point,
            batching_coefficients: &[Fr::from_u64(59)],
            challenges: &[],
        },
        &mut term_cost,
    );
    assert_eq!(terms.len(), COPY_LINK_TERM_COUNT);
    assert_eq!(term_cost.fr_mul, 58);
}
