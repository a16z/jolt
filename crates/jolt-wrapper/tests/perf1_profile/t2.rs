use ark_bn254::Fr as ArkFr;
use jolt_crypto::Bn254;
use jolt_field::{Field, Fr};
use jolt_hyperkzg::HyperKZGProverSetup;
use jolt_transcript::{Keccak256Transcript, Transcript};
use jolt_wrapper::limb_table::columns::Columns;
use jolt_wrapper::limb_table::dory::{FlattenedCheck, WireValues};
use jolt_wrapper::limb_table::relation::{RowRelation, RowSumcheck};
use jolt_wrapper::limb_table::schedule::build;
use jolt_wrapper::limb_table::stream::StreamBuilder;
use jolt_wrapper::stream::prove_kzg_stage;
use rand::rngs::StdRng;
use rand::SeedableRng;

use super::{common, Report, ROWS_LOG};

pub struct T2Witness {
    relation: RowRelation,
    matrix: Vec<Vec<Fr>>,
}

pub fn witness() -> T2Witness {
    let opening = common::synthetic_opening(8, 5, 0x72);
    let sigma = opening.statement.challenges.beta.len();
    let n = opening.witness.commitments.len();
    let check = FlattenedCheck::derive(sigma, n);
    let values = WireValues::derive(&opening.statement, sigma, n, ArkFr::from(0x72u64));
    let layout = build(&check, &values, &opening.setup, &check.wires());
    let coordinates = opening.witness.coordinates_in(&layout.input_order);
    let evaluated = layout
        .program
        .evaluate(&coordinates)
        .expect("evaluate program");
    let columns = Columns::generate(&layout.program, &evaluated, ROWS_LOG);
    let mut rng = StdRng::seed_from_u64(0x72);
    let mut challenge = || Fr::random(&mut rng);
    let (xi, alpha, fp_root, beta, fp_combine, copy_root) = (
        challenge(),
        challenge(),
        challenge(),
        challenge(),
        challenge(),
        challenge(),
    );
    let tau = (0..ROWS_LOG).map(|_| challenge()).collect();
    let (gamma, lambda, lambda_lookup, constancy_root) =
        (challenge(), challenge(), challenge(), challenge());
    let mut builder = StreamBuilder::new(&layout, &columns, 16);
    let _ = builder.phase_1b();
    let _ = builder.phase_2a(xi, alpha);
    let _ = builder.phase_2b(fp_root);
    let _ = builder.phase_2c(beta, fp_combine, copy_root);
    let witness = builder.finish(tau, gamma, lambda, lambda_lookup, constancy_root, 0);
    T2Witness {
        relation: witness.relation,
        matrix: witness.matrix,
    }
}

pub fn profile(report: &mut Report, witness: &T2Witness, setup: &HyperKZGProverSetup<Bn254>) {
    let mut prover = report.measure("T2      construct (row matrix, 158 Fr/row)", || {
        RowSumcheck::new(&witness.relation, &witness.matrix)
    });
    let claim = report.measure("T2      input claim", || prover.input_claim());
    let mut transcript = Keccak256Transcript::<Fr>::new(b"perf1-t2");
    let _ = report.measure("T2      rounds 0..18 + KZG round commits + BDFG", || {
        prove_kzg_stage(&mut prover, claim, 5, setup, &mut transcript).expect("T2 stage")
    });
}
