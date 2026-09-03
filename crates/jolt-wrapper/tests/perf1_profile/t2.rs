use jolt_crypto::Bn254;
use jolt_field::{Field, Fr, Ring};
use jolt_hyperkzg::HyperKZGProverSetup;
use jolt_transcript::{Keccak256Transcript, Transcript};
use jolt_wrapper::limb_table::columns::{operand_columns, Columns};
use jolt_wrapper::limb_table::dory::{FlattenedCheck, WireValues};
use jolt_wrapper::limb_table::export::{free_column, pin_columns, ClaimedColumns};
use jolt_wrapper::limb_table::lookup::{LookupColumns, PublicColumns};
use jolt_wrapper::limb_table::relation::col::WIDTH;
use jolt_wrapper::limb_table::relation::{
    eq_tau_column, Challenges, LookupConstants, RowRelation, RowSumcheck, SLOTS,
};
use jolt_wrapper::limb_table::schedule::build;
use jolt_wrapper::limb_table::wiring::{copy_kernel_table, fingerprint_columns};
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
    let values = WireValues::derive(&opening.statement, sigma, n);
    let layout = build(&check, &values, &opening.setup, &check.wires());
    let coordinates = opening.witness.coordinates_in(&layout.input_order);
    let evaluated = layout
        .program
        .evaluate(&coordinates)
        .expect("evaluate program");
    let columns = Columns::generate(&layout.program, &evaluated, ROWS_LOG);
    let mut rng = StdRng::seed_from_u64(0x72);
    let challenges = Challenges {
        tau: (0..ROWS_LOG).map(|_| Fr::random(&mut rng)).collect(),
        xi: Fr::random(&mut rng),
        alpha: Fr::random(&mut rng),
        gamma: Fr::random(&mut rng),
        lambda: Fr::random(&mut rng),
        beta: Fr::random(&mut rng),
        fp_root: Fr::random(&mut rng),
        fp_combine: Fr::random(&mut rng),
        lambda_lookup: Fr::random(&mut rng),
        copy_root: Fr::random(&mut rng),
        constancy_root: Fr::random(&mut rng),
    };
    let relation = RowRelation::new(
        challenges,
        LookupConstants {
            one_row: layout.one_cell * 16,
        },
    );
    let public = PublicColumns::new(&layout);
    let z_xi = columns.xi_values(relation.challenges.xi);
    let operands = operand_columns(&layout.program, &z_xi, SLOTS);
    let fingerprints = fingerprint_columns(&layout.table_reads, &z_xi, &relation);
    let lookup = LookupColumns::new(
        &public,
        &operands,
        &fingerprints.0,
        &fingerprints.1,
        &relation,
    );
    let (helpers, multiplicities) =
        columns.logup_columns(relation.challenges.alpha, &public.digits);
    let claimed = ClaimedColumns::assemble(
        &columns,
        &public,
        operands,
        helpers,
        multiplicities
            .into_iter()
            .map(|value| Fr::from_u64(u64::from(value)))
            .collect(),
        PublicColumns::inverse_table(relation.challenges.alpha),
        lookup,
        fingerprints,
        pin_columns(&layout),
        free_column(&layout),
    );
    let eq_tau = eq_tau_column(&relation.challenges.tau);
    let copy = copy_kernel_table(
        &layout.program,
        &public.kinds,
        &layout.table_reads,
        &eq_tau,
        &relation,
    );
    let constancy = public.constancy_weights(&eq_tau);
    let (small, id) = PublicColumns::small_and_id();
    let mut matrix = claimed.columns;
    matrix.extend([
        eq_tau,
        copy,
        public.sel,
        public.is_gt,
        public.is_g1,
        public.is_g2,
        public.s0,
        public.coord,
        constancy,
        small,
        id,
    ]);
    assert_eq!(matrix.len(), WIDTH);
    T2Witness { relation, matrix }
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
