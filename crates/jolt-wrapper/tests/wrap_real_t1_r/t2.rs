use jolt_field::Fr;
use jolt_wrapper::limb_table::adapter::{from_jolt, AdapterError};
use jolt_wrapper::limb_table::columns::Columns;
use jolt_wrapper::limb_table::dory::FlattenedCheck;
use jolt_wrapper::limb_table::layout::LOG_ROWS;
use jolt_wrapper::limb_table::schedule::{build, Layout};
use jolt_wrapper::relation::{DoryScalar, Preprocessing, Proof, Relation, Witness};

pub struct Base {
    pub columns: Columns,
}

impl Base {
    pub fn new(
        preprocessing: &Preprocessing,
        proof: &Proof,
        relation: &Relation,
        witness: &Witness,
        theta: Fr,
    ) -> Result<(Self, Layout), AdapterError> {
        let inputs = from_jolt(
            &preprocessing.pcs_setup,
            &proof.commitments,
            &proof.joint_opening_proof,
            &relation.link.dory,
            &witness.values,
            theta,
        )?;
        let layout = build(
            &inputs.check,
            &inputs.values,
            &inputs.setup,
            &inputs.wire_order,
        );
        let coordinates = inputs.witness.coordinates_in(&layout.input_order);
        let values = layout
            .program
            .evaluate(&coordinates)
            .unwrap_or_else(|_| unreachable!("verified Dory inputs satisfy the limb program"));
        let columns = Columns::generate(&layout.program, &values, LOG_ROWS);
        Ok((Self { columns }, layout))
    }
}

#[test]
fn t2_consumed_scalars_match_the_relation_links() {
    let (sigma, commitments) = (11, 42);
    let consumed = FlattenedCheck::derive(sigma, commitments).wires();
    assert_eq!(consumed, DoryScalar::link_order(sigma, commitments));
    assert_eq!(consumed.len(), 173);
    for omitted in [DoryScalar::Chi(sigma), DoryScalar::S1Acc, DoryScalar::S2Acc] {
        assert!(!consumed.contains(&omitted));
    }
}
