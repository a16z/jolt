#![expect(clippy::expect_used, reason = "integration tests should fail loudly")]

mod support;

use jolt_blindfold::{r1cs, BlindFoldStage, BlindFoldStatement, CommittedClaimRows};
use jolt_claims::protocols::jolt::{
    formulas::{
        booleanity::{booleanity, BooleanityDimensions},
        claim_reductions::increments,
        ra::JoltRaPolynomialLayout,
        ram::{self, RamValCheckAdviceContribution, RamValCheckInit},
        registers,
    },
    JoltChallengeId, JoltExpr, JoltOpeningId, JoltPublicId, JoltStageClaims, ReadWriteDimensions,
    TraceDimensions,
};
use jolt_r1cs::{ClaimSourceTable, R1csBuilder};
use jolt_sumcheck::{SumcheckDomainSpec, SumcheckStatement};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use support::*;

#[derive(Clone, Debug)]
struct JoltSourceValues {
    openings: Vec<(JoltOpeningId, F)>,
    publics: Vec<(JoltPublicId, F)>,
    challenges: Vec<(JoltChallengeId, F)>,
}

impl JoltSourceValues {
    fn seeded(stage: &JoltStageClaims<F>, seed: u64) -> Self {
        let openings = stage
            .required_openings()
            .into_iter()
            .enumerate()
            .map(|(index, id)| (id, f(seed + 11 + index as u64)))
            .collect();
        let publics = stage
            .required_publics()
            .into_iter()
            .enumerate()
            .map(|(index, id)| (id, f(seed + 101 + index as u64)))
            .collect();
        let challenges = stage
            .required_challenges()
            .into_iter()
            .enumerate()
            .map(|(index, id)| (id, f(seed + 211 + index as u64)))
            .collect();
        Self {
            openings,
            publics,
            challenges,
        }
    }

    fn opening(&self, id: &JoltOpeningId) -> F {
        self.openings
            .iter()
            .find_map(|(candidate, value)| (candidate == id).then_some(*value))
            .expect("opening exists")
    }

    fn public(&self, id: &JoltPublicId) -> F {
        self.publics
            .iter()
            .find_map(|(candidate, value)| (candidate == id).then_some(*value))
            .expect("public exists")
    }

    fn challenge(&self, id: &JoltChallengeId) -> F {
        self.challenges
            .iter()
            .find_map(|(candidate, value)| (candidate == id).then_some(*value))
            .expect("challenge exists")
    }

    fn set_opening(&mut self, id: JoltOpeningId, value: F) {
        let (_, existing) = self
            .openings
            .iter_mut()
            .find(|(candidate, _)| *candidate == id)
            .expect("opening exists");
        *existing = value;
    }

    fn evaluate(&self, expression: &JoltExpr<F>) -> F {
        expression.evaluate(
            |id| self.opening(id),
            |id| self.challenge(id),
            |id| self.public(id),
        )
    }
}

fn solve_linear_output_opening(
    stage: &JoltStageClaims<F>,
    values: &mut JoltSourceValues,
    target: F,
) -> JoltOpeningId {
    stage
        .output
        .required_openings
        .iter()
        .find_map(|candidate| {
            if stage.input.required_openings.contains(candidate) {
                return None;
            }
            let original = values.opening(candidate);
            values.set_opening(*candidate, f(0));
            let base = values.evaluate(&stage.output.expression);
            values.set_opening(*candidate, f(1));
            let shifted = values.evaluate(&stage.output.expression);
            let delta = shifted - base;
            values.set_opening(*candidate, original);
            if delta != f(0) {
                let solved = (target - base) * inverse(delta);
                values.set_opening(*candidate, solved);
                Some(*candidate)
            } else {
                None
            }
        })
        .expect("stage output has a linearly solvable opening")
}

fn build_jolt_stage_relation(
    stage: &JoltStageClaims<F>,
    generated: &GeneratedStage,
    values: &JoltSourceValues,
) -> Result<(), usize> {
    let statement = generated.statement;
    let statement = BlindFoldStatement::new(
        vec![BlindFoldStage::new(
            "jolt-stage",
            statement,
            SumcheckDomainSpec::BooleanHypercube,
            stage_consistency(statement, &generated.proof),
            CommittedClaimRows::new(
                Vec::new(),
                statement.degree + 1,
                generated.proof.output_claims.clone(),
            ),
            stage.input.expression.clone(),
            stage.output.expression.clone(),
        )],
        Vec::new(),
    );

    let mut builder = R1csBuilder::<F>::new();
    let mut sources = ClaimSourceTable::<F, JoltOpeningId, JoltPublicId, JoltChallengeId>::new();
    for &(id, value) in &values.openings {
        sources.insert_opening(id, builder.alloc(value));
    }
    for &(id, value) in &values.challenges {
        sources.insert_challenge(id, value);
    }
    for &(id, value) in &values.publics {
        sources.insert_public(id, value);
    }

    let layout = r1cs::allocate_layout(&mut builder, &statement).expect("layout allocates");
    r1cs::append(&mut builder, &statement, &layout, &mut sources).expect("constraints append");
    assign_generated_stage(&mut builder, &layout.stages[0].sumcheck, generated);

    let witness = builder.witness().expect("all witnesses assigned");
    builder.into_matrices().check_witness(&witness)
}

fn generated_jolt_stage(
    stage: &JoltStageClaims<F>,
    seed: u64,
) -> (GeneratedStage, JoltSourceValues, JoltOpeningId) {
    let setup = pedersen_setup(stage.sumcheck.degree + 1);
    let statement = SumcheckStatement::new(stage.sumcheck.rounds, stage.sumcheck.degree);
    let mut values = JoltSourceValues::seeded(stage, seed);
    let input_claim = values.evaluate(&stage.input.expression);
    let mut prover = SumcheckTestProver::new(ChaCha20Rng::from_seed([seed as u8; 32]));
    let generated = prover.prove_stage_with_fresh_transcript(
        &setup,
        b"blindfold-r1cs-e2e",
        statement,
        input_claim,
    );
    let final_claim = *generated
        .claim_outs
        .last()
        .expect("stage has at least one round");
    let solved = solve_linear_output_opening(stage, &mut values, final_claim);
    (generated, values, solved)
}

#[test]
fn jolt_claims_pipeline_lowers_registers_read_write_relation() {
    let stage = registers::read_write_checking::<F>(ReadWriteDimensions::new(2, 2, 1, 1));
    let (generated, values, _) = generated_jolt_stage(&stage, 5);

    assert!(build_jolt_stage_relation(&stage, &generated, &values).is_ok());
}

#[test]
fn jolt_claims_pipeline_lowers_ram_val_check_with_decomposed_advice() {
    let init = RamValCheckInit::decomposed(
        f(17),
        [
            RamValCheckAdviceContribution::trusted(f(3)),
            RamValCheckAdviceContribution::untrusted(f(7)),
        ],
    );
    let stage = ram::val_check::<F>(TraceDimensions::new(3), init);
    let (generated, values, _) = generated_jolt_stage(&stage, 7);

    assert!(build_jolt_stage_relation(&stage, &generated, &values).is_ok());
}

#[test]
fn jolt_claims_pipeline_lowers_increment_claim_reduction() {
    let stage = increments::claim_reduction::<F>(TraceDimensions::new(3));
    let (generated, values, _) = generated_jolt_stage(&stage, 9);

    assert!(build_jolt_stage_relation(&stage, &generated, &values).is_ok());
}

#[test]
fn jolt_claims_pipeline_rejects_tampered_public_source() {
    let stage = increments::claim_reduction::<F>(TraceDimensions::new(3));
    let (generated, mut values, _) = generated_jolt_stage(&stage, 11);
    values.publics[0].1 += f(1);

    assert!(build_jolt_stage_relation(&stage, &generated, &values).is_err());
}

#[test]
fn jolt_claims_pipeline_rejects_tampered_opening_source() {
    let stage = registers::read_write_checking::<F>(ReadWriteDimensions::new(2, 2, 1, 1));
    let (generated, mut values, solved) = generated_jolt_stage(&stage, 13);
    values.set_opening(solved, values.opening(&solved) + f(1));

    assert!(build_jolt_stage_relation(&stage, &generated, &values).is_err());
}

#[test]
fn jolt_claims_pipeline_lowers_booleanity_relation() {
    let setup = pedersen_setup(1);
    let layout = JoltRaPolynomialLayout::new(1, 1, 1).expect("valid RA layout");
    let jolt_stage = booleanity::<F>(BooleanityDimensions {
        layout,
        log_t: 1,
        log_k_chunk: 1,
    });
    let statement = SumcheckStatement::new(jolt_stage.sumcheck.rounds, jolt_stage.sumcheck.degree);
    let generated = generate_zero_stage(&setup, statement.num_vars);
    let statement = BlindFoldStatement::new(
        vec![BlindFoldStage::new(
            "jolt-booleanity",
            statement,
            SumcheckDomainSpec::BooleanHypercube,
            stage_consistency(statement, &generated.proof),
            CommittedClaimRows::new(
                Vec::new(),
                statement.degree + 1,
                generated.proof.output_claims.clone(),
            ),
            jolt_stage.input.expression.clone(),
            jolt_stage.output.expression.clone(),
        )],
        Vec::new(),
    );

    let mut builder = R1csBuilder::<F>::new();
    let mut sources = ClaimSourceTable::<F, JoltOpeningId, JoltPublicId, JoltChallengeId>::new();
    for opening_id in jolt_stage.required_openings() {
        sources.insert_opening(opening_id, builder.alloc(f(0)));
    }
    for challenge_id in jolt_stage.required_challenges() {
        sources.insert_challenge(challenge_id, f(7));
    }
    for public_id in jolt_stage.required_publics() {
        sources.insert_public(public_id, f(11));
    }

    let r1cs_layout = r1cs::allocate_layout(&mut builder, &statement).expect("layout allocates");
    r1cs::append(&mut builder, &statement, &r1cs_layout, &mut sources).expect("constraints append");
    assign_generated_stage(&mut builder, &r1cs_layout.stages[0].sumcheck, &generated);

    let witness = builder.witness().expect("all witnesses assigned");
    assert!(builder.into_matrices().check_witness(&witness).is_ok());
}
