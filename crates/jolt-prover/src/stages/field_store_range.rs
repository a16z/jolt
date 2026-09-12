//! A malicious prover keeps every store bridge value consistently too wide,
//! then biases fresh sumcheck messages to pass round checks. Only the final
//! lookup binding can reject this witness; the tracer is never involved.

use rand_core::OsRng;
use std::num::NonZeroUsize;

use jolt_claims::protocols::jolt::geometry::instruction::InstructionReadRafDimensions;
use jolt_crypto::{Bn254, Bn254G1, Pedersen, PedersenSetup};
use jolt_field::{Field, Fr, One, Ring, Zero};
use jolt_kernels::reference::instruction_read_raf::{
    InstructionReadRafKernel, InstructionReadRafWitness,
};
use jolt_kernels::SumcheckKernel;
use jolt_lookup_tables::LookupTableKind;
use jolt_poly::UnivariatePoly;
use jolt_r1cs::constraints::field_constraints::{
    field_inline_trace_constraints, NUM_VARS_PER_CYCLE, V_CONST, V_FIELD_RS1_VALUE,
    V_IS_FIELD_STORE_TO_X, V_X_RD_WRITE_VALUE, V_X_RIGHT_LOOKUP_OPERAND,
};
use jolt_sumcheck::{
    prove_batch, BatchMember, BatchPrelude, ProveRounds, SequentialRounds, SumcheckError,
    SumcheckRecorder,
};
use jolt_transcript::{LegacyBlake2bTranscript as Blake2bTranscript, Transcript};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::stage5::instruction_read_raf::{
    InstructionReadRaf, InstructionReadRafInputClaims,
};
use jolt_witness::witnesses::{InstructionRafFlag, LookupIndex, TableIndex};

use crate::recorder::ProofMode;

struct BiasedLookup {
    kernel: InstructionReadRafKernel<Fr>,
    true_claim: Fr,
    previous: Option<UnivariatePoly<Fr>>,
}

impl ProveRounds<Fr> for BiasedLookup {
    fn num_rounds(&self) -> usize {
        self.kernel.num_rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<Fr>,
        round: usize,
        previous_claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        if let (Some(point), Some(previous)) = (bind, &self.previous) {
            self.true_claim = previous.evaluate(point);
        }
        let honest = self.kernel.prove_round(bind, round, self.true_claim)?;
        let mut coefficients = honest.coefficients().to_vec();
        coefficients[0] += (previous_claim - self.true_claim) * Fr::from_u64(2).inverse().unwrap();
        self.previous = Some(honest);
        Ok(UnivariatePoly::new(coefficients))
    }

    fn finish_rounds(&mut self, bind: Fr) -> Result<(), SumcheckError<Fr>> {
        self.kernel.finish_rounds(bind)
    }
}

#[test]
fn fresh_store_lookup_proof_rejects_synchronized_wide_bridge_values() {
    let dimensions = InstructionReadRafDimensions::new(2, 128, NonZeroUsize::new(4).unwrap());
    let relation = InstructionReadRaf::<Fr>::new(dimensions);
    let table = LookupTableKind::<64>::iter()
        .find(|table| matches!(table, LookupTableKind::RangeCheck(_)))
        .unwrap();
    let mut rng = OsRng;
    let setup = PedersenSetup::new(
        (0..16).map(|_| Bn254::random_g1(&mut rng)).collect(),
        Bn254::random_g1(&mut rng),
    );
    let mode = ProofMode::<Pedersen<Bn254G1>>::new(Some(&setup)).unwrap();
    for index in [u64::MAX as u128, (1u128 << 64) + 7] {
        let value = Fr::from_u128(index);
        let mut bridge = vec![Fr::zero(); NUM_VARS_PER_CYCLE];
        bridge[V_CONST] = Fr::one();
        bridge[V_IS_FIELD_STORE_TO_X] = Fr::one();
        bridge[V_FIELD_RS1_VALUE] = value;
        bridge[V_X_RD_WRITE_VALUE] = value;
        bridge[V_X_RIGHT_LOOKUP_OPERAND] = value;
        field_inline_trace_constraints::<Fr>()
            .check_witness(&bridge)
            .unwrap();
        // RV64's write-lookup row is also satisfied: RdWriteValue = LookupOutput.
        let inputs = InstructionReadRafInputClaims {
            lookup_output: value,
            left_lookup_operand: Fr::zero(),
            right_lookup_operand: value,
        };
        let points = InstructionReadRafInputClaims {
            lookup_output: vec![Fr::from_u64(3), Fr::from_u64(5)],
            left_lookup_operand: vec![Fr::from_u64(3), Fr::from_u64(5)],
            right_lookup_operand: vec![Fr::from_u64(3), Fr::from_u64(5)],
        };
        let mut transcript = Blake2bTranscript::new(b"field-store-range-attack");
        let challenges = relation.draw_challenges(&mut transcript).unwrap();
        let input_claim = relation.input_claim(&inputs, &challenges).unwrap();
        let true_inputs = InstructionReadRafInputClaims {
            lookup_output: Fr::from_u64(index as u64),
            ..inputs.clone()
        };
        let mut kernel = BiasedLookup {
            kernel: InstructionReadRafKernel::new(
                dimensions,
                &points.lookup_output,
                vec![
                    InstructionReadRafWitness {
                        lookup_index: LookupIndex(index),
                        table_index: TableIndex(Some(table.index())),
                        raf_flag: InstructionRafFlag(true),
                    };
                    4
                ],
                challenges.gamma,
            )
            .unwrap(),
            true_claim: relation.input_claim(&true_inputs, &challenges).unwrap(),
            previous: None,
        };
        let rounds = kernel.num_rounds();
        let degree = relation.degree();
        let prelude = BatchPrelude::new(
            vec![BatchMember {
                input_claim,
                coefficient: Fr::one(),
                rounds,
                offset: 0,
            }],
            rounds,
            degree,
        );
        let mut recorder = mode.recorder().unwrap();
        recorder.absorb_input_claims(&[input_claim], &mut transcript);
        let mut verifier_transcript = transcript.clone();
        let proved = prove_batch(
            &prelude,
            &mut [&mut kernel],
            &mut SequentialRounds,
            &mut recorder,
            &mut transcript,
        )
        .unwrap();
        let outputs = kernel.kernel.output_claims(&inputs).unwrap();
        let output_points = relation
            .derive_opening_points(&proved.challenges, &points)
            .unwrap();
        let expected = relation
            .expected_output(&points, &outputs, &output_points, &challenges)
            .unwrap();
        let recorded = recorder.finish(&[], &mut transcript).unwrap();
        let in_range = index <= u64::MAX as u128;
        #[cfg(not(feature = "zk"))]
        {
            let reduced = recorded
                .proof
                .verify_compressed_boolean(rounds, degree, input_claim, &mut verifier_transcript)
                .unwrap();
            assert_eq!(reduced.point, proved.challenges);
            assert_eq!(reduced.value == expected, in_range);
        }
        #[cfg(feature = "zk")]
        {
            use jolt_blindfold::BlindFoldProtocol;
            use jolt_claims::Expr;
            use jolt_sumcheck::{SumcheckDomainSpec, SumcheckStatement};
            let statement = SumcheckStatement::new(rounds, degree);
            let consistency = recorded
                .proof
                .as_committed()
                .unwrap()
                .verify_committed_consistency(statement, &mut verifier_transcript)
                .unwrap();
            let protocol = |output| {
                BlindFoldProtocol::<Fr, Bn254G1>::builder::<(), (), usize>()
                    .stage("store-range-lookup")
                    .sumcheck(statement)
                    .consistency(consistency.clone())
                    .input_claim(Expr::constant(input_claim))
                    .output_claim(Expr::constant(output))
                    .finish_stage()
                    .unwrap()
                    .build()
                    .unwrap()
            };
            // Assign the attacker's consistent round chain, then check it against
            // the real lookup endpoint, using the production BlindFold matrices.
            let attacker = protocol(proved.final_claim);
            let assigned = attacker
                .assign_witness(
                    &[SumcheckDomainSpec::BooleanHypercube],
                    &[recorded.committed_witness.as_ref().unwrap()],
                    &[],
                    &[],
                    &mut OsRng,
                )
                .unwrap();
            let verifier = protocol(expected);
            let mut witness = vec![Fr::one()];
            witness.extend(assigned.rows.into_iter().flatten());
            witness.truncate(verifier.r1cs.num_vars);
            assert_eq!(verifier.r1cs.check_witness(&witness).is_ok(), in_range);
        }
    }
}
