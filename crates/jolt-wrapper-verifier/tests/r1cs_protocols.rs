#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "integration tests may panic on assertion failures"
)]

use std::collections::BTreeSet;

use jolt_crypto::{
    r1cs::GrumpkinPointWithIdentityVar, Grumpkin, GrumpkinPoint, JoltGroup, Pedersen,
    PedersenSetup, VectorCommitment,
};
use jolt_field::{CanonicalBytes, Field, FixedByteSize, Fq, Fr, FromPrimitiveInt};
use jolt_hyrax::r1cs::{verify_opening, HyraxOpeningR1csInput};
use jolt_openings::r1cs::{reduce_same_point_opening_claims, OpeningClaimVar};
use jolt_poly::EqPolynomial;
use jolt_r1cs::{AssignedScalar, FqVar, LinearCombination, R1csBuilder, Variable};
use jolt_sumcheck::{
    allocate_sumcheck_r1cs_layout, append_sumcheck_r1cs_constraints,
    append_sumcheck_r1cs_gadget_constraints, SumcheckR1csGadgetRound, SumcheckR1csRound,
    SumcheckR1csRoundLayout, SumcheckStatement,
};
use jolt_transcript::r1cs::{
    PoseidonR1csTranscript, R1csJoltByteTranscript, R1csJoltTranscript, R1csTranscript,
};
use jolt_wrapper_verifier::{verify_r1cs_witness, WrapperR1csBuilder, WrapperR1csProtocol};

type Transcript = PoseidonR1csTranscript;
type TestVc = Pedersen<GrumpkinPoint>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum ProtocolComponent {
    PoseidonTranscript,
    PublicInputs,
    FrVerifierArithmetic,
    FqChallengeInjection,
    FqArithmetic,
    Sumcheck,
    OpeningReduction,
    HyraxPedersenOpening,
}

#[derive(Clone, Debug)]
struct ProtocolCase {
    name: &'static str,
    components: &'static [ProtocolComponent],
    protocol: WrapperR1csProtocol<Fr>,
    tamper_targets: Vec<TamperTarget>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TamperTarget {
    name: String,
    mutation: Mutation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mutation {
    Witness(usize),
    PublicInput(usize),
    TruncatePublicInputs,
    ExtendPublicInputs,
    PublicInputLayoutOutOfBounds(usize),
}

#[derive(Clone, Debug)]
struct VariableChallengeRound {
    degree: usize,
    challenge: LinearCombination<Fr>,
}

impl SumcheckR1csRound<Fr> for VariableChallengeRound {
    fn degree(&self) -> usize {
        self.degree
    }

    fn challenge(&self) -> LinearCombination<Fr> {
        self.challenge.clone()
    }
}

#[test]
fn r1cs_protocol_cases_accept_valid_witnesses() {
    for case in protocol_cases() {
        verify_r1cs_witness(&case.protocol)
            .unwrap_or_else(|error| panic!("{} rejected valid witness: {error}", case.name));
    }
}

#[test]
fn public_input_witness_slots_are_constrained_by_r1cs() {
    let mut builder = WrapperR1csBuilder::<Fr, Transcript>::new(b"WrapperPublicBinding");
    let _ = builder.alloc_public_scalar(Fr::from_u64(11));
    let protocol = builder.finish().expect("protocol builds");
    let public_variable = protocol.layout.public_inputs[0];

    assert!(protocol.r1cs.check_witness(&protocol.witness).is_ok());

    let mut tampered = protocol.clone();
    tampered.witness[public_variable.index()] += Fr::from_u64(1);

    assert_eq!(tampered.public_inputs, protocol.public_inputs);
    assert!(tampered.r1cs.check_witness(&tampered.witness).is_err());
}

#[test]
fn r1cs_protocol_cases_reject_all_registered_tampering() {
    for case in protocol_cases() {
        verify_r1cs_witness(&case.protocol)
            .unwrap_or_else(|error| panic!("{} baseline rejected: {error}", case.name));

        for target in &case.tamper_targets {
            let mut tampered = case.protocol.clone();
            target.mutation.apply(&mut tampered);

            assert!(
                verify_r1cs_witness(&tampered).is_err(),
                "{} accepted tampering target {}",
                case.name,
                target.name
            );
        }
    }
}

#[test]
fn r1cs_protocol_case_names_are_unique() {
    let mut names = BTreeSet::new();
    for case in protocol_cases() {
        assert!(
            names.insert(case.name),
            "duplicate protocol case {}",
            case.name
        );
    }
}

#[test]
fn r1cs_protocol_tamper_target_names_are_unique() {
    for case in protocol_cases() {
        let mut names = BTreeSet::new();
        for target in &case.tamper_targets {
            assert!(
                names.insert(target.name.clone()),
                "duplicate tamper target {} in {}",
                target.name,
                case.name
            );
        }
    }
}

#[test]
fn r1cs_protocol_component_coverage_is_explicit() {
    let covered = protocol_cases()
        .into_iter()
        .flat_map(|case| case.components.iter().copied())
        .collect::<BTreeSet<_>>();
    let expected = BTreeSet::from([
        ProtocolComponent::PoseidonTranscript,
        ProtocolComponent::PublicInputs,
        ProtocolComponent::FrVerifierArithmetic,
        ProtocolComponent::FqChallengeInjection,
        ProtocolComponent::FqArithmetic,
        ProtocolComponent::Sumcheck,
        ProtocolComponent::OpeningReduction,
        ProtocolComponent::HyraxPedersenOpening,
    ]);

    assert_eq!(covered, expected);
}

fn protocol_cases() -> Vec<ProtocolCase> {
    vec![
        with_public_input_tampering(poseidon_public_arithmetic_case()),
        with_public_input_tampering(poseidon_fq_sumcheck_opening_case()),
        with_public_input_tampering(poseidon_hyrax_pedersen_case()),
    ]
}

fn poseidon_public_arithmetic_case() -> ProtocolCase {
    let scalar_value = Fr::from_u64(17);
    let byte_values = [3, 1, 4, 1, 5, 9];
    let public_output = poseidon_public_arithmetic_output(scalar_value, &byte_values);
    let mut builder = WrapperR1csBuilder::<Fr, Transcript>::new(b"WrapperPoseidonPublic");
    let mut tamper_targets = Vec::new();

    let scalar = builder.alloc_witness_scalar(scalar_value);
    push_scalar_target(&mut tamper_targets, "scalar", &scalar);
    let bytes = byte_values
        .iter()
        .map(|&byte| builder.alloc_witness_byte(byte))
        .collect::<Vec<_>>();
    push_scalar_target(&mut tamper_targets, "byte[0]", &bytes[0]);

    builder
        .transcript
        .append_bytes(&mut builder.builder, b"fixture_bytes", &bytes);
    builder
        .transcript
        .append_scalar(&mut builder.builder, b"fixture_scalar", scalar.clone());
    let challenge = builder.transcript.challenge_scalar(&mut builder.builder);
    push_scalar_target(&mut tamper_targets, "poseidon_challenge", &challenge);

    let output = builder.alloc_public_scalar(public_output);
    let byte_sum = byte_sum(&bytes);
    let product = builder
        .builder
        .multiply(challenge.lc.clone(), scalar.lc.clone());
    builder
        .builder
        .assert_equal(product + byte_sum.lc, output.lc.clone());

    ProtocolCase {
        name: "poseidon_public_arithmetic",
        components: &[
            ProtocolComponent::PoseidonTranscript,
            ProtocolComponent::PublicInputs,
            ProtocolComponent::FrVerifierArithmetic,
        ],
        protocol: builder.finish().expect("protocol builds"),
        tamper_targets,
    }
}

fn poseidon_fq_sumcheck_opening_case() -> ProtocolCase {
    let mut builder = WrapperR1csBuilder::<Fr, Transcript>::new(b"WrapperFqSumcheckOpening");
    let mut tamper_targets = Vec::new();

    let public_seed = builder.alloc_public_scalar(Fr::from_u64(42));
    push_scalar_target(&mut tamper_targets, "public_seed_variable", &public_seed);
    let private_seed = builder.alloc_witness_scalar(Fr::from_u64(99));
    push_scalar_target(&mut tamper_targets, "private_seed", &private_seed);
    let bytes = [0x72, 0x31, 0x63, 0x73].map(|byte| builder.alloc_witness_byte(byte));
    push_scalar_target(&mut tamper_targets, "transcript_byte", &bytes[0]);

    builder
        .transcript
        .append_scalar(&mut builder.builder, b"public", public_seed.clone());
    builder
        .transcript
        .append_scalar(&mut builder.builder, b"private", private_seed.clone());
    builder
        .transcript
        .append_bytes(&mut builder.builder, b"bytes", &bytes);
    let challenge = builder.transcript.challenge_scalar(&mut builder.builder);
    push_scalar_target(&mut tamper_targets, "poseidon_challenge", &challenge);

    append_fq_relation(&mut builder.builder, &challenge, &mut tamper_targets);
    let fq_sumcheck_challenge = FqVar::inject_fr_challenge(&mut builder.builder, &challenge);
    append_nonnative_sumcheck(
        &mut builder.builder,
        &fq_sumcheck_challenge,
        &mut tamper_targets,
    );
    append_fq_opening_reduction(&mut builder.builder, &challenge, &mut tamper_targets);
    let sumcheck_output =
        append_native_sumcheck(&mut builder.builder, &challenge, &mut tamper_targets);

    let expected_public =
        public_seed.value + private_seed.value * challenge.value + sumcheck_output;
    let public_output = builder.alloc_public_scalar(expected_public);
    let private_term = builder
        .builder
        .multiply(private_seed.lc.clone(), challenge.lc.clone());
    builder.builder.assert_equal(
        public_output.lc,
        public_seed.lc + private_term + LinearCombination::constant(sumcheck_output),
    );

    ProtocolCase {
        name: "poseidon_fq_sumcheck_opening_reduction",
        components: &[
            ProtocolComponent::PoseidonTranscript,
            ProtocolComponent::PublicInputs,
            ProtocolComponent::FrVerifierArithmetic,
            ProtocolComponent::FqChallengeInjection,
            ProtocolComponent::FqArithmetic,
            ProtocolComponent::Sumcheck,
            ProtocolComponent::OpeningReduction,
        ],
        protocol: builder.finish().expect("protocol builds"),
        tamper_targets,
    }
}

fn poseidon_hyrax_pedersen_case() -> ProtocolCase {
    let mut builder = WrapperR1csBuilder::<Fr, Transcript>::new(b"WrapperHyraxPedersen");
    let mut tamper_targets = Vec::new();

    let public_seed = builder.alloc_public_scalar(Fr::from_u64(7));
    push_scalar_target(&mut tamper_targets, "public_seed_variable", &public_seed);
    builder.transcript.append_scalar(
        &mut builder.builder,
        b"hyrax_public_seed",
        public_seed.clone(),
    );
    let challenge = builder.transcript.challenge_scalar(&mut builder.builder);
    push_scalar_target(&mut tamper_targets, "poseidon_challenge", &challenge);
    let row_challenge = FqVar::inject_fr_challenge(&mut builder.builder, &challenge);
    push_fq_limb_target(&mut tamper_targets, "row_challenge_limb", &row_challenge);

    let case = HyraxNativeCase::new(fr_to_fq(challenge.value));
    let input = HyraxAllocatedCase::alloc(&mut builder.builder, &case, row_challenge);
    input.push_tamper_targets(&mut tamper_targets);
    verify_opening::<TestVc>(&mut builder.builder, &case.setup, &input.input)
        .expect("valid Hyrax Pedersen opening");

    ProtocolCase {
        name: "poseidon_hyrax_pedersen_opening",
        components: &[
            ProtocolComponent::PoseidonTranscript,
            ProtocolComponent::PublicInputs,
            ProtocolComponent::FqChallengeInjection,
            ProtocolComponent::HyraxPedersenOpening,
        ],
        protocol: builder.finish().expect("protocol builds"),
        tamper_targets,
    }
}

fn with_public_input_tampering(mut case: ProtocolCase) -> ProtocolCase {
    for index in 0..case.protocol.public_inputs.len() {
        case.tamper_targets.push(TamperTarget {
            name: format!("public_inputs[{index}]"),
            mutation: Mutation::PublicInput(index),
        });
    }

    case.tamper_targets.push(TamperTarget {
        name: "public_inputs.truncate".to_string(),
        mutation: Mutation::TruncatePublicInputs,
    });
    case.tamper_targets.push(TamperTarget {
        name: "public_inputs.extend".to_string(),
        mutation: Mutation::ExtendPublicInputs,
    });
    case.tamper_targets.push(TamperTarget {
        name: "public_input_layout.oob".to_string(),
        mutation: Mutation::PublicInputLayoutOutOfBounds(0),
    });

    case
}

impl Mutation {
    fn apply(self, protocol: &mut WrapperR1csProtocol<Fr>) {
        match self {
            Self::Witness(index) => {
                protocol.witness[index] += Fr::from_u64(1);
            }
            Self::PublicInput(index) => {
                protocol.public_inputs[index] += Fr::from_u64(1);
            }
            Self::TruncatePublicInputs => {
                let _ = protocol.public_inputs.pop();
            }
            Self::ExtendPublicInputs => {
                protocol.public_inputs.push(Fr::from_u64(123));
            }
            Self::PublicInputLayoutOutOfBounds(index) => {
                protocol.layout.public_inputs[index] = Variable::new(protocol.witness.len());
            }
        }
    }
}

fn append_fq_relation(
    builder: &mut R1csBuilder<Fr>,
    challenge: &AssignedScalar<Fr>,
    tamper_targets: &mut Vec<TamperTarget>,
) {
    let fq_challenge = FqVar::inject_fr_challenge(builder, challenge);
    let seed = FqVar::alloc(builder, Fq::from_u64(17));
    let scale = FqVar::alloc(builder, Fq::from_u64(23));
    let offset = FqVar::alloc(builder, Fq::from_u64(11));

    let computed = fq_challenge
        .add(builder, &seed)
        .mul(builder, &scale)
        .sub(builder, &offset);
    let expected_value =
        (fr_to_fq(challenge.value) + Fq::from_u64(17)) * Fq::from_u64(23) - Fq::from_u64(11);
    let expected = FqVar::alloc(builder, expected_value);

    computed.assert_equal(builder, &expected);
    push_fq_limb_target(tamper_targets, "fq_challenge_limb", &fq_challenge);
    push_fq_limb_target(tamper_targets, "fq_seed_limb", &seed);
    push_fq_limb_target(tamper_targets, "fq_scale_limb", &scale);
    push_fq_limb_target(tamper_targets, "fq_offset_limb", &offset);
    push_fq_limb_target(tamper_targets, "fq_computed_limb", &computed);
    push_fq_limb_target(tamper_targets, "fq_expected_limb", &expected);
}

fn append_fq_opening_reduction(
    builder: &mut R1csBuilder<Fr>,
    challenge: &AssignedScalar<Fr>,
    tamper_targets: &mut Vec<TamperTarget>,
) {
    let gamma = FqVar::inject_fr_challenge(builder, challenge);
    let point0 = FqVar::alloc(builder, Fq::from_u64(7));
    let point1 = FqVar::alloc(builder, Fq::from_u64(8));
    let point1_copy = FqVar::alloc(builder, Fq::from_u64(8));
    let claim0 = FqVar::alloc(builder, Fq::from_u64(10));
    let claim1 = FqVar::alloc(builder, Fq::from_u64(20));
    let claim2 = FqVar::alloc(builder, Fq::from_u64(30));
    let claims = vec![
        OpeningClaimVar::new(0usize, vec![point0.clone(), point1], claim0),
        OpeningClaimVar::new(1usize, vec![point0.clone(), point1_copy.clone()], claim1),
        OpeningClaimVar::new(
            2usize,
            vec![point0, FqVar::constant(Fq::from_u64(8))],
            claim2,
        ),
    ];
    let reduced = reduce_same_point_opening_claims(builder, &claims, &gamma)
        .expect("same-point opening claims reduce");
    let expected_value = Fq::from_u64(10)
        + Fq::from_u64(20) * fr_to_fq(challenge.value)
        + Fq::from_u64(30) * fr_to_fq(challenge.value) * fr_to_fq(challenge.value);
    let expected = FqVar::alloc(builder, expected_value);
    reduced.opening_claim.assert_equal(builder, &expected);

    push_fq_limb_target(tamper_targets, "opening_gamma_limb", &gamma);
    push_fq_limb_target(tamper_targets, "opening_point_limb", &point1_copy);
    push_fq_limb_target(
        tamper_targets,
        "opening_claim_limb",
        &claims[1].opening_claim,
    );
    push_fq_limb_target(
        tamper_targets,
        "opening_reduced_limb",
        &reduced.opening_claim,
    );
    push_fq_limb_target(tamper_targets, "opening_expected_limb", &expected);
}

fn append_native_sumcheck(
    builder: &mut R1csBuilder<Fr>,
    challenge: &AssignedScalar<Fr>,
    tamper_targets: &mut Vec<TamperTarget>,
) -> Fr {
    let statement = SumcheckStatement::new(1, 2);
    let rounds = [VariableChallengeRound {
        degree: 2,
        challenge: challenge.lc.clone(),
    }];
    let layout = allocate_sumcheck_r1cs_layout(builder, statement, &rounds)
        .expect("sumcheck layout should allocate");
    push_variable_target(
        tamper_targets,
        "native_sumcheck_input_claim",
        layout.input_claim,
    );
    push_variable_target(
        tamper_targets,
        "native_sumcheck_coefficient",
        layout.rounds[0].coefficients[1],
    );
    push_variable_target(
        tamper_targets,
        "native_sumcheck_output_claim",
        layout.output_claim,
    );

    let output_claim = assign_native_sumcheck_witness(
        builder,
        &layout.rounds[0],
        layout.input_claim,
        challenge.value,
    );
    append_sumcheck_r1cs_constraints(builder, statement, &rounds, &layout)
        .expect("sumcheck constraints should build");
    output_claim
}

fn append_nonnative_sumcheck(
    builder: &mut R1csBuilder<Fr>,
    challenge: &FqVar,
    tamper_targets: &mut Vec<TamperTarget>,
) {
    let coefficients = [Fq::from_u64(2), Fq::from_u64(3), Fq::from_u64(5)]
        .map(|coefficient| FqVar::alloc(builder, coefficient));
    let input_claim = FqVar::alloc(
        builder,
        Fq::from_u64(2) + (Fq::from_u64(2) + Fq::from_u64(3) + Fq::from_u64(5)),
    );
    let output_claim = FqVar::alloc(
        builder,
        Fq::from_u64(2)
            + Fq::from_u64(3) * challenge.witness_value()
            + Fq::from_u64(5) * challenge.witness_value() * challenge.witness_value(),
    );
    let round = SumcheckR1csGadgetRound::new(
        input_claim.clone(),
        coefficients.to_vec(),
        challenge.clone(),
        output_claim.clone(),
    );

    append_sumcheck_r1cs_gadget_constraints(builder, SumcheckStatement::new(1, 2), &[round])
        .expect("non-native sumcheck constraints should build");
    push_fq_limb_target(
        tamper_targets,
        "nonnative_sumcheck_input_limb",
        &input_claim,
    );
    push_fq_limb_target(
        tamper_targets,
        "nonnative_sumcheck_coefficient_limb",
        &coefficients[1],
    );
    push_fq_limb_target(
        tamper_targets,
        "nonnative_sumcheck_output_limb",
        &output_claim,
    );
}

fn assign_native_sumcheck_witness(
    builder: &mut R1csBuilder<Fr>,
    round: &SumcheckR1csRoundLayout,
    input_claim: Variable,
    challenge: Fr,
) -> Fr {
    let c0 = Fr::from_u64(3);
    let c1 = Fr::from_u64(4);
    let c2 = Fr::from_u64(5);
    let output_claim = c0 + c1 * challenge + c2 * challenge * challenge;

    assign(builder, input_claim, c0 + (c0 + c1 + c2));
    assign(builder, round.coefficients[0], c0);
    assign(builder, round.coefficients[1], c1);
    assign(builder, round.coefficients[2], c2);
    assign(builder, round.claim_out, output_claim);

    output_claim
}

#[derive(Clone, Debug)]
struct HyraxNativeCase {
    setup: PedersenSetup<GrumpkinPoint>,
    row_commitments: Vec<GrumpkinPoint>,
    row_point_tail: Fq,
    entry_point: Vec<Fq>,
    combined_row: Vec<Fq>,
    combined_blinding: Fq,
    claimed_eval: Fq,
}

impl HyraxNativeCase {
    fn new(row_point_head: Fq) -> Self {
        let generator = Grumpkin::generator();
        let setup = PedersenSetup::new(
            (1..=4)
                .map(|index| generator.scalar_mul(&Fq::from_u64(10 + index)))
                .collect(),
            generator.scalar_mul(&Fq::from_u64(99)),
        );
        let rows = [
            [2, 3, 5, 7],
            [11, 13, 17, 19],
            [23, 29, 31, 37],
            [41, 43, 47, 53],
        ]
        .map(|row| row.map(Fq::from_u64).to_vec());
        let row_blindings = [
            Fq::from_u64(101),
            Fq::from_u64(103),
            Fq::from_u64(107),
            Fq::from_u64(109),
        ];
        let row_point_tail = Fq::from_u64(5);
        let row_point = vec![row_point_head, row_point_tail];
        let entry_point = vec![Fq::from_u64(7), Fq::from_u64(11)];

        let row_commitments = rows
            .iter()
            .zip(row_blindings)
            .map(|(row, blinding)| TestVc::commit(&setup, row, &blinding))
            .collect::<Vec<_>>();
        let row_weights = EqPolynomial::new(row_point).evaluations();
        let entry_weights = EqPolynomial::new(entry_point.clone()).evaluations();
        let zero = Fq::from_u64(0);
        let mut combined_row = vec![zero; 4];
        for (row, row_weight) in rows.iter().zip(&row_weights) {
            for (combined, row_entry) in combined_row.iter_mut().zip(row) {
                *combined += *row_weight * *row_entry;
            }
        }
        let combined_blinding = row_blindings
            .iter()
            .zip(&row_weights)
            .fold(zero, |acc, (blinding, row_weight)| {
                acc + *blinding * *row_weight
            });
        let claimed_eval = combined_row
            .iter()
            .zip(&entry_weights)
            .fold(zero, |acc, (entry, entry_weight)| {
                acc + *entry * *entry_weight
            });

        Self {
            setup,
            row_commitments,
            row_point_tail,
            entry_point,
            combined_row,
            combined_blinding,
            claimed_eval,
        }
    }
}

#[derive(Clone, Debug)]
struct HyraxAllocatedCase {
    input: HyraxOpeningR1csInput<FqVar, GrumpkinPointWithIdentityVar>,
}

impl HyraxAllocatedCase {
    fn alloc(builder: &mut R1csBuilder<Fr>, case: &HyraxNativeCase, row_point_head: FqVar) -> Self {
        Self {
            input: HyraxOpeningR1csInput {
                row_commitments: case
                    .row_commitments
                    .iter()
                    .map(|commitment| GrumpkinPointWithIdentityVar::alloc(builder, commitment))
                    .collect(),
                row_point: vec![row_point_head, FqVar::alloc(builder, case.row_point_tail)],
                entry_point: alloc_fq_vec(builder, &case.entry_point),
                combined_row: alloc_fq_vec(builder, &case.combined_row),
                combined_blinding: FqVar::alloc(builder, case.combined_blinding),
                claimed_eval: FqVar::alloc(builder, case.claimed_eval),
            },
        }
    }

    fn push_tamper_targets(&self, tamper_targets: &mut Vec<TamperTarget>) {
        push_scalar_target(
            tamper_targets,
            "hyrax_row_commitment_x",
            &self.input.row_commitments[0].x,
        );
        push_scalar_target(
            tamper_targets,
            "hyrax_row_commitment_y",
            &self.input.row_commitments[0].y,
        );
        push_scalar_target(
            tamper_targets,
            "hyrax_row_commitment_identity_flag",
            &self.input.row_commitments[0].is_identity,
        );
        push_fq_limb_target(
            tamper_targets,
            "hyrax_row_point_limb",
            &self.input.row_point[1],
        );
        push_fq_limb_target(
            tamper_targets,
            "hyrax_entry_point_limb",
            &self.input.entry_point[0],
        );
        push_fq_limb_target(
            tamper_targets,
            "hyrax_combined_row_limb",
            &self.input.combined_row[0],
        );
        push_fq_limb_target(
            tamper_targets,
            "hyrax_combined_blinding_limb",
            &self.input.combined_blinding,
        );
        push_fq_limb_target(
            tamper_targets,
            "hyrax_claimed_eval_limb",
            &self.input.claimed_eval,
        );
    }
}

fn poseidon_public_arithmetic_output(scalar: Fr, bytes: &[u8]) -> Fr {
    let mut probe = WrapperR1csBuilder::<Fr, Transcript>::new(b"WrapperPoseidonPublic");
    let scalar = probe.alloc_witness_scalar(scalar);
    let bytes = bytes
        .iter()
        .map(|&byte| probe.alloc_witness_byte(byte))
        .collect::<Vec<_>>();
    let byte_sum = byte_sum(&bytes);
    probe
        .transcript
        .append_bytes(&mut probe.builder, b"fixture_bytes", &bytes);
    probe
        .transcript
        .append_scalar(&mut probe.builder, b"fixture_scalar", scalar.clone());
    let challenge = probe.transcript.challenge_scalar(&mut probe.builder);
    challenge.value * scalar.value + byte_sum.value
}

fn byte_sum<F: Field>(bytes: &[AssignedScalar<F>]) -> AssignedScalar<F> {
    let mut value = F::zero();
    let mut lc = LinearCombination::zero();
    for byte in bytes {
        value += byte.value;
        lc = lc + byte.lc.clone();
    }
    AssignedScalar::new(value, lc)
}

fn alloc_fq_vec(builder: &mut R1csBuilder<Fr>, values: &[Fq]) -> Vec<FqVar> {
    values
        .iter()
        .copied()
        .map(|value| FqVar::alloc(builder, value))
        .collect()
}

fn assign(builder: &mut R1csBuilder<Fr>, variable: Variable, value: Fr) {
    builder
        .assign(variable, value)
        .expect("witness variable should be assignable");
}

fn push_scalar_target(
    targets: &mut Vec<TamperTarget>,
    name: impl Into<String>,
    scalar: &AssignedScalar<Fr>,
) {
    push_variable_target(targets, name, variable(scalar));
}

fn push_fq_limb_target(targets: &mut Vec<TamperTarget>, name: impl Into<String>, value: &FqVar) {
    push_scalar_target(targets, name, &value.limbs()[0]);
}

fn push_variable_target(
    targets: &mut Vec<TamperTarget>,
    name: impl Into<String>,
    variable: Variable,
) {
    targets.push(TamperTarget {
        name: name.into(),
        mutation: Mutation::Witness(variable.index()),
    });
}

fn variable(scalar: &AssignedScalar<Fr>) -> Variable {
    let [(variable, coefficient)] = scalar.lc.terms.as_slice() else {
        panic!("assigned scalar should be backed by one variable");
    };
    assert_eq!(*coefficient, Fr::from_u64(1));
    *variable
}

fn fr_to_fq(value: Fr) -> Fq {
    let mut bytes = [0u8; Fr::NUM_BYTES];
    value.to_bytes_le(&mut bytes);
    Fq::from_le_bytes_mod_order(&bytes)
}
