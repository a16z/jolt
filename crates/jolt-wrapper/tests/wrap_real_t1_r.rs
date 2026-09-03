#![expect(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::print_stdout,
    clippy::type_complexity,
    reason = "manual real-fixture integration gate"
)]

use std::path::Path;
use std::time::Instant;

use bincode::config::standard;
use bincode::serde::{decode_from_slice, encode_to_vec};
use common::jolt_device::JoltDevice;
use jolt_crypto::Bn254;
use jolt_field::{Fr, One, Ring, Zero};
use jolt_hyperkzg::{HyperKZGScheme, HyperKZGVerifierSetup};
use jolt_verifier::{JoltProof, JoltVerifierPreprocessing};
use jolt_wrapper::carry::CarryProver;
use jolt_wrapper::hash_table::terms::{
    challenge125, challenge_scalar128, AffineForm as HashAffineForm, LinkMap, WIRED_BIT_BASE,
    WIRED_WORD_BASE,
};
use jolt_wrapper::hash_table::{
    Decoder, Members as HashMembers, StreamColumns, StreamTermExporter, T1Challenges, VkColumn,
};
use jolt_wrapper::relation::{Pcs, ScheduleEntry, SqueezeKind, Vc};
use jolt_wrapper::relation_table::{
    CopyLink, CopyLinkSide, CopyLinkTermExporter, CopyLinkTermSide, RelationTable,
    RelationTableProver, RelationTermExporter, FIXED_COLUMNS, WIRES,
};
use jolt_wrapper::stream::{
    combine_packed_phases, commit_packed, commitment_prefix_challenges, prove_assembly, AffineForm,
    AssemblyMemberStatement, AssemblyStatement, Column, ColumnId, Commitment, CommitmentPhase,
    StageMember, StageMemberSpec, StageProof, Term, TermContext, TermExporter, TermObserver,
    VerifierCost, WrapperProof,
};
use jolt_wrapper::wrap::{verify_wrapped_with_key, WrapConfig, WrapPreparation, WrapVerifierKey};

type Proof = JoltProof<Pcs, Vc>;
type Preprocessing = JoltVerifierPreprocessing<Pcs, Vc>;

const FIXTURE: &str = "/Volumes/Dev/scratch/wrapper-fixtures/fibonacci_2_18_blake3.bin";
const LOG_ROWS: usize = 18;
const ROWS: usize = 1 << LOG_ROWS;
const K: usize = 16;
const PHASE_1_CHALLENGES: usize = 4 + T1Challenges::count(LOG_ROWS);
const PHASE_2_CHALLENGES: usize = 3 * LOG_ROWS + 6;

struct CarryTerms {
    member: usize,
    column: ColumnId,
    source_point: Vec<Fr>,
}

impl TermExporter for CarryTerms {
    fn terms(&self, context: &TermContext<'_>) -> Vec<Term> {
        self.export(context, &mut jolt_hyperkzg::NoopVerifierObserver)
    }

    fn terms_observed(
        &self,
        context: &TermContext<'_>,
        observer: &mut dyn TermObserver,
    ) -> Vec<Term> {
        self.export(context, observer)
    }
}

impl CarryTerms {
    fn export(&self, context: &TermContext<'_>, observer: &mut dyn TermObserver) -> Vec<Term> {
        let eq = self.source_point.iter().zip(context.row_point).fold(
            Fr::one(),
            |result, (&left, &right)| {
                let both = observer.fr_mul(left, right);
                let neither = observer.fr_mul(Fr::one() - left, Fr::one() - right);
                observer.fr_mul(result, both + neither)
            },
        );
        vec![Term {
            coefficient: observer.fr_mul(context.batching_coefficients[self.member], eq),
            factors: vec![column_form(self.column)],
        }]
    }
}

#[test]
#[ignore = "manual real fibonacci 2^18 wrapper gate"]
fn real_t1_relation_table_round_trip_and_tampers() {
    let uptime = std::process::Command::new("uptime")
        .output()
        .expect("uptime")
        .stdout;
    let started = Instant::now();
    let (preprocessing, public_io, original_proof) = fixture();
    let preparation = WrapPreparation::new(
        &preprocessing,
        &public_io,
        &original_proof,
        WrapConfig::default(),
    )
    .expect("prepare real wrapper inputs");
    let prepare_ms = started.elapsed().as_millis();

    let started = Instant::now();
    let setup = HyperKZGScheme::<Bn254>::setup_from_secret(
        Fr::from_u64(0x5eed),
        ROWS * K,
        Bn254::g1_generator(),
        Bn254::g2_generator(),
    );
    let setup_ms = started.elapsed().as_millis();

    let started = Instant::now();
    let hash_columns = StreamColumns::new(&preparation.hash_table, K, 0);
    assert_eq!(hash_columns.group_count, 22);
    let relation_table =
        RelationTable::from_relation(&preparation.relation, ROWS).expect("lower R table");
    let mut relation_witness = relation_table
        .wire_witness(&preparation.relation_witness.values)
        .expect("R wires");
    let links = LinkMap::new(&preparation.hash_key);
    let (copy_link, link_columns, left_values, right_values) = challenge_copy_link(
        &links,
        &preparation.relation,
        relation_table.cell_layout().challenge_base,
        &preparation.hash_table,
        relation_witness.evaluations()[0].clone(),
    );
    let adapt_ms = started.elapsed().as_millis();

    let mut phase_1_columns = hash_columns.columns;
    let relation_fixed_base = phase_1_columns.len();
    phase_1_columns.extend(relation_table.fixed_columns());
    pad_fr(&mut phase_1_columns);
    let relation_wire_base = phase_1_columns.len();
    phase_1_columns.extend(
        relation_witness.evaluations()[..WIRES]
            .iter()
            .cloned()
            .map(Column::Fr),
    );
    pad_fr(&mut phase_1_columns);
    let link_fixed_base = phase_1_columns.len();
    phase_1_columns.extend(link_columns.into_iter().map(Column::Fr));
    pad_fr(&mut phase_1_columns);
    let phase_1_groups = phase_1_columns.len() / K;
    assert_eq!(phase_1_groups, 25);

    let started = Instant::now();
    let phase_1 = commit_packed(&phase_1_columns, K, &setup).expect("phase 1 commitments");
    let phase_1_commit_ms = started.elapsed().as_millis();
    let phase_1_values = commitment_prefix_challenges(
        &preparation.profile_digest,
        &preparation.public_known,
        &[(&phase_1.commitments, PHASE_1_CHALLENGES)],
    );
    let relation_beta = phase_1_values[0];
    let relation_gamma = phase_1_values[1];
    let copy_beta = phase_1_values[2];
    let copy_gamma = phase_1_values[3];
    let hash_challenges = T1Challenges::from_challenges(&phase_1_values[4..], LOG_ROWS);
    let hash_relation = hash_challenges.relation();

    let started = Instant::now();
    relation_table
        .add_copy_helpers(&mut relation_witness, relation_beta, relation_gamma)
        .expect("R helpers");
    relation_table
        .check_witness(&relation_witness, relation_beta, relation_gamma)
        .expect("R witness");
    let copy_witness = copy_link
        .witness(left_values, right_values, copy_beta, copy_gamma)
        .expect("T1-R challenge link");
    copy_link
        .check(&copy_witness, copy_beta, copy_gamma)
        .expect("T1-R challenge equality");
    let mut phase_2_columns = relation_witness.evaluations()[WIRES..]
        .iter()
        .cloned()
        .map(Column::Fr)
        .collect::<Vec<_>>();
    phase_2_columns.extend(copy_witness.helpers.iter().cloned().map(Column::Fr));
    phase_2_columns.push(Column::Fr(vec![Fr::zero(); ROWS]));
    pad_fr(&mut phase_2_columns);
    let helper_ms = started.elapsed().as_millis();
    let started = Instant::now();
    let phase_2 = commit_packed(&phase_2_columns, K, &setup).expect("phase 2 commitments");
    let phase_2_commit_ms = started.elapsed().as_millis();
    let phase_2_group = phase_1_groups;
    let full_challenges = commitment_prefix_challenges(
        &preparation.profile_digest,
        &preparation.public_known,
        &[
            (&phase_1.commitments, PHASE_1_CHALLENGES),
            (&phase_2.commitments, PHASE_2_CHALLENGES),
        ],
    );
    assert_eq!(&full_challenges[..PHASE_1_CHALLENGES], phase_1_values);
    let mut cursor = PHASE_1_CHALLENGES;
    let tau_relation = take_point(&full_challenges, &mut cursor);
    let tau_copy = take_point(&full_challenges, &mut cursor);
    let tau_t2 = take_point(&full_challenges, &mut cursor);
    let relation_weights = take_array(&full_challenges, &mut cursor);
    let copy_weights = take_array(&full_challenges, &mut cursor);
    assert_eq!(cursor, full_challenges.len());

    let packed = combine_packed_phases(vec![phase_1, phase_2]).expect("combine phases");
    let relation_columns = std::array::from_fn(|column| {
        if column < FIXED_COLUMNS {
            physical_id(relation_fixed_base + column)
        } else if column < FIXED_COLUMNS + WIRES {
            physical_id(relation_wire_base + column - FIXED_COLUMNS)
        } else {
            physical_id((phase_2_group * K) + column - FIXED_COLUMNS - WIRES)
        }
    });
    let link_left_selectors = std::array::from_fn(|wire| physical_id(link_fixed_base + wire));
    let link_left_ids =
        std::array::from_fn(|wire| column_form(physical_id(link_fixed_base + WIRES + wire)));
    let link_right_selectors =
        std::array::from_fn(|wire| physical_id(link_fixed_base + 2 * WIRES + wire));
    let link_right_ids =
        std::array::from_fn(|wire| column_form(physical_id(link_fixed_base + 3 * WIRES + wire)));
    let relation_a = relation_columns[FIXED_COLUMNS];
    let copy_helpers = [
        physical_id(phase_2_group * K + 2),
        physical_id(phase_2_group * K + 3),
    ];
    let t2_column = physical_id(phase_2_group * K + 4);

    let HashMembers {
        rows: mut hash_rows,
        wiring: mut hash_wiring,
        input_claims: hash_input_claims,
    } = HashMembers::new(&preparation.hash_table, &hash_relation, &hash_challenges);
    let mut relation_rows = RelationTableProver::new(
        &relation_table,
        &relation_witness,
        tau_relation.clone(),
        relation_beta,
        relation_gamma,
        relation_weights,
    );
    let mut copy_rows = copy_link.prover(
        &copy_witness,
        tau_copy.clone(),
        copy_beta,
        copy_gamma,
        copy_weights,
    );
    let mut t2_standin =
        CarryProver::new(&vec![Fr::zero(); ROWS], &tau_t2, Fr::zero()).expect("T2 stand-in");
    assert!(hash_rows.input_claim().is_zero());
    assert!(relation_rows.input_claim().is_zero());
    assert!(copy_rows.input_claim().is_zero());

    let input_claims = [
        hash_input_claims[0],
        hash_input_claims[1],
        relation_rows.input_claim(),
        copy_rows.input_claim(),
        Fr::zero(),
    ];
    let statement = AssemblyStatement {
        key_digest: preparation.profile_digest,
        public_inputs: preparation.public_known.clone(),
        rows: ROWS,
        column_count: packed.layout.column_count,
        k: K,
        members: input_claims
            .iter()
            .enumerate()
            .map(|(index, &input_claim)| AssemblyMemberStatement {
                input_claim,
                spec: StageMemberSpec {
                    rounds: LOG_ROWS,
                    degree: [3, 3, 5, 5, 2][index],
                    offset: 0,
                },
            })
            .collect(),
        commitment_phases: vec![
            CommitmentPhase {
                group_count: phase_1_groups,
                challenge_count: PHASE_1_CHALLENGES,
            },
            CommitmentPhase {
                group_count: 1,
                challenge_count: PHASE_2_CHALLENGES,
            },
        ],
    };
    let verifier_key = WrapVerifierKey::new(
        statement,
        preparation.hash_key.clone(),
        [20, 21, relation_fixed_base / K, link_fixed_base / K]
            .into_iter()
            .map(|group| (group, packed.commitments[group]))
            .collect(),
    );
    assert_eq!(verifier_key.hash_links(), &links);
    assert_eq!(verifier_key.hash_schedule(), &preparation.hash_key);

    let hash_exporter = StreamTermExporter {
        log_rows: LOG_ROWS,
        challenge_offset: 4,
        public: &preparation.hash_public,
        columns: &hash_columns.ids,
        row_member: 0,
        wiring_member: 1,
    };
    let relation_exporter = RelationTermExporter {
        rows: ROWS,
        columns: relation_columns,
        tau: &tau_relation,
        beta: relation_beta,
        gamma: relation_gamma,
        relation_weights,
        member_index: 2,
    };
    let copy_exporter = CopyLinkTermExporter {
        link: &copy_link,
        left: CopyLinkTermSide {
            selectors: link_left_selectors,
            ids: link_left_ids,
            values: [
                map_hash_form(&challenge125(), &hash_columns.ids),
                map_hash_form(&challenge_scalar128(), &hash_columns.ids),
                AffineForm {
                    constant: Fr::zero(),
                    weights: Vec::new(),
                },
            ],
            helper: copy_helpers[0],
        },
        right: CopyLinkTermSide {
            selectors: link_right_selectors,
            ids: link_right_ids,
            values: [
                column_form(relation_a),
                column_form(relation_a),
                AffineForm {
                    constant: Fr::zero(),
                    weights: Vec::new(),
                },
            ],
            helper: copy_helpers[1],
        },
        tau: &tau_copy,
        beta: copy_beta,
        gamma: copy_gamma,
        relation_weights: copy_weights,
        member_index: 3,
    };
    let t2_exporter = CarryTerms {
        member: 4,
        column: t2_column,
        source_point: tau_t2,
    };
    let exporters: [&dyn TermExporter; 4] = [
        &hash_exporter,
        &relation_exporter,
        &copy_exporter,
        &t2_exporter,
    ];
    let mut members = [
        StageMember {
            prover: &mut hash_rows,
            input_claim: input_claims[0],
            degree: 3,
            offset: 0,
        },
        StageMember {
            prover: &mut hash_wiring,
            input_claim: input_claims[1],
            degree: 3,
            offset: 0,
        },
        StageMember {
            prover: &mut relation_rows,
            input_claim: input_claims[2],
            degree: 5,
            offset: 0,
        },
        StageMember {
            prover: &mut copy_rows,
            input_claim: input_claims[3],
            degree: 5,
            offset: 0,
        },
        StageMember {
            prover: &mut t2_standin,
            input_claim: input_claims[4],
            degree: 2,
            offset: 0,
        },
    ];
    let started = Instant::now();
    let wrapped = prove_assembly(
        &packed,
        &verifier_key.statement,
        &mut members,
        &exporters,
        &setup,
    )
    .expect("prove real T1/R wrapper");
    let prove_ms = started.elapsed().as_millis();
    let verifier_setup = HyperKZGVerifierSetup::from(&setup);
    let started = Instant::now();
    let (_, cost) = verify_wrapped_with_key(&verifier_key, &wrapped, &exporters, &verifier_setup)
        .expect("verify real T1/R wrapper");
    let verify_ms = started.elapsed().as_millis();

    tamper_suite(&wrapped, |proof| {
        verify_wrapped_with_key(&verifier_key, proof, &exporters, &verifier_setup).is_err()
    });
    report(
        &wrapped,
        phase_1_groups,
        cost,
        [
            prepare_ms,
            setup_ms,
            adapt_ms,
            phase_1_commit_ms,
            helper_ms,
            phase_2_commit_ms,
            prove_ms,
            verify_ms,
        ],
        &uptime,
    );
}

fn fixture() -> (Preprocessing, JoltDevice, Proof) {
    let bytes = std::fs::read(Path::new(FIXTURE)).expect("cached fibonacci fixture");
    decode_from_slice(&bytes, standard())
        .expect("decode cached fibonacci fixture")
        .0
}

fn take_point(challenges: &[Fr], cursor: &mut usize) -> Vec<Fr> {
    let point = challenges[*cursor..*cursor + LOG_ROWS].to_vec();
    *cursor += LOG_ROWS;
    point
}

fn take_array(challenges: &[Fr], cursor: &mut usize) -> [Fr; 3] {
    let values = challenges[*cursor..*cursor + 3]
        .try_into()
        .expect("three weights");
    *cursor += 3;
    values
}

fn physical_id(index: usize) -> ColumnId {
    ColumnId {
        group: index / K,
        slot: index % K,
    }
}

fn column_form(column: ColumnId) -> AffineForm {
    AffineForm {
        constant: Fr::zero(),
        weights: vec![(column, Fr::one())],
    }
}

fn map_hash_form(form: &HashAffineForm, columns: &[ColumnId]) -> AffineForm {
    AffineForm {
        constant: form.constant,
        weights: form
            .weights
            .iter()
            .map(|&(column, weight)| (columns[column], weight))
            .collect(),
    }
}

fn pad_fr(columns: &mut Vec<Column>) {
    while !columns.len().is_multiple_of(K) {
        columns.push(Column::Fr(vec![Fr::zero(); ROWS]));
    }
}

fn challenge_copy_link(
    links: &LinkMap,
    relation: &jolt_wrapper::relation::Relation,
    relation_base: usize,
    table: &jolt_wrapper::hash_table::HashTable,
    relation_a: Vec<Fr>,
) -> (CopyLink, Vec<Vec<Fr>>, [Vec<Fr>; 3], [Vec<Fr>; 3]) {
    let relation_squeezes = relation
        .link
        .schedule
        .iter()
        .filter_map(|entry| match entry {
            ScheduleEntry::Squeeze { kind, .. } => Some(*kind),
            ScheduleEntry::Bytes(_) | ScheduleEntry::Fr(_) | ScheduleEntry::Opaque { .. } => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(links.challenges.len(), relation_squeezes.len());
    let mut left_selectors = std::array::from_fn(|_| vec![Fr::zero(); ROWS]);
    let mut left_ids = std::array::from_fn(|_| vec![Fr::zero(); ROWS]);
    let mut right_selectors = std::array::from_fn(|_| vec![Fr::zero(); ROWS]);
    let mut right_ids = std::array::from_fn(|_| vec![Fr::zero(); ROWS]);
    for (index, ((squeeze, left_row), kind)) in
        links.challenges.iter().zip(relation_squeezes).enumerate()
    {
        let wire = match (squeeze.decoder, kind) {
            (Decoder::Challenge125, SqueezeKind::Challenge) => 0,
            (Decoder::Scalar128, SqueezeKind::Scalar) => 1,
            _ => panic!("T1/R squeeze decoder mismatch at {index}"),
        };
        let id = Fr::from_u64(index as u64 + 1);
        left_selectors[wire][*left_row] = Fr::one();
        left_ids[wire][*left_row] = id;
        right_selectors[wire][relation_base + index] = Fr::one();
        right_ids[wire][relation_base + index] = id;
    }
    let left = CopyLinkSide::new(left_selectors.clone(), left_ids.clone()).expect("T1 link side");
    let right = CopyLinkSide::new(right_selectors.clone(), right_ids.clone()).expect("R link side");
    let copy = CopyLink::new(left, right).expect("challenge link");
    let left_values = [
        materialize_hash_form(&challenge125(), table),
        materialize_hash_form(&challenge_scalar128(), table),
        vec![Fr::zero(); ROWS],
    ];
    let right_values = [relation_a.clone(), relation_a, vec![Fr::zero(); ROWS]];
    let columns = left_selectors
        .into_iter()
        .chain(left_ids)
        .chain(right_selectors)
        .chain(right_ids)
        .collect();
    (copy, columns, left_values, right_values)
}

fn materialize_hash_form(
    form: &HashAffineForm,
    table: &jolt_wrapper::hash_table::HashTable,
) -> Vec<Fr> {
    let mut values = vec![form.constant; ROWS];
    for &(column, weight) in &form.weights {
        for (row, value) in values.iter_mut().enumerate() {
            *value += weight * hash_column_value(table, column, row);
        }
    }
    values
}

fn hash_column_value(table: &jolt_wrapper::hash_table::HashTable, column: usize, row: usize) -> Fr {
    if column < WIRED_BIT_BASE {
        Fr::from_u64(u64::from(table.bits[column][row]))
    } else if column < WIRED_WORD_BASE {
        Fr::from_u64(u64::from(table.wired_bits[column - WIRED_BIT_BASE][row]))
    } else if column < WIRED_WORD_BASE + table.wired_words.len() {
        Fr::from_u64(u64::from(table.wired_words[column - WIRED_WORD_BASE][row]))
    } else {
        let vk = VkColumn::ALL[column - WIRED_WORD_BASE - table.wired_words.len()];
        Fr::from_u64(u64::from(table.vk.value(vk, row)))
    }
}

fn tamper_suite(proof: &WrapperProof, rejected: impl Fn(&WrapperProof) -> bool) {
    let original = proof.clone();
    let tamper = |edit: &dyn Fn(&mut WrapperProof)| {
        let mut candidate = original.clone();
        edit(&mut candidate);
        assert!(rejected(&candidate));
    };
    tamper(&|candidate| candidate.commitments[0] = Commitment::new(original.opening.com[0]));
    tamper(&|candidate| candidate.commitments[21] = Commitment::new(original.opening.com[0]));
    tamper(&|candidate| {
        candidate.stages[0]
            .committed_rounds
            .as_mut()
            .expect("stage A")
            .sum_at_zero += Fr::one();
    });
    tamper(&|candidate| {
        candidate.stages[1]
            .committed_rounds
            .as_mut()
            .expect("term stage")
            .round_commitments[0] = Bn254::g1_generator();
    });
    tamper(&|candidate| candidate.term_evaluations[0] += Fr::one());
    tamper(&|candidate| candidate.reduced_claims[0] += Fr::one());
    let mut opening = original;
    opening.opening.v[0][0] += Fr::one();
    assert!(rejected(&opening));
}

fn report(
    proof: &WrapperProof,
    phase_1_groups: usize,
    cost: VerifierCost,
    times: [u128; 8],
    uptime: &[u8],
) {
    let stage_a = committed_stage_bytes(&proof.stages[0]);
    let term_stage = committed_stage_bytes(&proof.stages[1]);
    let shared = 96 * usize::from(proof.round_opening.is_some());
    let ell = 32 * proof.term_evaluations.len();
    let stage_b = clear_stage_bytes(&proof.stages[2]);
    let reduced = 32 * proof.reduced_claims.len();
    let phase_1 = 32 * phase_1_groups;
    let phase_2 = 32 * (proof.commitments.len() - phase_1_groups);
    let opening = proof.payload_bytes()
        - phase_1
        - phase_2
        - stage_a
        - term_stage
        - shared
        - ell
        - stage_b
        - reduced;
    let serialized = encode_to_vec(proof, standard()).expect("serialize wrapper");
    assert_eq!(serialized.len(), proof.bincode_bytes());
    let [prepare, setup, adapt, commit_1, helpers, commit_2, prove, verify] = times;
    println!("uptime={}", String::from_utf8_lossy(uptime).trim());
    println!(
        "phases_ms prepare={prepare} setup={setup} adapt={adapt} commit1={commit_1} helpers={helpers} commit2={commit_2} prove={prove} verify={verify}"
    );
    println!(
        "bytes phase1={phase_1} phase2={phase_2} stage_a={stage_a} term={term_stage} shared_bdfg={shared} ell={ell} stage_b={stage_b} reduced={reduced} hyperkzg={opening} io_challenges=0 proof={} bincode={} public_known={}",
        proof.payload_bytes(),
        proof.bincode_bytes(),
        32 * 7,
    );
    println!("cost={cost:?} gas={}", estimated_gas(cost, proof));
}

fn committed_stage_bytes(stage: &StageProof) -> usize {
    let committed = stage.committed_rounds.as_ref().expect("committed stage");
    32 * (committed.round_commitments.len() + committed.round_claims.len() + 1)
}

fn clear_stage_bytes(stage: &StageProof) -> usize {
    32 * stage
        .round_polynomials
        .round_polynomials
        .iter()
        .map(|round| round.coeffs_except_linear_term().len())
        .sum::<usize>()
}

fn estimated_gas(cost: VerifierCost, proof: &WrapperProof) -> usize {
    let proof_g1 = proof.commitments.len()
        + proof
            .stages
            .iter()
            .filter_map(|stage| stage.committed_rounds.as_ref())
            .map(|stage| stage.round_commitments.len() + 3 * usize::from(stage.opening.is_some()))
            .sum::<usize>()
        + 3 * usize::from(proof.round_opening.is_some())
        + proof.opening.com.len()
        + 1;
    let calldata = proof.payload_bytes() + 32 * proof_g1 + 7 * 32;
    21_000
        + 16 * calldata
        + 7_700 * cost.ec_mul
        + 20 * cost.fr_mul
        + batched_inversion_gas(cost.fr_inv)
        + 100 * cost.keccak
        + 2 * 114_700
        + 183_400
}

fn batched_inversion_gas(inversions: usize) -> usize {
    if inversions == 0 {
        return 0;
    }
    let multiplication_complexity = 32usize.div_ceil(8).pow(2);
    let iteration_count = 253;
    let modexp = (multiplication_complexity * iteration_count / 3).max(200);
    modexp + 3 * (inversions - 1) * 20
}
