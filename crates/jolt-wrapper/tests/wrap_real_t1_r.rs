#![expect(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::print_stdout,
    clippy::type_complexity,
    reason = "manual real-fixture integration gate"
)]

use std::path::Path;
use std::process::Command;
use std::time::Instant;

use bincode::config::standard;
use bincode::serde::{decode_from_slice, encode_to_vec};
use common::jolt_device::JoltDevice;
use jolt_crypto::Bn254;
use jolt_field::{Fr, One, Ring, Zero};
use jolt_hyperkzg::{HyperKZGScheme, HyperKZGVerifierSetup};
use jolt_verifier::{JoltProof, JoltVerifierPreprocessing};
use jolt_wrapper::hash_table::terms::{
    challenge125, challenge_scalar128, AffineForm as HashAffineForm, LinkMap, WIRED_BIT_BASE,
    WIRED_WORD_BASE,
};
use jolt_wrapper::hash_table::{
    Decoder, HashTable, Members as HashMembers, StreamColumns, StreamTermExporter, T1Challenges,
    VkColumn,
};
use jolt_wrapper::limb_table::digit_link::LinkMember;
use jolt_wrapper::limb_table::lookup::omega_column;
use jolt_wrapper::limb_table::relation::{LookupConstants, RowRelation, RowSumcheck};
use jolt_wrapper::relation::{Pcs, Relation, ScheduleEntry, SqueezeKind, Vc};
use jolt_wrapper::relation_table::{
    CopyLink, CopyLinkSide, CopyLinkTermExporter, CopyLinkTermSide, DoryScalarLink,
    DoryScalarTermExporter, RelationTable, RelationTableProver, RelationTermExporter,
    FIXED_COLUMNS, WIRES,
};
use jolt_wrapper::stream::{
    commit_packed, AffineForm, AssemblyMemberStatement, AssemblyStatement, Column, ColumnId,
    Commitment, CommitmentPhase, StageMember, StageMemberSpec, StageProof, TermContext,
    TermExporter, VerifierCost, WrapperProof,
};
use jolt_wrapper::wrap::{
    commit_wrap_phase_one, commit_wrap_phase_two, verify_wrapped_with_key, wrap as wrap_proof,
    DoryLinkPlacement, T1Placement, WrapConfig, WrapError, WrapHashKey, WrapPreparation,
    WrapVerifierKey,
};

#[path = "wrap_real_t1_r/t2.rs"]
mod t2;
use t2::{
    Base as T2Base, Exporter as T2Exporter, LinkProver, NegatingExporter, CHALLENGE_COUNT,
    PHASE_ONE_COLUMNS, PHASE_TWO_COLUMNS,
};

type Proof = JoltProof<Pcs, Vc>;
type Preprocessing = JoltVerifierPreprocessing<Pcs, Vc>;

const FIXTURE: &str = "/Volumes/Dev/scratch/wrapper-fixtures/fibonacci_2_18_blake3.bin";
const LOG_ROWS: usize = 18;
const ROWS: usize = 1 << LOG_ROWS;
const PHASE_1_CHALLENGES: usize = 4 + T1Challenges::count(LOG_ROWS) + CHALLENGE_COUNT;
const PHASE_2_CHALLENGES: usize = 2 * LOG_ROWS + 6;

#[test]
#[ignore = "manual real fibonacci 2^18 wrapper gate"]
fn real_t1_relation_table_round_trip_and_tampers() {
    let k = std::env::var("WRAP_K")
        .map_or(Ok(32), |value| value.parse())
        .expect("WRAP_K is an integer");
    assert!(matches!(k, 16 | 32));
    let config = WrapConfig {
        common_log_rows: LOG_ROWS,
        packing_factor: k,
    };
    let uptime = Command::new("uptime").output().expect("uptime").stdout;
    let started = Instant::now();
    let (preprocessing, public_io, original_proof) = fixture();
    let setup = HyperKZGScheme::<Bn254>::setup_from_secret(
        Fr::from_u64(0x5eed),
        ROWS * k,
        Bn254::g1_generator(),
        Bn254::g2_generator(),
    );
    let setup_ms = started.elapsed().as_millis();
    let started = Instant::now();
    let hash_key = WrapHashKey::from_reference(
        &preprocessing,
        &public_io,
        &original_proof,
        config,
        T1Placement {
            group_offset: 0,
            challenge_offset: 4,
            members: [0, 1],
        },
        &setup,
    )
    .expect("build trusted T1 key");
    let key_profile_ms = started.elapsed().as_millis();
    let started = Instant::now();
    let mut preparation = WrapPreparation::new(
        &preprocessing,
        &public_io,
        &original_proof,
        config,
        &hash_key,
    )
    .expect("prepare real wrapper inputs");
    let mut wrong_shape = original_proof.clone();
    wrong_shape.trace_length *= 2;
    assert!(matches!(
        WrapPreparation::new(&preprocessing, &public_io, &wrong_shape, config, &hash_key,),
        Err(WrapError::ProfileMismatch)
    ));
    let prepare_ms = started.elapsed().as_millis();

    let started = Instant::now();
    t2::retain_used_links(&original_proof, &mut preparation.relation);
    let hash_columns = StreamColumns::new(&preparation.hash_table, k, 0);
    assert_eq!(hash_columns.vk_groups.end, hash_columns.group_count);
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
    let t2_base = T2Base::new(
        &preprocessing,
        &original_proof,
        &preparation.relation,
        &preparation.relation_witness,
    )
    .expect("adapt real Dory opening");
    let adapt_ms = started.elapsed().as_millis();

    let mut phase_1_columns = hash_columns.columns;
    let relation_fixed_base = phase_1_columns.len();
    phase_1_columns.extend(relation_table.fixed_columns());
    pad_fr(&mut phase_1_columns, k);
    let relation_wire_base = phase_1_columns.len();
    phase_1_columns.extend(
        relation_witness.evaluations()[..WIRES]
            .iter()
            .cloned()
            .map(Column::Fr),
    );
    pad_fr(&mut phase_1_columns, k);
    let link_fixed_base = phase_1_columns.len();
    phase_1_columns.extend(link_columns.into_iter().map(Column::Fr));
    pad_fr(&mut phase_1_columns, k);
    let t2_phase_one_base = phase_1_columns.len();
    phase_1_columns.extend(t2_base.phase_one());
    pad_fr(&mut phase_1_columns, k);
    let t2_vk_base = phase_1_columns.len();
    phase_1_columns.extend(t2_base.vk());
    pad_fr(&mut phase_1_columns, k);
    let phase_1_groups = phase_1_columns.len() / k;
    let phase_2_groups = 1 + PHASE_TWO_COLUMNS.div_ceil(k);

    let started = Instant::now();
    let pinned_groups = [relation_fixed_base / k, link_fixed_base / k, t2_vk_base / k];
    let pinned_commitments = pinned_groups
        .into_iter()
        .map(|group| {
            let start = group * k;
            let packed = commit_packed(&phase_1_columns[start..start + k], k, &setup)
                .expect("verifier-key group commitment");
            (group, packed.commitments[0])
        })
        .collect();
    let statement = AssemblyStatement {
        key_digest: preparation.profile_digest,
        public_inputs: preparation.public_known.clone(),
        rows: ROWS,
        column_count: phase_1_columns.len() + phase_2_groups * k,
        k,
        members: [3, 3, 5, 5, 5, 2]
            .into_iter()
            .map(|degree| AssemblyMemberStatement {
                input_claim: Fr::zero(),
                spec: StageMemberSpec {
                    rounds: LOG_ROWS,
                    degree,
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
                group_count: phase_2_groups,
                challenge_count: PHASE_2_CHALLENGES,
            },
        ],
        pinned_commitments: Vec::new(),
    };
    let verifier_key = WrapVerifierKey::new(
        statement,
        hash_key,
        preparation.hash_public.clone(),
        Some(DoryLinkPlacement {
            challenge: PHASE_1_CHALLENGES - 1,
            member: 5,
            scalar_count: t2_base.inputs.wire_order.len(),
        }),
        pinned_commitments,
    );
    assert_eq!(verifier_key.hash_links(), &links);
    assert_eq!(verifier_key.hash_schedule(), &preparation.hash_key);
    let key_commit_ms = started.elapsed().as_millis();

    let started = Instant::now();
    let phase_1 = commit_wrap_phase_one(&phase_1_columns, &verifier_key, &setup)
        .expect("phase 1 commitments");
    let phase_1_commit_ms = started.elapsed().as_millis();
    let phase_1_values = phase_1.challenges().to_vec();
    let relation_beta = phase_1_values[0];
    let relation_gamma = phase_1_values[1];
    let copy_beta = phase_1_values[2];
    let copy_gamma = phase_1_values[3];
    let t2_challenge_start = 4 + T1Challenges::count(LOG_ROWS);
    let hash_challenges =
        T1Challenges::from_challenges(&phase_1_values[4..t2_challenge_start], LOG_ROWS);
    let hash_relation = hash_challenges.relation();
    let (t2_challenges, t2_rho) = t2::challenges(&phase_1_values[t2_challenge_start..]);
    let t2_relation = RowRelation::new(
        t2_challenges,
        LookupConstants {
            one_row: t2_base.layout.one_cell * 16,
        },
    );

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
    let t2_claimed = t2_base.claimed(&t2_relation);
    let t2_matrix = t2_base.matrix(&t2_relation, &t2_claimed);
    let mut phase_2_columns = relation_witness.evaluations()[WIRES..]
        .iter()
        .cloned()
        .map(Column::Fr)
        .collect::<Vec<_>>();
    phase_2_columns.extend(copy_witness.helpers.iter().cloned().map(Column::Fr));
    pad_fr(&mut phase_2_columns, k);
    let t2_phase_two_base = phase_1_columns.len() + phase_2_columns.len();
    let mut t2_phase_two = t2_claimed
        .phase_two()
        .iter()
        .cloned()
        .map(Column::Fr)
        .collect::<Vec<_>>();
    t2::bit_reverse_columns(&mut t2_phase_two);
    phase_2_columns.extend(t2_phase_two);
    pad_fr(&mut phase_2_columns, k);
    assert_eq!(phase_2_columns.len(), phase_2_groups * k);
    let helper_ms = started.elapsed().as_millis();
    let started = Instant::now();
    let committed = commit_wrap_phase_two(phase_1, &phase_2_columns, &verifier_key, &setup)
        .expect("phase 2 commitments");
    let phase_2_commit_ms = started.elapsed().as_millis();
    let phase_2_group = phase_1_groups;
    let full_challenges = committed.challenges();
    assert_eq!(&full_challenges[..PHASE_1_CHALLENGES], phase_1_values);
    let mut cursor = PHASE_1_CHALLENGES;
    let tau_relation = take_point(full_challenges, &mut cursor);
    let tau_copy = take_point(full_challenges, &mut cursor);
    let relation_weights = take_array(full_challenges, &mut cursor);
    let copy_weights = take_array(full_challenges, &mut cursor);
    assert_eq!(cursor, full_challenges.len());
    let assembly_challenges = full_challenges.to_vec();

    let relation_columns = std::array::from_fn(|column| {
        if column < FIXED_COLUMNS {
            physical_id(relation_fixed_base + column, k)
        } else if column < FIXED_COLUMNS + WIRES {
            physical_id(relation_wire_base + column - FIXED_COLUMNS, k)
        } else {
            physical_id((phase_2_group * k) + column - FIXED_COLUMNS - WIRES, k)
        }
    });
    let link_left_selectors = std::array::from_fn(|wire| physical_id(link_fixed_base + wire, k));
    let link_left_ids =
        std::array::from_fn(|wire| column_form(physical_id(link_fixed_base + WIRES + wire, k)));
    let link_right_selectors =
        std::array::from_fn(|wire| physical_id(link_fixed_base + 2 * WIRES + wire, k));
    let link_right_ids =
        std::array::from_fn(|wire| column_form(physical_id(link_fixed_base + 3 * WIRES + wire, k)));
    let relation_a = relation_columns[FIXED_COLUMNS];
    let copy_helpers = [
        physical_id(phase_2_group * k + 2, k),
        physical_id(phase_2_group * k + 3, k),
    ];
    let t2_columns = t2::column_ids(t2_phase_one_base, t2_phase_two_base, t2_vk_base, k);

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
    let mut t2_rows = RowSumcheck::new(&t2_relation, &t2_matrix);
    let scalar_link = DoryScalarLink::new(ROWS, relation_table.cell_layout(), t2_rho);
    let scalar_prover = scalar_link.prover(&relation_witness);
    let digit_prover = LinkMember::new(
        omega_column(&t2_base.layout, t2_rho),
        &t2_base.public.digit_values,
    );
    let dory_link_claim =
        (0..t2_base.inputs.wire_order.len()).fold(Fr::one(), |power, _| power * t2_rho);
    let mut dory_link = LinkProver::new(digit_prover, scalar_prover, dory_link_claim);
    assert!(hash_rows.input_claim().is_zero());
    assert!(relation_rows.input_claim().is_zero());
    assert!(copy_rows.input_claim().is_zero());
    assert!(t2_rows.input_claim().is_zero());

    let input_claims = [
        hash_input_claims[0],
        hash_input_claims[1],
        relation_rows.input_claim(),
        copy_rows.input_claim(),
        Fr::zero(),
        dory_link_claim,
    ];
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
    let t2_exporter = T2Exporter {
        layout: &t2_base.layout,
        relation: &t2_relation,
        columns: &t2_columns,
        rho: t2_rho,
        row_member: 4,
        digit_member: 5,
    };
    let scalar_exporter = DoryScalarTermExporter {
        link: &scalar_link,
        wire: relation_a,
        member_index: 5,
    };
    let negative_scalar_exporter = NegatingExporter(&scalar_exporter);
    let exporters: [&dyn TermExporter; 5] = [
        &hash_exporter,
        &relation_exporter,
        &copy_exporter,
        &t2_exporter,
        &negative_scalar_exporter,
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
            prover: &mut t2_rows,
            input_claim: input_claims[4],
            degree: 5,
            offset: 0,
        },
        StageMember {
            prover: &mut dory_link,
            input_claim: input_claims[5],
            degree: 2,
            offset: 0,
        },
    ];
    let started = Instant::now();
    let wrapped = wrap_proof(committed, &verifier_key, &mut members, &exporters, &setup)
        .expect("prove real T1/T2/R wrapper");
    let prove_ms = started.elapsed().as_millis();
    let verifier_setup = HyperKZGVerifierSetup::from(&setup);
    let started = Instant::now();
    let (results, cost) =
        verify_wrapped_with_key(&verifier_key, &wrapped, &exporters, &verifier_setup)
            .expect("verify real T1/T2/R wrapper");
    let verify_ms = started.elapsed().as_millis();
    let term_context = TermContext {
        row_point: &results[0].point,
        batching_coefficients: &results[0].coefficients,
        challenges: &assembly_challenges,
    };
    let term_count = exporters
        .iter()
        .map(|exporter| exporter.terms(&term_context).len())
        .sum();

    let t2_wire_index = hash_columns.group_count - hash_columns.vk_groups.len() + 1;
    tamper_suite(&wrapped, t2_wire_index, |proof| {
        verify_wrapped_with_key(&verifier_key, proof, &exporters, &verifier_setup).is_err()
    });
    report(
        &wrapped,
        phase_1_groups - 5,
        term_count,
        cost,
        [
            key_profile_ms,
            prepare_ms,
            setup_ms,
            adapt_ms,
            key_commit_ms,
            phase_1_commit_ms,
            helper_ms,
            phase_2_commit_ms,
            prove_ms,
            verify_ms,
        ],
        &uptime,
    );
    println!(
        "groups k={k} t1_sent={} t1_vk={} r=2 copy_vk=1 t2_phase1={} t2_vk=1 r_helpers=1 t2_phase2={} phase1_full={phase_1_groups} phase1_wire={} phase2={phase_2_groups}",
        hash_columns.group_count - hash_columns.vk_groups.len(),
        hash_columns.vk_groups.len(),
        PHASE_ONE_COLUMNS.div_ceil(k),
        PHASE_TWO_COLUMNS.div_ceil(k),
        phase_1_groups - 5,
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

fn physical_id(index: usize, k: usize) -> ColumnId {
    ColumnId {
        group: index / k,
        slot: index % k,
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

fn pad_fr(columns: &mut Vec<Column>, k: usize) {
    while !columns.len().is_multiple_of(k) {
        columns.push(Column::Fr(vec![Fr::zero(); ROWS]));
    }
}

fn challenge_copy_link(
    links: &LinkMap,
    relation: &Relation,
    relation_base: usize,
    table: &HashTable,
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

fn materialize_hash_form(form: &HashAffineForm, table: &HashTable) -> Vec<Fr> {
    let mut values = vec![form.constant; ROWS];
    for &(column, weight) in &form.weights {
        for (row, value) in values.iter_mut().enumerate() {
            *value += weight * hash_column_value(table, column, row);
        }
    }
    values
}

fn hash_column_value(table: &HashTable, column: usize, row: usize) -> Fr {
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

fn tamper_suite(
    proof: &WrapperProof,
    t2_wire_index: usize,
    rejected: impl Fn(&WrapperProof) -> bool,
) {
    let original = proof.clone();
    let tamper = |edit: &dyn Fn(&mut WrapperProof)| {
        let mut candidate = original.clone();
        edit(&mut candidate);
        assert!(rejected(&candidate));
    };
    tamper(&|candidate| candidate.commitments[0] = Commitment::new(original.opening.com[0]));
    tamper(&|candidate| {
        candidate.commitments[t2_wire_index] = Commitment::new(original.opening.com[0]);
    });
    tamper(&|candidate| {
        let last = candidate.commitments.len() - 1;
        candidate.commitments[last] = Commitment::new(original.opening.com[0]);
    });
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
    term_count: usize,
    cost: VerifierCost,
    times: [u128; 10],
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
    let [key_profile, prepare, setup, adapt, key_commit, commit_1, helpers, commit_2, prove, verify] =
        times;
    println!("uptime={}", String::from_utf8_lossy(uptime).trim());
    println!(
        "phases_ms key_profile={key_profile} prepare={prepare} setup={setup} adapt={adapt} key_commit={key_commit} commit1={commit_1} helpers={helpers} commit2={commit_2} prove={prove} verify={verify}"
    );
    println!(
        "bytes phase1={phase_1} phase2={phase_2} stage_a={stage_a} term={term_stage} shared_bdfg={shared} ell={ell} stage_b={stage_b} reduced={reduced} hyperkzg={opening} io_challenges=0 proof={} bincode={} public_known={}",
        proof.payload_bytes(),
        proof.bincode_bytes(),
        32 * 7,
    );
    println!(
        "terms={term_count} term_rounds={}",
        proof.stages[1]
            .committed_rounds
            .as_ref()
            .expect("term stage")
            .round_commitments
            .len()
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
