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
    Decoder, HashTable, Members as HashMembers, StreamColumns as HashStreamColumns,
    StreamTermExporter as HashStreamTermExporter, T1Challenges, VkColumn,
};
use jolt_wrapper::limb_table::relation::Col as T2Col;
use jolt_wrapper::limb_table::stream::{
    commitment_phases as t2_commitment_phases, link_input_claim,
    vk_group_range as t2_vk_group_range, LimbTableKey, Members as T2Members,
    StreamBuilder as T2StreamBuilder, StreamTermExporter as T2StreamTermExporter, StreamWitness,
    T2Challenges,
};
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
    verify_wrapped_with_key, wrap as wrap_proof, DoryLinkPlacement, DoryLinkedProver,
    NegatingTermExporter, T1Placement, WrapCommitments, WrapConfig, WrapError, WrapHashKey,
    WrapLimbKey, WrapPreparation, WrapVerifierKey,
};

#[path = "wrap_real_t1_r/t2.rs"]
mod t2;
use t2::Base as T2Base;

type Proof = JoltProof<Pcs, Vc>;
type Preprocessing = JoltVerifierPreprocessing<Pcs, Vc>;

const FIXTURE: &str = "/Volumes/Dev/scratch/wrapper-fixtures/fibonacci_2_18_blake3.bin";
const LOG_ROWS: usize = 18;
const ROWS: usize = 1 << LOG_ROWS;
const T1_CHALLENGE_OFFSET: usize = 4;
const THETA_OFFSET: usize = T1_CHALLENGE_OFFSET + T1Challenges::count(LOG_ROWS);
const RHO_OFFSET: usize = THETA_OFFSET + 1;
const T2_CHALLENGE_OFFSET: usize = RHO_OFFSET + 1;
const R_STAGE_CHALLENGE_OFFSET: usize = T2_CHALLENGE_OFFSET + T2Challenges::count();

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
            challenge_offset: T1_CHALLENGE_OFFSET,
            members: [0, 1],
        },
        &setup,
    )
    .expect("build trusted T1 key");
    let key_profile_ms = started.elapsed().as_millis();

    let started = Instant::now();
    let preparation = WrapPreparation::new(
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
        WrapPreparation::new(&preprocessing, &public_io, &wrong_shape, config, &hash_key),
        Err(WrapError::ProfileMismatch)
    ));
    let prepare_ms = started.elapsed().as_millis();

    let started = Instant::now();
    let hash_columns = HashStreamColumns::new(&preparation.hash_table, k, 0);
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
    let adapt_r_ms = started.elapsed().as_millis();

    let mut phase_1a_columns = hash_columns.columns;
    let relation_fixed_base = phase_1a_columns.len();
    phase_1a_columns.extend(relation_table.fixed_columns());
    pad_fr(&mut phase_1a_columns, k);
    let relation_wire_base = phase_1a_columns.len();
    phase_1a_columns.extend(
        relation_witness.evaluations()[..WIRES]
            .iter()
            .cloned()
            .map(Column::Fr),
    );
    pad_fr(&mut phase_1a_columns, k);
    let link_fixed_base = phase_1a_columns.len();
    phase_1a_columns.extend(link_columns.into_iter().map(Column::Fr));
    pad_fr(&mut phase_1a_columns, k);
    let phase_1a_groups = phase_1a_columns.len() / k;
    let t2_group_offset = phase_1a_groups;
    let t2_phases = t2_commitment_phases(k);
    let t2_vk_groups = t2_vk_group_range(k, 0).len();
    let relation_helper_count = relation_witness.evaluations().len() - WIRES;
    let relation_helper_groups = (relation_helper_count + 2).div_ceil(k);

    let mut commitment_phases = vec![CommitmentPhase {
        group_count: phase_1a_groups,
        challenge_count: T2_CHALLENGE_OFFSET,
    }];
    commitment_phases.extend(t2_phases);
    let last = commitment_phases.last_mut().expect("T2 phase 2c");
    last.group_count += relation_helper_groups;
    last.challenge_count += 2 * LOG_ROWS + 6;
    let total_groups = commitment_phases
        .iter()
        .map(|phase| phase.group_count)
        .sum::<usize>();
    let statement = AssemblyStatement {
        key_digest: preparation.profile_digest,
        public_inputs: preparation.public_known.clone(),
        rows: ROWS,
        column_count: total_groups * k,
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
        commitment_phases,
        pinned_commitments: Vec::new(),
    };

    let started = Instant::now();
    let pinned_groups = [relation_fixed_base / k, link_fixed_base / k];
    let pinned_commitments = pinned_groups
        .into_iter()
        .map(|group| {
            let start = group * k;
            let packed = commit_packed(&phase_1a_columns[start..start + k], k, &setup)
                .expect("verifier-key group commitment");
            (group, packed.commitments[0])
        })
        .collect::<Vec<_>>();
    let fixed_key_commit_ms = started.elapsed().as_millis();

    let started = Instant::now();
    let mut commitments = WrapCommitments::new()
        .commit(&phase_1a_columns, &statement, &setup)
        .expect("phase 1a commitments");
    let phase_1a_commit_ms = started.elapsed().as_millis();
    let phase_1_values = commitments.challenges().to_vec();
    let relation_beta = phase_1_values[0];
    let relation_gamma = phase_1_values[1];
    let copy_beta = phase_1_values[2];
    let copy_gamma = phase_1_values[3];
    let hash_challenges =
        T1Challenges::from_challenges(&phase_1_values[T1_CHALLENGE_OFFSET..THETA_OFFSET], LOG_ROWS);
    let hash_relation = hash_challenges.relation();
    let theta = phase_1_values[THETA_OFFSET];
    let t2_rho = phase_1_values[RHO_OFFSET];

    let started = Instant::now();
    let (t2_base, t2_layout) = T2Base::new(
        &preprocessing,
        &original_proof,
        &preparation.relation,
        &preparation.relation_witness,
        theta,
    )
    .expect("adapt real Dory opening");
    let adapt_t2_ms = started.elapsed().as_millis();

    let mut t2_builder = T2StreamBuilder::new(&t2_layout, &t2_base.columns, k);
    let started = Instant::now();
    commitments = commitments
        .commit(t2_builder.phase_1b(), &statement, &setup)
        .expect("T2 phase 1b commitments");
    let phase_1b_commit_ms = started.elapsed().as_millis();

    let [xi, alpha] = commitments.challenges()[T2_CHALLENGE_OFFSET..]
        .try_into()
        .expect("T2 phase 1b challenges");
    let started = Instant::now();
    commitments = commitments
        .commit(t2_builder.phase_2a(xi, alpha), &statement, &setup)
        .expect("T2 phase 2a commitments");
    let phase_2a_commit_ms = started.elapsed().as_millis();

    let fp_root = commitments.challenges()[T2_CHALLENGE_OFFSET + 2];
    let started = Instant::now();
    commitments = commitments
        .commit(t2_builder.phase_2b(fp_root), &statement, &setup)
        .expect("T2 phase 2b commitments");
    let phase_2b_commit_ms = started.elapsed().as_millis();

    let known = commitments.challenges()[T2_CHALLENGE_OFFSET..].to_vec();

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
    let mut relation_helper_columns = relation_witness.evaluations()[WIRES..]
        .iter()
        .cloned()
        .map(Column::Fr)
        .collect::<Vec<_>>();
    relation_helper_columns.extend(copy_witness.helpers.iter().cloned().map(Column::Fr));
    pad_fr(&mut relation_helper_columns, k);
    let helper_ms = started.elapsed().as_millis();

    let mut final_phase_columns = t2_builder.phase_2c(known[3], known[4], known[5]).to_vec();
    final_phase_columns.extend(relation_helper_columns);
    let started = Instant::now();
    commitments = commitments
        .commit(&final_phase_columns, &statement, &setup)
        .expect("T2 phase 2c, VK and relation-helper commitments");
    let phase_2c_commit_ms = started.elapsed().as_millis();
    let committed = commitments
        .finish(&statement)
        .expect("all commitment phases");
    let full_challenges = committed.challenges();
    assert_eq!(&full_challenges[..T2_CHALLENGE_OFFSET], phase_1_values);
    let t2_phase_challenges = &full_challenges[T2_CHALLENGE_OFFSET..R_STAGE_CHALLENGE_OFFSET];
    let t2_challenges = T2Challenges::from_challenges(theta, t2_phase_challenges, t2_rho);
    let row = t2_challenges.row;
    let t2_witness = t2_builder.finish(
        row.tau,
        row.gamma,
        row.lambda,
        row.lambda_lookup,
        row.constancy_root,
        t2_group_offset,
    );
    let mut cursor = R_STAGE_CHALLENGE_OFFSET;
    let tau_relation = take_point(full_challenges, &mut cursor);
    let tau_copy = take_point(full_challenges, &mut cursor);
    let relation_weights = take_array(full_challenges, &mut cursor);
    let copy_weights = take_array(full_challenges, &mut cursor);
    assert_eq!(cursor, full_challenges.len());
    let assembly_challenges = full_challenges.to_vec();

    let wrong_layout = t2_base.layout();
    let started = Instant::now();
    let t2_key = LimbTableKey::new(t2_layout, k, &setup).expect("T2 verifier key");
    let wrong_t2_key = LimbTableKey::new(wrong_layout, k, &setup).expect("wrong-key fixture");
    let wrong_hash_key = hash_key.clone();
    let link_placement = DoryLinkPlacement {
        challenge: RHO_OFFSET,
        theta: THETA_OFFSET,
        member: 5,
    };
    let verifier_key = WrapVerifierKey::new(
        statement.clone(),
        hash_key,
        preparation.hash_public.clone(),
        WrapLimbKey::new(t2_key, t2_group_offset),
        Some(link_placement),
        pinned_commitments.clone(),
    );
    let mut wrong_pins = pinned_commitments;
    wrong_pins.insert(
        0,
        (
            t2_witness.stream.vk_groups.start,
            Commitment::new(Bn254::g1_generator()),
        ),
    );
    let wrong_verifier_key = WrapVerifierKey::new(
        statement,
        wrong_hash_key,
        preparation.hash_public.clone(),
        WrapLimbKey::new(wrong_t2_key, t2_group_offset),
        Some(link_placement),
        wrong_pins,
    );
    assert_eq!(verifier_key.hash_links(), &links);
    assert_eq!(verifier_key.hash_schedule(), &preparation.hash_key);
    let key_commit_ms = fixed_key_commit_ms + started.elapsed().as_millis();

    let relation_columns = std::array::from_fn(|column| {
        if column < FIXED_COLUMNS {
            physical_id(relation_fixed_base + column, k)
        } else if column < FIXED_COLUMNS + WIRES {
            physical_id(relation_wire_base + column - FIXED_COLUMNS, k)
        } else {
            physical_id(
                t2_witness.stream.vk_groups.end * k + column - FIXED_COLUMNS - WIRES,
                k,
            )
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
        physical_id(
            t2_witness.stream.vk_groups.end * k + relation_helper_count,
            k,
        ),
        physical_id(
            t2_witness.stream.vk_groups.end * k + relation_helper_count + 1,
            k,
        ),
    ];

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
    let T2Members {
        rows: mut t2_rows,
        link: t2_digit_link,
    } = T2Members::new(
        &t2_witness.relation,
        &t2_witness.matrix,
        verifier_key.limb_layout(),
        &t2_witness.matrix[T2Col::D],
        t2_rho,
    );
    let scalar_link = DoryScalarLink::new(
        ROWS,
        relation_table.cell_layout(),
        verifier_key.limb_layout(),
        t2_rho,
    );
    let scalar_prover = scalar_link.prover(&relation_witness);
    let dory_link_claim = link_input_claim(Fr::zero(), t2_rho, theta, verifier_key.limb_layout());
    let mut dory_link = DoryLinkedProver::new(t2_digit_link, scalar_prover, dory_link_claim);
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
    let hash_exporter = HashStreamTermExporter {
        log_rows: LOG_ROWS,
        challenge_offset: T1_CHALLENGE_OFFSET,
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
    let t2_exporter = T2StreamTermExporter {
        layout: verifier_key.limb_layout(),
        challenge_offset: T2_CHALLENGE_OFFSET,
        theta_offset: THETA_OFFSET,
        rho_offset: RHO_OFFSET,
        columns: &t2_witness.stream.ids,
        row_member: 4,
        link_member: 5,
    };
    let scalar_exporter = DoryScalarTermExporter {
        link: &scalar_link,
        wire: relation_a,
        member_index: 5,
    };
    let negative_scalar_exporter = NegatingTermExporter(&scalar_exporter);
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

    let wire_phase_groups = [
        phase_1a_groups - hash_columns.vk_groups.len() - 2,
        t2_phases[0].group_count,
        t2_phases[1].group_count,
        t2_phases[2].group_count,
        t2_phases[3].group_count - t2_vk_groups + relation_helper_groups,
    ];
    tamper_suite(&wrapped, wire_phase_groups, k, |proof| {
        verify_wrapped_with_key(&verifier_key, proof, &exporters, &verifier_setup).is_err()
    });
    assert!(
        verify_wrapped_with_key(&wrong_verifier_key, &wrapped, &exporters, &verifier_setup,)
            .is_err()
    );
    assert_t2_row_tamper_rejected(
        &t2_witness,
        T2Col::FLAG,
        verifier_key.limb_layout().sign_rows[0].1 as usize,
    );
    assert_t2_row_tamper_rejected(
        &t2_witness,
        T2Col::CHUNKS,
        verifier_key.limb_layout().q_halves[0] as usize * 8,
    );
    assert_t2_row_tamper_rejected(
        &t2_witness,
        T2Col::D,
        verifier_key.limb_layout().digit_ops[0].first_row as usize,
    );

    report(
        &wrapped,
        wire_phase_groups,
        term_count,
        cost,
        &[
            ("setup", setup_ms),
            ("key_profile", key_profile_ms),
            ("prepare", prepare_ms),
            ("adapt_r", adapt_r_ms),
            ("adapt_t2", adapt_t2_ms),
            ("key_commit", key_commit_ms),
            ("commit_1a", phase_1a_commit_ms),
            ("commit_1b", phase_1b_commit_ms),
            ("commit_2a", phase_2a_commit_ms),
            ("commit_2b", phase_2b_commit_ms),
            ("helpers", helper_ms),
            ("commit_2c", phase_2c_commit_ms),
            ("prove", prove_ms),
            ("verify", verify_ms),
        ],
        &uptime,
    );
    println!(
        "groups k={k} t1_sent={} t1_vk={} r=2 copy_vk=1 t2_1b={} t2_2a={} t2_2b={} t2_2c={} t2_vk={} r_helpers={} full={} wire={}",
        hash_columns.group_count - hash_columns.vk_groups.len(),
        hash_columns.vk_groups.len(),
        t2_phases[0].group_count,
        t2_phases[1].group_count,
        t2_phases[2].group_count,
        t2_phases[3].group_count,
        t2_vk_groups,
        relation_helper_groups,
        total_groups,
        wire_phase_groups.iter().sum::<usize>(),
    );
}

fn assert_t2_row_tamper_rejected(witness: &StreamWitness, column: usize, row: usize) {
    let mut values = witness
        .matrix
        .iter()
        .map(|values| values[row])
        .collect::<Vec<_>>();
    values[column] += Fr::one();
    assert!(witness
        .relation
        .constraint_values(&values)
        .into_iter()
        .any(|(_, value)| !value.is_zero()));
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
    wire_phase_groups: [usize; 5],
    k: usize,
    rejected: impl Fn(&WrapperProof) -> bool,
) {
    let original = proof.clone();
    let tamper = |edit: &dyn Fn(&mut WrapperProof)| {
        let mut candidate = original.clone();
        edit(&mut candidate);
        assert!(rejected(&candidate));
    };
    tamper(&|candidate| candidate.commitments[0] = Commitment::new(original.opening.com[0]));
    let phase_1b = wire_phase_groups[0];
    let phase_2a = phase_1b + wire_phase_groups[1];
    let phase_2b = phase_2a + wire_phase_groups[2];
    let phase_2c = phase_2b + wire_phase_groups[3];
    tamper(&|candidate| {
        candidate.commitments[phase_1b] = Commitment::new(original.opening.com[0]);
    });
    let sign_group = phase_1b + T2Col::FLAG / k;
    tamper(&|candidate| {
        candidate.commitments[sign_group] = Commitment::new(original.opening.com[0]);
    });
    let psi_group = phase_1b + usize::from(wire_phase_groups[1] > 1);
    tamper(&|candidate| {
        candidate.commitments[psi_group] = Commitment::new(original.opening.com[0]);
    });
    tamper(&|candidate| {
        candidate.commitments[phase_2b] = Commitment::new(original.opening.com[0]);
    });
    tamper(&|candidate| {
        candidate.commitments[phase_2c] = Commitment::new(original.opening.com[0]);
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
    wire_phase_groups: [usize; 5],
    term_count: usize,
    cost: VerifierCost,
    times: &[(&str, u128)],
    uptime: &[u8],
) {
    let stage_a = committed_stage_bytes(&proof.stages[0]);
    let term_stage = committed_stage_bytes(&proof.stages[1]);
    let shared = 96 * usize::from(proof.round_opening.is_some());
    let ell = 32 * proof.term_evaluations.len();
    let stage_b = clear_stage_bytes(&proof.stages[2]);
    let reduced = 32 * proof.reduced_claims.len();
    let commitment_bytes = wire_phase_groups.map(|groups| 32 * groups);
    assert_eq!(
        wire_phase_groups.iter().sum::<usize>(),
        proof.commitments.len()
    );
    let opening = proof.payload_bytes()
        - commitment_bytes.iter().sum::<usize>()
        - stage_a
        - term_stage
        - shared
        - ell
        - stage_b
        - reduced;
    let serialized = encode_to_vec(proof, standard()).expect("serialize wrapper");
    assert_eq!(serialized.len(), proof.bincode_bytes());
    println!("uptime={}", String::from_utf8_lossy(uptime).trim());
    let phases = times
        .iter()
        .map(|(name, ms)| format!("{name}={ms}"))
        .collect::<Vec<_>>()
        .join(" ");
    println!("phases_ms {phases}");
    println!(
        "bytes phase1a={} phase1b={} phase2a={} phase2b={} phase2c={} stage_a={stage_a} term={term_stage} shared_bdfg={shared} ell={ell} stage_b={stage_b} reduced={reduced} hyperkzg={opening} io_challenges=0 proof={} bincode={} public_known={}",
        commitment_bytes[0],
        commitment_bytes[1],
        commitment_bytes[2],
        commitment_bytes[3],
        commitment_bytes[4],
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
