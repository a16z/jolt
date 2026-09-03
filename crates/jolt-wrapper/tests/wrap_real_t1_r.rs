#![expect(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::print_stdout,
    reason = "manual real-fixture integration gate"
)]

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::process::Command;
use std::time::Instant;

use bincode::config::standard;
use bincode::serde::{decode_from_slice, encode_to_vec};
use common::jolt_device::JoltDevice;
use jolt_crypto::Bn254;
use jolt_field::{CanonicalEncoding, Fr, One, Ring, Zero};
use jolt_hyperkzg::{HyperKZGProverSetup, HyperKZGScheme, HyperKZGVerifierSetup};
use jolt_poly::CompressedPoly;
use jolt_verifier::{JoltProof, JoltVerifierPreprocessing};
use jolt_wrapper::hash_table::layout::MESSAGE;
use jolt_wrapper::hash_table::terms::{
    challenge125, challenge_scalar128, fr_word, fr_word_shifted, AffineForm as HashAffineForm,
    LinkMap, WIRED_BIT_BASE, WIRED_WORD_BASE,
};
use jolt_wrapper::hash_table::{ByteSource, ElementKind as HashElementKind};
use jolt_wrapper::hash_table::{
    Decoder, HashTable, Members as HashMembers, StreamColumns as HashStreamColumns, T1Challenges,
    VkColumn,
};
use jolt_wrapper::limb_table::columns::Columns as T2Columns;
use jolt_wrapper::limb_table::dory::{input_elements, ElementKind as T2ElementKind, InputElement};
use jolt_wrapper::limb_table::relation::Col as T2Col;
use jolt_wrapper::limb_table::schedule::{Layout as T2Layout, WINDOW_ROW_BASE};
use jolt_wrapper::limb_table::stream::{
    commitment_phases as t2_commitment_phases, link_input_claim,
    vk_group_range as t2_vk_group_range, LimbTableKey, Members as T2Members,
    StreamBuilder as T2StreamBuilder, StreamWitness, T2Challenges,
};
use jolt_wrapper::relation::{Pcs, Relation, ScheduleEntry, SqueezeKind, Vc};
use jolt_wrapper::relation_table::{
    CopyLink, CopyLinkSide, CopyLinkTermSide, DoryScalarLink, RelationTable, RelationTableProver,
    FIXED_COLUMNS, WIRES,
};
use jolt_wrapper::stream::{
    commit_packed, AffineForm, AssemblyMemberStatement, AssemblyStatement, Column, ColumnId,
    Commitment, CommitmentPhase, StageMember, StageMemberSpec, StageProof, TermContext,
    VerifierCost, WrapperProof,
};
use jolt_wrapper::wrap::{
    hash_public_statement, verify_wrapped_with_key, wrap as wrap_proof, CopyExporterPlan,
    DoryLinkPlacement, DoryLinkedProver, LimbExporterPlan, PublicCopyPlan, RelationExporterPlan,
    ScalarExporterPlan, T1Placement, WrapAssemblyPlan, WrapCommitments, WrapConfig, WrapError,
    WrapHashKey, WrapLimbKey, WrapPreparation, WrapVerifierKey,
};

#[path = "wrap_real_t1_r/t2.rs"]
mod t2;
use t2::Base as T2Base;

type Proof = JoltProof<Pcs, Vc>;
type Preprocessing = JoltVerifierPreprocessing<Pcs, Vc>;

const FIXTURE: &str = "/Volumes/Dev/scratch/wrapper-fixtures/fibonacci_2_18_blake3.bin";
const LOG_ROWS: usize = 18;
const ROWS: usize = 1 << LOG_ROWS;
#[test]
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
    let profile_hash_key = WrapHashKey::from_reference(
        &preprocessing,
        &public_io,
        &original_proof,
        config,
        T1Placement {
            group_offset: 0,
            challenge_offset: 0,
            members: [0, 1],
        },
        &setup,
    )
    .expect("build trusted T1 key");
    let mut key_profile_ms = started.elapsed().as_millis();

    let started = Instant::now();
    let preparation = WrapPreparation::new(
        &preprocessing,
        &public_io,
        &original_proof,
        config,
        &profile_hash_key,
    )
    .expect("prepare real wrapper inputs");
    let mut wrong_shape = original_proof.clone();
    wrong_shape.trace_length *= 2;
    assert!(matches!(
        WrapPreparation::new(
            &preprocessing,
            &public_io,
            &wrong_shape,
            config,
            &profile_hash_key,
        ),
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
    let (reference_base, reference_layout) = T2Base::new(
        &preprocessing,
        &original_proof,
        &preparation.relation,
        &preparation.relation_witness,
        Fr::zero(),
    )
    .expect("build reference T2 link layout");
    let cells = relation_table.cell_layout();
    let mut copy_specs = vec![
        challenge_copy_spec(&links, &preparation.relation, cells.challenge_base),
        absorbed_word_copy_spec(&links, cells.absorbed_word_base, ROWS),
        public_copy_spec(cells.public_input_base, &preparation.public_known, ROWS),
    ];
    copy_specs.extend(element_copy_specs(
        &links,
        &reference_layout,
        &t1_commitment_order(&original_proof),
        &preparation.hash_table,
        &reference_base.columns,
        ROWS,
    ));
    let relation_a = relation_witness.evaluations()[0].clone();
    for (index, spec) in copy_specs.iter().enumerate() {
        spec.assert_values_match(
            &preparation.hash_table,
            &relation_a,
            &reference_base.columns,
            index,
        );
    }
    let t1_challenge_offset = 2 + 2 * copy_specs.len();
    let theta_offset = t1_challenge_offset + T1Challenges::count(LOG_ROWS);
    let rho_offset = theta_offset + 1;
    let t2_challenge_offset = rho_offset + 1;
    let r_stage_challenge_offset = t2_challenge_offset + T2Challenges::count();
    let started_key = Instant::now();
    let hash_key = WrapHashKey::from_reference(
        &preprocessing,
        &public_io,
        &original_proof,
        config,
        T1Placement {
            group_offset: 0,
            challenge_offset: t1_challenge_offset,
            members: [0, 1],
        },
        &setup,
    )
    .expect("build linked T1 key");
    key_profile_ms += started_key.elapsed().as_millis();
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
    let copy_fixed_bases = copy_specs
        .iter()
        .map(|spec| {
            let base = phase_1a_columns.len();
            phase_1a_columns.extend(spec.fixed_columns().map(Column::Fr));
            pad_fr(&mut phase_1a_columns, k);
            base
        })
        .collect::<Vec<_>>();
    let phase_1a_groups = phase_1a_columns.len() / k;
    let t2_group_offset = phase_1a_groups;
    let t2_phases = t2_commitment_phases(k);
    let t2_vk_groups = t2_vk_group_range(k, 0).len();
    let relation_helper_count = relation_witness.evaluations().len() - WIRES;
    let relation_helper_groups = (relation_helper_count + 2 * copy_specs.len()).div_ceil(k);

    let mut commitment_phases = vec![CommitmentPhase {
        group_count: phase_1a_groups,
        challenge_count: t2_challenge_offset,
    }];
    commitment_phases.extend(t2_phases);
    let last = commitment_phases.last_mut().expect("T2 phase 2c");
    last.group_count += relation_helper_groups;
    last.challenge_count += (1 + copy_specs.len()) * (LOG_ROWS + 3);
    let total_groups = commitment_phases
        .iter()
        .map(|phase| phase.group_count)
        .sum::<usize>();
    let mut public_inputs = preparation.public_known.clone();
    public_inputs.extend(hash_public_statement(&preparation.hash_public));
    let member_degrees = std::iter::once(3)
        .chain(std::iter::once(3))
        .chain(std::iter::once(5))
        .chain(std::iter::repeat_n(5, copy_specs.len()))
        .chain([5, 2]);
    let statement = AssemblyStatement {
        key_digest: preparation.profile_digest,
        public_inputs,
        rows: ROWS,
        column_count: total_groups * k,
        k,
        members: member_degrees
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
    let pinned_groups = std::iter::once(relation_fixed_base / k)
        .chain(copy_fixed_bases.iter().map(|base| base / k));
    let pinned_commitments = pinned_groups
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
    let copy_challenges = (0..copy_specs.len())
        .map(|index| (phase_1_values[2 + 2 * index], phase_1_values[3 + 2 * index]))
        .collect::<Vec<_>>();
    let hash_challenges =
        T1Challenges::from_challenges(&phase_1_values[t1_challenge_offset..theta_offset], LOG_ROWS);
    let hash_relation = hash_challenges.relation();
    let theta = phase_1_values[theta_offset];
    let t2_rho = phase_1_values[rho_offset];

    let started = Instant::now();
    let (t2_base, t2_layout) = T2Base::new(
        &preprocessing,
        &original_proof,
        &preparation.relation,
        &preparation.relation_witness,
        theta,
    )
    .expect("adapt real Dory opening");
    assert_eq!(t2_layout.input_order, reference_layout.input_order);
    assert_eq!(
        t2_layout.program.input_rows,
        reference_layout.program.input_rows
    );
    assert_eq!(t2_layout.sign_rows, reference_layout.sign_rows);
    let adapt_t2_ms = started.elapsed().as_millis();

    let mut t2_builder = T2StreamBuilder::new(&t2_layout, &t2_base.columns, k);
    let started = Instant::now();
    commitments = commitments
        .commit(t2_builder.phase_1b(), &statement, &setup)
        .expect("T2 phase 1b commitments");
    let phase_1b_commit_ms = started.elapsed().as_millis();

    let [xi, alpha] = commitments.challenges()[t2_challenge_offset..]
        .try_into()
        .expect("T2 phase 1b challenges");
    let started = Instant::now();
    commitments = commitments
        .commit(t2_builder.phase_2a(xi, alpha), &statement, &setup)
        .expect("T2 phase 2a commitments");
    let phase_2a_commit_ms = started.elapsed().as_millis();

    let fp_root = commitments.challenges()[t2_challenge_offset + 2];
    let started = Instant::now();
    commitments = commitments
        .commit(t2_builder.phase_2b(fp_root), &statement, &setup)
        .expect("T2 phase 2b commitments");
    let phase_2b_commit_ms = started.elapsed().as_millis();

    let known = commitments.challenges()[t2_challenge_offset..].to_vec();

    let started = Instant::now();
    relation_table
        .add_copy_helpers(&mut relation_witness, relation_beta, relation_gamma)
        .expect("R helpers");
    relation_table
        .check_witness(&relation_witness, relation_beta, relation_gamma)
        .expect("R witness");
    let copy_witnesses = copy_specs
        .iter()
        .zip(&copy_challenges)
        .enumerate()
        .map(|(index, (spec, &(beta, gamma)))| {
            let (left, right) = spec.values(&preparation.hash_table, &relation_a, &t2_base.columns);
            let witness = spec
                .link
                .witness(left, right, beta, gamma)
                .expect("linked transcript witness");
            spec.link
                .check(&witness, beta, gamma)
                .unwrap_or_else(|error| panic!("linked transcript equality {index}: {error}"));
            witness
        })
        .collect::<Vec<_>>();
    let mut relation_helper_columns = relation_witness.evaluations()[WIRES..]
        .iter()
        .cloned()
        .map(Column::Fr)
        .collect::<Vec<_>>();
    relation_helper_columns.extend(
        copy_witnesses
            .iter()
            .flat_map(|witness| witness.helpers.iter().cloned())
            .map(Column::Fr),
    );
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
    assert_eq!(&full_challenges[..t2_challenge_offset], phase_1_values);
    let t2_phase_challenges = &full_challenges[t2_challenge_offset..r_stage_challenge_offset];
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
    let mut cursor = r_stage_challenge_offset;
    let tau_relation = take_point(full_challenges, &mut cursor);
    let tau_copies = (0..copy_specs.len())
        .map(|_| take_point(full_challenges, &mut cursor))
        .collect::<Vec<_>>();
    let relation_weights = take_array(full_challenges, &mut cursor);
    let copy_weights = (0..copy_specs.len())
        .map(|_| take_array(full_challenges, &mut cursor))
        .collect::<Vec<_>>();
    assert_eq!(cursor, full_challenges.len());
    let assembly_challenges = full_challenges.to_vec();

    let wrong_layout = t2_base.layout();
    let started = Instant::now();
    let t2_key = LimbTableKey::new(t2_layout, k, &setup).expect("T2 verifier key");
    let wrong_t2_key = LimbTableKey::new(wrong_layout, k, &setup).expect("wrong-key fixture");
    let wrong_hash_key = hash_key.clone();
    let public_hash_key = hash_key.clone();
    let program_hash_key = hash_key.clone();
    let link_placement = DoryLinkPlacement {
        challenge: rho_offset,
        theta: theta_offset,
        member: 4 + copy_specs.len(),
    };
    let mut wrong_pins = pinned_commitments.clone();
    wrong_pins.insert(
        0,
        (
            t2_witness.stream.vk_groups.start,
            Commitment::new(Bn254::g1_generator()),
        ),
    );
    let key_commit_started = started;

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
    let relation_a = relation_columns[FIXED_COLUMNS];
    let copy_term_sides = copy_specs
        .iter()
        .zip(&copy_fixed_bases)
        .enumerate()
        .map(|(index, (spec, &base))| {
            let left = CopyLinkTermSide {
                selectors: std::array::from_fn(|wire| physical_id(base + wire, k)),
                ids: std::array::from_fn(|wire| column_form(physical_id(base + WIRES + wire, k))),
                values: spec.left.clone().map(|source| match source {
                    LeftLinkValue::Hash(form) => map_hash_form(&form, &hash_columns.ids),
                    LeftLinkValue::Public | LeftLinkValue::Zero => zero_form(),
                }),
                helper: physical_id(
                    t2_witness.stream.vk_groups.end * k + relation_helper_count + 2 * index,
                    k,
                ),
            };
            let right = CopyLinkTermSide {
                selectors: std::array::from_fn(|wire| physical_id(base + 2 * WIRES + wire, k)),
                ids: std::array::from_fn(|wire| {
                    column_form(physical_id(base + 3 * WIRES + wire, k))
                }),
                values: spec
                    .right
                    .map(|source| t2_link_form(source, relation_a, &t2_witness)),
                helper: physical_id(
                    t2_witness.stream.vk_groups.end * k + relation_helper_count + 2 * index + 1,
                    k,
                ),
            };
            (left, right)
        })
        .collect::<Vec<_>>();
    let t2_member = 3 + copy_specs.len();
    let dory_member = t2_member + 1;
    let weights_offset = r_stage_challenge_offset + (1 + copy_specs.len()) * LOG_ROWS;
    let assembly_plan = WrapAssemblyPlan {
        hash_columns: hash_columns.ids.clone(),
        relation: RelationExporterPlan {
            rows: ROWS,
            columns: relation_columns,
            tau: r_stage_challenge_offset..r_stage_challenge_offset + LOG_ROWS,
            beta: 0,
            gamma: 1,
            weights: weights_offset..weights_offset + 3,
            member: 2,
        },
        copies: copy_specs
            .iter()
            .zip(&copy_term_sides)
            .enumerate()
            .map(|(index, (spec, (left, right)))| CopyExporterPlan {
                link: spec.link.clone(),
                left: left.clone(),
                right: right.clone(),
                tau: r_stage_challenge_offset + (index + 1) * LOG_ROWS
                    ..r_stage_challenge_offset + (index + 2) * LOG_ROWS,
                beta: 2 + 2 * index,
                gamma: 3 + 2 * index,
                weights: weights_offset + 3 * (index + 1)..weights_offset + 3 * (index + 2),
                member: 3 + index,
                public: spec.public.as_ref().map(|(wire, rows, _)| PublicCopyPlan {
                    wire: *wire,
                    rows: rows.clone(),
                    values: 0..preparation.public_known.len(),
                }),
            })
            .collect(),
        limb: LimbExporterPlan {
            challenge_offset: t2_challenge_offset,
            theta_offset,
            rho_offset,
            columns: t2_witness.stream.ids.clone(),
            row_member: t2_member,
            link_member: dory_member,
        },
        scalar: ScalarExporterPlan {
            rows: ROWS,
            cells: relation_table.cell_layout(),
            rho_offset,
            wire: relation_a,
            member: dory_member,
        },
    };
    let verifier_key = WrapVerifierKey::new(
        statement.clone(),
        hash_key,
        preparation.hash_public.clone(),
        WrapLimbKey::new(t2_key, t2_group_offset),
        Some(link_placement),
        assembly_plan.clone(),
        pinned_commitments.clone(),
    )
    .expect("wrapper verifier key");
    let wrong_verifier_key = WrapVerifierKey::new(
        statement.clone(),
        wrong_hash_key,
        preparation.hash_public.clone(),
        WrapLimbKey::new(wrong_t2_key, t2_group_offset),
        Some(link_placement),
        assembly_plan.clone(),
        wrong_pins,
    )
    .expect("wrong-pin wrapper key");
    let mut public_statement = statement.clone();
    public_statement.public_inputs[0] += Fr::one();
    let public_verifier_key = WrapVerifierKey::new(
        public_statement,
        public_hash_key,
        preparation.hash_public.clone(),
        WrapLimbKey::new(
            LimbTableKey::new(t2_base.layout(), k, &setup).expect("public-mismatch T2 key"),
            t2_group_offset,
        ),
        Some(link_placement),
        assembly_plan.clone(),
        pinned_commitments.clone(),
    )
    .expect("public-mismatch wrapper key");
    let mut program_statement = statement.clone();
    program_statement.key_digest[0] ^= 1;
    let program_verifier_key = WrapVerifierKey::new(
        program_statement,
        program_hash_key,
        preparation.hash_public.clone(),
        WrapLimbKey::new(
            LimbTableKey::new(t2_base.layout(), k, &setup).expect("program-mismatch T2 key"),
            t2_group_offset,
        ),
        Some(link_placement),
        assembly_plan,
        pinned_commitments.clone(),
    );
    assert!(matches!(
        program_verifier_key,
        Err(WrapError::StatementMismatch)
    ));
    assert_eq!(verifier_key.hash_links(), &links);
    assert_eq!(verifier_key.hash_schedule(), &preparation.hash_key);
    let key_commit_ms = fixed_key_commit_ms + key_commit_started.elapsed().as_millis();

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
    let mut copy_rows = copy_specs
        .iter()
        .zip(&copy_witnesses)
        .zip(&tau_copies)
        .zip(&copy_challenges)
        .zip(&copy_weights)
        .map(|((((spec, witness), tau), &(beta, gamma)), &weights)| {
            spec.link.prover(witness, tau.clone(), beta, gamma, weights)
        })
        .collect::<Vec<_>>();
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
    assert!(copy_rows.iter().all(|rows| rows.input_claim().is_zero()));
    assert!(t2_rows.input_claim().is_zero());

    let mut input_claims = vec![
        hash_input_claims[0],
        hash_input_claims[1],
        relation_rows.input_claim(),
    ];
    input_claims.extend(copy_rows.iter().map(|rows| rows.input_claim()));
    input_claims.extend([Fr::zero(), dory_link_claim]);
    let mut members = vec![
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
    ];
    for (index, rows) in copy_rows.iter_mut().enumerate() {
        members.push(StageMember {
            prover: rows,
            input_claim: input_claims[3 + index],
            degree: 5,
            offset: 0,
        });
    }
    members.extend([
        StageMember {
            prover: &mut t2_rows,
            input_claim: input_claims[t2_member],
            degree: 5,
            offset: 0,
        },
        StageMember {
            prover: &mut dory_link,
            input_claim: input_claims[dory_member],
            degree: 2,
            offset: 0,
        },
    ]);

    let started = Instant::now();
    let wrapped = wrap_proof(committed, &verifier_key, &mut members, &setup)
        .expect("prove real T1/T2/R wrapper");
    let prove_ms = started.elapsed().as_millis();
    let verifier_setup = HyperKZGVerifierSetup::from(&setup);
    let started = Instant::now();
    let (results, cost) = verify_wrapped_with_key(&verifier_key, &wrapped, &verifier_setup)
        .expect("verify real T1/T2/R wrapper");
    let verify_ms = started.elapsed().as_millis();
    let term_context = TermContext {
        row_point: &results[0].point,
        batching_coefficients: &results[0].coefficients,
        challenges: &assembly_challenges,
    };
    let term_count = verifier_key.term_count(&term_context);

    let wire_phase_groups = [
        phase_1a_groups - hash_columns.vk_groups.len() - 1 - copy_specs.len(),
        t2_phases[0].group_count,
        t2_phases[1].group_count,
        t2_phases[2].group_count,
        t2_phases[3].group_count - t2_vk_groups + relation_helper_groups,
    ];
    tamper_suite(&wrapped, wire_phase_groups, k, |proof| {
        verify_wrapped_with_key(&verifier_key, proof, &verifier_setup).is_err()
    });
    assert_t2_commitment_row_tamper_rejected(
        &wrapped,
        &t2_witness,
        wire_phase_groups,
        k,
        &setup,
        WINDOW_ROW_BASE as usize,
        |proof| verify_wrapped_with_key(&verifier_key, proof, &verifier_setup).is_err(),
    );
    assert_t2_commitment_row_tamper_rejected(
        &wrapped,
        &t2_witness,
        wire_phase_groups,
        k,
        &setup,
        verifier_key.limb_layout().program.input_rows[0] as usize,
        |proof| verify_wrapped_with_key(&verifier_key, proof, &verifier_setup).is_err(),
    );
    let pinned_phase_1a = hash_columns
        .vk_groups
        .clone()
        .chain(std::iter::once(relation_fixed_base / k))
        .chain(copy_fixed_bases.iter().map(|base| base / k))
        .collect::<Vec<_>>();
    assert_r_absorbed_word_commitment_tamper_rejected(
        &wrapped,
        &phase_1a_columns,
        relation_wire_base,
        cells.absorbed_word_base,
        &pinned_phase_1a,
        k,
        &setup,
        |proof| verify_wrapped_with_key(&verifier_key, proof, &verifier_setup).is_err(),
    );
    assert!(verify_wrapped_with_key(&wrong_verifier_key, &wrapped, &verifier_setup).is_err());
    assert!(verify_wrapped_with_key(&public_verifier_key, &wrapped, &verifier_setup).is_err());
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
        statement.public_inputs.len(),
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
        "groups k={k} t1_sent={} t1_vk={} r=2 copy_vk={} t2_1b={} t2_2a={} t2_2b={} t2_2c={} t2_vk={} r_helpers={} full={} wire={}",
        hash_columns.group_count - hash_columns.vk_groups.len(),
        hash_columns.vk_groups.len(),
        copy_specs.len(),
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

fn assert_t2_commitment_row_tamper_rejected(
    proof: &WrapperProof,
    witness: &StreamWitness,
    wire_phase_groups: [usize; 5],
    k: usize,
    setup: &HyperKZGProverSetup<Bn254>,
    row: usize,
    rejected: impl Fn(&WrapperProof) -> bool,
) {
    let id = witness.stream.ids[T2Col::CHUNKS];
    let group_offset = witness
        .stream
        .ids
        .iter()
        .map(|id| id.group)
        .min()
        .expect("T2 stream columns");
    let local_group = id.group - group_offset;
    assert!(local_group < wire_phase_groups[1]);
    let start = local_group * k;
    let mut columns = witness.stream.columns[start..start + k].to_vec();
    let Column::U16(values) = &mut columns[id.slot] else {
        panic!("T2 chunk column is u16");
    };
    values[row] ^= 1;
    let commitment = commit_packed(&columns, k, setup)
        .expect("commit tampered T2 window row")
        .commitments[0];
    let mut candidate = proof.clone();
    candidate.commitments[wire_phase_groups[0] + local_group] = commitment;
    assert!(rejected(&candidate));
}

#[expect(clippy::too_many_arguments, reason = "proof-level R row tamper")]
fn assert_r_absorbed_word_commitment_tamper_rejected(
    proof: &WrapperProof,
    phase_1a: &[Column],
    relation_wire_base: usize,
    row: usize,
    pinned_groups: &[usize],
    k: usize,
    setup: &HyperKZGProverSetup<Bn254>,
    rejected: impl Fn(&WrapperProof) -> bool,
) {
    let global_group = relation_wire_base / k;
    let mut columns = phase_1a[global_group * k..(global_group + 1) * k].to_vec();
    let Column::Fr(values) = &mut columns[relation_wire_base % k] else {
        panic!("R wire column is a field column");
    };
    values[row] += Fr::one();
    let commitment = commit_packed(&columns, k, setup)
        .expect("commit tampered R absorbed word")
        .commitments[0];
    let wire_group = global_group
        - pinned_groups
            .iter()
            .filter(|&&group| group < global_group)
            .count();
    let mut candidate = proof.clone();
    candidate.commitments[wire_group] = commitment;
    assert!(rejected(&candidate));
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

#[derive(Clone)]
enum LeftLinkValue {
    Hash(HashAffineForm),
    Public,
    Zero,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum T2LinkValue {
    Relation,
    Chunk(usize),
    Sign,
    Zero,
}

struct CopySpec {
    link: CopyLink,
    fixed: Vec<Vec<Fr>>,
    left: [LeftLinkValue; WIRES],
    right: [T2LinkValue; WIRES],
    public: Option<(usize, Vec<usize>, Vec<Fr>)>,
}

impl CopySpec {
    fn values(
        &self,
        hash: &HashTable,
        relation_a: &[Fr],
        t2: &T2Columns,
    ) -> ([Vec<Fr>; WIRES], [Vec<Fr>; WIRES]) {
        let rows = relation_a.len();
        let left = self.left.clone().map(|source| match source {
            LeftLinkValue::Hash(form) => materialize_hash_form(&form, hash),
            LeftLinkValue::Public => {
                let (_, public_rows, values) = self.public.as_ref().expect("public copy link");
                let mut column = vec![Fr::zero(); rows];
                for (&row, &value) in public_rows.iter().zip(values) {
                    column[row] = value;
                }
                column
            }
            LeftLinkValue::Zero => vec![Fr::zero(); rows],
        });
        let right = self.right.map(|source| match source {
            T2LinkValue::Relation => relation_a.to_vec(),
            T2LinkValue::Chunk(chunk) => t2
                .chunk_column(chunk)
                .into_iter()
                .map(|value| Fr::from_u64(u64::from(value)))
                .collect(),
            T2LinkValue::Sign => t2
                .flags
                .iter()
                .map(|value| Fr::from_u64(u64::from(*value)))
                .collect(),
            T2LinkValue::Zero => vec![Fr::zero(); rows],
        });
        (left, right)
    }

    fn fixed_columns(&self) -> impl Iterator<Item = Vec<Fr>> + '_ {
        self.fixed.iter().cloned()
    }

    fn assert_values_match(
        &self,
        hash: &HashTable,
        relation_a: &[Fr],
        t2: &T2Columns,
        index: usize,
    ) {
        let (left, right) = self.values(hash, relation_a, t2);
        let selected = |side: &CopyLinkSide, values: &[Vec<Fr>; WIRES]| {
            let mut entries = HashMap::new();
            for (wire, ((selectors, ids), values)) in
                side.selectors.iter().zip(&side.ids).zip(values).enumerate()
            {
                for (row, ((selector, id), value)) in
                    selectors.iter().zip(ids).zip(values).enumerate()
                {
                    if !selector.is_zero() {
                        assert!(entries.insert(*id, (*value, wire, row)).is_none());
                    }
                }
            }
            entries
        };
        let left = selected(&self.link.left, &left);
        let right = selected(&self.link.right, &right);
        assert_eq!(left.len(), right.len(), "copy link {index} edge count");
        for (id, left) in left {
            let right = right.get(&id);
            assert_eq!(
                right.map(|entry| entry.0),
                Some(left.0),
                "copy link {index}, edge {}, left wire/row {}/{}, right {:?}",
                id.to_u64_checked().unwrap_or(u64::MAX),
                left.1,
                left.2,
                right.map(|entry| (entry.1, entry.2)),
            );
        }
    }
}

fn fixed_copy_columns(link: &CopyLink) -> Vec<Vec<Fr>> {
    link.left
        .selectors
        .iter()
        .chain(&link.left.ids)
        .chain(&link.right.selectors)
        .chain(&link.right.ids)
        .cloned()
        .collect()
}

fn hash_halfword(half: usize, swapped: bool, bits: usize) -> HashAffineForm {
    let mut form = HashAffineForm::default();
    for output in 0..bits {
        let input = if swapped { output ^ 8 } else { output };
        form.add(MESSAGE + 16 * half + input, Fr::from_u64(1 << output));
    }
    form
}

fn hash_bit(bit: usize) -> HashAffineForm {
    let mut form = HashAffineForm::default();
    form.add(MESSAGE + bit, Fr::one());
    form
}

fn zero_form() -> AffineForm {
    AffineForm {
        constant: Fr::zero(),
        weights: Vec::new(),
    }
}

fn t2_link_form(source: T2LinkValue, relation_a: ColumnId, t2: &StreamWitness) -> AffineForm {
    match source {
        T2LinkValue::Relation => column_form(relation_a),
        T2LinkValue::Chunk(chunk) => column_form(t2.stream.ids[T2Col::CHUNKS + chunk]),
        T2LinkValue::Sign => column_form(t2.stream.ids[T2Col::FLAG]),
        T2LinkValue::Zero => zero_form(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum HashLinkValue {
    Half {
        half: usize,
        swapped: bool,
        bits: usize,
    },
    Bit(usize),
}

impl HashLinkValue {
    fn form(self) -> HashAffineForm {
        match self {
            Self::Half {
                half,
                swapped,
                bits,
            } => hash_halfword(half, swapped, bits),
            Self::Bit(bit) => hash_bit(bit),
        }
    }
}

#[derive(Clone, Copy)]
struct LinkEdge {
    left_row: usize,
    right_row: usize,
    left: HashLinkValue,
    right: T2LinkValue,
    id: u64,
}

struct ElementTarget {
    rows: Vec<usize>,
    sign_row: Option<usize>,
}

fn t1_commitment_order(proof: &Proof) -> Vec<usize> {
    let instruction = proof.commitments.instruction_ra.len();
    let ram = proof.commitments.ram_ra.len();
    let bytecode = proof.commitments.bytecode_ra.len();
    let mut order = vec![1, 0];
    order.extend(2..2 + instruction);
    order.extend((0..ram).map(|index| 2 + instruction + bytecode + index));
    order.extend((0..bytecode).map(|index| 2 + instruction + index));
    order
}

fn element_targets(
    layout: &T2Layout,
    commitment_order: &[usize],
) -> BTreeMap<(HashElementKind, u32), ElementTarget> {
    let mut by_element = HashMap::new();
    let mut cursor = 0;
    for &element in &layout.input_order {
        let coordinates = element.kind().coords();
        let rows = layout.program.input_rows[cursor..cursor + coordinates]
            .iter()
            .map(|row| *row as usize)
            .collect();
        cursor += coordinates;
        let sign_row = layout
            .sign_rows
            .iter()
            .find_map(|(candidate, row)| (*candidate == element).then_some(*row as usize));
        assert!(by_element
            .insert(element, ElementTarget { rows, sign_row })
            .is_none());
    }
    assert_eq!(cursor, layout.program.input_rows.len());
    let mut targets = BTreeMap::new();
    let mut dory = [0u32; 3];
    for element in input_elements(layout.check.sigma, layout.check.n) {
        let key = if let InputElement::Commitment(index) = element {
            let transcript_index = commitment_order
                .iter()
                .position(|&ordered| ordered == index)
                .expect("commitment appears in the transcript");
            (HashElementKind::CommitmentGt, transcript_index as u32)
        } else {
            let slot = match element.kind() {
                T2ElementKind::Gt => 0,
                T2ElementKind::G1 => 1,
                T2ElementKind::G2 => 2,
            };
            let kind = [
                HashElementKind::DoryGt,
                HashElementKind::DoryG1,
                HashElementKind::DoryG2,
            ][slot];
            let key = (kind, dory[slot]);
            dory[slot] += 1;
            key
        };
        let target = by_element
            .remove(&element)
            .expect("every transcript element has T2 input rows");
        assert!(targets.insert(key, target).is_none());
    }
    assert!(by_element.is_empty());
    targets
}

fn assert_element_edge(
    edge: &LinkEdge,
    hash: &HashTable,
    t2: &T2Columns,
    kind: HashElementKind,
    index: u32,
    byte: usize,
) {
    let form = edge.left.form();
    let left = form
        .weights
        .iter()
        .fold(form.constant, |value, &(column, weight)| {
            value + weight * hash_column_value(hash, column, edge.left_row)
        });
    let right = match edge.right {
        T2LinkValue::Relation => unreachable!("element links do not read R"),
        T2LinkValue::Chunk(chunk) => t2.chunk(edge.right_row, chunk),
        T2LinkValue::Sign => Fr::from_u64(u64::from(t2.flags[edge.right_row])),
        T2LinkValue::Zero => Fr::zero(),
    };
    assert_eq!(
        left, right,
        "element {kind:?}[{index}] byte {byte}, T1 row {}, T2 row {}, {:?}",
        edge.left_row, edge.right_row, edge.right
    );
}

fn element_copy_specs(
    links: &LinkMap,
    layout: &T2Layout,
    commitment_order: &[usize],
    hash: &HashTable,
    t2: &T2Columns,
    rows: usize,
) -> Vec<CopySpec> {
    let targets = element_targets(layout, commitment_order);
    let mut positions = HashMap::new();
    for &(source, row, byte_in_word) in &links.bytes {
        if let ByteSource::Element { kind, index, byte } = source {
            assert!(positions
                .insert((kind, index, byte), (row, usize::from(byte_in_word)))
                .is_none());
        }
    }
    let mut edges = Vec::new();
    let mut next_id = 1u64;
    for (&(kind, index), target) in &targets {
        let bytes = match kind {
            HashElementKind::CommitmentGt | HashElementKind::DoryGt => 384,
            HashElementKind::DoryG1 => 32,
            HashElementKind::DoryG2 => 64,
        };
        for byte in (0..bytes).step_by(2) {
            let (left_row, first) = positions[&(kind, index, byte as u16)];
            let (second_row, second) = positions[&(kind, index, byte as u16 + 1)];
            assert_eq!(left_row, second_row);
            assert_eq!(second, first + 1);
            assert!(matches!(first, 0 | 2));
            let source_half = byte / 2;
            let (target_half, swapped) = if kind == HashElementKind::CommitmentGt {
                (bytes / 2 - 1 - source_half, true)
            } else {
                (source_half, false)
            };
            let coordinate = target_half / 16;
            let chunk = target_half % 16;
            let is_curve_top = matches!(kind, HashElementKind::DoryG1 | HashElementKind::DoryG2)
                && source_half == bytes / 2 - 1;
            let edge = LinkEdge {
                left_row,
                right_row: target.rows[coordinate],
                left: HashLinkValue::Half {
                    half: first / 2,
                    swapped,
                    bits: if is_curve_top { 14 } else { 16 },
                },
                right: T2LinkValue::Chunk(chunk),
                id: next_id,
            };
            assert_element_edge(&edge, hash, t2, kind, index, byte);
            edges.push(edge);
            next_id += 1;
            if is_curve_top {
                let sign_row = target.sign_row.expect("curve input sign row");
                let edge = LinkEdge {
                    left_row,
                    right_row: sign_row,
                    left: HashLinkValue::Bit(8 * second + 7),
                    right: T2LinkValue::Sign,
                    id: next_id,
                };
                assert_element_edge(&edge, hash, t2, kind, index, byte + 1);
                edges.push(edge);
                next_id += 1;
                let edge = LinkEdge {
                    left_row,
                    right_row: sign_row,
                    left: HashLinkValue::Bit(8 * second + 6),
                    right: T2LinkValue::Zero,
                    id: next_id,
                };
                assert_element_edge(&edge, hash, t2, kind, index, byte + 1);
                edges.push(edge);
                next_id += 1;
            }
        }
    }
    assert_eq!(positions.len(), 45_152);

    let mut groups: Vec<(Vec<HashLinkValue>, Vec<T2LinkValue>, Vec<LinkEdge>)> = Vec::new();
    for edge in edges {
        let group = groups.iter().position(|(left, right, _)| {
            (left.contains(&edge.left) || left.len() < WIRES)
                && (right.contains(&edge.right) || right.len() < WIRES)
        });
        let index = group.unwrap_or_else(|| {
            groups.push((Vec::new(), Vec::new(), Vec::new()));
            groups.len() - 1
        });
        let group = &mut groups[index];
        if !group.0.contains(&edge.left) {
            group.0.push(edge.left);
        }
        if !group.1.contains(&edge.right) {
            group.1.push(edge.right);
        }
        group.2.push(edge);
    }
    groups
        .into_iter()
        .map(|(mut left_keys, mut right_keys, edges)| {
            left_keys.resize(WIRES, HashLinkValue::Bit(0));
            right_keys.resize(WIRES, T2LinkValue::Zero);
            let mut left_selectors = std::array::from_fn(|_| vec![Fr::zero(); rows]);
            let mut left_ids = std::array::from_fn(|_| vec![Fr::zero(); rows]);
            let mut right_selectors = std::array::from_fn(|_| vec![Fr::zero(); rows]);
            let mut right_ids = std::array::from_fn(|_| vec![Fr::zero(); rows]);
            for edge in edges {
                let left = left_keys
                    .iter()
                    .position(|key| *key == edge.left)
                    .expect("left value form");
                let right = right_keys
                    .iter()
                    .position(|key| *key == edge.right)
                    .expect("right value form");
                let id = Fr::from_u64(edge.id);
                left_selectors[left][edge.left_row] = Fr::one();
                left_ids[left][edge.left_row] = id;
                right_selectors[right][edge.right_row] = Fr::one();
                right_ids[right][edge.right_row] = id;
            }
            let link = CopyLink::new(
                CopyLinkSide::new(left_selectors, left_ids).expect("T1 element side"),
                CopyLinkSide::new(right_selectors, right_ids).expect("T2 element side"),
            )
            .expect("T1/T2 element link");
            let left_keys: [HashLinkValue; WIRES] =
                left_keys.try_into().expect("three T1 value forms");
            CopySpec {
                fixed: fixed_copy_columns(&link),
                link,
                left: left_keys.map(|key| LeftLinkValue::Hash(key.form())),
                right: right_keys.try_into().expect("three T2 value forms"),
                public: None,
            }
        })
        .collect()
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

fn challenge_copy_spec(links: &LinkMap, relation: &Relation, relation_base: usize) -> CopySpec {
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
    let left = CopyLinkSide::new(left_selectors, left_ids).expect("T1 link side");
    let right = CopyLinkSide::new(right_selectors, right_ids).expect("R link side");
    let copy = CopyLink::new(left, right).expect("challenge link");
    CopySpec {
        fixed: fixed_copy_columns(&copy),
        link: copy,
        left: [
            LeftLinkValue::Hash(challenge125()),
            LeftLinkValue::Hash(challenge_scalar128()),
            LeftLinkValue::Zero,
        ],
        right: [
            T2LinkValue::Relation,
            T2LinkValue::Relation,
            T2LinkValue::Zero,
        ],
        public: None,
    }
}

fn absorbed_word_copy_spec(links: &LinkMap, base: usize, rows: usize) -> CopySpec {
    let mut left_selectors = std::array::from_fn(|_| vec![Fr::zero(); rows]);
    let mut left_ids = std::array::from_fn(|_| vec![Fr::zero(); rows]);
    let mut right_selectors = std::array::from_fn(|_| vec![Fr::zero(); rows]);
    let mut right_ids = std::array::from_fn(|_| vec![Fr::zero(); rows]);
    for &(index, row) in &links.wires {
        let id = Fr::from_u64(u64::from(index) + 1);
        left_selectors[0][row] = Fr::one();
        left_ids[0][row] = id;
        right_selectors[0][base + index as usize] = Fr::one();
        right_ids[0][base + index as usize] = id;
    }
    for &(index, row) in &links.wires_shifted {
        let id = Fr::from_u64(u64::from(index) + 1);
        left_selectors[1][row] = Fr::one();
        left_ids[1][row] = id;
        right_selectors[0][base + index as usize] = Fr::one();
        right_ids[0][base + index as usize] = id;
    }
    assert_eq!(links.wires.len() + links.wires_shifted.len(), 1_199);
    let link = CopyLink::new(
        CopyLinkSide::new(left_selectors, left_ids).expect("T1 absorbed-word side"),
        CopyLinkSide::new(right_selectors, right_ids).expect("R absorbed-word side"),
    )
    .expect("absorbed-word link");
    CopySpec {
        fixed: fixed_copy_columns(&link),
        link,
        left: [
            LeftLinkValue::Hash(fr_word()),
            LeftLinkValue::Hash(fr_word_shifted()),
            LeftLinkValue::Zero,
        ],
        right: [T2LinkValue::Relation, T2LinkValue::Zero, T2LinkValue::Zero],
        public: None,
    }
}

fn public_copy_spec(base: usize, values: &[Fr], rows: usize) -> CopySpec {
    let public_rows = (0..values.len()).collect::<Vec<_>>();
    let mut left_selectors = std::array::from_fn(|_| vec![Fr::zero(); rows]);
    let mut left_ids = std::array::from_fn(|_| vec![Fr::zero(); rows]);
    let mut right_selectors = std::array::from_fn(|_| vec![Fr::zero(); rows]);
    let mut right_ids = std::array::from_fn(|_| vec![Fr::zero(); rows]);
    for (index, &row) in public_rows.iter().enumerate() {
        let id = Fr::from_u64(index as u64 + 1);
        left_selectors[0][row] = Fr::one();
        left_ids[0][row] = id;
        right_selectors[0][base + index] = Fr::one();
        right_ids[0][base + index] = id;
    }
    let link = CopyLink::new(
        CopyLinkSide::new(left_selectors, left_ids).expect("public statement side"),
        CopyLinkSide::new(right_selectors, right_ids).expect("R public-input side"),
    )
    .expect("public-input link");
    CopySpec {
        fixed: fixed_copy_columns(&link),
        link,
        left: [
            LeftLinkValue::Public,
            LeftLinkValue::Zero,
            LeftLinkValue::Zero,
        ],
        right: [T2LinkValue::Relation, T2LinkValue::Zero, T2LinkValue::Zero],
        public: Some((0, public_rows, values.to_vec())),
    }
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
    _wire_phase_groups: [usize; 5],
    _k: usize,
    rejected: impl Fn(&WrapperProof) -> bool,
) {
    let original = proof.clone();
    let tamper = |edit: &dyn Fn(&mut WrapperProof)| {
        let mut candidate = original.clone();
        edit(&mut candidate);
        assert!(rejected(&candidate));
    };
    for challenge in 0..original.public_challenges.len() {
        tamper(&|candidate| candidate.public_challenges[challenge][0] ^= 1);
    }
    for commitment in 0..original.commitments.len() {
        tamper(&|candidate| {
            candidate.commitments[commitment] = Commitment::new(original.opening.com[0]);
        });
    }
    for stage in 0..original.stages.len() {
        for round in 0..original.stages[stage]
            .round_polynomials
            .round_polynomials
            .len()
        {
            for coefficient in 0..original.stages[stage].round_polynomials.round_polynomials[round]
                .coeffs_except_linear_term()
                .len()
            {
                tamper(&|candidate| {
                    let polynomial =
                        &mut candidate.stages[stage].round_polynomials.round_polynomials[round];
                    let mut coefficients = polynomial.coeffs_except_linear_term().to_vec();
                    coefficients[coefficient] += Fr::one();
                    *polynomial = CompressedPoly::new(coefficients);
                });
            }
        }
        if let Some(committed) = &original.stages[stage].committed_rounds {
            for round in 0..committed.round_commitments.len() {
                tamper(&|candidate| {
                    candidate.stages[stage]
                        .committed_rounds
                        .as_mut()
                        .expect("committed stage")
                        .round_commitments[round] += Bn254::g1_generator();
                });
            }
            for round in 0..committed.round_claims.len() {
                tamper(&|candidate| {
                    candidate.stages[stage]
                        .committed_rounds
                        .as_mut()
                        .expect("committed stage")
                        .round_claims[round] += Fr::one();
                });
            }
            tamper(&|candidate| {
                candidate.stages[stage]
                    .committed_rounds
                    .as_mut()
                    .expect("committed stage")
                    .sum_at_zero += Fr::one();
            });
            if committed.opening.is_some() {
                tamper(&|candidate| {
                    candidate.stages[stage]
                        .committed_rounds
                        .as_mut()
                        .expect("committed stage")
                        .opening
                        .as_mut()
                        .expect("stage opening")
                        .shifted_commitment += Bn254::g1_generator();
                });
                tamper(&|candidate| {
                    candidate.stages[stage]
                        .committed_rounds
                        .as_mut()
                        .expect("committed stage")
                        .opening
                        .as_mut()
                        .expect("stage opening")
                        .quotient_commitment += Bn254::g1_generator();
                });
                tamper(&|candidate| {
                    candidate.stages[stage]
                        .committed_rounds
                        .as_mut()
                        .expect("committed stage")
                        .opening
                        .as_mut()
                        .expect("stage opening")
                        .evaluation_witness += Bn254::g1_generator();
                });
            }
        }
    }
    if original.round_opening.is_some() {
        tamper(&|candidate| {
            candidate
                .round_opening
                .as_mut()
                .expect("shared round opening")
                .shifted_commitment += Bn254::g1_generator();
        });
        tamper(&|candidate| {
            candidate
                .round_opening
                .as_mut()
                .expect("shared round opening")
                .quotient_commitment += Bn254::g1_generator();
        });
        tamper(&|candidate| {
            candidate
                .round_opening
                .as_mut()
                .expect("shared round opening")
                .evaluation_witness += Bn254::g1_generator();
        });
    }
    for stage in 0..original.stage_claims.len() {
        for claim in 0..original.stage_claims[stage].len() {
            tamper(&|candidate| candidate.stage_claims[stage][claim] += Fr::one());
        }
    }
    for evaluation in 0..original.term_evaluations.len() {
        tamper(&|candidate| candidate.term_evaluations[evaluation] += Fr::one());
    }
    for claim in 0..original.reduced_claims.len() {
        tamper(&|candidate| candidate.reduced_claims[claim] += Fr::one());
    }
    for commitment in 0..original.opening.com.len() {
        tamper(&|candidate| candidate.opening.com[commitment] += Bn254::g1_generator());
    }
    tamper(&|candidate| candidate.opening.w += Bn254::g1_generator());
    for row in 0..original.opening.v.len() {
        for evaluation in 0..original.opening.v[row].len() {
            tamper(&|candidate| candidate.opening.v[row][evaluation] += Fr::one());
        }
    }
    tamper(&|candidate| candidate.opening.p0_at_r_squared += Fr::one());
}

fn report(
    proof: &WrapperProof,
    wire_phase_groups: [usize; 5],
    term_count: usize,
    cost: VerifierCost,
    statement_fields: usize,
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
        "bytes phase1a={} phase1b={} phase2a={} phase2b={} phase2c={} stage_a={stage_a} term={term_stage} shared_bdfg={shared} ell={ell} stage_b={stage_b} reduced={reduced} hyperkzg={opening} proof={} bincode={} statement={}",
        commitment_bytes[0],
        commitment_bytes[1],
        commitment_bytes[2],
        commitment_bytes[3],
        commitment_bytes[4],
        proof.payload_bytes(),
        proof.bincode_bytes(),
        32 * statement_fields,
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
