#![expect(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::print_stdout,
    reason = "manual real-fixture integration gate"
)]

use std::path::Path;
use std::process::{self, Command};
use std::time::{Duration, Instant};

use bincode::config::standard;
use bincode::serde::{decode_from_slice, encode_to_vec};
use common::jolt_device::JoltDevice;
use jolt_crypto::Bn254;
use jolt_field::{Fr, One, Ring, Zero};
use jolt_hyperkzg::{HyperKZGProverSetup, HyperKZGScheme, HyperKZGVerifierSetup};
use jolt_poly::{CompressedPoly, UnivariatePoly};
use jolt_sumcheck::prover::ProveRounds;
use jolt_sumcheck::SumcheckError;
use jolt_verifier::{JoltProof, JoltVerifierPreprocessing};
use jolt_wrapper::hash_table::{
    LinkMap, Members as HashMembers, StreamColumns as HashStreamColumns, T1Challenges,
};
use jolt_wrapper::limb_table::digit_link::LinkMember;
use jolt_wrapper::limb_table::lookup::link_weights;
use jolt_wrapper::limb_table::relation::{Col as T2Col, RowSumcheck as T2RowSumcheck};
use jolt_wrapper::limb_table::schedule::WINDOW_ROW_BASE;
use jolt_wrapper::limb_table::stream::{
    commitment_phases as t2_commitment_phases, link_input_claim,
    vk_group_range as t2_vk_group_range, LimbTableKey, Members as T2Members,
    StreamBuilder as T2StreamBuilder, StreamWitness, T2Challenges, PHASE_CHALLENGES,
};
use jolt_wrapper::links::DoryScalarLinkProver;
use jolt_wrapper::profile::WrapperProfile;
use jolt_wrapper::relation::{Pcs, ScheduleEntry, Vc};
use jolt_wrapper::stream::{
    commit_packed, Column, Commitment, StageMember, StageProof, TermContext, VerifierCost,
    WrapperProof,
};
use jolt_wrapper::wrap::{
    verify_wrapped_with_key, wrap as wrap_proof, WrapCommitments, WrapConfig, WrapError,
    WrapHashKey, WrapPreparation, WrapVerifierKey,
};
use jolt_wrapper::SpartanError;

#[path = "wrap_real_t1_r/t2.rs"]
mod t2;
use t2::Base as T2Base;

type Proof = JoltProof<Pcs, Vc>;
type Preprocessing = JoltVerifierPreprocessing<Pcs, Vc>;

struct DoryLinkedProver {
    digit: LinkMember,
    scalar: DoryScalarLinkProver,
    digit_claim: Fr,
    scalar_claim: Fr,
    pending: Option<(UnivariatePoly<Fr>, UnivariatePoly<Fr>)>,
}

struct TimedProver<P> {
    inner: P,
    elapsed: Duration,
}

impl<P> TimedProver<P> {
    fn new(inner: P) -> Self {
        Self {
            inner,
            elapsed: Duration::ZERO,
        }
    }
}

impl<P: ProveRounds<Fr>> ProveRounds<Fr> for TimedProver<P> {
    fn num_rounds(&self) -> usize {
        self.inner.num_rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<Fr>,
        round: usize,
        previous_claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        let started = Instant::now();
        let result = self.inner.prove_round(bind, round, previous_claim);
        self.elapsed += started.elapsed();
        result
    }

    fn finish_rounds(&mut self, bind: Fr) -> Result<(), SumcheckError<Fr>> {
        let started = Instant::now();
        let result = self.inner.finish_rounds(bind);
        self.elapsed += started.elapsed();
        result
    }

    fn append_bound_values(&self, values: &mut Vec<Fr>) -> Result<(), SumcheckError<Fr>> {
        self.inner.append_bound_values(values)
    }
}

impl DoryLinkedProver {
    fn new(digit: LinkMember, scalar: DoryScalarLinkProver, input_claim: Fr) -> Self {
        let digit_claim = digit.input_claim();
        let scalar_claim = scalar.input_claim();
        assert_eq!(
            digit_claim - scalar_claim,
            input_claim,
            "digit={digit_claim:?} scalar={scalar_claim:?}"
        );
        Self {
            digit,
            scalar,
            digit_claim,
            scalar_claim,
            pending: None,
        }
    }
}

impl ProveRounds<Fr> for DoryLinkedProver {
    fn num_rounds(&self) -> usize {
        self.digit.num_rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<Fr>,
        round: usize,
        previous_claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        if let Some(challenge) = bind {
            let (digit, scalar) = self
                .pending
                .take()
                .unwrap_or_else(|| unreachable!("prior round supplies bind polynomials"));
            self.digit_claim = digit.evaluate(challenge);
            self.scalar_claim = scalar.evaluate(challenge);
        }
        if self.digit_claim - self.scalar_claim != previous_claim {
            return Err(SumcheckError::RoundCheckFailed {
                round,
                expected: previous_claim,
                actual: self.digit_claim - self.scalar_claim,
            });
        }
        let digit = self.digit.prove_round(bind, round, self.digit_claim)?;
        let scalar = self.scalar.prove_round(bind, round, self.scalar_claim)?;
        let combined = &digit - &scalar;
        self.pending = Some((digit, scalar));
        Ok(combined)
    }

    fn finish_rounds(&mut self, bind: Fr) -> Result<(), SumcheckError<Fr>> {
        self.digit.finish_rounds(bind)?;
        self.scalar.finish_rounds(bind)
    }

    fn append_bound_values(&self, _values: &mut Vec<Fr>) -> Result<(), SumcheckError<Fr>> {
        // Both sides reuse columns supplied by the T2 and carry members.
        Ok(())
    }
}

const FIXTURE: &str = "/Volumes/Dev/scratch/wrapper-fixtures/fibonacci_2_18_blake3.bin";
const LOG_ROWS: usize = 18;
const ROWS: usize = 1 << LOG_ROWS;
#[test]
fn real_wrapper_round_trip_and_tampers() {
    assert_eq!(WrapConfig::default().packing_factor, 16);
    let k = std::env::var("WRAP_K")
        .map_or(Ok(WrapConfig::default().packing_factor), |value| {
            value.parse()
        })
        .expect("WRAP_K is an integer");
    assert!(matches!(k, 16 | 32));
    let config = WrapConfig {
        common_log_rows: LOG_ROWS,
        packing_factor: k,
    };
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
    let profile_hash_key =
        WrapHashKey::from_reference(&preprocessing, &public_io, &original_proof, config, &setup)
            .expect("build trusted T1 key");
    let key_profile_ms = started.elapsed().as_millis();

    let key_preparation = WrapPreparation::new(
        &preprocessing,
        &public_io,
        &original_proof,
        config,
        &profile_hash_key,
    )
    .expect("prepare verifier-key inputs");
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
    let (_, reference_layout) = T2Base::new(
        &preprocessing,
        &original_proof,
        &key_preparation.relation,
        &key_preparation.relation_witness,
        Fr::zero(),
    )
    .expect("build reference T2 link layout");
    let hash_key = profile_hash_key.clone();
    let wrong_hash_key = hash_key.clone();
    let public_hash_key = hash_key.clone();
    let statement_hash_key = hash_key.clone();
    let program_hash_key = hash_key.clone();
    let started_key = Instant::now();
    let verifier_key = WrapVerifierKey::new(
        &key_preparation.profile,
        hash_key,
        key_preparation.hash_public.clone(),
        LimbTableKey::new(reference_layout.clone(), k, &setup).expect("T2 verifier key"),
        key_preparation.public_known.clone(),
        &setup,
    )
    .expect("wrapper verifier key");
    let fixed_key_commit_ms = started_key.elapsed().as_millis();
    let mut wrong_layout = reference_layout.clone();
    wrong_layout.one_cell += 1;
    let wrong_verifier_key = WrapVerifierKey::new(
        &key_preparation.profile,
        wrong_hash_key,
        key_preparation.hash_public.clone(),
        LimbTableKey::new(wrong_layout, k, &setup).expect("wrong-key fixture"),
        key_preparation.public_known.clone(),
        &setup,
    )
    .expect("wrong-pin wrapper key");
    let mut wrong_statement_hash_public = key_preparation.hash_public.clone();
    wrong_statement_hash_public.state_in[0] ^= 1;
    let statement_verifier_key = WrapVerifierKey::new(
        &key_preparation.profile,
        statement_hash_key,
        wrong_statement_hash_public.clone(),
        LimbTableKey::new(reference_layout.clone(), k, &setup).expect("statement-mismatch T2 key"),
        key_preparation.public_known.clone(),
        &setup,
    )
    .expect("statement-mismatch wrapper key");
    let mut wrong_public = key_preparation.public_known.clone();
    wrong_public[0] += Fr::one();
    let public_verifier_key = WrapVerifierKey::new(
        &key_preparation.profile,
        public_hash_key,
        key_preparation.hash_public.clone(),
        LimbTableKey::new(reference_layout.clone(), k, &setup).expect("public-mismatch T2 key"),
        wrong_public,
        &setup,
    )
    .expect("public-mismatch wrapper key");
    let mut wrong_profile = key_preparation.profile.clone();
    wrong_profile.bytecode_ra_commitments += 1;
    let program_verifier_key = WrapVerifierKey::new(
        &wrong_profile,
        program_hash_key,
        key_preparation.hash_public.clone(),
        LimbTableKey::new(reference_layout.clone(), k, &setup).expect("program-mismatch T2 key"),
        key_preparation.public_known.clone(),
        &setup,
    );
    assert!(matches!(
        program_verifier_key,
        Err(WrapError::StatementMismatch)
    ));
    let copy_count = verifier_key.copy_count();
    let t1_challenge_offset = 0;
    let theta_offset = t1_challenge_offset + T1Challenges::count(LOG_ROWS);
    let t2_challenge_offset = theta_offset + 1;
    let copy_challenge_offset = t2_challenge_offset + PHASE_CHALLENGES[0];
    let rho_offset = copy_challenge_offset + 2 * copy_count;
    let t2_after_1b_offset = rho_offset + 1;
    let r_stage_challenge_offset = t2_after_1b_offset + T2Challenges::count() - PHASE_CHALLENGES[0];
    let statement = verifier_key.assembly_statement().clone();

    let uptime_start = uptime();
    let cpu_start_s = process_cpu_seconds();
    let honest_started = Instant::now();
    let mut previous = Duration::ZERO;
    let preparation = WrapPreparation::new(
        &preprocessing,
        &public_io,
        &original_proof,
        config,
        &profile_hash_key,
    )
    .expect("prepare real wrapper inputs");
    let prepare_ms = lap_ms(&honest_started, &mut previous);

    let hash_columns = HashStreamColumns::new(&preparation.hash_table, k, 0);
    let public_columns = 1 + preparation.relation.public.num_public;
    let witness = &preparation.relation_witness.values[public_columns..];
    let mut witness_values = vec![Fr::zero(); ROWS];
    witness_values[..witness.len()].copy_from_slice(witness);
    let links = LinkMap::new(&preparation.hash_key);

    let mut phase_1a_columns = hash_columns.columns;
    let witness_base = phase_1a_columns.len();
    phase_1a_columns.push(Column::Fr(witness_values.clone()));
    pad_fr(&mut phase_1a_columns, k);
    let copy_fixed_base = phase_1a_columns.len();
    let copy_fixed_columns = verifier_key.copy_fixed_columns();
    let copy_vk_groups = copy_fixed_columns.len().div_ceil(k);
    phase_1a_columns.extend(copy_fixed_columns);
    pad_fr(&mut phase_1a_columns, k);
    let phase_1a_groups = phase_1a_columns.len() / k;
    let t2_group_offset = phase_1a_groups;
    let t2_phases = t2_commitment_phases(k);
    let challenge_counts = statement
        .commitment_phases
        .iter()
        .map(|phase| phase.challenge_count)
        .collect::<Vec<_>>();
    assert_eq!(
        challenge_counts,
        vec![39, 23, 1, 3, 232],
        "wrapper Fiat-Shamir phase schedule"
    );
    let t2_vk_groups = t2_vk_group_range(k, 0).len();
    let total_groups = statement
        .commitment_phases
        .iter()
        .map(|phase| phase.group_count)
        .sum::<usize>();
    assert_eq!(statement.commitment_phases[0].group_count, phase_1a_groups);
    let adapt_r_ms = lap_ms(&honest_started, &mut previous);

    let mut commitments = WrapCommitments::new()
        .commit(&phase_1a_columns, &statement, &setup)
        .expect("phase 1a commitments");
    let phase_1_values = commitments.challenges().to_vec();
    let hash_challenges =
        T1Challenges::from_challenges(&phase_1_values[t1_challenge_offset..theta_offset], LOG_ROWS);
    let hash_relation = hash_challenges.relation();
    let theta = phase_1_values[theta_offset];
    let phase_1a_commit_ms = lap_ms(&honest_started, &mut previous);

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
    let adapt_t2_ms = lap_ms(&honest_started, &mut previous);

    let mut t2_builder = T2StreamBuilder::new(&t2_layout, &t2_base.columns, k);
    commitments = commitments
        .commit(t2_builder.phase_1b(), &statement, &setup)
        .expect("T2 phase 1b commitments");

    let phase_1b_values = commitments.challenges();
    let [xi, alpha] = phase_1b_values[t2_challenge_offset..copy_challenge_offset]
        .try_into()
        .expect("T2 phase 1b challenges");
    let copy_challenges = (0..copy_count)
        .map(|index| {
            (
                phase_1b_values[copy_challenge_offset + 2 * index],
                phase_1b_values[copy_challenge_offset + 2 * index + 1],
            )
        })
        .collect::<Vec<_>>();
    let t2_rho = phase_1b_values[rho_offset];
    let phase_1b_commit_ms = lap_ms(&honest_started, &mut previous);
    commitments = commitments
        .commit(t2_builder.phase_2a(xi, alpha), &statement, &setup)
        .expect("T2 phase 2a commitments");

    let fp_root = commitments.challenges()[t2_after_1b_offset];
    let phase_2a_commit_ms = lap_ms(&honest_started, &mut previous);
    commitments = commitments
        .commit(t2_builder.phase_2b(fp_root), &statement, &setup)
        .expect("T2 phase 2b commitments");

    let beta = commitments.challenges()[t2_after_1b_offset + 1];
    let fp_combine = commitments.challenges()[t2_after_1b_offset + 2];
    let copy_root = commitments.challenges()[t2_after_1b_offset + 3];
    let phase_2b_commit_ms = lap_ms(&honest_started, &mut previous);

    let copy_witnesses = verifier_key
        .copy_witnesses(
            &preparation.hash_table,
            &witness_values,
            &t2_base.columns,
            &copy_challenges,
        )
        .expect("linked transcript witnesses");
    for (index, (witness, &(beta, gamma))) in
        copy_witnesses.iter().zip(&copy_challenges).enumerate()
    {
        let link = verifier_key.copy_link(index).expect("canonical copy link");
        link.check(witness, beta, gamma)
            .unwrap_or_else(|error| panic!("linked transcript equality {index}: {error}"));
    }
    let helper_columns = copy_witnesses
        .iter()
        .flat_map(|witness| witness.helper_columns())
        .map(Column::Fr)
        .collect::<Vec<_>>();
    let helper_ms = lap_ms(&honest_started, &mut previous);

    let final_phase_columns = t2_builder.phase_2c(beta, fp_combine, copy_root, helper_columns);
    commitments = commitments
        .commit(final_phase_columns, &statement, &setup)
        .expect("T2 phase 2c, VK and relation-helper commitments");
    let committed = commitments
        .finish(&statement)
        .expect("all commitment phases");
    let full_challenges = committed.challenges();
    assert_eq!(&full_challenges[..=theta_offset], phase_1_values);
    let mut t2_phase_challenges =
        full_challenges[t2_challenge_offset..copy_challenge_offset].to_vec();
    t2_phase_challenges
        .extend_from_slice(&full_challenges[t2_after_1b_offset..r_stage_challenge_offset]);
    let t2_challenges = T2Challenges::from_challenges(theta, &t2_phase_challenges, t2_rho);
    let row = t2_challenges.row;
    let phase_2c_commit_ms = lap_ms(&honest_started, &mut previous);
    let t2_witness = t2_builder.finish(
        row.tau,
        row.gamma,
        row.lambda,
        row.lambda_lookup,
        row.constancy_root,
        t2_group_offset,
    );
    let mut cursor = r_stage_challenge_offset;
    let tau_copies = (0..copy_count)
        .map(|_| take_point(full_challenges, &mut cursor))
        .collect::<Vec<_>>();
    let copy_weights = (0..copy_count)
        .map(|_| take_array(full_challenges, &mut cursor))
        .collect::<Vec<_>>();
    assert_eq!(cursor, full_challenges.len());
    let assembly_challenges = full_challenges.to_vec();
    let t2_finish_ms = lap_ms(&honest_started, &mut previous);

    let t2_member = 2 + copy_count;
    let dory_member = t2_member + 1;
    assert_eq!(verifier_key.hash_links(), &links);
    assert_eq!(verifier_key.hash_schedule(), &preparation.hash_key);
    let key_commit_ms = fixed_key_commit_ms;

    let HashMembers {
        rows: mut hash_rows,
        wiring: mut hash_wiring,
        input_claims: hash_input_claims,
    } = HashMembers::new(&preparation.hash_table, &hash_relation, &hash_challenges);
    let mut copy_rows = (0..copy_count)
        .zip(&copy_witnesses)
        .zip(&tau_copies)
        .zip(&copy_challenges)
        .zip(&copy_weights)
        .map(|((((index, witness), tau), &(beta, gamma)), &weights)| {
            verifier_key
                .copy_link(index)
                .expect("canonical copy link")
                .prover(witness, tau.clone(), beta, gamma, weights)
        })
        .collect::<Vec<_>>();
    let t2_member_started = Instant::now();
    let T2Members {
        rows: t2_rows,
        link: t2_digit_link,
    } = T2Members::new(
        &t2_witness.relation,
        &t2_witness.matrix,
        verifier_key.limb_layout(),
        t2_rho,
    );
    let t2_member_ms = t2_member_started.elapsed().as_millis();
    let scalar_link = verifier_key.dory_scalar_link(t2_rho);
    let scalar_prover = scalar_link.prover(&witness_values);
    let weights = link_weights(verifier_key.limb_layout(), t2_rho);
    let expected_scalar = preparation
        .relation
        .link
        .dory
        .scalars
        .iter()
        .zip(weights)
        .map(|((_, variable), weight)| {
            weight * preparation.relation_witness.values[variable.index()]
        })
        .sum::<Fr>();
    assert_eq!(scalar_prover.input_claim(), expected_scalar);
    let dory_link_claim = link_input_claim(Fr::zero(), t2_rho, theta, verifier_key.limb_layout());
    let mut dory_link = DoryLinkedProver::new(t2_digit_link, scalar_prover, dory_link_claim);
    assert!(hash_rows.input_claim().is_zero());
    assert!(copy_rows.iter().all(|rows| rows.input_claim().is_zero()));
    assert!(t2_rows.input_claim().is_zero());
    let mut timed_t2_rows = TimedProver::new(t2_rows);

    let mut input_claims = vec![hash_input_claims[0], hash_input_claims[1]];
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
    ];
    for (index, rows) in copy_rows.iter_mut().enumerate() {
        members.push(StageMember {
            prover: rows,
            input_claim: input_claims[2 + index],
            degree: 5,
            offset: 0,
        });
    }
    members.extend([
        StageMember {
            prover: &mut timed_t2_rows,
            input_claim: input_claims[t2_member],
            degree: T2RowSumcheck::degree(),
            offset: 0,
        },
        StageMember {
            prover: &mut dory_link,
            input_claim: input_claims[dory_member],
            degree: 2,
            offset: 0,
        },
    ]);
    let member_ms = lap_ms(&honest_started, &mut previous);

    let wrapped = wrap_proof(
        committed,
        &verifier_key,
        &preparation.relation_witness,
        members,
        &setup,
    )
    .expect("prove real T1/T2/R wrapper");
    let t2_stage_a_ms = timed_t2_rows.elapsed.as_millis();
    let prove_ms = lap_ms(&honest_started, &mut previous);
    let honest_online_ms = honest_started.elapsed().as_millis();
    let cpu_seconds = process_cpu_seconds() - cpu_start_s;
    let uptime_end = uptime();
    let online_phase_ms = [
        prepare_ms,
        adapt_r_ms,
        phase_1a_commit_ms,
        adapt_t2_ms,
        phase_1b_commit_ms,
        phase_2a_commit_ms,
        phase_2b_commit_ms,
        helper_ms,
        phase_2c_commit_ms,
        t2_finish_ms,
        member_ms,
        prove_ms,
    ];
    let phase_sum_ms = online_phase_ms.into_iter().sum::<u128>();
    assert!(
        phase_sum_ms.abs_diff(honest_online_ms) * 100 <= honest_online_ms * 2,
        "online phase sum {phase_sum_ms}ms differs from wall {honest_online_ms}ms"
    );
    let verifier_setup = HyperKZGVerifierSetup::from(&setup);
    let started = Instant::now();
    let (results, cost) = verify_wrapped_with_key(&verifier_key, &wrapped, &verifier_setup)
        .expect("verify real T1/T2/R wrapper");
    let verify_ms = started.elapsed().as_millis();
    let term_context = TermContext {
        row_point: &results[2].point,
        batching_coefficients: &results[2].coefficients,
        challenges: &assembly_challenges,
    };
    let term_count = verifier_key.term_count(&term_context);

    let wire_phase_groups = [
        phase_1a_groups - hash_columns.vk_groups.len() - copy_vk_groups,
        t2_phases[0].group_count,
        t2_phases[1].group_count,
        t2_phases[2].group_count,
        statement.commitment_phases[4].group_count - t2_vk_groups,
    ];
    let (expected_wire_groups, expected_total_groups) = match k {
        16 => ([21, 5, 4, 1, 2], 44),
        32 => ([12, 3, 2, 1, 1], 26),
        _ => panic!("unsupported packing factor"),
    };
    assert_eq!(wire_phase_groups, expected_wire_groups);
    assert_eq!(total_groups, expected_total_groups);
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
        .chain(copy_fixed_base / k..copy_fixed_base / k + copy_vk_groups)
        .collect::<Vec<_>>();
    let absorbed_row = preparation
        .relation
        .link
        .schedule
        .iter()
        .find_map(|entry| match entry {
            ScheduleEntry::Fr(variable) => Some(variable.index() - public_columns),
            ScheduleEntry::Bytes(_)
            | ScheduleEntry::Opaque { .. }
            | ScheduleEntry::Squeeze { .. } => None,
        })
        .expect("absorbed R witness");
    assert_r_absorbed_word_commitment_tamper_rejected(
        &wrapped,
        &phase_1a_columns,
        witness_base,
        absorbed_row,
        &pinned_phase_1a,
        k,
        &setup,
        |proof| verify_wrapped_with_key(&verifier_key, proof, &verifier_setup).is_err(),
    );
    assert!(verify_wrapped_with_key(&wrong_verifier_key, &wrapped, &verifier_setup).is_err());
    let original_t1_claims = hash_challenges.input_claims(&preparation.hash_public);
    let changed_t1_claims = hash_challenges.input_claims(&wrong_statement_hash_public);
    assert_ne!(original_t1_claims[1], changed_t1_claims[1]);
    assert_eq!(
        verifier_key.assembly_statement().pinned_commitments,
        statement_verifier_key
            .assembly_statement()
            .pinned_commitments
    );
    assert_ne!(
        verifier_key.assembly_statement().public_inputs,
        statement_verifier_key.assembly_statement().public_inputs
    );
    assert!(matches!(
        verify_wrapped_with_key(&statement_verifier_key, &wrapped, &verifier_setup),
        Err(WrapError::Spartan(SpartanError::OuterFinalClaim))
    ));
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
        k,
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
            ("t2_finish", t2_finish_ms),
            ("members", member_ms),
            ("t2_member", t2_member_ms),
            ("prove", prove_ms),
            ("t2_stage_a", t2_stage_a_ms),
            ("verify", verify_ms),
        ],
        (&uptime_start, &uptime_end),
        (honest_online_ms, phase_sum_ms, cpu_seconds),
    );
    let matrix_nnz = preparation
        .relation
        .matrices
        .a
        .iter()
        .chain(&preparation.relation.matrices.b)
        .chain(&preparation.relation.matrices.c)
        .map(Vec::len)
        .sum::<usize>();
    println!(
        "r1cs constraints={} variables={} public=7 witness={} matrix_nnz={}",
        preparation.relation.matrices.num_constraints,
        preparation.relation.matrices.num_vars,
        witness_values
            .len()
            .min(preparation.relation.matrices.num_vars - public_columns),
        matrix_nnz,
    );
    println!(
        "links challenges={} absorbed_fr={} element_bytes=45152 copy_links={} copy_terms={} dory_scalars=173",
        links.challenges.len(),
        links.wires.len() + links.wires_shifted.len(),
        copy_count,
        10 * copy_count,
    );
    println!(
        "groups k={k} t1_sent={} t1_vk={} w=1 copy_vk={} t2_1b={} t2_2a={} t2_2b={} t2_2c={} t2_vk={} helpers={} full={} wire={}",
        hash_columns.group_count - hash_columns.vk_groups.len(),
        hash_columns.vk_groups.len(),
        copy_vk_groups,
        t2_phases[0].group_count,
        t2_phases[1].group_count,
        t2_phases[2].group_count,
        t2_phases[3].group_count,
        t2_vk_groups,
        0,
        total_groups,
        wire_phase_groups.iter().sum::<usize>(),
    );
}

#[test]
fn commitment_link_order_handles_distinct_ram_and_bytecode_families() {
    let (preprocessing, _, mut proof) = fixture();
    let _ = proof
        .commitments
        .bytecode_ra
        .pop()
        .expect("fixture has bytecode commitments");
    let profile = WrapperProfile::new(&preprocessing, &proof).expect("wrapper profile");
    let instruction = profile.instruction_ra_commitments;
    let ram = profile.ram_ra_commitments;
    let bytecode = profile.bytecode_ra_commitments;
    assert_ne!(ram, bytecode);

    let order = profile.commitment_link_order();
    assert_eq!(&order[..2], &[1, 0]);
    assert_eq!(
        &order[2..2 + instruction],
        &(2..2 + instruction).collect::<Vec<_>>()
    );
    assert_eq!(
        &order[2 + instruction..2 + instruction + bytecode],
        &(2 + instruction + ram..2 + instruction + ram + bytecode).collect::<Vec<_>>()
    );
    assert_eq!(
        &order[2 + instruction + bytecode..],
        &(2 + instruction..2 + instruction + ram).collect::<Vec<_>>()
    );
}

fn assert_t2_row_tamper_rejected(witness: &StreamWitness, column: usize, row: usize) {
    let mut values = (0..T2Col::WIDTH)
        .map(|column| witness.matrix.value(column, row))
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
    let mut columns = vec![Column::Bits(vec![0; ROWS]); k];
    for (local, column_id) in witness.stream.ids.iter().enumerate() {
        if column_id.group == id.group {
            columns[column_id.slot] = witness.matrix.column(local);
        }
    }
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

fn uptime() -> Vec<u8> {
    Command::new("uptime").output().expect("uptime").stdout
}

fn process_cpu_seconds() -> f64 {
    let output = Command::new("ps")
        .args(["-o", "time=", "-p"])
        .arg(process::id().to_string())
        .output()
        .expect("process CPU time");
    let value = String::from_utf8(output.stdout).expect("process CPU time is UTF-8");
    let value = value.trim();
    let (days, clock) = value.split_once('-').map_or((0, value), |(days, clock)| {
        (days.parse::<u64>().expect("CPU days"), clock)
    });
    let parts = clock
        .split(':')
        .map(|part| part.parse::<f64>().expect("CPU clock component"))
        .collect::<Vec<_>>();
    let (hours, minutes, seconds) = match parts.as_slice() {
        [minutes, seconds] => (0.0, *minutes, *seconds),
        [hours, minutes, seconds] => (*hours, *minutes, *seconds),
        _ => panic!("unexpected process CPU time: {value}"),
    };
    days as f64 * 86_400.0 + hours * 3_600.0 + minutes * 60.0 + seconds
}

fn lap_ms(started: &Instant, previous: &mut Duration) -> u128 {
    let elapsed = started.elapsed();
    let milliseconds = elapsed.saturating_sub(*previous).as_millis();
    *previous = elapsed;
    milliseconds
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

fn pad_fr(columns: &mut Vec<Column>, k: usize) {
    while !columns.len().is_multiple_of(k) {
        columns.push(Column::Fr(vec![Fr::zero(); ROWS]));
    }
}

include!("wrap_real_t1_r/tamper.rs");
