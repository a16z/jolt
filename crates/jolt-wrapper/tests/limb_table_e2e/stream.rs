/// Milestone 2: through the stream interface — phase groups built in
/// protocol order, physical ids, the two members and the `TermExporter` —
/// the exported terms at the members' claims equal the batched final claim,
/// every member final equals the packed columns' evaluation at the stage
/// point, and the digit link's input claim is R's weighted scalar claim plus
/// the constant-one and offset terms.
#[test]
fn stream_exporter_terms_match_the_members() {
    use jolt_crypto::Bn254;
    use jolt_hyperkzg::HyperKZGScheme;
    use jolt_wrapper::limb_table::export::phases;
    use jolt_wrapper::limb_table::stream::{
        commitment_phases, prover_group_count, Members, StreamTermExporter, T2Challenges,
    };
    use jolt_wrapper::stream::{commit_packed, Column, TermContext, TermExporter};
    let w = witness(0xE2E);
    let mut rng = ChaCha20Rng::seed_from_u64(0x57E4);
    let theta = Fr::from(w.values.get(&Wire::Offset));
    let phase_challenges: Vec<Fr> = (0..T2Challenges::count()).map(|_| fr(&mut rng)).collect();
    let rho = fr(&mut rng);
    let challenges = T2Challenges::from_challenges(theta, &phase_challenges, rho);

    // Columns: kinds and ids in phase order, group counts as declared.
    let packing = 4;
    let witness = staged(&w, &challenges.row, packing, 3);
    let first_group = witness
        .stream
        .ids
        .iter()
        .map(|id| id.group)
        .min()
        .unwrap();
    let mut physical_columns = vec![
        Column::Bits(vec![0; 1usize << LOG_ROWS]);
        witness.stream.group_count * packing
    ];
    for (local, id) in witness.stream.ids.iter().enumerate() {
        let physical = (id.group - first_group) * packing + id.slot;
        physical_columns[physical] = witness.matrix.column(local);
    }
    let StreamWitness {
        relation,
        matrix,
        stream,
    } = witness;
    let columns = (0..Col::WIDTH)
        .map(|column| matrix.field_column(column))
        .collect::<Vec<_>>();
    let declared: usize = commitment_phases(packing)
        .iter()
        .map(|p| p.group_count)
        .sum();
    assert_eq!(
        stream.vk_groups.end,
        3 + declared,
        "phases cover every group"
    );
    assert_eq!(stream.vk_groups.start, 3 + prover_group_count(4));
    assert_eq!(stream.vk_groups.end, 3 + stream.group_count);
    let physical = |local: usize| {
        let id = stream.ids[local];
        (id.group - 3) * packing + id.slot
    };
    for spec in phases() {
        for local in spec.columns {
            match (&physical_columns[physical(local)], local) {
                (Column::U16(values), l) if l < Col::DIGITS => {
                    assert!(values
                        .iter()
                        .zip(&columns[l])
                        .all(|(v, c)| Fr::from_u64(u64::from(*v)) == *c));
                }
                (Column::Bits(values), l) => {
                    assert!(
                        values
                            .iter()
                            .zip(&columns[l])
                            .all(|(v, c)| Fr::from_u64(u64::from(*v)) == *c),
                        "column {l}"
                    );
                }
                (Column::U32(values), l) => {
                    assert!(
                        values
                            .iter()
                            .zip(&columns[l])
                            .all(|(v, c)| Fr::from_u64(u64::from(*v)) == *c),
                        "column {l}"
                    );
                }
                (Column::Fr(values), l) => assert_eq!(values, &columns[l], "column {l}"),
                (Column::U16(_), l) => panic!("column {l} is not a chunk"),
            }
        }
    }

    // Members driven jointly; the exporter's terms at their claims.
    let mut members = Members::new(&relation, &matrix, &w.layout, rho);
    assert_eq!(
        members.link.input_claim(),
        link_input_claim(r_link_claim(&w, rho), rho, theta, &w.layout),
        "digit link pairs with R's scalar link claim"
    );
    assert_eq!(members.rows.input_claim(), Fr::zero());
    let mut driver = ChaCha20Rng::seed_from_u64(99);
    let (point, row_claim) = drive(&mut members.rows, Fr::zero(), &mut driver);
    let link_input = members.link.input_claim();
    let mut driver = ChaCha20Rng::seed_from_u64(99);
    let (r_link, link_claim) = drive(&mut members.link, link_input, &mut driver);
    assert_eq!(point, r_link);
    let claims = members.rows.claims();
    let digit_final = members.link.final_values().digit;
    drop(members);
    let batching = [fr(&mut rng), fr(&mut rng)];
    let mut all_challenges = vec![theta];
    all_challenges.extend(phase_challenges.iter().copied());
    all_challenges.push(rho);
    let exporter = StreamTermExporter {
        layout: &w.layout,
        challenge_offset: 1,
        theta_offset: 0,
        rho_offset: 1 + T2Challenges::count(),
        columns: &stream.ids,
        row_member: 0,
        link_member: 1,
    };
    let mut cost = VerifierCost::default();
    let terms = exporter.terms_observed(
        &TermContext {
            row_point: &point,
            batching_coefficients: &batching,
            challenges: &all_challenges,
        },
        &mut cost,
    );
    println!(
        "stream exporter: {} terms, {} fr_mul",
        terms.len(),
        cost.fr_mul
    );
    let value: Fr = terms
        .iter()
        .map(|term| {
            term.factors.iter().fold(term.coefficient, |acc, form| {
                acc * form
                    .weights
                    .iter()
                    .fold(form.constant, |acc, (id, weight)| {
                        let local = stream.ids.iter().position(|i| i == id).unwrap();
                        acc + *weight * claims[local]
                    })
            })
        })
        .sum();
    assert_eq!(value, batching[0] * row_claim + batching[1] * link_claim);

    // Every member final is the packed columns' evaluation at the stage point
    // (the stream opens the packed groups there, big-endian).
    let rows = 1usize << LOG_ROWS;
    let setup = HyperKZGScheme::<Bn254>::setup_from_secret(
        Fr::from_u64(97),
        rows * packing,
        Bn254::g1_generator(),
        Bn254::g2_generator(),
    );
    let packed = commit_packed(&physical_columns, packing, &setup).expect("packed columns");
    let evaluations = packed.column_evaluations(&point).expect("evaluations");
    for (local, claim) in claims.iter().enumerate() {
        assert_eq!(evaluations[physical(local)], *claim, "column {local}");
    }
    assert_eq!(
        evaluations[physical(Col::D)],
        digit_final,
        "digit link final"
    );
}

/// Review #3 blocker: two chains sharing a scalar (`θ`, read by every
/// offset chain) cannot recode it differently. The digit link weighs every
/// chain-base occurrence with its own `ρ` power, so cancelling `±1` digit
/// shifts in one window of two chains change `Σ ω·D` away from the input
/// claim the verifier derives from R's scalar claim.
#[test]
fn shared_scalar_recoded_differently_per_chain_is_rejected() {
    let w = witness(0xE2E);
    let mut rng = ChaCha20Rng::seed_from_u64(0xD1617);
    let public = PublicColumns::new(&w.layout);
    let offset_kd = w.layout.digit_bases - 1;
    let pair = (0..64u32).find_map(|window| {
        let ops: Vec<_> = w
            .layout
            .digit_ops
            .iter()
            .filter(|op| op.kd == offset_kd && op.w == window)
            .collect();
        let digit = public.digit_values[ops.first()?.first_row as usize];
        (ops.len() >= 2 && digit != Fr::from_i64(-8) && digit != Fr::from_u64(7))
            .then(|| (*ops[0], *ops[1]))
    });
    let (plus, minus) = pair.expect("an interior offset digit shared by two chains");
    assert_ne!(plus.link, minus.link, "distinct occurrences");
    let rho = fr(&mut rng);
    let theta = Fr::from(w.values.get(&Wire::Offset));
    let expected = link_input_claim(r_link_claim(&w, rho), rho, theta, &w.layout);
    let chunks = window_chunks(&w);
    assert_eq!(
        LinkMember::new(&w.layout, rho, &public.digit_values, &chunks).input_claim(),
        expected,
        "honest recodings"
    );
    let mut altered = public.digit_values.clone();
    altered[plus.first_row as usize] += Fr::from_u64(1);
    altered[minus.first_row as usize] -= Fr::from_u64(1);
    assert_ne!(
        LinkMember::new(&w.layout, rho, &altered, &chunks).input_claim(),
        expected,
        "each occurrence is bound to the scalar on its own"
    );
}

/// Review #4 blocker: `s ± r` recodes to another valid signed digit string of
/// the same residue. The window check admits one recoding per scalar: with
/// honest window rows an aliased occurrence's link claim leaves the one the
/// verifier derives from R, and window rows matching the alias need a chunk
/// outside `[0, 2^16)`, which the row member's range LogUp rejects.
#[test]
fn modulus_alias_recodings_are_rejected() {
    let w = witness(0xE2E);
    let mut rng = ChaCha20Rng::seed_from_u64(0xA11A5);
    let ch = challenges(&mut rng);
    let rho = fr(&mut rng);
    let relation = RowRelation::new(
        ch,
        LookupConstants {
            one_row: w.layout.one_cell * 16,
        },
    );
    let columns = matrix(&w, &relation);
    let one_kd = w.layout.digit_bases - 2;
    let occurrence = w
        .layout
        .digit_ops
        .iter()
        .find(|op| op.kd == one_kd)
        .expect("constant-one occurrence")
        .link;
    let ops: Vec<_> = w
        .layout
        .digit_ops
        .iter()
        .filter(|op| op.link == occurrence)
        .collect();
    assert_eq!(ops.len(), 64);
    let modulus = BigInt::from_biguint(
        Sign::Plus,
        BigUint::from_bytes_le(&ArkFr::MODULUS.to_bytes_le()),
    );
    let theta = Fr::from(w.values.get(&Wire::Offset));
    let expected = link_input_claim(r_link_claim(&w, rho), rho, theta, &w.layout);
    for alias in [BigInt::from(1) + &modulus, BigInt::from(1) - &modulus] {
        let digits = recode(&alias);
        let mut aliased = columns.clone();
        for op in &ops {
            let d = digits[63 - op.w as usize];
            let bits = digit_bits(u8::try_from(d + 8).unwrap());
            let first = op.first_row as usize;
            for (b, bit) in bits.iter().enumerate() {
                for value in &mut aliased[Col::DIGITS + b][first..first + usize::from(op.rows)] {
                    *value = Fr::from_u64(u64::from(*bit));
                }
            }
            aliased[Col::D][op.first_row as usize] = Fr::from_i64(d);
        }
        // Honest window rows: the link's claim is not the verifier's.
        let link = LinkMember::new(
            &w.layout,
            rho,
            &aliased[Col::D],
            &aliased[Col::CHUNKS..Col::CHUNKS + 8],
        );
        assert_ne!(
            link.input_claim(),
            expected,
            "alias {alias} with honest window rows"
        );
        // Window rows matching the alias: `V_hi` outside `0..=WINDOW_BOUND`
        // only fits the identities with an out-of-range chunk.
        let v_hi = digits[48..]
            .iter()
            .rev()
            .fold(BigInt::zero(), |acc, d| acc * 16 + d);
        let v = fr_from_bigint(&v_hi);
        let row = WINDOW_ROW_BASE as usize + occurrence as usize;
        let mut forged = aliased;
        for j in 0..8 {
            forged[Col::CHUNKS + j][row] = Fr::zero();
        }
        forged[Col::CHUNKS][row] = v;
        forged[Col::CHUNKS + 4][row] = Fr::from_u64(WINDOW_BOUND) - v;
        let link = LinkMember::new(
            &w.layout,
            rho,
            &forged[Col::D],
            &forged[Col::CHUNKS..Col::CHUNKS + 8],
        );
        assert_eq!(
            link.input_claim(),
            expected,
            "alias {alias} with matching window rows satisfies the link"
        );
        rejects(forged, &relation, &w.layout, |_| {});
    }
}

/// Every phase slice the builder returns has the group count
/// `commitment_phases` declares, at every packing the assembly uses.
#[test]
fn stream_builder_phase_slices_match_declared_geometry() {
    use jolt_wrapper::limb_table::stream::commitment_phases;

    let w = witness(0xE2E);
    let mut rng = ChaCha20Rng::seed_from_u64(0x00C0_1117);
    let ch = challenges(&mut rng);
    for packing in [4, 16, 32] {
        let declared = commitment_phases(packing);
        let mut builder = StreamBuilder::new(&w.layout, &w.columns, packing);
        assert_eq!(builder.phase_1b().len() / packing, declared[0].group_count);
        assert_eq!(
            builder.phase_2a(ch.xi, ch.alpha).len() / packing,
            declared[1].group_count
        );
        assert_eq!(
            builder.phase_2b(ch.fp_root).len() / packing,
            declared[2].group_count
        );
        assert_eq!(
            builder
                .phase_2c(ch.beta, ch.fp_combine, ch.copy_root, Vec::new())
                .len()
                / packing,
            declared[3].group_count
        );
    }
    assert_eq!(GROUP_SIZE, 4);
    assert_eq!(
        commitment_phases(32).map(|phase| phase.group_count),
        [3, 2, 1, 2]
    );
}

/// Every packed group, including the pinned verifier-key suffix, must belong
/// to a commitment phase or `AssemblyStatement` rejects the table shape.
#[test]
fn commitment_phases_cover_verifier_key_groups() {
    use jolt_wrapper::limb_table::stream::{commitment_phases, prover_group_count, vk_group_range};

    for packing in [4, 16, 32] {
        let declared: usize = commitment_phases(packing)
            .iter()
            .map(|phase| phase.group_count)
            .sum();
        let all_groups = prover_group_count(packing) + vk_group_range(packing, 0).len();
        assert_eq!(declared, all_groups, "packing {packing}");
    }
}

/// Review #5: copying one valid `(V, V')` row onto another occurrence keeps
/// every chunk in range and `V + V' = WINDOW_BOUND`, but the row's `ρ^o`
/// coefficient ties it to its own occurrence's top digits: the link rejects.
#[test]
fn one_window_row_cannot_be_reused_for_two_occurrences() {
    let w = witness(0xE2E);
    let mut rng = ChaCha20Rng::seed_from_u64(0x5E05E);
    let ch = challenges(&mut rng);
    let rho = fr(&mut rng);
    let relation = RowRelation::new(
        ch,
        LookupConstants {
            one_row: w.layout.one_cell * 16,
        },
    );
    let columns = matrix(&w, &relation);
    let theta = Fr::from(w.values.get(&Wire::Offset));
    let expected = link_input_claim(r_link_claim(&w, rho), rho, theta, &w.layout);
    let rows =
        WINDOW_ROW_BASE as usize..WINDOW_ROW_BASE as usize + w.layout.link_occurrences as usize;
    let (source, target) = rows
        .clone()
        .flat_map(|source| rows.clone().map(move |target| (source, target)))
        .find(|&(source, target)| {
            source != target
                && (0..8)
                    .any(|j| columns[Col::CHUNKS + j][source] != columns[Col::CHUNKS + j][target])
        })
        .expect("two occurrences with different top windows");
    let mut reused = columns.clone();
    for j in 0..8 {
        reused[Col::CHUNKS + j][target] = columns[Col::CHUNKS + j][source];
    }
    assert_ne!(
        LinkMember::new(
            &w.layout,
            rho,
            &reused[Col::D],
            &reused[Col::CHUNKS..Col::CHUNKS + 8],
        )
        .input_claim(),
        expected
    );
}
