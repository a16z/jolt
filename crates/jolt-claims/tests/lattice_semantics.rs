//! Semantic integration tests for the lattice module: build concrete one-hot
//! witness data and check the identities the relations claim — native OneHotTrace
//! member shape, fused-inc chunk reconstruction, and auxiliary
//! advice/bytecode/program-image decodes — against real multilinear evaluations.

use jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode::{
    committed_lane_vars, BYTECODE_LANE_LAYOUT, COMMITTED_BYTECODE_LANE_CAPACITY,
};
use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
use jolt_claims::protocols::jolt::geometry::ra::JoltRaPolynomialLayout;
use jolt_claims::protocols::jolt::lattice::geometry::{
    balanced_inc_value, byte_decode_weight, selector_block_weight,
};
use jolt_claims::protocols::jolt::lattice::{
    one_hot_trace_columns, BalancedIncChunking, OneHotTraceShape,
};
use jolt_claims::protocols::jolt::{BytecodeRegisterLane, JoltCommittedPolynomial};
use jolt_field::{Fr, Ring};
use jolt_lookup_tables::{LookupTableKind, XLEN};
use jolt_poly::{boolean_point_msb, eq_index_msb, EqPolynomial, Polynomial};
use jolt_riscv::{NUM_CIRCUIT_FLAGS, NUM_INSTRUCTION_FLAGS};
use jolt_utils::Math;

fn fr(value: u64) -> Fr {
    Fr::from_u64(value)
}

/// MLE evaluation via the library's own (msb-first) convention — the same one
/// production code uses, so the tests pin the packing against it.
fn eval_mle(evals: &[Fr], point: &[Fr]) -> Fr {
    Polynomial::new(evals.to_vec()).evaluate(point)
}

/// A deterministic, non-boolean evaluation point (distinct small primes).
fn point(len: usize, seed: u64) -> Vec<Fr> {
    const PRIMES: [u64; 16] = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59];
    (0..len)
        .map(|i| fr(PRIMES[(i + seed as usize) % PRIMES.len()] + seed))
        .collect()
}

/// One-hot evaluations over `(hot value ‖ instance)`: index
/// `(hot << log_rows) | row`.
fn one_hot_evals(value_bits: usize, log_rows: usize, hot: &[usize]) -> Vec<Fr> {
    assert_eq!(hot.len(), 1 << log_rows);
    let mut data = vec![fr(0); 1 << (value_bits + log_rows)];
    for (row, &value) in hot.iter().enumerate() {
        assert!(value < (1 << value_bits));
        data[(value << log_rows) | row] = fr(1);
    }
    data
}

fn digit_zero_evals(value_bits: usize, log_rows: usize, hot: &[usize]) -> Vec<Fr> {
    let mut data = one_hot_evals(value_bits, log_rows, hot);
    for (row, value) in hot.iter().copied().enumerate() {
        if value == 0 {
            data[row] = fr(0);
        }
    }
    data
}

/// Byte one-hot evaluations over `(byte ‖ place ‖ word)` for 8-byte words:
/// index `((byte << 3) | place) << word_vars | word`.
fn word_byte_evals(words: &[u64], word_vars: usize) -> Vec<Fr> {
    assert_eq!(words.len(), 1 << word_vars);
    let mut data = vec![fr(0); 1 << (8 + 3 + word_vars)];
    for (w, &word) in words.iter().enumerate() {
        for place in 0..8 {
            let byte = ((word >> (8 * place)) & 0xff) as usize;
            data[(((byte << 3) | place) << word_vars) | w] = fr(1);
        }
    }
    data
}

/// The word decode over row-partial evaluations: `Σ_{byte, place}
/// byte_decode_weight · partials[(byte << place_bits) | place]`.
fn byte_decode_sum(partials: &[Fr], place_bits: usize) -> Fr {
    let mut sum = fr(0);
    for byte in 0..256 {
        for place in 0..(1 << place_bits) {
            sum += byte_decode_weight(
                &boolean_point_msb::<Fr>(8, byte),
                &boolean_point_msb::<Fr>(place_bits, place),
            ) * partials[(byte << place_bits) | place];
        }
    }
    sum
}

/// Every semantic column uses the same `K x T` domain before the digit-zero row is
/// omitted from the commitment.
#[test]
#[expect(clippy::unwrap_used)]
fn one_hot_trace_columns_share_a_uniform_semantic_domain() {
    let log_t = 3;
    let log_k_chunk = 4;
    let shape = OneHotTraceShape {
        ra_layout: JoltRaPolynomialLayout::new(32, 1, 1).unwrap(),
        log_t,
        log_k_chunk,
    };
    let members = one_hot_trace_columns(&shape).unwrap();

    let chunk_count = BalancedIncChunking::new(log_k_chunk).unwrap().chunk_count();
    assert_eq!(chunk_count, 16);
    assert_eq!(members.len(), 51);
    assert_eq!(members[0], JoltCommittedPolynomial::InstructionRa(0));
    assert_eq!(members[48], JoltCommittedPolynomial::BalancedIncCarry);
    assert_eq!(members.last(), Some(&JoltCommittedPolynomial::RamRa(0)));

    for (index, polynomial) in members.iter().enumerate() {
        let hot = if *polynomial == JoltCommittedPolynomial::BalancedIncCarry {
            (0..1 << log_t).map(|t| t % 2).collect::<Vec<_>>()
        } else {
            (0..1 << log_t)
                .map(|t| (7 * t + 3 * index + 1) % (1 << log_k_chunk))
                .collect::<Vec<_>>()
        };
        let data = one_hot_evals(log_k_chunk, log_t, &hot);
        assert_eq!(data.len(), 1 << (log_k_chunk + log_t));
        for cycle in 0..1 << log_t {
            let hamming_weight = (0..1 << log_k_chunk)
                .map(|address| data[(address << log_t) | cycle])
                .sum::<Fr>();
            assert_eq!(hamming_weight, fr(1), "{polynomial:?}, cycle {cycle}");
        }
    }
}

#[test]
fn digit_zero_reconstruction_matches_semantic_one_hot_at_random_points() {
    let log_t = 3;
    let value_bits = 4;
    let rows = 1 << log_t;
    let r_address = point(value_bits, 2);
    let r_cycle = point(log_t, 5);
    let eq_cycle = EqPolynomial::<Fr>::evals(&r_cycle, None);
    let eq_zero = eq_index_msb::<Fr>(&r_address, 0);

    for hot in [
        vec![
            Some(0),
            Some(3),
            Some(0),
            Some(15),
            Some(1),
            Some(0),
            Some(8),
            Some(2),
        ],
        vec![
            None,
            Some(0),
            Some(7),
            None,
            Some(1),
            Some(0),
            None,
            Some(12),
        ],
    ] {
        let mut semantic = vec![fr(0); 1 << (value_bits + log_t)];
        let mut sparse = semantic.clone();
        let mut activation = vec![fr(0); rows];
        for (cycle, row) in hot.iter().copied().enumerate() {
            if let Some(row) = row {
                semantic[(row << log_t) | cycle] = fr(1);
                activation[cycle] = fr(1);
                if row != 0 {
                    sparse[(row << log_t) | cycle] = fr(1);
                }
            }
        }

        let full_point = [r_address.as_slice(), r_cycle.as_slice()].concat();
        let semantic_eval = eval_mle(&semantic, &full_point);
        let activation_eval = eval_mle(&activation, &r_cycle);
        let sparse_by_row = (0..1 << value_bits)
            .map(|row| {
                (0..rows)
                    .map(|cycle| eq_cycle[cycle] * sparse[(row << log_t) | cycle])
                    .sum::<Fr>()
            })
            .collect::<Vec<_>>();
        let recentered = eq_zero * activation_eval
            + sparse_by_row
                .iter()
                .enumerate()
                .map(|(row, value)| {
                    *value * (eq_index_msb::<Fr>(&r_address, row as u128) - eq_zero)
                })
                .sum::<Fr>();
        assert_eq!(recentered, semantic_eval);
    }
}

/// Centered radix digits and their signed carry reconstruct the fused
/// increment when digit-zero entries are absent from the commitment.
#[test]
#[expect(clippy::unwrap_used)]
fn balanced_chunk_decomposition_reconstructs_signed_increments() {
    let log_t = 3;
    let chunking = BalancedIncChunking::new(8).unwrap();
    let count = chunking.chunk_count();
    assert_eq!(count, 8);

    let values: [i128; 8] = [
        5,
        -7,
        0,
        (1 << 63) - 1,
        -(1 << 63),
        123_456_789,
        -987_654_321,
        0,
    ];

    let radix = 1i128 << chunking.chunk_width();
    let bias = (radix / 2) * (((1i128 << 64) - 1) / (radix - 1));
    let mask = radix - 1;
    let mut chunk_hot = vec![vec![0usize; values.len()]; count];
    let mut carry_hot = Vec::with_capacity(values.len());
    let mut fused_data = Vec::with_capacity(values.len());
    for (t, &value) in values.iter().enumerate() {
        let biased = value + bias;
        for (j, hot) in chunk_hot.iter_mut().enumerate() {
            let standard = (biased >> (chunking.chunk_width() * j)) & mask;
            hot[t] = ((standard + radix / 2) & mask) as usize;
        }
        carry_hot.push((biased >> 64).rem_euclid(radix) as usize);
        fused_data.push(Fr::from_i128(value));
    }
    let chunk_polynomials: Vec<Vec<Fr>> = chunk_hot
        .iter()
        .map(|hot| digit_zero_evals(8, log_t, hot))
        .collect();
    let carry_polynomial = digit_zero_evals(8, log_t, &carry_hot);

    let r_cycle = point(log_t, 1);
    let eq_cycle = EqPolynomial::<Fr>::evals(&r_cycle, None);

    let partial = |chunk: &[Fr]| -> Vec<Fr> {
        (0..256)
            .map(|a| {
                (0..values.len())
                    .map(|t| eq_cycle[t] * chunk[(a << log_t) | t])
                    .sum()
            })
            .collect::<Vec<Fr>>()
    };

    let mut reconstructed = fr(0);
    for (j, chunk) in chunk_polynomials.iter().enumerate() {
        let partials = partial(chunk);
        let decoded: Fr = partials
            .iter()
            .enumerate()
            .map(|(a, value)| balanced_inc_value(&boolean_point_msb::<Fr>(8, a)) * *value)
            .sum();
        reconstructed += chunking.place_value::<Fr>(j) * decoded;
    }
    let carry: Fr = partial(&carry_polynomial)
        .iter()
        .enumerate()
        .map(|(a, value)| balanced_inc_value(&boolean_point_msb::<Fr>(8, a)) * *value)
        .sum();

    let fused = eval_mle(&fused_data, &r_cycle);
    assert_eq!(reconstructed + Fr::pow2(64) * carry, fused);
}

/// The bytecode chunk decode identity behind `BytecodeChunkReconstruction`: a
/// committed `BytecodeChunk` polynomial built row-by-row from the lane layout
/// must equal the weighted sum of its per-lane polynomials over their
/// hot-value indices, with every weight sourced from the lane-eq
/// (`eq_index_msb`) and `byte_decode_weight` semantics — pinning the lane
/// offsets, the byte decodes, the plain-0/1 flag encoding, and the zeroing of
/// capacity-padding lanes and padded lookup table indices.
#[test]
fn bytecode_lane_decode_matches_committed_chunk() {
    let log_rows = 2;
    let rows = 1 << log_rows;
    let imm_byte_width = 16;
    let table_count = LookupTableKind::<XLEN>::COUNT;
    let register_count = 1usize << REGISTER_ADDRESS_BITS;
    let layout = BYTECODE_LANE_LAYOUT;
    let lane_weight = |lane_point: &[Fr], lane: usize| eq_index_msb::<Fr>(lane_point, lane as u128);

    struct Row {
        rs1: usize,
        rs2: usize,
        rd: usize,
        unexp_pc: u64,
        imm: u64,
        lookup: usize,
        raf: bool,
    }
    let row_data: Vec<Row> = (0..rows)
        .map(|row| Row {
            rs1: (11 * row + 3) % register_count,
            rs2: (17 * row + 5) % register_count,
            rd: (23 * row + 7) % register_count,
            unexp_pc: 0x8000_0000 + 4 * row as u64,
            imm: [7, 1 << 40, 0, 123_456_789][row],
            lookup: (5 * row + 1) % table_count,
            raf: row % 2 == 1,
        })
        .collect();
    let circuit_flag = |row: usize, flag: usize| (row + flag).is_multiple_of(3);
    let instruction_flag = |row: usize, flag: usize| (row + flag).is_multiple_of(4);

    // The committed chunk polynomial: value[(lane << log_rows) | row].
    let mut chunk_data = vec![fr(0); COMMITTED_BYTECODE_LANE_CAPACITY << log_rows];
    let mut lane = |lane: usize, row: usize, value: Fr| {
        chunk_data[(lane << log_rows) | row] = value;
    };
    for (row, data) in row_data.iter().enumerate() {
        lane(layout.rs1_start + data.rs1, row, fr(1));
        lane(layout.rs2_start + data.rs2, row, fr(1));
        lane(layout.rd_start + data.rd, row, fr(1));
        lane(layout.unexp_pc_idx, row, fr(data.unexp_pc));
        lane(layout.imm_idx, row, fr(data.imm));
        for flag in 0..NUM_CIRCUIT_FLAGS {
            lane(
                layout.circuit_start + flag,
                row,
                fr(u64::from(circuit_flag(row, flag))),
            );
        }
        for flag in 0..NUM_INSTRUCTION_FLAGS {
            lane(
                layout.instr_start + flag,
                row,
                fr(u64::from(instruction_flag(row, flag))),
            );
        }
        lane(layout.lookup_start + data.lookup, row, fr(1));
        lane(layout.raf_flag_idx, row, fr(u64::from(data.raf)));
    }

    let lane_point = point(committed_lane_vars(), 2);
    let row_point = point(log_rows, 4);
    let eq_row = EqPolynomial::<Fr>::evals(&row_point, None);

    // Partial evaluation over the row variables: one value per hot-value
    // index.
    let row_partials = |data: &[Fr]| -> Vec<Fr> {
        let indices = data.len() >> log_rows;
        (0..indices)
            .map(|index| {
                (0..rows)
                    .map(|t| eq_row[t] * data[(index << log_rows) | t])
                    .sum()
            })
            .collect()
    };

    let mut decoded = fr(0);

    // Register selector lanes: one-hot over the register alphabet.
    for (lane, start) in [
        (BytecodeRegisterLane::Rs1, layout.rs1_start),
        (BytecodeRegisterLane::Rs2, layout.rs2_start),
        (BytecodeRegisterLane::Rd, layout.rd_start),
    ] {
        let hot: Vec<usize> = row_data
            .iter()
            .map(|row| match lane {
                BytecodeRegisterLane::Rs1 => row.rs1,
                BytecodeRegisterLane::Rs2 => row.rs2,
                BytecodeRegisterLane::Rd => row.rd,
            })
            .collect();
        let partials = row_partials(&one_hot_evals(REGISTER_ADDRESS_BITS, log_rows, &hot));
        for (reg, partial) in partials.iter().enumerate() {
            decoded += lane_weight(&lane_point, start + reg) * *partial;
        }
        // The selector-block weight MLE at a Boolean register point is
        // exactly that register's lane weight — the derived and the
        // per-index weights agree.
        assert_eq!(
            selector_block_weight(
                &lane_point,
                start,
                &boolean_point_msb::<Fr>(REGISTER_ADDRESS_BITS, 3),
                register_count
            ),
            lane_weight(&lane_point, start + 3),
        );
    }

    // Direct 0/1 flag lanes.
    for flag in 0..NUM_CIRCUIT_FLAGS {
        let flags: Vec<Fr> = (0..rows)
            .map(|row| fr(u64::from(circuit_flag(row, flag))))
            .collect();
        decoded += lane_weight(&lane_point, layout.circuit_start + flag) * row_partials(&flags)[0];
    }
    for flag in 0..NUM_INSTRUCTION_FLAGS {
        let flags: Vec<Fr> = (0..rows)
            .map(|row| fr(u64::from(instruction_flag(row, flag))))
            .collect();
        decoded += lane_weight(&lane_point, layout.instr_start + flag) * row_partials(&flags)[0];
    }
    let raf_flags: Vec<Fr> = (0..rows)
        .map(|row| fr(u64::from(row_data[row].raf)))
        .collect();
    decoded += lane_weight(&lane_point, layout.raf_flag_idx) * row_partials(&raf_flags)[0];

    // Lookup selector lane: one-hot over the (padded) table alphabet.
    let lookup_hot: Vec<usize> = row_data.iter().map(|row| row.lookup).collect();
    let lookup_partials = row_partials(&one_hot_evals(table_count.log_2(), log_rows, &lookup_hot));
    for (sym, partial) in lookup_partials.iter().enumerate() {
        decoded += lane_weight(&lane_point, layout.lookup_start + sym) * *partial;
    }

    // Byte lanes: little-endian byte one-hot decode, weights from
    // `byte_decode_weight` at Boolean indices.
    let pc_partials = row_partials(&word_byte_evals(
        &row_data.iter().map(|row| row.unexp_pc).collect::<Vec<_>>(),
        log_rows,
    ));
    decoded += lane_weight(&lane_point, layout.unexp_pc_idx) * byte_decode_sum(&pc_partials, 3);

    // 16 canonical little-endian byte places per row (high places hold the
    // zero byte for u64-sized values): (byte ‖ place ‖ row) with 4 place
    // bits.
    let mut imm_bytes = vec![fr(0); 1 << (8 + 4 + log_rows)];
    for (row, r) in row_data.iter().enumerate() {
        for place in 0..imm_byte_width {
            let byte = if place < 8 {
                ((r.imm >> (8 * place)) & 0xff) as usize
            } else {
                0
            };
            imm_bytes[(((byte << 4) | place) << log_rows) | row] = fr(1);
        }
    }
    let imm_partials = row_partials(&imm_bytes);
    decoded += lane_weight(&lane_point, layout.imm_idx) * byte_decode_sum(&imm_partials, 4);

    let mut full_point = lane_point.clone();
    full_point.extend_from_slice(&row_point);
    assert_eq!(
        eval_mle(&chunk_data, &full_point),
        decoded,
        "lane decode weights must rebuild the committed chunk evaluation"
    );
}

/// The advice word decode identity behind the advice reconstruction
/// relations, and the byte-validity hamming leg, over concrete words —
/// including a zero "padding" word (one-hot of zero, so the per-place hamming
/// sum stays 1).
#[test]
fn advice_bytes_reconstruct_words_and_keep_hamming_one() {
    let word_vars = 2;
    let words = [0x0123_4567_89ab_cdef, 42, 0, u64::MAX];
    let bytes = word_byte_evals(&words, word_vars);
    let word_evals: Vec<Fr> = words.iter().map(|&w| fr(w)).collect();

    let r_word = point(word_vars, 3);
    let eq_word = EqPolynomial::<Fr>::evals(&r_word, None);

    // Partial evaluation over the word variables: one value per (byte ‖ place).
    let partials: Vec<Fr> = (0..1 << 11)
        .map(|index| {
            (0..words.len())
                .map(|w| eq_word[w] * bytes[(index << word_vars) | w])
                .sum()
        })
        .collect();

    let decoded = byte_decode_sum(&partials, 3);
    assert_eq!(decoded, eval_mle(&word_evals, &r_word));

    // Hamming at a random (place ‖ word) point: Σ_byte B = 1.
    let r_place = point(3, 6);
    let eq_place = EqPolynomial::<Fr>::evals(&r_place, None);
    let hamming: Fr = (0..256)
        .map(|byte| {
            (0..8)
                .map(|place| eq_place[place] * partials[(byte << 3) | place])
                .sum::<Fr>()
        })
        .sum();
    assert_eq!(hamming, fr(1));
}
