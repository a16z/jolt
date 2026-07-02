//! Semantic integration tests for the lattice module: build concrete one-hot
//! witness data, pack it, and check the *identities* the relations and views
//! claim — sizes, slot placement, the fused-inc chunk reconstruction, and the
//! decode views — against real multilinear evaluations.

use jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode::{
    committed_lane_vars, committed_lanes, BYTECODE_LANE_LAYOUT,
};
use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
use jolt_claims::protocols::jolt::geometry::ra::JoltRaPolynomialLayout;
use jolt_claims::protocols::jolt::lattice::views::{
    advice_word_decode_terms, bytecode_chunk_decode_terms, DecodeTerm,
};
use jolt_claims::protocols::jolt::lattice::{
    proof_packed_columns, LatticeColumn, ProofPackingShape, UnsignedIncChunking,
};
use jolt_claims::protocols::jolt::{JoltAdviceKind, JoltCommittedPolynomial};
use jolt_field::{Fr, FromPrimitiveInt, RingCore};
use jolt_lookup_tables::{LookupTableKind, XLEN};
use jolt_openings::PrefixPacking;
use jolt_poly::EqPolynomial;
use jolt_riscv::{NUM_CIRCUIT_FLAGS, NUM_INSTRUCTION_FLAGS};

fn fr(value: u64) -> Fr {
    Fr::from_u64(value)
}

/// Inner-product MLE evaluation: `Σ_i eq(point, bits(i)) · evals[i]`, with
/// index bits msb-first relative to the point (the packing convention).
fn eval_mle(evals: &[Fr], point: &[Fr]) -> Fr {
    let table = EqPolynomial::<Fr>::evals(point, None);
    assert_eq!(
        table.len(),
        evals.len(),
        "point arity must match table size"
    );
    table.iter().zip(evals).map(|(w, v)| *w * *v).sum()
}

/// A deterministic, non-boolean evaluation point (distinct small primes).
fn point(len: usize, seed: u64) -> Vec<Fr> {
    const PRIMES: [u64; 16] = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59];
    (0..len)
        .map(|i| fr(PRIMES[(i + seed as usize) % PRIMES.len()] + seed))
        .collect()
}

/// One-hot column over `(symbol ‖ row)`: cell `(sym << log_rows) | row`.
fn one_hot_column(log_symbols: usize, log_rows: usize, hot: &[usize]) -> Vec<Fr> {
    assert_eq!(hot.len(), 1 << log_rows);
    let mut data = vec![fr(0); 1 << (log_symbols + log_rows)];
    for (row, &sym) in hot.iter().enumerate() {
        assert!(sym < (1 << log_symbols));
        data[(sym << log_rows) | row] = fr(1);
    }
    data
}

/// Byte one-hot column over `(symbol ‖ limb ‖ word)` for 8-byte words:
/// cell `((byte << 3) | limb) << word_vars | word`.
fn word_byte_column(words: &[u64], word_vars: usize) -> Vec<Fr> {
    assert_eq!(words.len(), 1 << word_vars);
    let mut data = vec![fr(0); 1 << (8 + 3 + word_vars)];
    for (w, &word) in words.iter().enumerate() {
        for limb in 0..8 {
            let byte = ((word >> (8 * limb)) & 0xff) as usize;
            data[(((byte << 3) | limb) << word_vars) | w] = fr(1);
        }
    }
    data
}

/// Every logical column, packed at its assigned slot, must satisfy
/// `P_i(x) = W(prefix_i ‖ x)` — this pins the slot placement, the cell-order
/// convention, and the declared arities end to end.
#[test]
#[expect(clippy::unwrap_used)]
fn packed_witness_slots_reproduce_every_logical_column() {
    let log_t = 3;
    let log_k_chunk = 4;
    let word_vars = 2;
    let shape = ProofPackingShape {
        ra_layout: JoltRaPolynomialLayout::new(1, 1, 1).unwrap(),
        log_t,
        log_k_chunk,
        untrusted_advice_word_vars: Some(word_vars),
    };
    let columns = proof_packed_columns(&shape).unwrap();
    let packing = PrefixPacking::new(columns.clone()).unwrap();

    // Size accounting: 3 Ra columns + 16 chunk columns at 2^7 cells, the msb
    // at 2^3, the advice bytes at 2^13; 2432 + 8 + 8192 = 10632 cells round
    // up to a 2^14 packed hypercube.
    let chunk_count = UnsignedIncChunking::new(log_k_chunk).unwrap().chunk_count();
    assert_eq!(chunk_count, 16);
    let total_cells: usize = columns.iter().map(|(_, vars)| 1usize << vars).sum();
    assert_eq!(total_cells, 19 * (1 << 7) + (1 << 3) + (1 << 13));
    assert_eq!(packing.packed_num_vars, 14);
    for (column, vars) in &columns {
        assert_eq!(packing[column].prefix.len(), 14 - vars);
    }

    // Concrete witness data per column.
    let mut column_data: Vec<(LatticeColumn, Vec<Fr>)> = Vec::new();
    for (index, (column, vars)) in columns.iter().enumerate() {
        let data = match column {
            LatticeColumn::Committed(JoltCommittedPolynomial::UnsignedIncMsb) => {
                (0..1 << log_t).map(|t| fr((t % 2) as u64)).collect()
            }
            LatticeColumn::Committed(JoltCommittedPolynomial::UntrustedAdviceBytes) => {
                word_byte_column(&[0x0123_4567_89ab_cdef, 42, 0, u64::MAX], word_vars)
            }
            _ => {
                let hot: Vec<usize> = (0..1 << log_t)
                    .map(|t| (7 * t + 3 * index + 1) % (1 << log_k_chunk))
                    .collect();
                one_hot_column(log_k_chunk, log_t, &hot)
            }
        };
        assert_eq!(data.len(), 1 << vars, "declared arity must match the data");
        column_data.push((*column, data));
    }

    // Assemble the packed witness exactly as PrefixPacking's slot layout
    // dictates: cell `local` of a slot lands at `(prefix_value << vars) | local`.
    let mut packed = vec![fr(0); 1 << packing.packed_num_vars];
    for (column, data) in &column_data {
        let slot = &packing[column];
        let prefix_value = slot
            .prefix
            .iter()
            .fold(0usize, |acc, bit| (acc << 1) | usize::from(*bit));
        let base = prefix_value << slot.num_vars;
        packed[base..base + data.len()].copy_from_slice(data);
    }

    for (index, (column, data)) in column_data.iter().enumerate() {
        let slot = &packing[column];
        let x = point(slot.num_vars, index as u64);
        let mut packed_point: Vec<Fr> = slot.prefix.iter().map(|bit| fr(u64::from(*bit))).collect();
        packed_point.extend_from_slice(&x);
        assert_eq!(
            eval_mle(data, &x),
            eval_mle(&packed, &packed_point),
            "column {column:?} must be readable through its packed slot"
        );
    }
}

/// The inc-chain identities over a concrete trace of signed increments:
/// per-row hamming weight, and the reconstruction
/// `Σ_j place_j · Σ_a id(a) · chunk_j(a, r_cycle) = FusedInc + 2^64·(1 − msb)`
/// — including the padding subtlety that a zero increment encodes as msb = 1
/// with all chunks hot at symbol 0 (an all-zero or msb-cold padding row would
/// falsify both legs).
#[test]
#[expect(clippy::unwrap_used)]
fn chunk_decomposition_reconstructs_signed_increments() {
    let log_t = 3;
    let chunking = UnsignedIncChunking::new(8).unwrap();
    let count = chunking.chunk_count();
    assert_eq!(count, 8);

    // A trace mixing positive, negative, extreme, and zero ("padding")
    // increments.
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

    let mut chunk_hot = vec![vec![0usize; values.len()]; count];
    let mut msb_data = Vec::with_capacity(values.len());
    let mut fused_data = Vec::with_capacity(values.len());
    for (t, &value) in values.iter().enumerate() {
        let unsigned = (value + (1i128 << 64)) as u128;
        let msb = (unsigned >> 64) as u64;
        let lower = (unsigned & ((1u128 << 64) - 1)) as u64;
        for (j, hot) in chunk_hot.iter_mut().enumerate() {
            hot[t] = ((lower >> (8 * j)) & 0xff) as usize;
        }
        if value == 0 {
            // The padding encoding: one-hot of zero with a HOT msb (0 + 2^64
            // has bit 64 set). Witness generation must not zero these rows.
            assert_eq!(msb, 1);
            assert!(chunk_hot.iter().all(|hot| hot[t] == 0));
        }
        msb_data.push(fr(msb));
        fused_data.push(Fr::from_i128(value));
    }
    let chunk_columns: Vec<Vec<Fr>> = chunk_hot
        .iter()
        .map(|hot| one_hot_column(8, log_t, hot))
        .collect();

    let r_cycle = point(log_t, 1);
    let eq_cycle = EqPolynomial::<Fr>::evals(&r_cycle, None);

    // Partial evaluation of chunk j at (a, r_cycle) for every symbol a.
    let partial = |column: &[Fr]| -> Vec<Fr> {
        (0..256)
            .map(|a| {
                (0..values.len())
                    .map(|t| eq_cycle[t] * column[(a << log_t) | t])
                    .sum()
            })
            .collect::<Vec<Fr>>()
    };

    let mut reconstructed = fr(0);
    for (j, column) in chunk_columns.iter().enumerate() {
        let partials = partial(column);
        // Hamming: exactly one hot symbol per row extends to Σ_a chunk(a, r) = 1
        // at any cycle point.
        assert_eq!(partials.iter().copied().sum::<Fr>(), fr(1));
        // Identity decode of the chunk's symbol value.
        let decoded: Fr = partials
            .iter()
            .enumerate()
            .map(|(a, value)| fr(a as u64) * *value)
            .sum();
        reconstructed += chunking.place_value::<Fr>(j) * decoded;
    }

    let fused = eval_mle(&fused_data, &r_cycle);
    let msb = eval_mle(&msb_data, &r_cycle);
    assert_eq!(reconstructed, fused + Fr::pow2(64) * (fr(1) - msb));
}

/// The bytecode lane decode view: a committed `BytecodeChunk` polynomial
/// built row-by-row from the lane layout must equal the weighted sum of its
/// decomposed sub-columns at every `(lane_point ‖ row_point)` — pinning the
/// lane offsets, the byte decodes, the plain-0/1 flag encoding, and the
/// zeroing of capacity-padding lanes and padded lookup symbols.
#[test]
#[expect(clippy::unwrap_used, clippy::panic)]
fn bytecode_lane_decode_matches_committed_chunk() {
    let log_rows = 2;
    let rows = 1 << log_rows;
    let chunk = 0;
    let imm_byte_width = 16;
    let table_count = LookupTableKind::<XLEN>::COUNT;
    let register_count = 1usize << REGISTER_ADDRESS_BITS;
    let layout = BYTECODE_LANE_LAYOUT;

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
    let mut chunk_data = vec![fr(0); committed_lanes() << log_rows];
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

    // The decomposed sub-columns, keyed the way the decode terms address
    // them: cell index over the (symbol ‖ limb) prefix, row as suffix.
    let sub_column = |column: &LatticeColumn| -> Vec<Fr> {
        match *column {
            LatticeColumn::BytecodeRegisterSelector { lane, .. } => {
                let hot: Vec<usize> = row_data
                    .iter()
                    .map(|row| {
                        use jolt_claims::protocols::jolt::lattice::BytecodeRegisterLane as L;
                        match lane {
                            L::Rs1 => row.rs1,
                            L::Rs2 => row.rs2,
                            L::Rd => row.rd,
                        }
                    })
                    .collect();
                one_hot_column(REGISTER_ADDRESS_BITS, log_rows, &hot)
            }
            LatticeColumn::BytecodeCircuitFlag { flag, .. } => (0..rows)
                .map(|row| fr(u64::from(circuit_flag(row, flag))))
                .collect(),
            LatticeColumn::BytecodeInstructionFlag { flag, .. } => (0..rows)
                .map(|row| fr(u64::from(instruction_flag(row, flag))))
                .collect(),
            LatticeColumn::BytecodeLookupSelector { .. } => {
                let symbol_bits = table_count.next_power_of_two().trailing_zeros() as usize;
                let hot: Vec<usize> = row_data.iter().map(|row| row.lookup).collect();
                one_hot_column(symbol_bits, log_rows, &hot)
            }
            LatticeColumn::BytecodeRafFlag { .. } => (0..rows)
                .map(|row| fr(u64::from(row_data[row].raf)))
                .collect(),
            LatticeColumn::BytecodeUnexpandedPcBytes { .. } => word_byte_column(
                &row_data.iter().map(|row| row.unexp_pc).collect::<Vec<_>>(),
                log_rows,
            ),
            LatticeColumn::BytecodeImmBytes { .. } => {
                // 16 canonical little-endian byte limbs per row (high limbs
                // zero for u64-sized values): (byte ‖ limb ‖ row) with a
                // 4-bit limb index.
                let mut data = vec![fr(0); 1 << (8 + 4 + log_rows)];
                for (row, r) in row_data.iter().enumerate() {
                    for limb in 0..imm_byte_width {
                        let byte = if limb < 8 {
                            ((r.imm >> (8 * limb)) & 0xff) as usize
                        } else {
                            0
                        };
                        data[(((byte << 4) | limb) << log_rows) | row] = fr(1);
                    }
                }
                data
            }
            _ => panic!("unexpected column in bytecode decode terms: {column:?}"),
        }
    };

    let lane_point = point(committed_lane_vars(), 2);
    let row_point = point(log_rows, 4);
    let eq_row = EqPolynomial::<Fr>::evals(&row_point, None);

    let terms: Vec<DecodeTerm<Fr>> =
        bytecode_chunk_decode_terms(chunk, &lane_point, imm_byte_width).unwrap();

    // Materialize each referenced sub-column once, partially evaluated over
    // the row variables.
    let mut column_partials: Vec<(LatticeColumn, Vec<Fr>)> = Vec::new();
    for term in &terms {
        if !column_partials.iter().any(|(c, _)| c == &term.column) {
            let data = sub_column(&term.column);
            let cells = data.len() >> log_rows;
            let partials: Vec<Fr> = (0..cells)
                .map(|cell| {
                    (0..rows)
                        .map(|t| eq_row[t] * data[(cell << log_rows) | t])
                        .sum()
                })
                .collect();
            column_partials.push((term.column, partials));
        }
    }
    let decoded: Fr = terms
        .iter()
        .map(|term| {
            let (_, partials) = column_partials
                .iter()
                .find(|(c, _)| c == &term.column)
                .unwrap();
            term.weight * partials[term.cell]
        })
        .sum();

    let mut full_point = lane_point.clone();
    full_point.extend_from_slice(&row_point);
    assert_eq!(
        eval_mle(&chunk_data, &full_point),
        decoded,
        "lane decode terms must reconstruct the committed chunk evaluation"
    );
}

/// The advice word decode view and the byte-validity hamming leg over
/// concrete words, including a zero "padding" word (one-hot of zero, so the
/// per-position hamming sum stays 1).
#[test]
fn advice_byte_view_decodes_words_and_keeps_hamming_one() {
    let word_vars = 2;
    let words = [0x0123_4567_89ab_cdef, 42, 0, u64::MAX];
    let column = word_byte_column(&words, word_vars);
    let word_evals: Vec<Fr> = words.iter().map(|&w| fr(w)).collect();

    let r_word = point(word_vars, 3);
    let eq_word = EqPolynomial::<Fr>::evals(&r_word, None);

    // Partial evaluation over the word variables: one value per (symbol ‖ limb).
    let partials: Vec<Fr> = (0..1 << 11)
        .map(|cell| {
            (0..words.len())
                .map(|w| eq_word[w] * column[(cell << word_vars) | w])
                .sum()
        })
        .collect();

    let decoded: Fr = advice_word_decode_terms::<Fr>(JoltAdviceKind::Untrusted)
        .iter()
        .map(|term| term.weight * partials[term.cell])
        .sum();
    assert_eq!(decoded, eval_mle(&word_evals, &r_word));

    // Hamming at a random (limb ‖ word) point: Σ_sym cell = 1.
    let r_limb = point(3, 6);
    let eq_limb = EqPolynomial::<Fr>::evals(&r_limb, None);
    let hamming: Fr = (0..256)
        .map(|sym| {
            (0..8)
                .map(|limb| eq_limb[limb] * partials[(sym << 3) | limb])
                .sum::<Fr>()
        })
        .sum();
    assert_eq!(hamming, fr(1));
}
