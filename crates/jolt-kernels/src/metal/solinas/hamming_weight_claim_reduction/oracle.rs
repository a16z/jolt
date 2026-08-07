//! Scalar correctness oracle with no Metal dependency.

use jolt_field::Field;

use super::model::HammingWeightCensus;
use super::*;

const PACKED_PC_MASK: u64 = (1 << 56) - 1;
pub const HAMMING_WEIGHT_UNFACTORED_ORACLE_MAX_ROWS: usize = 1 << 16;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HammingWeightOracleOutput<F> {
    pub masses: Vec<F>,
    pub audit_rows: Vec<HammingWeightAuditRow>,
    pub census: HammingWeightCensus,
}

/// Direct definition of the recentered pushforwards. This fixture evaluates
/// `eq(r_cycle, row)` from all cycle coordinates for every row; it does not
/// construct or consume the shader's `E_out * E_in` factorization.
pub fn unfactored_recentered_pushforwards<F: Field>(
    rows: &[HammingWeightResidentRow],
    reference_cycle: &[F],
    shape: HammingWeightShape,
) -> Result<Vec<F>, HammingWeightSuccessorError> {
    check_length("resident rows", rows.len(), shape.rows())?;
    check_length(
        "cycle reference variables",
        reference_cycle.len(),
        shape.rows().ilog2() as usize,
    )?;
    if rows.len() > HAMMING_WEIGHT_UNFACTORED_ORACLE_MAX_ROWS {
        return Err(HammingWeightSuccessorError::OracleFixtureTooLarge {
            rows: rows.len(),
            maximum: HAMMING_WEIGHT_UNFACTORED_ORACLE_MAX_ROWS,
        });
    }

    let mut masses = vec![F::zero(); HAMMING_WEIGHT_SELECTORS * HAMMING_WEIGHT_BINS];
    for (row_index, row) in rows.iter().copied().enumerate() {
        let weight = unfactored_eq_at(reference_cycle, row_index);
        for selector in 0..HAMMING_WEIGHT_SELECTORS {
            let Some(hot) = hot_index(row, selector) else {
                continue;
            };
            if hot != 0 {
                masses[selector * HAMMING_WEIGHT_BINS + hot] += weight;
            }
        }
    }
    Ok(masses)
}

pub fn recentered_pushforwards<F: Field>(
    rows: &[HammingWeightResidentRow],
    e_in: &[F],
    e_out: &[F],
    shape: HammingWeightShape,
) -> Result<HammingWeightOracleOutput<F>, HammingWeightSuccessorError> {
    check_length("resident rows", rows.len(), shape.rows())?;
    check_length("inner equality", e_in.len(), shape.inner_length())?;
    check_length("outer equality", e_out.len(), shape.outer_length())?;

    let mut masses = vec![F::zero(); HAMMING_WEIGHT_SELECTORS * HAMMING_WEIGHT_BINS];
    let mut audit_rows = Vec::with_capacity(shape.outer_length());

    for (outer, outer_weight) in e_out.iter().copied().enumerate() {
        let mut local =
            vec![F::zero(); HAMMING_WEIGHT_SELECTORS * HAMMING_WEIGHT_BINS].into_boxed_slice();
        let mut pc_present = 0u32;
        let mut ram_present = 0u32;
        let mut retained_nonzero_contributions = 0u32;
        let mut occupied_outer_bins = 0u32;
        let row_start = outer * shape.inner_length();
        for inner in 0..shape.inner_length() {
            let row = rows[row_start + inner];
            let words = row.words();
            pc_present += u32::from((words[4] & PACKED_PC_MASK) != 0);
            ram_present += u32::from(words[2] != 0);
            for selector in 0..HAMMING_WEIGHT_SELECTORS {
                if let Some(hot) = hot_index(row, selector) {
                    if hot == 0 {
                        continue;
                    }
                    local[selector * HAMMING_WEIGHT_BINS + hot] += e_in[inner];
                    retained_nonzero_contributions += 1;
                }
            }
        }
        for selector in 0..HAMMING_WEIGHT_SELECTORS {
            for hot in 1..HAMMING_WEIGHT_BINS {
                let index = selector * HAMMING_WEIGHT_BINS + hot;
                if !local[index].is_zero() {
                    masses[index] += outer_weight * local[index];
                    occupied_outer_bins += 1;
                }
            }
        }
        audit_rows.push(HammingWeightAuditRow {
            rows_seen: shape.inner_length() as u32,
            pc_present,
            ram_present,
            retained_nonzero_contributions,
            occupied_outer_bins,
            reserved: [0; 3],
        });
    }

    let census =
        HammingWeightCensus::from_audit_rows(&audit_rows, HammingWeightStatus::default(), shape)?;

    Ok(HammingWeightOracleOutput {
        masses,
        audit_rows,
        census,
    })
}

pub fn hot_index(row: HammingWeightResidentRow, selector: usize) -> Option<usize> {
    let words = row.words();
    match selector {
        0..=7 => Some(((words[1] >> (8 * (7 - selector))) & 0xff) as usize),
        8..=15 => Some(((words[0] >> (8 * (15 - selector))) & 0xff) as usize),
        16..=17 => {
            let plus_one = words[4] & PACKED_PC_MASK;
            (plus_one != 0).then(|| (((plus_one - 1) >> (8 * (17 - selector))) & 0xff) as usize)
        }
        18..=19 => {
            (words[2] != 0).then(|| (((words[2] - 1) >> (8 * (19 - selector))) & 0xff) as usize)
        }
        20..=27 => {
            let (biased, _) = biased_inc(words);
            let standard = (biased >> (8 * (selector - 20))) & 0xff;
            Some(((standard + 128) & 0xff) as usize)
        }
        28 => {
            let (_, carry) = biased_inc(words);
            Some(carry.rem_euclid(HAMMING_WEIGHT_BINS as i32) as usize)
        }
        _ => None,
    }
}

fn unfactored_eq_at<F: Field>(point: &[F], row: usize) -> F {
    point
        .iter()
        .enumerate()
        .fold(F::one(), |weight, (coordinate, value)| {
            let shift = point.len() - coordinate - 1;
            if row & (1usize << shift) == 0 {
                weight * (F::one() - *value)
            } else {
                weight * *value
            }
        })
}

fn biased_inc(words: [u64; 5]) -> (u64, i32) {
    let magnitude = words[3];
    if words[4] >> 63 != 0 {
        (
            HAMMING_WEIGHT_BALANCED_INC_BIAS.wrapping_sub(magnitude),
            -i32::from(magnitude > HAMMING_WEIGHT_BALANCED_INC_BIAS),
        )
    } else {
        let biased = HAMMING_WEIGHT_BALANCED_INC_BIAS.wrapping_add(magnitude);
        (biased, i32::from(biased < HAMMING_WEIGHT_BALANCED_INC_BIAS))
    }
}

fn check_length(
    name: &'static str,
    got: usize,
    expected: usize,
) -> Result<(), HammingWeightSuccessorError> {
    if got == expected {
        Ok(())
    } else {
        Err(HammingWeightSuccessorError::StorageLength {
            name,
            expected,
            got,
        })
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "oracle fixtures use checked shapes")]
mod tests {
    use jolt_field::AkitaField;

    use super::*;

    fn fixture_shape() -> HammingWeightShape {
        HammingWeightShape::new(
            1 << 15,
            HammingWeightSuccessorConfig {
                inner_log2: 9,
                stage_rows: HAMMING_WEIGHT_STAGE_ROWS,
                threads_per_threadgroup: HAMMING_WEIGHT_THREADS,
                trace_cutoff: 1 << 15,
            },
        )
        .unwrap()
    }

    fn eq_table(point: &[AkitaField]) -> Vec<AkitaField> {
        (0..1usize << point.len())
            .map(|index| unfactored_eq_at(point, index))
            .collect()
    }

    fn fixture_rows(rows: usize) -> Vec<HammingWeightResidentRow> {
        (0..rows)
            .map(|index| {
                let lookup_lo = (index as u64).wrapping_mul(0x0102_0304_0506_0708);
                let lookup_hi = (!(index as u64)).rotate_left(17);
                let ram = if index % 3 != 0 {
                    (index & 0xffff) as u64 + 1
                } else {
                    0
                };
                let magnitude = (index as u64).wrapping_mul(0x1_0001);
                let pc = if index % 5 != 0 {
                    ((index * 7) & 0xffff) as u64 + 1
                } else {
                    0
                };
                let sign = u64::from(index % 7 == 0) << 63;
                HammingWeightResidentRow::from_words([
                    lookup_lo,
                    lookup_hi,
                    ram,
                    magnitude,
                    pc | sign,
                ])
            })
            .collect()
    }

    #[test]
    fn split_topology_matches_unfactored_cycle_eq() {
        let shape = fixture_shape();
        let rows = fixture_rows(shape.rows());
        let point = (0..shape.rows().ilog2())
            .map(|index| AkitaField::from_u64(3 + u64::from(index)))
            .collect::<Vec<_>>();
        let split = point.len() - shape.inner_log2();
        let e_out = eq_table(&point[..split]);
        let e_in = eq_table(&point[split..]);

        let direct = unfactored_recentered_pushforwards(&rows, &point, shape).unwrap();
        let factored = recentered_pushforwards(&rows, &e_in, &e_out, shape).unwrap();
        assert_eq!(direct, factored.masses);
        assert_eq!(factored.audit_rows.len(), shape.outer_length());
        assert_eq!(factored.census.rows, shape.rows() as u64);
        assert!(direct
            .chunks_exact(HAMMING_WEIGHT_BINS)
            .all(|selector| selector[0].is_zero()));
    }

    #[test]
    fn selector_decoder_covers_optional_rows_and_signed_carry() {
        let negative = HammingWeightResidentRow::from_words([
            0x0807_0605_0403_0201,
            0x100f_0e0d_0c0b_0a09,
            0,
            HAMMING_WEIGHT_BALANCED_INC_BIAS + 1,
            1 << 63,
        ]);
        assert_eq!(hot_index(negative, 0), Some(16));
        assert_eq!(hot_index(negative, 7), Some(9));
        assert_eq!(hot_index(negative, 8), Some(8));
        assert_eq!(hot_index(negative, 15), Some(1));
        assert_eq!(hot_index(negative, 16), None);
        assert_eq!(hot_index(negative, 18), None);
        assert_eq!(hot_index(negative, 28), Some(255));

        let positive = HammingWeightResidentRow::from_words([
            0,
            0,
            0x1235,
            u64::MAX - HAMMING_WEIGHT_BALANCED_INC_BIAS + 1,
            0x4322,
        ]);
        assert_eq!(hot_index(positive, 16), Some(0x43));
        assert_eq!(hot_index(positive, 17), Some(0x21));
        assert_eq!(hot_index(positive, 18), Some(0x12));
        assert_eq!(hot_index(positive, 19), Some(0x34));
        assert_eq!(hot_index(positive, 28), Some(1));
        assert_eq!(hot_index(positive, HAMMING_WEIGHT_SELECTORS), None);
    }

    #[test]
    fn unfactored_fixture_is_bounded() {
        let shape = HammingWeightShape::new(
            1 << 17,
            HammingWeightSuccessorConfig {
                inner_log2: 11,
                stage_rows: HAMMING_WEIGHT_STAGE_ROWS,
                threads_per_threadgroup: HAMMING_WEIGHT_THREADS,
                trace_cutoff: 1 << 17,
            },
        )
        .unwrap();
        let rows = vec![HammingWeightResidentRow::default(); shape.rows()];
        let point = vec![AkitaField::from_u64(2); 17];
        assert!(matches!(
            unfactored_recentered_pushforwards(&rows, &point, shape),
            Err(HammingWeightSuccessorError::OracleFixtureTooLarge {
                rows: 131_072,
                maximum: HAMMING_WEIGHT_UNFACTORED_ORACLE_MAX_ROWS,
            })
        ));
    }
}
