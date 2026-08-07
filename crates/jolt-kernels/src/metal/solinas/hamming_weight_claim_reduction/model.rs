//! Analytical work, traffic, and acceptance model.

use super::*;

/// Retained M4 Max streaming-copy control, not a measurement of this shader.
pub const M4_MAX_COPY_BYTES_PER_SECOND: u64 = 451_701_710_520;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HammingWeightCensus {
    pub rows: u64,
    pub pc_present: u64,
    pub ram_present: u64,
    pub retained_nonzero_contributions: u64,
    pub occupied_outer_bins: u64,
}

impl HammingWeightCensus {
    pub fn from_audit_rows(
        audit_rows: &[HammingWeightAuditRow],
        status: HammingWeightStatus,
        shape: HammingWeightShape,
    ) -> Result<Self, HammingWeightSuccessorError> {
        if audit_rows.len() != shape.outer_length() {
            return Err(HammingWeightSuccessorError::StorageLength {
                name: "audit rows",
                expected: shape.outer_length(),
                got: audit_rows.len(),
            });
        }
        check_zero(
            "unsupported dispatches",
            status.unsupported_dispatches as u64,
        )?;
        check_reserved("status reserved", &status.reserved)?;
        for audit in audit_rows {
            check_exact(
                "rows per audit shard",
                shape.inner_length() as u64,
                audit.rows_seen as u64,
            )?;
            check_at_most(
                "PC-present rows per shard",
                audit.pc_present as u64,
                shape.inner_length() as u64,
            )?;
            check_at_most(
                "RAM-present rows per shard",
                audit.ram_present as u64,
                shape.inner_length() as u64,
            )?;
            check_at_most(
                "retained contributions per shard",
                audit.retained_nonzero_contributions as u64,
                HAMMING_WEIGHT_SELECTORS as u64 * shape.inner_length() as u64,
            )?;
            check_at_most(
                "occupied bins per shard",
                audit.occupied_outer_bins as u64,
                HAMMING_WEIGHT_SELECTORS as u64 * HAMMING_WEIGHT_RETAINED_BINS as u64,
            )?;
            check_reserved("audit reserved", &audit.reserved)?;
        }
        let census = Self {
            rows: checked_sum_u32(audit_rows.iter().map(|audit| audit.rows_seen))?,
            pc_present: checked_sum_u32(audit_rows.iter().map(|audit| audit.pc_present))?,
            ram_present: checked_sum_u32(audit_rows.iter().map(|audit| audit.ram_present))?,
            retained_nonzero_contributions: checked_sum_u32(
                audit_rows
                    .iter()
                    .map(|audit| audit.retained_nonzero_contributions),
            )?,
            occupied_outer_bins: checked_sum_u32(
                audit_rows.iter().map(|audit| audit.occupied_outer_bins),
            )?,
        };
        census.validate(shape)?;
        Ok(census)
    }

    pub fn validate(self, shape: HammingWeightShape) -> Result<(), HammingWeightSuccessorError> {
        let rows = shape.rows() as u64;
        check_at_most("rows", self.rows, rows)?;
        if self.rows != rows {
            return Err(HammingWeightSuccessorError::AuditMismatch {
                name: "rows",
                expected: rows,
                got: self.rows,
            });
        }
        check_at_most("PC-present rows", self.pc_present, rows)?;
        check_at_most("RAM-present rows", self.ram_present, rows)?;
        check_at_most(
            "retained nonzero contributions",
            self.retained_nonzero_contributions,
            raw_contribution_upper_bound(rows),
        )?;
        check_at_most(
            "occupied outer bins",
            self.occupied_outer_bins,
            outer_product_upper_bound(shape),
        )?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HammingWeightWorkModel {
    pub raw_contributions: u64,
    pub retained_histogram_adds: u64,
    pub finalize_adds: u64,
    pub device_outer_products: u64,
    pub device_outer_product_upper_bound: u64,
    pub host_split_eq_products: u64,
    pub threadgroup_hot_bytes: u64,
    pub threadgroup_weight_write_bytes: u64,
    pub threadgroup_weight_read_bytes: u64,
    pub threadgroup_audit_bytes: u64,
    pub threadgroup_logical_bytes: u64,
}

impl HammingWeightWorkModel {
    pub fn new(
        shape: HammingWeightShape,
        census: HammingWeightCensus,
    ) -> Result<Self, HammingWeightSuccessorError> {
        census.validate(shape)?;
        let pc_contributions = 2u64
            .checked_mul(census.pc_present)
            .ok_or(HammingWeightSuccessorError::Overflow)?;
        let ram_contributions = 2u64
            .checked_mul(census.ram_present)
            .ok_or(HammingWeightSuccessorError::Overflow)?;
        let raw_contributions = 25u64
            .checked_mul(census.rows)
            .and_then(|value| value.checked_add(pc_contributions))
            .and_then(|value| value.checked_add(ram_contributions))
            .ok_or(HammingWeightSuccessorError::Overflow)?;
        let finalize_adds = outer_product_upper_bound(shape);
        let host_split_eq_products = (shape.inner_length() as u64 - 1)
            .checked_add(shape.outer_length() as u64 - 1)
            .ok_or(HammingWeightSuccessorError::Overflow)?;
        let threadgroup_hot_bytes = 2u64
            .checked_mul(HAMMING_WEIGHT_SELECTORS as u64)
            .and_then(|value| value.checked_mul(census.rows))
            .ok_or(HammingWeightSuccessorError::Overflow)?;
        let threadgroup_weight_write_bytes = (HAMMING_WEIGHT_FIELD_BYTES as u64)
            .checked_mul(census.rows)
            .ok_or(HammingWeightSuccessorError::Overflow)?;
        let threadgroup_weight_read_bytes = (HAMMING_WEIGHT_FIELD_BYTES as u64)
            .checked_mul(census.retained_nonzero_contributions)
            .ok_or(HammingWeightSuccessorError::Overflow)?;
        let stages = census.rows / HAMMING_WEIGHT_STAGE_ROWS as u64;
        let loader_audit_bytes = 2u64
            .checked_mul(HAMMING_WEIGHT_STAGE_AUDIT_BYTES as u64)
            .and_then(|value| value.checked_mul(stages))
            .ok_or(HammingWeightSuccessorError::Overflow)?;
        let occupied_audit_bytes = 2u64
            .checked_mul(HAMMING_WEIGHT_SELECTORS as u64)
            .and_then(|value| value.checked_mul(std::mem::size_of::<u32>() as u64))
            .and_then(|value| value.checked_mul(shape.outer_length() as u64))
            .ok_or(HammingWeightSuccessorError::Overflow)?;
        let threadgroup_audit_bytes = loader_audit_bytes
            .checked_add(occupied_audit_bytes)
            .ok_or(HammingWeightSuccessorError::Overflow)?;
        let threadgroup_logical_bytes = threadgroup_hot_bytes
            .checked_add(threadgroup_weight_write_bytes)
            .and_then(|value| value.checked_add(threadgroup_weight_read_bytes))
            .and_then(|value| value.checked_add(threadgroup_audit_bytes))
            .ok_or(HammingWeightSuccessorError::Overflow)?;
        Ok(Self {
            raw_contributions,
            retained_histogram_adds: census.retained_nonzero_contributions,
            finalize_adds,
            device_outer_products: census.occupied_outer_bins,
            device_outer_product_upper_bound: outer_product_upper_bound(shape),
            host_split_eq_products,
            threadgroup_hot_bytes,
            threadgroup_weight_write_bytes,
            threadgroup_weight_read_bytes,
            threadgroup_audit_bytes,
            threadgroup_logical_bytes,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HammingWeightTrafficModel {
    pub borrowed_row_allocation_bytes: u64,
    pub sequence_owned_allocation_bytes: u64,
    pub partial_bytes: u64,
    pub audit_bytes: u64,
    pub status_bytes: u64,
    pub output_readback_bytes: u64,
    pub shader_issued_bytes: u64,
    pub compulsory_bytes: u64,
    pub optimistic_offchip_bytes: u64,
    pub issued_copy_floor_ns: u64,
    pub issued_eighty_percent_copy_cap_ns: u64,
    pub optimistic_offchip_eighty_percent_cap_ns: u64,
}

impl HammingWeightTrafficModel {
    pub fn new(shape: HammingWeightShape) -> Result<Self, HammingWeightSuccessorError> {
        let rows = shape.rows() as u64;
        let inner = shape.inner_length() as u64;
        let outer = shape.outer_length() as u64;
        let selectors = HAMMING_WEIGHT_SELECTORS as u64;
        let bins = HAMMING_WEIGHT_BINS as u64;
        let retained_bins = HAMMING_WEIGHT_RETAINED_BINS as u64;
        let field_bytes = HAMMING_WEIGHT_FIELD_BYTES as u64;

        let row_bytes = checked_product(&[HAMMING_WEIGHT_ROW_BYTES as u64, rows])?;
        let inner_weight_bytes = checked_product(&[field_bytes, inner])?;
        let logical_inner_weight_reads = checked_product(&[field_bytes, rows])?;
        let outer_weight_bytes = checked_product(&[field_bytes, outer])?;
        let logical_outer_weight_reads = checked_product(&[field_bytes, selectors, outer])?;
        let partial_bytes = checked_product(&[field_bytes, selectors, retained_bins, outer])?;
        let output_bytes = checked_product(&[field_bytes, selectors, bins])?;
        let audit_bytes =
            checked_product(&[std::mem::size_of::<HammingWeightAuditRow>() as u64, outer])?;
        let status_bytes = std::mem::size_of::<HammingWeightStatus>() as u64;
        let output_readback_bytes = output_bytes
            .checked_add(audit_bytes)
            .and_then(|value| value.checked_add(status_bytes))
            .ok_or(HammingWeightSuccessorError::Overflow)?;
        let partial_handoff_bytes = partial_bytes
            .checked_mul(2)
            .ok_or(HammingWeightSuccessorError::Overflow)?;
        let output_handoff_bytes = output_bytes
            .checked_mul(2)
            .ok_or(HammingWeightSuccessorError::Overflow)?;
        let audit_handoff_bytes = audit_bytes
            .checked_mul(2)
            .ok_or(HammingWeightSuccessorError::Overflow)?;
        let status_handoff_bytes = status_bytes
            .checked_mul(2)
            .ok_or(HammingWeightSuccessorError::Overflow)?;

        let sequence_owned_allocation_bytes = inner_weight_bytes
            .checked_add(outer_weight_bytes)
            .and_then(|value| value.checked_add(partial_bytes))
            .and_then(|value| value.checked_add(output_bytes))
            .and_then(|value| value.checked_add(audit_bytes))
            .and_then(|value| value.checked_add(status_bytes))
            .ok_or(HammingWeightSuccessorError::Overflow)?;
        let shader_issued_bytes = row_bytes
            .checked_add(logical_inner_weight_reads)
            .and_then(|value| value.checked_add(logical_outer_weight_reads))
            .and_then(|value| value.checked_add(partial_handoff_bytes))
            .and_then(|value| value.checked_add(output_handoff_bytes))
            .and_then(|value| value.checked_add(audit_handoff_bytes))
            .and_then(|value| value.checked_add(status_handoff_bytes))
            .ok_or(HammingWeightSuccessorError::Overflow)?;
        let optimistic_offchip_bytes = row_bytes
            .checked_add(inner_weight_bytes)
            .and_then(|value| value.checked_add(outer_weight_bytes))
            .and_then(|value| value.checked_add(partial_handoff_bytes))
            .and_then(|value| value.checked_add(output_handoff_bytes))
            .and_then(|value| value.checked_add(audit_handoff_bytes))
            .and_then(|value| value.checked_add(status_handoff_bytes))
            .ok_or(HammingWeightSuccessorError::Overflow)?;

        Ok(Self {
            borrowed_row_allocation_bytes: row_bytes,
            sequence_owned_allocation_bytes,
            partial_bytes,
            audit_bytes,
            status_bytes,
            output_readback_bytes,
            shader_issued_bytes,
            compulsory_bytes: optimistic_offchip_bytes,
            optimistic_offchip_bytes,
            issued_copy_floor_ns: rate_floor_ns(shader_issued_bytes, M4_MAX_COPY_BYTES_PER_SECOND),
            issued_eighty_percent_copy_cap_ns: rate_floor_ns(
                shader_issued_bytes,
                M4_MAX_COPY_BYTES_PER_SECOND * 4 / 5,
            ),
            optimistic_offchip_eighty_percent_cap_ns: rate_floor_ns(
                optimistic_offchip_bytes,
                M4_MAX_COPY_BYTES_PER_SECOND * 4 / 5,
            ),
        })
    }
}

pub const fn raw_contribution_upper_bound(rows: u64) -> u64 {
    HAMMING_WEIGHT_SELECTORS as u64 * rows
}

pub const fn outer_product_upper_bound(shape: HammingWeightShape) -> u64 {
    HAMMING_WEIGHT_SELECTORS as u64
        * HAMMING_WEIGHT_RETAINED_BINS as u64
        * shape.outer_length() as u64
}

pub const fn clears_five_x(member_ns: u64) -> bool {
    member_ns <= HAMMING_WEIGHT_TARGET_FIVE_X_NS
}

pub const fn clears_eight_x(member_ns: u64) -> bool {
    member_ns <= HAMMING_WEIGHT_TARGET_EIGHT_X_NS
}

fn checked_product(values: &[u64]) -> Result<u64, HammingWeightSuccessorError> {
    values.iter().try_fold(1u64, |product, value| {
        product
            .checked_mul(*value)
            .ok_or(HammingWeightSuccessorError::Overflow)
    })
}

fn rate_floor_ns(work: u64, rate_per_second: u64) -> u64 {
    let numerator = u128::from(work) * 1_000_000_000u128;
    numerator.div_ceil(u128::from(rate_per_second)) as u64
}

fn check_at_most(
    name: &'static str,
    got: u64,
    maximum: u64,
) -> Result<(), HammingWeightSuccessorError> {
    if got <= maximum {
        Ok(())
    } else {
        Err(HammingWeightSuccessorError::AuditMismatch {
            name,
            expected: maximum,
            got,
        })
    }
}

fn check_exact(
    name: &'static str,
    expected: u64,
    got: u64,
) -> Result<(), HammingWeightSuccessorError> {
    if got == expected {
        Ok(())
    } else {
        Err(HammingWeightSuccessorError::AuditMismatch {
            name,
            expected,
            got,
        })
    }
}

fn check_zero(name: &'static str, got: u64) -> Result<(), HammingWeightSuccessorError> {
    check_exact(name, 0, got)
}

fn check_reserved(name: &'static str, words: &[u32]) -> Result<(), HammingWeightSuccessorError> {
    let got = words.iter().copied().map(u64::from).sum();
    check_zero(name, got)
}

fn checked_sum_u32(
    mut values: impl Iterator<Item = u32>,
) -> Result<u64, HammingWeightSuccessorError> {
    values.try_fold(0u64, |sum, value| {
        sum.checked_add(u64::from(value))
            .ok_or(HammingWeightSuccessorError::Overflow)
    })
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests use checked production shapes")]
mod tests {
    use super::*;

    #[test]
    fn log_26_work_and_traffic_are_exact() {
        let shape = HammingWeightShape::new(
            HAMMING_WEIGHT_TARGET_ROWS,
            HammingWeightSuccessorConfig::default(),
        )
        .unwrap();
        let census = HammingWeightCensus {
            rows: HAMMING_WEIGHT_TARGET_ROWS as u64,
            pc_present: HAMMING_WEIGHT_TARGET_ROWS as u64,
            ram_present: HAMMING_WEIGHT_TARGET_ROWS as u64 / 4,
            retained_nonzero_contributions: 1_588_505_707,
            occupied_outer_bins: 1_000_000,
        };
        let work = HammingWeightWorkModel::new(shape, census).unwrap();
        let traffic = HammingWeightTrafficModel::new(shape).unwrap();

        assert_eq!(work.raw_contributions, 1_845_493_760);
        assert_eq!(work.finalize_adds, 1_893_120);
        assert_eq!(work.host_split_eq_products, 262_398);
        assert_eq!(work.threadgroup_hot_bytes, 3_892_314_112);
        assert_eq!(work.threadgroup_weight_write_bytes, 1_073_741_824);
        assert_eq!(work.threadgroup_weight_read_bytes, 25_416_091_312);
        assert_eq!(work.threadgroup_audit_bytes, 50_391_040);
        assert_eq!(work.threadgroup_logical_bytes, 30_432_538_288);
        assert_eq!(traffic.borrowed_row_allocation_bytes, 2_684_354_560);
        assert_eq!(traffic.sequence_owned_allocation_bytes, 34_615_312);
        assert_eq!(traffic.partial_bytes, 30_289_920);
        assert_eq!(traffic.audit_bytes, 8_192);
        assert_eq!(traffic.status_bytes, 16);
        assert_eq!(traffic.output_readback_bytes, 126_992);
        assert_eq!(traffic.shader_issued_bytes, 3_819_048_992);
        assert_eq!(traffic.optimistic_offchip_bytes, 2_749_386_784);
        assert!(clears_five_x(HAMMING_WEIGHT_TARGET_FIVE_X_NS));
        assert!(clears_eight_x(HAMMING_WEIGHT_TARGET_EIGHT_X_NS));
    }

    #[test]
    fn sharded_audit_survives_log_28_contribution_counts() {
        let shape =
            HammingWeightShape::new(1 << 28, HammingWeightSuccessorConfig::default()).unwrap();
        let retained_per_shard =
            u32::try_from(HAMMING_WEIGHT_SELECTORS * shape.inner_length()).unwrap();
        let audit = HammingWeightAuditRow {
            rows_seen: u32::try_from(shape.inner_length()).unwrap(),
            pc_present: u32::try_from(shape.inner_length()).unwrap(),
            ram_present: 0,
            retained_nonzero_contributions: retained_per_shard,
            occupied_outer_bins: u32::try_from(
                HAMMING_WEIGHT_SELECTORS * HAMMING_WEIGHT_RETAINED_BINS,
            )
            .unwrap(),
            reserved: [0; 3],
        };
        let rows = vec![audit; shape.outer_length()];
        let census =
            HammingWeightCensus::from_audit_rows(&rows, HammingWeightStatus::default(), shape)
                .unwrap();

        assert_eq!(census.rows, 1 << 28);
        assert_eq!(census.retained_nonzero_contributions, 7_784_628_224);
        assert!(census.retained_nonzero_contributions > u64::from(u32::MAX));
    }

    #[test]
    fn audit_rejects_bad_shards_and_status() {
        let shape =
            HammingWeightShape::new(1 << 18, HammingWeightSuccessorConfig::default()).unwrap();
        let mut rows = vec![
            HammingWeightAuditRow {
                rows_seen: u32::try_from(shape.inner_length()).unwrap(),
                ..HammingWeightAuditRow::default()
            };
            shape.outer_length()
        ];
        rows[0].rows_seen -= 1;
        assert!(
            HammingWeightCensus::from_audit_rows(&rows, HammingWeightStatus::default(), shape,)
                .is_err()
        );

        rows[0].rows_seen += 1;
        assert!(HammingWeightCensus::from_audit_rows(
            &rows,
            HammingWeightStatus {
                unsupported_dispatches: 1,
                reserved: [0; 3],
            },
            shape,
        )
        .is_err());
    }
}
