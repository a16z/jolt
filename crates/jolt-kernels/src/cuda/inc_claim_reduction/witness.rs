use jolt_witness::witnesses::{RamInc, RdInc};
use jolt_witness::WitnessBundle;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, WitnessBundle)]
pub struct IncClaimReductionWitness {
    #[opening(committed = RamInc)]
    pub ram: RamInc,
    #[opening(committed = RdInc)]
    pub rd: RdInc,
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_witness::witnesses::{RamInc, RdInc};

    use super::IncClaimReductionWitness;
    use crate::cuda::common::context::shared_context;

    fn sample_rows() -> Vec<IncClaimReductionWitness> {
        let increments: [(i128, i128); 6] = [
            (0, 0),
            (1, -1),
            (-1, 1),
            (u64::MAX as i128, -(u64::MAX as i128)),
            (-(u64::MAX as i128), u64::MAX as i128),
            (0, -(1i128 << 63)),
        ];
        increments
            .into_iter()
            .map(|(ram, rd)| IncClaimReductionWitness {
                ram: RamInc(ram),
                rd: RdInc(rd),
            })
            .collect()
    }

    #[test]
    fn sample_rows_exercise_both_increment_signs() {
        let rows = sample_rows();
        for (name, values) in [
            ("ram", rows.iter().map(|row| row.ram.0).collect::<Vec<_>>()),
            ("rd", rows.iter().map(|row| row.rd.0).collect::<Vec<_>>()),
        ] {
            assert!(
                values.iter().any(|value| *value < 0),
                "no synthetic row carries a negative {name} increment",
            );
            assert!(
                values.iter().any(|value| *value > 0),
                "no synthetic row carries a positive {name} increment",
            );
            assert!(
                values.contains(&0),
                "no synthetic row carries an idle {name} cycle",
            );
        }
    }

    #[test]
    fn synthetic_device_columns_match_the_host_conversion() {
        let Some(context) = shared_context() else {
            return;
        };
        let rows = sample_rows();
        let expected: Vec<Vec<Fr>> = vec![
            rows.iter().map(|row| Fr::from_i128(row.ram.0)).collect(),
            rows.iter().map(|row| Fr::from_i128(row.rd.0)).collect(),
        ];
        let convert = |values: Vec<i128>| {
            let mut limbs = Vec::with_capacity(values.len() * 2);
            for value in &values {
                let bits = *value as u128;
                limbs.push(bits as u64);
                limbs.push((bits >> 64) as u64);
            }
            let uploaded = context.upload_u64_slice(&limbs).expect("upload limbs");
            context
                .i128_to_montgomery_device(&uploaded, values.len())
                .expect("device conversion")
                .to_host()
                .expect("download")
        };
        let got: Vec<Vec<Fr>> = vec![
            convert(rows.iter().map(|row| row.ram.0).collect()),
            convert(rows.iter().map(|row| row.rd.0).collect()),
        ];
        assert_eq!(got, expected, "device increment columns diverged");
    }
}
