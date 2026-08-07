use std::sync::Arc;

use jolt_field::{AkitaField, FromPrimitiveInt};

use super::super::registers::{
    CertifiedRegisterOwner, RegisterOwnerRead, RegisterOwnerRow, RegisterOwnerWrite,
    REGISTER_CSR_COLUMNS,
};
use super::*;

fn field(value: u64) -> AkitaField {
    AkitaField::from_u64(value)
}

fn fixture() -> (Vec<RegisterOwnerRow>, [u64; REGISTER_CSR_COLUMNS]) {
    let mut initial = [0; REGISTER_CSR_COLUMNS];
    initial[1] = 5;
    initial[2] = 9;
    let rows = vec![
        RegisterOwnerRow {
            rs1: Some(RegisterOwnerRead {
                register: 1,
                value: 5,
            }),
            rs2: Some(RegisterOwnerRead {
                register: 2,
                value: 9,
            }),
            rd: Some(RegisterOwnerWrite {
                register: 1,
                pre_value: 5,
                post_value: 8,
            }),
        },
        RegisterOwnerRow {
            rs1: Some(RegisterOwnerRead {
                register: 1,
                value: 8,
            }),
            rd: Some(RegisterOwnerWrite {
                register: 2,
                pre_value: 9,
                post_value: 4,
            }),
            ..RegisterOwnerRow::default()
        },
        RegisterOwnerRow {
            rd: Some(RegisterOwnerWrite {
                register: 3,
                pre_value: 0,
                post_value: 7,
            }),
            ..RegisterOwnerRow::default()
        },
        RegisterOwnerRow {
            rd: Some(RegisterOwnerWrite {
                register: 3,
                pre_value: 7,
                post_value: 7,
            }),
            ..RegisterOwnerRow::default()
        },
        RegisterOwnerRow {
            rs2: Some(RegisterOwnerRead {
                register: 1,
                value: 8,
            }),
            ..RegisterOwnerRow::default()
        },
        RegisterOwnerRow {
            rd: Some(RegisterOwnerWrite {
                register: 1,
                pre_value: 8,
                post_value: 2,
            }),
            ..RegisterOwnerRow::default()
        },
        RegisterOwnerRow {
            rs1: Some(RegisterOwnerRead {
                register: 1,
                value: 2,
            }),
            ..RegisterOwnerRow::default()
        },
        RegisterOwnerRow::default(),
    ];
    (rows, initial)
}

fn carrier(
    rows: &[RegisterOwnerRow],
    initial: &[u64; REGISTER_CSR_COLUMNS],
) -> (Arc<CertifiedRegisterOwner>, RegisterFamilyCarrier) {
    let owner = Arc::new(CertifiedRegisterOwner::build(rows, initial, rows.len()).unwrap());
    let identity =
        RegisterOwnerIdentity::new(11, RegisterOwnerSourceKind::OwnedRandomAccess, 13, 17).unwrap();
    let carrier = RegisterFamilyCarrier::new(identity, Arc::clone(&owner)).unwrap();
    (owner, carrier)
}

#[test]
fn carrier_shares_one_owner_and_reports_actual_views() {
    let (rows, initial) = fixture();
    let (owner, carrier) = carrier(&rows, &initial);

    assert!(core::ptr::eq(owner.as_ref(), carrier.owner()));
    assert_eq!(Arc::strong_count(&owner), 2);
    assert_eq!(carrier.geometry().cycles(), rows.len());
    assert_eq!(carrier.geometry().prefix_bits(), 2);
    assert_eq!(carrier.geometry().suffix_bits(), 1);
    assert_eq!(
        carrier.storage().full_owner_bytes,
        owner.csr().storage_bytes()
    );
    assert!(carrier.storage().claim_midpoint_bytes < carrier.storage().value_owner_bytes);
    assert!(carrier.storage().value_owner_bytes < carrier.storage().full_owner_bytes);
}

#[test]
fn owner_claim_oracle_matches_an_independent_dense_sumcheck() {
    let (rows, initial) = fixture();
    let (_, carrier) = carrier(&rows, &initial);
    let tau = [field(3), field(5), field(7)];
    let gamma = field(19);
    let challenges = [field(23), field(29), field(31)];
    let components = claim_components_from_owner(&carrier, &tau).unwrap();
    let got = claim_sumcheck_oracle(&carrier, &components, gamma, &challenges).unwrap();

    let mut equality = test_eq_evaluations(&tau);
    let mut columns = dense_claim_columns(&rows);
    let gamma_sq = gamma * gamma;
    let input_claim = test_dense_claim(&equality, &columns, gamma, gamma_sq);
    let mut messages = Vec::new();
    for &challenge in &challenges {
        messages.push(test_quadratic_message(&equality, &columns, gamma, gamma_sq));
        test_bind(&mut equality, challenge);
        for column in &mut columns {
            test_bind(column, challenge);
        }
    }
    let outputs = ClaimOutputValues {
        rd_write_value: columns[0][0],
        rs1_value: columns[1][0],
        rs2_value: columns[2][0],
    };

    assert_eq!(got.input_claim, input_claim);
    assert_eq!(got.messages, messages);
    assert_eq!(got.outputs, outputs);
    assert_eq!(
        got.opening_point,
        challenges.iter().rev().copied().collect::<Vec<_>>()
    );
    assert_eq!(
        got.terminal_claim,
        test_dense_claim(&equality, &columns, gamma, gamma_sq)
    );
}

#[test]
fn owner_value_oracle_preserves_zero_increment_wa_and_low_bind_order() {
    let (rows, initial) = fixture();
    let (_, carrier) = carrier(&rows, &initial);
    let address = [
        field(2),
        field(3),
        field(5),
        field(7),
        field(11),
        field(13),
        field(17),
    ];
    let cycle = vec![field(19), field(23), field(29)];
    let point = RegisterValuePoint::new(&carrier, &address, cycle.clone()).unwrap();
    let first = value_first_message_oracle(&carrier, &point).unwrap();

    let address_eq = test_eq_evaluations(&address);
    let mut inc = Vec::new();
    let mut wa = Vec::new();
    let mut lt = Vec::new();
    for (index, row) in rows.iter().enumerate() {
        inc.push(AkitaField::from_i128(row.rd.map_or(0, |write| {
            i128::from(write.post_value) - i128::from(write.pre_value)
        })));
        wa.push(row.rd.map_or(AkitaField::zero(), |write| {
            address_eq[usize::from(write.register)]
        }));
        lt.push(test_lt_at_boolean_index(index, &cycle));
    }
    let expected_first = test_cubic_message(&inc, &wa, &lt);
    assert_eq!(first.samples, expected_first);
    assert_eq!(first.relation_claim, expected_first.claim_identity());

    let challenge = field(37);
    let expected_inc = test_bound(&inc, challenge);
    let expected_wa = test_bound(&wa, challenge);
    let expected_lt = test_bound(&lt, challenge);
    let mut emitted = Vec::new();
    let transition =
        value_first_transition_oracle(&carrier, &point, first.samples, challenge, |index, row| {
            emitted.push((index, row));
        })
        .unwrap();
    let expected_rows = expected_inc
        .iter()
        .zip(&expected_wa)
        .enumerate()
        .map(|(index, (&rd_inc, &rd_wa))| (index, ValueBoundRow { rd_inc, rd_wa }))
        .collect::<Vec<_>>();

    assert_eq!(emitted, expected_rows);
    assert_eq!(
        transition.next_message,
        Some(test_cubic_message(
            &expected_inc,
            &expected_wa,
            &expected_lt
        ))
    );
    assert_eq!(
        transition.bound_claim,
        expected_inc
            .iter()
            .zip(&expected_wa)
            .zip(&expected_lt)
            .fold(AkitaField::zero(), |sum, ((&inc, &wa), &lt)| sum
                + inc * wa * lt)
    );

    assert_eq!(inc[3], AkitaField::zero());
    assert_ne!(wa[3], AkitaField::zero());
}

#[test]
fn typed_points_reject_owner_and_order_mismatches() {
    let (rows, initial) = fixture();
    let (_, first_owner) = carrier(&rows, &initial);
    let owner = Arc::new(CertifiedRegisterOwner::build(&rows, &initial, rows.len()).unwrap());
    let second_identity =
        RegisterOwnerIdentity::new(11, RegisterOwnerSourceKind::OwnedRandomAccess, 13, 99).unwrap();
    let second_owner = RegisterFamilyCarrier::new(second_identity, owner).unwrap();
    let address = [field(2); REGISTER_ADDRESS_BITS];
    let point = RegisterValuePoint::new(&first_owner, &address, vec![field(3), field(5), field(7)])
        .unwrap();
    assert_eq!(
        point.validate_owner(&second_owner),
        Err(RegisterFamilyModelError::OwnerIdentityMismatch)
    );

    let challenges = [field(11), field(13), field(17)];
    let output = point.output_point(&challenges).unwrap();
    assert_eq!(&output[..REGISTER_ADDRESS_BITS], &address);
    assert_eq!(
        &output[REGISTER_ADDRESS_BITS..],
        challenges.iter().rev().copied().collect::<Vec<_>>()
    );
}

fn dense_claim_columns(rows: &[RegisterOwnerRow]) -> [Vec<AkitaField>; 3] {
    let mut columns = core::array::from_fn(|_| Vec::with_capacity(rows.len()));
    for row in rows {
        columns[0].push(
            row.rd
                .map_or(AkitaField::zero(), |write| field(write.post_value)),
        );
        columns[1].push(row.rs1.map_or(AkitaField::zero(), |read| field(read.value)));
        columns[2].push(row.rs2.map_or(AkitaField::zero(), |read| field(read.value)));
    }
    columns
}

fn test_dense_claim(
    equality: &[AkitaField],
    columns: &[Vec<AkitaField>; 3],
    gamma: AkitaField,
    gamma_sq: AkitaField,
) -> AkitaField {
    (0..equality.len()).fold(AkitaField::zero(), |sum, index| {
        sum + equality[index]
            * (columns[0][index] + gamma * columns[1][index] + gamma_sq * columns[2][index])
    })
}

fn test_quadratic_message(
    equality: &[AkitaField],
    columns: &[Vec<AkitaField>; 3],
    gamma: AkitaField,
    gamma_sq: AkitaField,
) -> QuadraticSamples<AkitaField> {
    let mut sums = [AkitaField::zero(); 3];
    for pair in 0..equality.len() / 2 {
        let index = 2 * pair;
        let eq = test_linear_samples(equality[index], equality[index + 1]);
        let rd = test_linear_samples(columns[0][index], columns[0][index + 1]);
        let rs1 = test_linear_samples(columns[1][index], columns[1][index + 1]);
        let rs2 = test_linear_samples(columns[2][index], columns[2][index + 1]);
        for sample in 0..3 {
            sums[sample] +=
                eq[sample] * (rd[sample] + gamma * rs1[sample] + gamma_sq * rs2[sample]);
        }
    }
    QuadraticSamples {
        at_0: sums[0],
        at_1: sums[1],
        at_2: sums[2],
    }
}

fn test_eq_evaluations(point: &[AkitaField]) -> Vec<AkitaField> {
    (0..1usize << point.len())
        .map(|index| {
            point
                .iter()
                .enumerate()
                .fold(AkitaField::one(), |value, (position, &coordinate)| {
                    let bit = (index >> (point.len() - 1 - position)) & 1;
                    value
                        * if bit == 0 {
                            AkitaField::one() - coordinate
                        } else {
                            coordinate
                        }
                })
        })
        .collect()
}

fn test_bind(values: &mut Vec<AkitaField>, challenge: AkitaField) {
    let bound = test_bound(values, challenge);
    *values = bound;
}

fn test_bound(values: &[AkitaField], challenge: AkitaField) -> Vec<AkitaField> {
    values
        .chunks_exact(2)
        .map(|pair| pair[0] + challenge * (pair[1] - pair[0]))
        .collect()
}

fn test_linear_samples(low: AkitaField, high: AkitaField) -> [AkitaField; 3] {
    [low, high, high + high - low]
}

fn test_cubic_message(
    inc: &[AkitaField],
    wa: &[AkitaField],
    lt: &[AkitaField],
) -> CubicSamples<AkitaField> {
    let mut sums = [AkitaField::zero(); 4];
    for index in 0..inc.len() / 2 {
        let row = 2 * index;
        let inc = test_cubic_linear_samples(inc[row], inc[row + 1]);
        let wa = test_cubic_linear_samples(wa[row], wa[row + 1]);
        let lt = test_cubic_linear_samples(lt[row], lt[row + 1]);
        for sample in 0..4 {
            sums[sample] += inc[sample] * wa[sample] * lt[sample];
        }
    }
    CubicSamples {
        at_0: sums[0],
        at_1: sums[1],
        at_2: sums[2],
        at_3: sums[3],
    }
}

fn test_cubic_linear_samples(low: AkitaField, high: AkitaField) -> [AkitaField; 4] {
    let delta = high - low;
    [low, high, high + delta, high + delta + delta]
}

fn test_lt_at_boolean_index(index: usize, point: &[AkitaField]) -> AkitaField {
    let mut lt = AkitaField::zero();
    let mut equality = AkitaField::one();
    for (position, &coordinate) in point.iter().enumerate() {
        let bit = (index >> (point.len() - 1 - position)) & 1;
        if bit == 0 {
            lt += coordinate * equality;
            equality *= AkitaField::one() - coordinate;
        } else {
            equality *= coordinate;
        }
    }
    lt
}
