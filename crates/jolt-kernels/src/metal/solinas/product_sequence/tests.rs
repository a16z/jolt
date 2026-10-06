#![expect(clippy::unwrap_used, reason = "test assertions")]

use super::*;
use jolt_field::{FromPrimitiveInt, One, Zero};

#[test]
fn product_messages_and_bindings_match_direct_field_evaluation() {
    let context = SolinasMetal::for_akita().unwrap();
    for factors in [5, 9] {
        let mut elements = 1024;
        let mut tables: Vec<AkitaField> = (0..factors * elements)
            .map(|i| {
                if i % 3 == 0 {
                    -AkitaField::from_u64((i * 7919 + 23) as u64)
                } else {
                    AkitaField::from_u64((i * 65537 + 7) as u64)
                }
            })
            .collect();
        let weights = |count| {
            (0..count)
                .map(|i| AkitaField::from_u64((i * 13 + 11) as u64))
                .collect::<Vec<_>>()
        };
        let mut e_in = weights(4);
        let mut e_out = weights(elements / 8);
        let config = ProductSequenceConfig::default();
        let mut sequence = if factors == 5 {
            context.prepare_product_sequence(&tables, elements, &e_in, &e_out, config)
        } else {
            context.prepare_product_sequence_from_fn(
                elements,
                factors,
                &e_in,
                &e_out,
                config,
                |factor, position| tables[factor * elements + position],
            )
        }
        .unwrap();
        let mut message = sequence.message(&e_in, &e_out).unwrap();
        loop {
            let expected: Vec<AkitaField> = (0..factors)
                .map(|sample| {
                    (0..elements / 2)
                        .map(|row| {
                            let product: AkitaField = (0..factors)
                                .map(|factor| {
                                    let lo = tables[factor * elements + 2 * row];
                                    let hi = tables[factor * elements + 2 * row + 1];
                                    if sample + 1 == factors {
                                        hi - lo
                                    } else {
                                        lo + AkitaField::from_u64((sample + 1) as u64) * (hi - lo)
                                    }
                                })
                                .fold(AkitaField::one(), |a, b| a * b);
                            product * e_in[row % e_in.len()] * e_out[row / e_in.len()]
                        })
                        .fold(AkitaField::zero(), |a, b| a + b)
                })
                .collect();
            assert_eq!(message, expected, "factors={factors}, elements={elements}");
            let mut readback = vec![AkitaField::zero(); tables.len()];
            sequence.read_current_tables(&mut readback).unwrap();
            assert_eq!(readback, tables);
            if elements == 8 {
                break;
            }
            let challenge = -AkitaField::from_u64((elements + 37) as u64);
            tables = tables
                .chunks_exact(elements)
                .flat_map(|factor| {
                    factor
                        .chunks_exact(2)
                        .map(|pair| pair[0] + challenge * (pair[1] - pair[0]))
                })
                .collect();
            elements /= 2;
            e_in = weights(2);
            e_out = weights(elements / 4);
            message = sequence.bind_and_message(challenge, &e_in, &e_out).unwrap();
        }
    }
}
