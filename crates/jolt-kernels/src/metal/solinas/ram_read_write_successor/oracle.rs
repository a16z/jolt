//! Small independent checks for the successor's grouped-message transform.
//!
//! The full relation oracle remains in the predecessor design packet. These
//! fixtures isolate the new claim: an eq weight can move outside one cycle
//! group's address sum without changing either Gruen hint.

const MODULUS: u64 = 97;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct F(u64);

impl F {
    const ZERO: Self = Self(0);

    fn new(value: u64) -> Self {
        Self(value % MODULUS)
    }

    fn add(self, rhs: Self) -> Self {
        Self::new(self.0 + rhs.0)
    }

    fn sub(self, rhs: Self) -> Self {
        Self::new(self.0 + MODULUS - rhs.0)
    }

    fn mul(self, rhs: Self) -> Self {
        Self::new(self.0 * rhs.0)
    }
}

#[derive(Clone, Copy)]
struct Event {
    ra_low: F,
    ra_high: F,
    value_low: F,
    value_high: F,
}

fn value_term(value: F, increment: F, gamma: F) -> F {
    value.add(gamma.mul(increment.add(value)))
}

fn inner(event: Event, increment_low: F, increment_high: F, gamma: F) -> [F; 2] {
    let ra_delta = event.ra_high.sub(event.ra_low);
    let value_delta = event.value_high.sub(event.value_low);
    let increment_delta = increment_high.sub(increment_low);
    [
        event
            .ra_low
            .mul(value_term(event.value_low, increment_low, gamma)),
        ra_delta.mul(value_term(value_delta, increment_delta, gamma)),
    ]
}

fn flat(
    events: &[Event],
    increment_low: F,
    increment_high: F,
    gamma: F,
    e_in: F,
    e_out: F,
) -> [F; 2] {
    let weight = e_in.mul(e_out);
    events.iter().fold([F::ZERO; 2], |sum, &event| {
        let value = inner(event, increment_low, increment_high, gamma);
        [
            sum[0].add(weight.mul(value[0])),
            sum[1].add(weight.mul(value[1])),
        ]
    })
}

fn grouped(
    events: &[Event],
    increment_low: F,
    increment_high: F,
    gamma: F,
    e_in: F,
    e_out: F,
) -> [F; 2] {
    let sum = events.iter().fold([F::ZERO; 2], |sum, &event| {
        let value = inner(event, increment_low, increment_high, gamma);
        [sum[0].add(value[0]), sum[1].add(value[1])]
    });
    let weight = e_in.mul(e_out);
    [weight.mul(sum[0]), weight.mul(sum[1])]
}

fn opening_point(cycle: &[u64], address: &[u64]) -> Vec<u64> {
    address
        .iter()
        .rev()
        .chain(cycle.iter().rev())
        .copied()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grouped_weight_matches_flat_event_weighting() {
        let events = [
            Event {
                ra_low: F::new(1),
                ra_high: F::new(0),
                value_low: F::new(11),
                value_high: F::new(19),
            },
            Event {
                ra_low: F::new(7),
                ra_high: F::new(13),
                value_low: F::new(23),
                value_high: F::new(29),
            },
            Event {
                ra_low: F::new(0),
                ra_high: F::new(1),
                value_low: F::new(31),
                value_high: F::new(37),
            },
        ];
        let args = (F::new(41), F::new(43), F::new(47), F::new(53), F::new(59));
        assert_eq!(
            flat(&events, args.0, args.1, args.2, args.3, args.4),
            grouped(&events, args.0, args.1, args.2, args.3, args.4)
        );
    }

    #[test]
    fn empty_group_contributes_zero() {
        assert_eq!(
            grouped(&[], F::new(3), F::new(5), F::new(7), F::new(11), F::new(13)),
            [F::ZERO; 2]
        );
    }

    #[test]
    fn output_point_keeps_address_then_cycle_big_endian_orientation() {
        assert_eq!(
            opening_point(&[1, 2, 3, 4], &[10, 11, 12]),
            vec![12, 11, 10, 4, 3, 2, 1]
        );
    }
}
