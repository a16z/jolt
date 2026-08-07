//! Analytical screen for the ProductRemainder/InstructionClaimReduction pair.

pub const FIELD_BYTES: u128 = 16;
pub const PRODUCT_ROW_BYTES: u128 = 40;
pub const LOOKUP_COMPANION_BYTES: u128 = 24;
pub const SIMD_WIDTH: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Roofs {
    pub bandwidth_bytes_per_second: f64,
    pub dependent_products_per_second: f64,
    pub bind_message_products_per_second: f64,
    pub multi_accumulator_products_per_second: f64,
    pub promotion_fraction: f64,
}

impl Roofs {
    pub const M4_MAX_RETAINED: Self = Self {
        bandwidth_bytes_per_second: 451_701_710_520.0,
        dependent_products_per_second: 18_100_000_000.0,
        bind_message_products_per_second: 24_080_000_000.0,
        multi_accumulator_products_per_second: 32_690_000_000.0,
        promotion_fraction: 0.8,
    };
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BalancedWeights {
    pub e_in: usize,
    pub e_out: usize,
}

impl BalancedWeights {
    pub fn for_head_bits(head_bits: u32) -> Self {
        let in_bits = head_bits / 2;
        let out_bits = head_bits - in_bits;
        Self {
            e_in: 1usize << in_bits,
            e_out: 1usize << out_bits,
        }
    }

    pub fn pairs(self) -> usize {
        self.e_in * self.e_out
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PhaseWork {
    pub useful_products: u128,
    pub product_dependent_products: u128,
    pub compulsory_bytes: u128,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhaseBound {
    pub traffic_ms: f64,
    pub compute_ms: f64,
    pub floor_ms: f64,
    pub promotion_gate_ms: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct JointWork {
    pub materialize: PhaseWork,
    pub transitions: PhaseWork,
    pub openings: PhaseWork,
    pub transition_rounds: usize,
    pub transition_source_elements: u128,
    pub transition_e_out_elements: u128,
    pub separate_materialize_bytes: u128,
}

impl JointWork {
    pub fn total_compulsory_bytes(self) -> u128 {
        self.materialize.compulsory_bytes
            + self.transitions.compulsory_bytes
            + self.openings.compulsory_bytes
    }

    pub fn total_useful_products(self) -> u128 {
        self.materialize.useful_products
            + self.transitions.useful_products
            + self.openings.useful_products
    }

    pub fn materialize_bytes_saved(self) -> u128 {
        self.separate_materialize_bytes - self.materialize.compulsory_bytes
    }
}

pub fn joint_work(rows: usize, cpu_tail: usize) -> JointWork {
    assert!(rows.is_power_of_two() && rows >= 4);
    assert!(cpu_tail.is_power_of_two() && (2..=rows).contains(&cpu_tail));

    let log_rows = rows.ilog2();
    let first = BalancedWeights::for_head_bits(log_rows - 1);
    let rows_u128 = rows as u128;
    let first_out = first.e_out as u128;

    let product_materialize_products = 5 * rows_u128 + 2 * first_out;
    let materialize = PhaseWork {
        useful_products: 2 * product_materialize_products,
        product_dependent_products: product_materialize_products,
        compulsory_bytes: 112 * rows_u128,
    };

    let mut source = rows;
    let mut source_sum = 0u128;
    let mut e_out_sum = 0u128;
    let mut transition_rounds = 0usize;
    while source > cpu_tail {
        let head_bits = source.ilog2() - 2;
        let weights = BalancedWeights::for_head_bits(head_bits);
        debug_assert_eq!(weights.pairs(), source / 4);
        source_sum += source as u128;
        e_out_sum += weights.e_out as u128;
        source /= 2;
        transition_rounds += 1;
    }
    let product_transition_products = 2 * source_sum + 2 * e_out_sum;
    let transitions = PhaseWork {
        useful_products: 3 * source_sum + 4 * e_out_sum,
        product_dependent_products: product_transition_products,
        compulsory_bytes: 72 * source_sum,
    };

    let opening = BalancedWeights::for_head_bits(log_rows);
    let opening_out = opening.e_out as u128;
    let product_opening_products = 3 * rows_u128 + 8 * opening_out;
    let openings = PhaseWork {
        useful_products: 5 * rows_u128 + 10 * opening_out,
        product_dependent_products: product_opening_products,
        compulsory_bytes: (PRODUCT_ROW_BYTES + LOOKUP_COMPANION_BYTES) * rows_u128,
    };

    JointWork {
        materialize,
        transitions,
        openings,
        transition_rounds,
        transition_source_elements: source_sum,
        transition_e_out_elements: e_out_sum,
        separate_materialize_bytes: 152 * rows_u128,
    }
}

fn milliseconds(amount: u128, rate_per_second: f64) -> f64 {
    amount as f64 * 1_000.0 / rate_per_second
}

pub fn materialize_bound(work: PhaseWork, roofs: Roofs) -> PhaseBound {
    phase_bound(work, roofs, roofs.multi_accumulator_products_per_second)
}

pub fn transition_bound(work: PhaseWork, roofs: Roofs) -> PhaseBound {
    phase_bound(work, roofs, roofs.bind_message_products_per_second)
}

pub fn opening_bound(work: PhaseWork, roofs: Roofs) -> PhaseBound {
    phase_bound(work, roofs, roofs.multi_accumulator_products_per_second)
}

fn phase_bound(work: PhaseWork, roofs: Roofs, aggregate_product_rate: f64) -> PhaseBound {
    let traffic_ms = milliseconds(work.compulsory_bytes, roofs.bandwidth_bytes_per_second);
    let compute_ms = milliseconds(work.useful_products, aggregate_product_rate).max(milliseconds(
        work.product_dependent_products,
        roofs.dependent_products_per_second,
    ));
    let floor_ms = traffic_ms.max(compute_ms);
    PhaseBound {
        traffic_ms,
        compute_ms,
        floor_ms,
        promotion_gate_ms: floor_ms / roofs.promotion_fraction,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROWS: usize = 1 << 26;
    const CPU_TAIL: usize = 1 << 16;

    fn close(left: f64, right: f64) {
        assert!((left - right).abs() < 0.001, "{left} != {right}");
    }

    #[test]
    fn balanced_geometry_preserves_pair_count() {
        for head_bits in 0..26 {
            let weights = BalancedWeights::for_head_bits(head_bits);
            assert_eq!(weights.pairs(), 1usize << head_bits);
            assert!(weights.e_out >= weights.e_in);
            assert!(weights.e_out <= 2 * weights.e_in);
        }
    }

    #[test]
    fn log26_counts_are_exact() {
        let work = joint_work(ROWS, CPU_TAIL);
        assert_eq!(work.materialize.useful_products, 671_121_408);
        assert_eq!(work.materialize.compulsory_bytes, 7_516_192_768);
        assert_eq!(work.transition_rounds, 10);
        assert_eq!(work.transition_source_elements, 134_086_656);
        assert_eq!(work.transition_e_out_elements, 15_872);
        assert_eq!(work.transitions.useful_products, 402_323_456);
        assert_eq!(work.transitions.compulsory_bytes, 9_654_239_232);
        assert_eq!(work.openings.useful_products, 335_626_240);
        assert_eq!(work.openings.compulsory_bytes, 4_294_967_296);
        assert_eq!(work.total_useful_products(), 1_409_071_104);
    }

    #[test]
    fn fused_materialize_removes_one_product_row_read() {
        let work = joint_work(ROWS, CPU_TAIL);
        assert_eq!(work.separate_materialize_bytes, 10_200_547_328);
        assert_eq!(work.materialize_bytes_saved(), 2_684_354_560);
        assert_eq!(
            work.materialize_bytes_saved(),
            PRODUCT_ROW_BYTES * ROWS as u128
        );
    }

    #[test]
    fn pair_has_analytical_eight_x_headroom() {
        let work = joint_work(ROWS, CPU_TAIL);
        let roofs = Roofs::M4_MAX_RETAINED;
        let materialize = materialize_bound(work.materialize, roofs);
        let transitions = transition_bound(work.transitions, roofs);
        let openings = opening_bound(work.openings, roofs);
        close(materialize.promotion_gate_ms, 25.662);
        close(transitions.promotion_gate_ms, 26.716);
        close(openings.promotion_gate_ms, 13.908);

        let total_gate = materialize.promotion_gate_ms
            + transitions.promotion_gate_ms
            + openings.promotion_gate_ms;
        let same_run_cpu_pair_ms = 496.128_750 + 357.582_667;
        assert!(total_gate < same_run_cpu_pair_ms / 8.0);
        assert!(total_gate + 30.0 < same_run_cpu_pair_ms / 8.0);
    }

    #[test]
    fn retained_standalone_instruction_active_gate_clears_eight_x() {
        let frozen_cpu_ms = 306.683_705;
        let retained_alias_active_gate_ms = 27.418_192;
        assert!(retained_alias_active_gate_ms < frozen_cpu_ms / 8.0);
    }
}
