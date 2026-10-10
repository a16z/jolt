use crate::optimized::support::bind_pairs;
use jolt_field::JoltField;
use jolt_poly::{EqPolynomial, UnivariatePoly};
use jolt_sumcheck::SumcheckError;

#[derive(Default)]
#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField")
)]
pub(crate) struct RegisterAddressState<F: JoltField> {
    pub(crate) ra: Vec<F>,
    pub(crate) wa: Vec<F>,
    pub(crate) val: Vec<F>,
    pub(crate) eq_scalar: F,
    pub(crate) inc_scalar: F,
}

impl<F: JoltField> RegisterAddressState<F> {
    /// Address-round message over the K-sized dense arrays. Cheap enough to
    /// sample all `degree + 1` points directly, so the naive tier's running
    /// claim self-check is kept.
    pub(crate) fn round_message(
        &self,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        let half = self.ra.len() / 2;
        let mut evals = [F::zero(); 4];
        for y in 0..half {
            let pair = |table: &[F]| {
                let lo = table[2 * y];
                (lo, table[2 * y + 1] - lo)
            };
            let (ra_0, ra_m) = pair(&self.ra);
            let (wa_0, wa_m) = pair(&self.wa);
            let (val_0, val_m) = pair(&self.val);
            let (mut ra_t, mut wa_t, mut val_t) = (ra_0, wa_0, val_0);
            for eval in &mut evals {
                *eval += wa_t * (self.inc_scalar + val_t) + ra_t * val_t;
                ra_t += ra_m;
                wa_t += wa_m;
                val_t += val_m;
            }
        }
        let evals = evals.map(|eval| self.eq_scalar * eval);
        let round_sum = evals[0] + evals[1];
        if round_sum != previous_claim {
            return Err(SumcheckError::RoundCheckFailed {
                round,
                expected: previous_claim,
                actual: round_sum,
            });
        }
        Ok(UnivariatePoly::from_evals(&evals))
    }

    pub(crate) fn bind(&mut self, challenge: F) {
        for table in [&mut self.ra, &mut self.wa, &mut self.val] {
            bind_pairs(table, challenge);
        }
    }
}

/// Split the joint cycle/address equality table without a K*T allocation.
pub(crate) struct OperandEq<F> {
    pub(crate) hi: Vec<F>,
    pub(crate) lo: Vec<F>,
    pub(crate) cycle_bits_in_lo: usize,
    pub(crate) addr_bits: usize,
}

impl<F: JoltField> OperandEq<F> {
    pub(crate) fn new(r_address: &[F], r_cycle: &[F]) -> Self {
        let addr_bits = r_address.len();
        let n = r_cycle.len() + addr_bits;
        let hi_bits = r_cycle.len().min(n.div_ceil(2));
        let joint: Vec<F> = r_cycle.iter().chain(r_address).copied().collect();
        let (hi, lo) = joint.split_at(hi_bits);
        Self {
            hi: EqPolynomial::evals(hi, None),
            lo: EqPolynomial::evals(lo, None),
            cycle_bits_in_lo: n - hi_bits - addr_bits,
            addr_bits,
        }
    }

    pub(crate) fn low_index(&self, cycle: usize, register: u8) -> usize {
        ((cycle & ((1usize << self.cycle_bits_in_lo) - 1)) << self.addr_bits)
            | usize::from(register)
    }
}
