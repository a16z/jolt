use jolt_field::JoltField;
use jolt_poly::{try_eq_mle, LtPolynomial};

pub(crate) fn reversed<F: Copy>(sumcheck_point: &[F]) -> Vec<F> {
    sumcheck_point.iter().rev().copied().collect()
}

pub(crate) fn eq_at_point<F: JoltField>(opening_point: &[F], reference: &[F]) -> Result<F, String> {
    try_eq_mle(opening_point, reference).map_err(|error| error.to_string())
}

pub(crate) fn eq_at_cycle<F: JoltField>(
    fixed_cycle: &[F],
    opening_point: &[F],
    address_bits: usize,
    family: &str,
) -> Result<F, String> {
    let own_cycle = opening_point.get(address_bits..).ok_or_else(|| {
        format!("{family} read-write opening point is shorter than the {family} address width")
    })?;
    try_eq_mle(fixed_cycle, own_cycle).map_err(|error| error.to_string())
}

pub(crate) fn lt_at_cycle<F: JoltField>(
    own_point: &[F],
    upstream_point: &[F],
    address_bits: usize,
    family: &str,
) -> Result<F, String> {
    let own_cycle = own_point.get(address_bits..).ok_or_else(|| {
        format!("rd_inc opening point is shorter than the {family} address width")
    })?;
    let fixed_cycle = upstream_point.get(address_bits..).ok_or_else(|| {
        format!("{family} read-write opening point is shorter than the {family} address width")
    })?;
    Ok(LtPolynomial::evaluate(own_cycle, fixed_cycle))
}

pub(crate) fn val_evaluation_address<'a, F>(
    upstream_point: &'a [F],
    address_bits: usize,
    log_t: usize,
    family: &str,
) -> Result<&'a [F], String> {
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "the address width is a small constant and log_t an ilog2 result (< 64); the sum cannot overflow usize"
    )]
    let expected_len = address_bits + log_t;
    if upstream_point.len() != expected_len {
        return Err(format!(
            "{family} read-write opening point has {} variables, expected {expected_len}",
            upstream_point.len()
        ));
    }
    upstream_point
        .get(..address_bits)
        .ok_or_else(|| format!("{family} read-write opening point address prefix is out of range"))
}
