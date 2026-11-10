use super::*;

#[derive(Clone, Copy)]
pub(super) struct RoundWeights<'a> {
    pub e_in: &'a [F],
    pub e_out: &'a [F],
    pub need_one: bool,
}

impl<'a> RoundWeights<'a> {
    pub(super) fn new(eq: &'a GruenSplitEqPolynomial<F>) -> Self {
        Self {
            e_in: eq.e_in_current(),
            e_out: eq.e_out_current(),
            need_one: eq.current_linear_evals().1.is_zero() && !eq.current_scalar().is_zero(),
        }
    }
}

pub(super) struct MessageBuffers {
    pub pairs: usize,
    pub groups: usize,
    pub e_in: DeviceBuffer<F>,
    pub e_out: DeviceBuffer<F>,
    pub partials: DeviceBuffer<F>,
}

impl MessageBuffers {
    pub(super) fn new(
        gpu: &RegisterGpu,
        rows: usize,
        weights: RoundWeights<'_>,
        pairs_per_group: usize,
    ) -> Result<Self, MetalError> {
        let pairs = rows.div_ceil(2);
        if weights.e_in.is_empty()
            || !weights.e_in.len().is_power_of_two()
            || pairs.div_ceil(weights.e_in.len()) > weights.e_out.len()
        {
            return Err(invalid("equality tables do not cover sparse prefix"));
        }
        let groups = pairs.div_ceil(pairs_per_group);
        let partials = DeviceBuffer::zeroed(
            &gpu.device,
            groups
                .checked_mul(3)
                .ok_or_else(|| invalid("partial length overflow"))?,
        )?;
        Ok(Self {
            pairs,
            groups,
            e_in: DeviceBuffer::from_slice(&gpu.device, weights.e_in)?,
            e_out: DeviceBuffer::from_slice(&gpu.device, weights.e_out)?,
            partials,
        })
    }

    pub(super) fn totals(&mut self) -> Result<[F; 3], MetalError> {
        let mut sums = [F::zero(); 3];
        for part in self.partials.read()?.chunks_exact(3) {
            for (sum, value) in sums.iter_mut().zip(part) {
                *sum += *value;
            }
        }
        Ok(sums)
    }
}
