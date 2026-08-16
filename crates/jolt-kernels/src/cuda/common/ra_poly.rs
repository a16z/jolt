#![expect(
    dead_code,
    reason = "implementation target: the stage-5 relations wire this once their kernels land"
)]

use cudarc::driver::{CudaSlice, PushKernelArg};
use jolt_field::{Field, Fr};
use jolt_poly::BindingOrder;

use super::context::CudaKernelContext;
use super::device::{fr_limbs, require_fr, require_fr_slice, DeviceFrVec};
use super::error::CudaError;

pub const COLLAPSE_AFTER_ROUNDS: usize = 3;

pub use crate::cuda::common::pack::COLD;

pub struct DeviceRaPolynomial {
    indices: CudaSlice<u32>,
    tables: DeviceFrVec,
    addresses: usize,
    cycles: usize,
    order: BindingOrder,
    rounds_bound: usize,
    dense: Option<DeviceFrVec>,
}

fn reverse_bits(value: usize, bits: usize) -> usize {
    (0..bits).fold(0, |acc, bit| {
        acc | (((value >> bit) & 1) << (bits - 1 - bit))
    })
}

impl DeviceRaPolynomial {
    pub fn new<F: Field>(
        context: &CudaKernelContext,
        hot_indices: &[Option<usize>],
        eq_address: &[F],
        order: BindingOrder,
    ) -> Result<Self, CudaError> {
        let cycles = hot_indices.len();
        let addresses = eq_address.len();
        if !cycles.is_power_of_two() || !addresses.is_power_of_two() {
            return Err(CudaError::InvariantViolation {
                reason: "a one-hot polynomial needs power-of-two cycle and address counts",
            });
        }
        let mut raw = Vec::with_capacity(cycles);
        for hot in hot_indices {
            let encoded = match *hot {
                None => COLD,
                Some(address) if address < addresses => {
                    u32::try_from(address).map_err(|_| CudaError::LengthMismatch {
                        expected: addresses,
                        got: address,
                    })?
                }
                Some(address) => {
                    return Err(CudaError::LengthMismatch {
                        expected: addresses,
                        got: address,
                    })
                }
            };
            raw.push(encoded);
        }
        Self::from_words(context, &raw, eq_address, order)
    }

    pub fn from_words<F: Field>(
        context: &CudaKernelContext,
        words: &[u32],
        eq_address: &[F],
        order: BindingOrder,
    ) -> Result<Self, CudaError> {
        let tables = context.upload(require_fr_slice(eq_address)?)?;
        Self::from_device_tables(context, words, tables, order)
    }

    pub fn from_device_tables(
        context: &CudaKernelContext,
        words: &[u32],
        tables: DeviceFrVec,
        order: BindingOrder,
    ) -> Result<Self, CudaError> {
        let cycles = words.len();
        let addresses = tables.len();
        if !cycles.is_power_of_two() || !addresses.is_power_of_two() {
            return Err(CudaError::InvariantViolation {
                reason: "a one-hot polynomial needs power-of-two cycle and address counts",
            });
        }
        Ok(Self {
            indices: context.upload_u32_slice(words)?,
            tables,
            addresses,
            cycles,
            order,
            rounds_bound: 0,
            dense: None,
        })
    }

    pub const fn len(&self) -> usize {
        self.cycles >> self.rounds_bound
    }

    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub const fn order(&self) -> BindingOrder {
        self.order
    }

    pub const fn rounds_bound(&self) -> usize {
        self.rounds_bound
    }

    pub const fn is_collapsed(&self) -> bool {
        self.dense.is_some()
    }

    pub const fn dense(&self) -> Option<&DeviceFrVec> {
        self.dense.as_ref()
    }

    pub fn coefficients(&self, context: &CudaKernelContext) -> Result<DeviceFrVec, CudaError> {
        if let Some(dense) = &self.dense {
            return dense.try_clone();
        }
        self.gather(context, self.rounds_bound)
    }

    fn slot_bases(&self, rounds: usize) -> Vec<u32> {
        let slots = 1usize << rounds;
        let len = self.cycles >> rounds;
        (0..slots)
            .map(|slot| {
                let offset = match self.order {
                    BindingOrder::LowToHigh => slot,
                    BindingOrder::HighToLow => reverse_bits(slot, rounds) * len,
                };
                offset as u32
            })
            .collect()
    }

    fn gather(&self, context: &CudaKernelContext, rounds: usize) -> Result<DeviceFrVec, CudaError> {
        let len = self.cycles >> rounds;
        let mut output = context.alloc(len)?;
        if len == 0 {
            return Ok(output);
        }
        let bases = context.upload_u32_slice(&self.slot_bases(rounds))?;
        let slots = CudaKernelContext::count_of(1usize << rounds)?;
        let addresses = CudaKernelContext::count_of(self.addresses)?;
        let stride = CudaKernelContext::count_of(match self.order {
            BindingOrder::LowToHigh => 1usize << rounds,
            BindingOrder::HighToLow => 1,
        })?;
        let count = CudaKernelContext::count_of(len)?;
        let mut builder = context.stream().launch_builder(context.ra_gather());
        let _ = builder.arg(&self.indices);
        let _ = builder.arg(self.tables.limbs());
        let _ = builder.arg(&bases);
        let _ = builder.arg(&slots);
        let _ = builder.arg(&addresses);
        let _ = builder.arg(&stride);
        let _ = builder.arg(output.limbs_mut());
        let _ = builder.arg(&count);
        // SAFETY: thread `j < len` writes only `out[j]`, and reads
        // `indices[j * stride + bases[slot]]` for `slots` slots — every offset is
        // below `cycles` (`stride * len == cycles` for LowToHigh; for HighToLow
        // `reverse_bits(slot) * len + j < cycles`), and a non-`COLD` entry is
        // `< addresses` — checked by `new` per index, and by `from_words`'s
        // callers, which encode through `common::pack::encode_address` with the
        // same `addresses` bound — so the table read is inside `tables`'s
        // `slots * addresses`. `out` is a distinct allocation.
        let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
        context.stream().synchronize()?;
        Ok(output)
    }

    fn split_tables<F: Field>(
        &mut self,
        context: &CudaKernelContext,
        challenge: F,
    ) -> Result<(), CudaError> {
        let challenge = require_fr(challenge)?;
        let eq_zero = fr_limbs(Fr::from(1u64) - challenge);
        let eq_one = fr_limbs(challenge);
        let len = self.tables.len();
        let mut doubled = context.alloc(len * 2)?;
        let count = CudaKernelContext::count_of(len)?;
        let mut builder = context.stream().launch_builder(context.ra_split_tables());
        let _ = builder.arg(self.tables.limbs());
        for limb in &eq_zero {
            let _ = builder.arg(limb);
        }
        for limb in &eq_one {
            let _ = builder.arg(limb);
        }
        let _ = builder.arg(doubled.limbs_mut());
        let _ = builder.arg(&count);
        // SAFETY: thread `i < len` reads `in[i]` and writes exactly `out[i]` and
        // `out[len + i]` — disjoint across threads. The two eq weights arrive as
        // by-value limbs, so no device buffer backs them. `in` holds `len`, `out`
        // holds `2 * len`, and `out` is a distinct allocation, so no thread reads
        // another's write.
        let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
        self.tables = doubled;
        Ok(())
    }

    pub fn bind<F: Field>(
        &mut self,
        context: &CudaKernelContext,
        challenge: F,
    ) -> Result<(), CudaError> {
        if let Some(dense) = &self.dense {
            let bound = context.bind(dense, require_fr(challenge)?, self.order)?;
            self.dense = Some(bound);
            self.rounds_bound += 1;
            return Ok(());
        }
        if self.len() < 2 {
            return Err(CudaError::LengthMismatch {
                expected: 2,
                got: self.len(),
            });
        }
        self.split_tables(context, challenge)?;
        self.rounds_bound += 1;
        if self.rounds_bound >= COLLAPSE_AFTER_ROUNDS {
            self.dense = Some(self.gather(context, self.rounds_bound)?);
            self.tables = context.alloc(0)?;
            self.indices = context.alloc_u32(0)?;
        }
        Ok(())
    }

    pub fn final_claim(&self, context: &CudaKernelContext) -> Result<Fr, CudaError> {
        if self.len() != 1 {
            return Err(CudaError::LengthMismatch {
                expected: 1,
                got: self.len(),
            });
        }
        match &self.dense {
            Some(dense) => dense.first(),
            None => self.gather(context, self.rounds_bound)?.first(),
        }
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::{BindingOrder, EqPolynomial, Polynomial};
    use proptest::prelude::*;

    use super::super::context::shared_context;
    use super::super::testing::arb_point;
    use super::{reverse_bits, DeviceRaPolynomial, COLLAPSE_AFTER_ROUNDS};

    struct HostRaPolynomial {
        indices: Vec<Option<usize>>,
        tables: Vec<Vec<Fr>>,
        order: BindingOrder,
        rounds_bound: usize,
    }

    impl HostRaPolynomial {
        fn new(indices: &[Option<usize>], eq_address: &[Fr], order: BindingOrder) -> Self {
            Self {
                indices: indices.to_vec(),
                tables: vec![eq_address.to_vec()],
                order,
                rounds_bound: 0,
            }
        }

        fn len(&self) -> usize {
            self.indices.len() >> self.rounds_bound
        }

        fn coefficient(&self, j: usize) -> Fr {
            let fold = 1usize << self.rounds_bound;
            let mut total = Fr::from_u64(0);
            for (slot, table) in self.tables.iter().enumerate() {
                let position = match self.order {
                    BindingOrder::LowToHigh => fold * j + slot,
                    BindingOrder::HighToLow => {
                        j + reverse_bits(slot, self.rounds_bound) * self.len()
                    }
                };
                if let Some(Some(address)) = self.indices.get(position) {
                    total += table[*address];
                }
            }
            total
        }

        fn coefficients(&self) -> Vec<Fr> {
            (0..self.len()).map(|j| self.coefficient(j)).collect()
        }

        fn bind(&mut self, challenge: Fr) {
            let one = Fr::from_u64(1);
            let eq_zero = one - challenge;
            let eq_one = challenge;
            let mut split: Vec<Vec<Fr>> = Vec::with_capacity(self.tables.len() * 2);
            for table in &self.tables {
                split.push(table.iter().map(|value| *value * eq_zero).collect());
            }
            for table in &self.tables {
                split.push(table.iter().map(|value| *value * eq_one).collect());
            }
            self.tables = split;
            self.rounds_bound += 1;
        }
    }

    fn fixture(log_t: usize, log_k: usize, seed: u64) -> (Vec<Option<usize>>, Vec<Fr>) {
        let cycles = 1usize << log_t;
        let addresses = 1usize << log_k;
        let indices: Vec<Option<usize>> = (0..cycles)
            .map(|c| {
                if (c as u64 + seed) % 5 == 3 {
                    None
                } else {
                    Some(((c as u64 * 7 + seed) as usize) % addresses)
                }
            })
            .collect();
        let address: Vec<Fr> = (0..log_k)
            .map(|i| Fr::from_u64(11 + 5 * i as u64))
            .collect();
        (indices, EqPolynomial::new(address).evaluations())
    }

    #[test]
    fn host_model_matches_dense_binding() {
        for order in [BindingOrder::LowToHigh, BindingOrder::HighToLow] {
            let log_t = 5usize;
            let (indices, eq_address) = fixture(log_t, 3, 1);

            let mut model = HostRaPolynomial::new(&indices, &eq_address, order);
            let mut dense = Polynomial::new(model.coefficients());

            for round in 0..log_t {
                assert_eq!(
                    model.coefficients(),
                    dense.evals().to_vec(),
                    "host model diverged from dense at round {round} for {order:?}",
                );
                let challenge = Fr::from_u64(101 + 7 * round as u64);
                model.bind(challenge);
                dense.bind_with_order(challenge, order);
            }
        }
    }

    proptest! {
        #[test]
        fn ra_poly_matches_legacy_model(
            log_t in 1usize..=6,
            log_k in 1usize..=4,
            seed in any::<u64>(),
            challenges in arb_point(6),
        ) {
            let Some(context) = shared_context() else { return Ok(()); };
            for order in [BindingOrder::LowToHigh, BindingOrder::HighToLow] {
                let (indices, eq_address) = fixture(log_t, log_k, seed);

                let mut expected = HostRaPolynomial::new(&indices, &eq_address, order);
                let mut got = DeviceRaPolynomial::new(context, &indices, &eq_address, order)
                    .expect("device ra poly");

                for (round, &challenge) in challenges.iter().enumerate().take(log_t) {
                    let got_coeffs = got
                        .coefficients(context)
                        .expect("device coefficients")
                        .to_host()
                        .expect("download");
                    prop_assert_eq!(
                        got_coeffs,
                        expected.coefficients(),
                        "coefficients diverged at round {} for {:?}",
                        round,
                        order
                    );
                    prop_assert_eq!(got.len(), expected.len());
                    prop_assert_eq!(got.is_collapsed(), round >= COLLAPSE_AFTER_ROUNDS);
                    got.bind(context, challenge).expect("device bind");
                    expected.bind(challenge);
                }
                prop_assert_eq!(
                    got.final_claim(context).expect("final claim"),
                    expected.coefficients()[0]
                );
            }
        }
    }
}
