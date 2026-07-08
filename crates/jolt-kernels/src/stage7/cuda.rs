use jolt_field::{Field, Fr};

use crate::cuda::{CudaError, DeviceFrVec, HammingRoundPolyInputs};

pub(crate) struct CudaHammingWeightState {
    g: Vec<DeviceFrVec>,
    eq_bool: DeviceFrVec,
    eq_virt: Vec<DeviceFrVec>,
    gamma_powers: DeviceFrVec,
    scratch: DeviceFrVec,
    scale: Fr,
}

impl CudaHammingWeightState {
    pub(crate) fn new<F: Field>(
        g: &[Vec<F>],
        eq_bool: &[F],
        eq_virt: &[Vec<F>],
        gamma_powers: &[F],
        active_scale: F,
    ) -> Option<Self> {
        let ctx = crate::cuda::shared_ctx()?;
        if g.len() != eq_virt.len() {
            return None;
        }
        let g_refs = g
            .iter()
            .map(|poly| crate::cuda::as_fr_slice(poly))
            .collect::<Option<Vec<&[Fr]>>>()?;
        let eq_virt_refs = eq_virt
            .iter()
            .map(|poly| crate::cuda::as_fr_slice(poly))
            .collect::<Option<Vec<&[Fr]>>>()?;
        let g = ctx.upload_many(&g_refs).ok()?;
        let eq_virt = ctx.upload_many(&eq_virt_refs).ok()?;
        Some(Self {
            g,
            eq_bool: ctx.upload(crate::cuda::as_fr_slice(eq_bool)?).ok()?,
            eq_virt,
            gamma_powers: ctx.upload(crate::cuda::as_fr_slice(gamma_powers)?).ok()?,
            scratch: ctx.upload(&[]).ok()?,
            scale: crate::cuda::into_fr(active_scale)?,
        })
    }

    pub(crate) fn round_poly(&self) -> Option<[Fr; 2]> {
        let ctx = crate::cuda::shared_ctx()?;
        let g_refs: Vec<&DeviceFrVec> = self.g.iter().collect();
        let eq_virt_refs: Vec<&DeviceFrVec> = self.eq_virt.iter().collect();
        ctx.hamming_round_poly(HammingRoundPolyInputs {
            g: &g_refs,
            eq_virt: &eq_virt_refs,
            eq_bool: &self.eq_bool,
            gamma_powers: &self.gamma_powers,
            scale: self.scale,
        })
        .ok()
    }

    pub(crate) fn bind(&mut self, challenge: Fr) -> Result<(), CudaError> {
        let ctx = crate::cuda::shared_ctx().ok_or(CudaError::Pool)?;
        for g in &mut self.g {
            ctx.bind(g, &mut self.scratch, challenge)?;
        }
        ctx.bind(&mut self.eq_bool, &mut self.scratch, challenge)?;
        for eq in &mut self.eq_virt {
            ctx.bind(eq, &mut self.scratch, challenge)?;
        }
        Ok(())
    }

    pub(crate) fn g_first(&self, index: usize) -> Option<Result<Fr, CudaError>> {
        self.g.get(index).map(DeviceFrVec::first)
    }

    pub(crate) fn eq_bool_first(&self) -> Result<Fr, CudaError> {
        self.eq_bool.first()
    }

    pub(crate) fn eq_virt_first(&self, index: usize) -> Option<Result<Fr, CudaError>> {
        self.eq_virt.get(index).map(DeviceFrVec::first)
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::dense::bind_dense_evals_reuse;
    use proptest::prelude::*;

    fn fr_strategy() -> impl Strategy<Value = Fr> {
        any::<[u8; 32]>().prop_map(|bytes| Fr::from_bytes(&bytes))
    }

    proptest! {
        #[test]
        fn cuda_hamming_state_matches_host(
            log_len in 1usize..8,
            num_ra in 1usize..4,
            seed in fr_strategy(),
        ) {
            let len = 1usize << log_len;
            let g: Vec<Vec<Fr>> = (0..num_ra)
                .map(|i| (0..len).map(|j| seed + Fr::from_u64((i * len + j + 1) as u64)).collect())
                .collect();
            let eq_bool: Vec<Fr> = (0..len).map(|j| seed + Fr::from_u64((j + 7) as u64)).collect();
            let eq_virt: Vec<Vec<Fr>> = (0..num_ra)
                .map(|i| (0..len).map(|j| seed + Fr::from_u64((i * len + j + 13) as u64)).collect())
                .collect();
            let gamma_powers: Vec<Fr> =
                (0..3 * num_ra).map(|i| seed + Fr::from_u64((i + 2) as u64)).collect();
            let scale = seed + Fr::from_u64(3);

            let mut h_g = g.clone();
            let mut h_eq_bool = eq_bool.clone();
            let mut h_eq_virt = eq_virt.clone();

            let mut dev = CudaHammingWeightState::new(&g, &eq_bool, &eq_virt, &gamma_powers, scale)
                .unwrap();

            for round in 0..log_len {
                let half = h_g[0].len() / 2;
                let mut expected = [Fr::from_u64(0); 2];
                for row in 0..half {
                    let low = 2 * row;
                    let high = 2 * row + 1;
                    let two = Fr::from_u64(2);
                    let eb = [h_eq_bool[low], h_eq_bool[low] + (h_eq_bool[high] - h_eq_bool[low]) * two];
                    for index in 0..num_ra {
                        let g_ev = [
                            h_g[index][low],
                            h_g[index][low] + (h_g[index][high] - h_g[index][low]) * two,
                        ];
                        let ev = [
                            h_eq_virt[index][low],
                            h_eq_virt[index][low]
                                + (h_eq_virt[index][high] - h_eq_virt[index][low]) * two,
                        ];
                        for e in 0..2 {
                            expected[e] += g_ev[e]
                                * (gamma_powers[3 * index]
                                    + gamma_powers[3 * index + 1] * eb[e]
                                    + gamma_powers[3 * index + 2] * ev[e]);
                        }
                    }
                }
                expected[0] *= scale;
                expected[1] *= scale;

                let got = dev.round_poly().unwrap();
                prop_assert_eq!(got, expected, "round {}", round);

                let challenge = seed + Fr::from_u64((round + 41) as u64);
                let mut scratch = Vec::new();
                for poly in &mut h_g {
                    bind_dense_evals_reuse(poly, &mut scratch, challenge);
                }
                bind_dense_evals_reuse(&mut h_eq_bool, &mut scratch, challenge);
                for poly in &mut h_eq_virt {
                    bind_dense_evals_reuse(poly, &mut scratch, challenge);
                }
                dev.bind(challenge).unwrap();
            }
        }
    }
}
