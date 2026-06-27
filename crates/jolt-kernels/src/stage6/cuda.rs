use jolt_field::{Field, Fr};

use crate::cuda::{CudaError, DeviceFrVec, Gather8Inputs, HammingBooleanityInputs, RoundPolyTerms};

pub(crate) struct CudaHammingBooleanityState {
    hamming_weight: DeviceFrVec,
    scratch: DeviceFrVec,
}

impl CudaHammingBooleanityState {
    pub(crate) fn new<F: Field>(hamming_weight: &[F]) -> Option<Self> {
        let ctx = crate::cuda::shared_ctx()?;
        Some(Self {
            hamming_weight: ctx.upload(crate::cuda::as_fr_slice(hamming_weight)?).ok()?,
            scratch: ctx.upload(&[]).ok()?,
        })
    }

    pub(crate) fn round_poly_q<F: Field>(&self, e_in: &[F], e_out: &[F]) -> Option<[Fr; 2]> {
        let ctx = crate::cuda::shared_ctx()?;
        let e_in_dev = ctx.upload(crate::cuda::as_fr_slice(e_in)?).ok()?;
        let e_out_dev = ctx.upload(crate::cuda::as_fr_slice(e_out)?).ok()?;
        ctx.hamming_booleanity_round_poly(HammingBooleanityInputs {
            hamming_weight: &self.hamming_weight,
            e_in: &e_in_dev,
            e_out: &e_out_dev,
        })
        .ok()
    }

    pub(crate) fn bind(&mut self, challenge: Fr) -> Result<(), CudaError> {
        let ctx = crate::cuda::shared_ctx().ok_or(CudaError::Pool)?;
        ctx.bind(&mut self.hamming_weight, &mut self.scratch, challenge)
    }

    pub(crate) fn hamming_weight_first(&self) -> Result<Fr, CudaError> {
        self.hamming_weight.first()
    }
}

pub(crate) struct CudaIncState {
    eq_ram: DeviceFrVec,
    ram_inc: DeviceFrVec,
    eq_rd: DeviceFrVec,
    rd_inc: DeviceFrVec,
    scratch: DeviceFrVec,
    gamma2: Fr,
}

impl CudaIncState {
    pub(crate) fn new<F: Field>(
        eq_ram: &[F],
        ram_inc: &[F],
        eq_rd: &[F],
        rd_inc: &[F],
        gamma2: F,
    ) -> Option<Self> {
        let ctx = crate::cuda::shared_ctx()?;
        Some(Self {
            eq_ram: ctx.upload(crate::cuda::as_fr_slice(eq_ram)?).ok()?,
            ram_inc: ctx.upload(crate::cuda::as_fr_slice(ram_inc)?).ok()?,
            eq_rd: ctx.upload(crate::cuda::as_fr_slice(eq_rd)?).ok()?,
            rd_inc: ctx.upload(crate::cuda::as_fr_slice(rd_inc)?).ok()?,
            scratch: ctx.upload(&[]).ok()?,
            gamma2: crate::cuda::into_fr(gamma2)?,
        })
    }

    pub(crate) fn round_poly_evals(&self) -> Result<[Fr; 2], CudaError> {
        let ctx = crate::cuda::shared_ctx().ok_or(CudaError::Pool)?;
        let factors = [&self.eq_ram, &self.ram_inc, &self.eq_rd, &self.rd_inc];
        let term_coeffs = [Fr::from(1u64), self.gamma2];
        let term_factor_offsets = [0u32, 2, 4];
        let term_factor_indices = [0u32, 1, 2, 3];
        let evals = ctx.sum_of_products_round_poly(RoundPolyTerms {
            factors: &factors,
            term_coeffs: &term_coeffs,
            term_factor_offsets: &term_factor_offsets,
            term_factor_indices: &term_factor_indices,
            degree: 2,
        })?;
        Ok([evals[0], evals[1]])
    }

    pub(crate) fn bind(&mut self, challenge: Fr) -> Result<(), CudaError> {
        let ctx = crate::cuda::shared_ctx().ok_or(CudaError::Pool)?;
        for factor in [
            &mut self.eq_ram,
            &mut self.ram_inc,
            &mut self.eq_rd,
            &mut self.rd_inc,
        ] {
            ctx.bind(factor, &mut self.scratch, challenge)?;
        }
        Ok(())
    }

    pub(crate) fn factor_first(&self, factor: usize) -> Result<Fr, CudaError> {
        match factor {
            0 => self.eq_ram.first(),
            1 => self.ram_inc.first(),
            2 => self.eq_rd.first(),
            3 => self.rd_inc.first(),
            _ => Err(CudaError::Pool),
        }
    }

    fn final_relation_eval(&self) -> Result<Fr, CudaError> {
        let eq_ram = self.eq_ram.first()?;
        let ram_inc = self.ram_inc.first()?;
        let eq_rd = self.eq_rd.first()?;
        let rd_inc = self.rd_inc.first()?;
        Ok(eq_ram * ram_inc + self.gamma2 * eq_rd * rd_inc)
    }
}

pub(crate) fn inc_final_relation_eval<F: Field>(
    state: &super::IncClaimReductionStage6State<F>,
) -> Option<F> {
    let cuda = state.cuda.as_ref()?;
    crate::cuda::fr_into::<F>(cuda.final_relation_eval().ok()?)
}

pub(crate) fn materialize_gather8<F: Field, R: AsRef<[Option<u8>]>>(
    table_groups: &[&Vec<Vec<F>>; 8],
    indices: &[R],
) -> Option<Vec<Vec<F>>> {
    let ctx = crate::cuda::shared_ctx()?;
    let num_chunks = indices.len();
    if num_chunks == 0 {
        return None;
    }
    let new_len = indices[0].as_ref().len() / 8;
    let table_len = table_groups[0].first().map_or(0, Vec::len);
    if table_len == 0 || new_len == 0 {
        return None;
    }
    for group in table_groups {
        if group.len() != num_chunks || group.iter().any(|table| table.len() != table_len) {
            return None;
        }
    }

    let mut flat_groups: Vec<Vec<Fr>> = Vec::with_capacity(8);
    for group in table_groups {
        let mut flat = Vec::with_capacity(num_chunks * table_len);
        for table in *group {
            flat.extend_from_slice(crate::cuda::as_fr_slice(table)?);
        }
        flat_groups.push(flat);
    }
    let table_refs: [&[Fr]; 8] = std::array::from_fn(|g| flat_groups[g].as_slice());

    let mut flat_indices: Vec<i16> = Vec::with_capacity(num_chunks * new_len * 8);
    for chunk in indices {
        for entry in chunk.as_ref() {
            flat_indices.push(entry.map_or(-1, i16::from));
        }
    }

    let dense = ctx
        .gather8_materialize(Gather8Inputs {
            table_groups: table_refs,
            indices: &flat_indices,
            num_chunks,
            table_len,
            new_len,
        })
        .ok()?;

    dense
        .into_iter()
        .map(|chunk| {
            chunk
                .into_iter()
                .map(crate::cuda::fr_into::<F>)
                .collect::<Option<Vec<F>>>()
        })
        .collect::<Option<Vec<Vec<F>>>>()
}
