use jolt_field::{Field, Fr};

use crate::cuda::{
    CoreBooleanityAddressInputs, CoreBooleanityCycleInputs, CudaError, DeviceFrVec, Gather8Inputs,
    HammingBooleanityInputs, RaVirtualD4Inputs, RoundPolyTerms,
};

pub(crate) struct CudaBytecodeReadRafState {
    factors: Vec<DeviceFrVec>,
    scratch: DeviceFrVec,
    term_coeffs: Vec<Fr>,
    term_factor_offsets: Vec<u32>,
    term_factor_indices: Vec<u32>,
    points: Vec<Fr>,
    num_output_factors: usize,
}

impl CudaBytecodeReadRafState {
    pub(crate) fn new_address<F: Field>(
        stage_factors: &[Vec<F>],
        stage_values: &[Vec<F>],
        entry_trace: &[F],
        entry_expected: &[F],
        gamma_powers: &[F],
    ) -> Option<Self> {
        let ctx = crate::cuda::shared_ctx()?;
        let stages = stage_factors.len();
        let mut factors = Vec::with_capacity(2 * stages + 2);
        for factor in stage_factors {
            factors.push(ctx.upload(crate::cuda::as_fr_slice(factor)?).ok()?);
        }
        for value in stage_values {
            factors.push(ctx.upload(crate::cuda::as_fr_slice(value)?).ok()?);
        }
        factors.push(ctx.upload(crate::cuda::as_fr_slice(entry_trace)?).ok()?);
        factors.push(ctx.upload(crate::cuda::as_fr_slice(entry_expected)?).ok()?);

        let mut term_coeffs = Vec::with_capacity(stages + 1);
        let mut term_factor_offsets = vec![0u32];
        let mut term_factor_indices = Vec::new();
        for (stage, gamma) in gamma_powers.iter().take(stages).enumerate() {
            term_coeffs.push(crate::cuda::into_fr(*gamma)?);
            term_factor_indices.push(stage as u32);
            term_factor_indices.push((stages + stage) as u32);
            term_factor_offsets.push(term_factor_indices.len() as u32);
        }
        term_coeffs.push(crate::cuda::into_fr(gamma_powers[7])?);
        term_factor_indices.push((2 * stages) as u32);
        term_factor_indices.push((2 * stages + 1) as u32);
        term_factor_offsets.push(term_factor_indices.len() as u32);

        Some(Self {
            factors,
            scratch: ctx.upload(&[]).ok()?,
            term_coeffs,
            term_factor_offsets,
            term_factor_indices,
            points: vec![Fr::from(0u64), Fr::from(2u64)],
            num_output_factors: 0,
        })
    }

    pub(crate) fn new_cycle<F: Field>(
        cycle_chunks: &[Vec<F>],
        combined_eq: &[F],
        degree: usize,
    ) -> Option<Self> {
        let ctx = crate::cuda::shared_ctx()?;
        let num_chunks = cycle_chunks.len();
        let mut factors = Vec::with_capacity(num_chunks + 1);
        for chunk in cycle_chunks {
            factors.push(ctx.upload(crate::cuda::as_fr_slice(chunk)?).ok()?);
        }
        factors.push(ctx.upload(crate::cuda::as_fr_slice(combined_eq)?).ok()?);

        let term_factor_indices: Vec<u32> = (0..=num_chunks as u32).collect();
        let points: Vec<Fr> = (0..degree)
            .map(|e| Fr::from(if e == 0 { 0 } else { (e + 1) as u64 }))
            .collect();
        Some(Self {
            factors,
            scratch: ctx.upload(&[]).ok()?,
            term_coeffs: vec![Fr::from(1u64)],
            term_factor_offsets: vec![0, (num_chunks + 1) as u32],
            term_factor_indices,
            points,
            num_output_factors: num_chunks,
        })
    }

    pub(crate) fn round_poly_evals(&self) -> Option<Vec<Fr>> {
        let ctx = crate::cuda::shared_ctx()?;
        let factor_refs: Vec<&DeviceFrVec> = self.factors.iter().collect();
        ctx.sum_of_products_round_poly_at(
            RoundPolyTerms {
                factors: &factor_refs,
                term_coeffs: &self.term_coeffs,
                term_factor_offsets: &self.term_factor_offsets,
                term_factor_indices: &self.term_factor_indices,
                degree: self.points.len(),
            },
            &self.points,
        )
        .ok()
    }

    pub(crate) fn bind(&mut self, challenge: Fr) -> Result<(), CudaError> {
        let ctx = crate::cuda::shared_ctx().ok_or(CudaError::Pool)?;
        for factor in &mut self.factors {
            ctx.bind(factor, &mut self.scratch, challenge)?;
        }
        Ok(())
    }

    pub(crate) fn factor_first(&self, factor: usize) -> Option<Result<Fr, CudaError>> {
        self.factors.get(factor).map(DeviceFrVec::first)
    }

    pub(crate) fn output_factor_first(&self, factor: usize) -> Option<Result<Fr, CudaError>> {
        (factor < self.num_output_factors).then(|| self.factors[factor].first())
    }
}

pub(crate) struct CudaRaVirtualD4State {
    chunks: Vec<DeviceFrVec>,
    scratch: DeviceFrVec,
    gamma_powers: Vec<Fr>,
}

impl CudaRaVirtualD4State {
    pub(crate) fn new<F: Field>(chunks: &[Vec<F>], gamma_powers: &[F]) -> Option<Self> {
        let ctx = crate::cuda::shared_ctx()?;
        if chunks.is_empty() || chunks.len() != gamma_powers.len() * 4 {
            return None;
        }
        let device_chunks = chunks
            .iter()
            .map(|chunk| ctx.upload(crate::cuda::as_fr_slice(chunk)?).ok())
            .collect::<Option<Vec<DeviceFrVec>>>()?;
        let gamma_powers = gamma_powers
            .iter()
            .map(|g| crate::cuda::into_fr(*g))
            .collect::<Option<Vec<Fr>>>()?;
        Some(Self {
            chunks: device_chunks,
            scratch: ctx.upload(&[]).ok()?,
            gamma_powers,
        })
    }

    pub(crate) fn round_poly_evals<F: Field>(&self, e_in: &[F], e_out: &[F]) -> Option<[Fr; 4]> {
        let ctx = crate::cuda::shared_ctx()?;
        let e_in_dev = ctx.upload(crate::cuda::as_fr_slice(e_in)?).ok()?;
        let e_out_dev = ctx.upload(crate::cuda::as_fr_slice(e_out)?).ok()?;
        let chunk_refs: Vec<&DeviceFrVec> = self.chunks.iter().collect();
        ctx.ra_virtual_d4_round_poly(RaVirtualD4Inputs {
            chunks: &chunk_refs,
            gamma_powers: &self.gamma_powers,
            e_in: &e_in_dev,
            e_out: &e_out_dev,
        })
        .ok()
    }

    pub(crate) fn bind(&mut self, challenge: Fr) -> Result<(), CudaError> {
        let ctx = crate::cuda::shared_ctx().ok_or(CudaError::Pool)?;
        for chunk in &mut self.chunks {
            ctx.bind(chunk, &mut self.scratch, challenge)?;
        }
        Ok(())
    }

    pub(crate) fn chunk_first(&self, chunk: usize) -> Option<Result<Fr, CudaError>> {
        self.chunks.get(chunk).map(DeviceFrVec::first)
    }
}

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

pub(crate) struct CudaCoreBooleanityState {
    h: Vec<DeviceFrVec>,
    scratch: DeviceFrVec,
    rho: Vec<Fr>,
}

impl CudaCoreBooleanityState {
    pub(crate) fn from_sparse_round1<F: Field>(
        tables: &[Vec<F>],
        indices: &[jolt_witness::Stage6BooleanityRow],
        rho: &[F],
    ) -> Option<Self> {
        let ctx = crate::cuda::shared_ctx()?;
        let num_polys = tables.len();
        if num_polys == 0 || num_polys != rho.len() {
            return None;
        }
        let chunk_domain = tables.first().map_or(0, Vec::len);
        if chunk_domain == 0 || tables.iter().any(|t| t.len() != chunk_domain) {
            return None;
        }
        let rows = indices.len();
        let poly_stride = jolt_witness::STAGE6_BOOLEANITY_MAX_POLYS;

        let mut flat_tables = Vec::with_capacity(num_polys * chunk_domain);
        for table in tables {
            flat_tables.extend_from_slice(crate::cuda::as_fr_slice(table)?);
        }
        let mut present_mask = Vec::with_capacity(rows);
        let mut values = Vec::with_capacity(rows * poly_stride);
        for row in indices {
            present_mask.push(row.present_mask());
            values.extend_from_slice(row.values());
        }

        let h = ctx
            .core_booleanity_gather(crate::cuda::CoreBooleanityGatherInputs {
                tables: &flat_tables,
                present_mask: &present_mask,
                values: &values,
                num_polys,
                chunk_domain,
                rows,
                poly_stride,
            })
            .ok()?;
        let rho = rho
            .iter()
            .map(|r| crate::cuda::into_fr(*r))
            .collect::<Option<Vec<Fr>>>()?;
        Some(Self {
            h,
            scratch: ctx.upload(&[]).ok()?,
            rho,
        })
    }

    pub(crate) fn round_poly_q<F: Field>(&self, e_in: &[F], e_out: &[F]) -> Option<[Fr; 2]> {
        let ctx = crate::cuda::shared_ctx()?;
        let e_in_dev = ctx.upload(crate::cuda::as_fr_slice(e_in)?).ok()?;
        let e_out_dev = ctx.upload(crate::cuda::as_fr_slice(e_out)?).ok()?;
        let h_refs: Vec<&DeviceFrVec> = self.h.iter().collect();
        ctx.core_booleanity_cycle_round_poly(CoreBooleanityCycleInputs {
            h_polys: &h_refs,
            rho: &self.rho,
            e_in: &e_in_dev,
            e_out: &e_out_dev,
        })
        .ok()
    }

    pub(crate) fn bind(&mut self, challenge: Fr) -> Result<(), CudaError> {
        let ctx = crate::cuda::shared_ctx().ok_or(CudaError::Pool)?;
        for poly in &mut self.h {
            ctx.bind(poly, &mut self.scratch, challenge)?;
        }
        Ok(())
    }

    pub(crate) fn poly_first(&self, index: usize) -> Option<Result<Fr, CudaError>> {
        self.h.get(index).map(DeviceFrVec::first)
    }
}

pub(crate) struct CudaCoreBooleanityAddressState {
    g: Vec<DeviceFrVec>,
    gamma_squares: Vec<Fr>,
}

impl CudaCoreBooleanityAddressState {
    pub(crate) fn new<F: Field>(g: &[Vec<F>], gamma_squares: &[F]) -> Option<Self> {
        let ctx = crate::cuda::shared_ctx()?;
        if g.is_empty() || g.len() != gamma_squares.len() {
            return None;
        }
        let device_g = g
            .iter()
            .map(|poly| ctx.upload(crate::cuda::as_fr_slice(poly)?).ok())
            .collect::<Option<Vec<DeviceFrVec>>>()?;
        let gamma_squares = gamma_squares
            .iter()
            .map(|g| crate::cuda::into_fr(*g))
            .collect::<Option<Vec<Fr>>>()?;
        Some(Self {
            g: device_g,
            gamma_squares,
        })
    }

    pub(crate) fn round_poly_q<F: Field>(
        &self,
        f_values: &[F],
        e_in: &[F],
        e_out: &[F],
        m: usize,
    ) -> Option<[Fr; 2]> {
        let ctx = crate::cuda::shared_ctx()?;
        let f_dev = crate::cuda::as_fr_slice(f_values)?;
        let e_in_dev = ctx.upload(crate::cuda::as_fr_slice(e_in)?).ok()?;
        let e_out_dev = ctx.upload(crate::cuda::as_fr_slice(e_out)?).ok()?;
        let g_refs: Vec<&DeviceFrVec> = self.g.iter().collect();
        ctx.core_booleanity_address_round_poly(CoreBooleanityAddressInputs {
            g: &g_refs,
            f_values: f_dev,
            gamma_squares: &self.gamma_squares,
            e_in: &e_in_dev,
            e_out: &e_out_dev,
            m: m as u32,
        })
        .ok()
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
