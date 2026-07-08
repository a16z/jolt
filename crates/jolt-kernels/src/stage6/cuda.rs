use jolt_field::{Field, Fr};

use crate::cuda::{
    CoreBooleanityAddressInputs, CoreBooleanityCycleInputs, CudaError, DeviceFrVec, Gather8Inputs,
    HammingBooleanityInputs, RaVirtualD4Inputs, RoundPolyTerms,
};

pub(crate) enum CudaBytecodeReadRafState {
    SumOfProducts {
        factors: Vec<DeviceFrVec>,
        scratch: DeviceFrVec,
        term_coeffs: Vec<Fr>,
        term_factor_offsets: Vec<u32>,
        term_factor_indices: Vec<u32>,
        points: Vec<Fr>,
        num_output_factors: usize,
    },
    CycleSparse {
        round: u32,
        tables: crate::cuda::CudaSlice<u64>,
        values: crate::cuda::CudaSlice<i16>,
        combined_eq: DeviceFrVec,
        combined_eq_scratch: DeviceFrVec,
        num_chunks: usize,
        chunk_domain: usize,
        source_rows: usize,
        degree: usize,
    },
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
        let mut refs: Vec<&[Fr]> = Vec::with_capacity(2 * stages + 2);
        for factor in stage_factors {
            refs.push(crate::cuda::as_fr_slice(factor)?);
        }
        for value in stage_values {
            refs.push(crate::cuda::as_fr_slice(value)?);
        }
        refs.push(crate::cuda::as_fr_slice(entry_trace)?);
        refs.push(crate::cuda::as_fr_slice(entry_expected)?);
        let factors = ctx.upload_many(&refs).ok()?;

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

        Some(Self::SumOfProducts {
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
        let mut refs: Vec<&[Fr]> = Vec::with_capacity(num_chunks + 1);
        for chunk in cycle_chunks {
            refs.push(crate::cuda::as_fr_slice(chunk)?);
        }
        refs.push(crate::cuda::as_fr_slice(combined_eq)?);
        let factors = ctx.upload_many(&refs).ok()?;
        Self::from_device_cycle(factors, num_chunks, degree)
    }

    fn from_device_cycle(factors: Vec<DeviceFrVec>, num_chunks: usize, degree: usize) -> Option<Self> {
        let ctx = crate::cuda::shared_ctx()?;
        let term_factor_indices: Vec<u32> = (0..=num_chunks as u32).collect();
        let points: Vec<Fr> = (0..degree)
            .map(|e| Fr::from(if e == 0 { 0 } else { (e + 1) as u64 }))
            .collect();
        Some(Self::SumOfProducts {
            factors,
            scratch: ctx.upload(&[]).ok()?,
            term_coeffs: vec![Fr::from(1u64)],
            term_factor_offsets: vec![0, (num_chunks + 1) as u32],
            term_factor_indices,
            points,
            num_output_factors: num_chunks,
        })
    }

    pub(crate) fn new_cycle_sparse<F: Field>(
        tables: &[Vec<F>],
        indices: &[Vec<Option<u8>>],
        combined_eq: &[F],
        degree: usize,
    ) -> Option<Self> {
        let ctx = crate::cuda::shared_ctx()?;
        let num_chunks = tables.len();
        if num_chunks == 0 || indices.len() != num_chunks {
            return None;
        }
        let chunk_domain = tables.first().map_or(0, Vec::len);
        if chunk_domain == 0 || tables.iter().any(|t| t.len() != chunk_domain) {
            return None;
        }
        let source_rows = indices.first().map_or(0, Vec::len);
        if source_rows == 0 || indices.iter().any(|c| c.len() != source_rows) {
            return None;
        }
        if combined_eq.len() != source_rows {
            return None;
        }

        let mut flat_tables: Vec<u64> = Vec::with_capacity(num_chunks * chunk_domain * 4);
        for table in tables {
            for v in crate::cuda::as_fr_slice(table)? {
                flat_tables.extend_from_slice(&v.inner_limbs().0);
            }
        }
        let mut values: Vec<i16> = Vec::with_capacity(num_chunks * source_rows);
        for chunk in indices {
            for entry in chunk {
                values.push(entry.map_or(-1, i16::from));
            }
        }

        Some(Self::CycleSparse {
            round: 1,
            tables: ctx.upload_u64_slice(&flat_tables).ok()?,
            values: ctx.upload_i16_slice(&values).ok()?,
            combined_eq: ctx.upload(crate::cuda::as_fr_slice(combined_eq)?).ok()?,
            combined_eq_scratch: ctx.upload(&[]).ok()?,
            num_chunks,
            chunk_domain,
            source_rows,
            degree,
        })
    }

    pub(crate) fn round_poly_evals(&self) -> Option<Vec<Fr>> {
        let ctx = crate::cuda::shared_ctx()?;
        match self {
            Self::SumOfProducts {
                factors,
                term_coeffs,
                term_factor_offsets,
                term_factor_indices,
                points,
                ..
            } => {
                let factor_refs: Vec<&DeviceFrVec> = factors.iter().collect();
                ctx.sum_of_products_round_poly_at(
                    RoundPolyTerms {
                        factors: &factor_refs,
                        term_coeffs,
                        term_factor_offsets,
                        term_factor_indices,
                        degree: points.len(),
                    },
                    points,
                )
                .ok()
            }
            Self::CycleSparse {
                round,
                tables,
                values,
                combined_eq,
                num_chunks,
                chunk_domain,
                source_rows,
                degree,
                ..
            } => ctx
                .bytecode_cycle_sparse_round_poly(crate::cuda::BytecodeCycleSparseInputs {
                    tables,
                    values,
                    combined_eq,
                    num_chunks: *num_chunks,
                    chunk_domain: *chunk_domain,
                    source_rows: *source_rows,
                    degree: *degree,
                    round: *round,
                })
                .ok(),
        }
    }

    pub(crate) fn bind(&mut self, challenge: Fr) -> Result<(), CudaError> {
        let ctx = crate::cuda::shared_ctx().ok_or(CudaError::Pool)?;
        match self {
            Self::SumOfProducts {
                factors, scratch, ..
            } => {
                for factor in factors.iter_mut() {
                    ctx.bind(factor, scratch, challenge)?;
                }
                Ok(())
            }
            Self::CycleSparse {
                round,
                tables,
                values,
                combined_eq,
                combined_eq_scratch,
                num_chunks,
                chunk_domain,
                source_rows,
                degree,
            } => {
                let num_sets = 1usize << (*round - 1);
                let set_elems = *num_chunks * *chunk_domain;
                let bound = ctx.ra_virtual_d4_sparse_bind(tables, num_sets, set_elems, challenge)?;
                ctx.bind(combined_eq, combined_eq_scratch, challenge)?;
                if *round < 3 {
                    *tables = bound;
                    *round += 1;
                    Ok(())
                } else {
                    let out_len = *source_rows >> 3;
                    let chunks = ctx.ra_virtual_d4_sparse_collapse(
                        &bound,
                        values,
                        *num_chunks,
                        *chunk_domain,
                        *source_rows,
                        out_len,
                    )?;
                    let combined = std::mem::replace(combined_eq, ctx.upload(&[])?);
                    let mut factors = chunks;
                    factors.push(combined);
                    *self = Self::from_device_cycle(factors, *num_chunks, *degree)
                        .ok_or(CudaError::Pool)?;
                    Ok(())
                }
            }
        }
    }

    pub(crate) fn factor_first(&self, factor: usize) -> Option<Result<Fr, CudaError>> {
        match self {
            Self::SumOfProducts { factors, .. } => factors.get(factor).map(DeviceFrVec::first),
            Self::CycleSparse { .. } => None,
        }
    }

    pub(crate) fn output_factor_first(&self, factor: usize) -> Option<Result<Fr, CudaError>> {
        match self {
            Self::SumOfProducts {
                factors,
                num_output_factors,
                ..
            } => (factor < *num_output_factors).then(|| factors[factor].first()),
            Self::CycleSparse { .. } => None,
        }
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
        let refs = chunks
            .iter()
            .map(|chunk| crate::cuda::as_fr_slice(chunk))
            .collect::<Option<Vec<&[Fr]>>>()?;
        let device_chunks = ctx.upload_many(&refs).ok()?;
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

    pub(crate) fn from_device_chunks(chunks: Vec<DeviceFrVec>, gamma_powers: Vec<Fr>) -> Option<Self> {
        let ctx = crate::cuda::shared_ctx()?;
        Some(Self {
            chunks,
            scratch: ctx.upload(&[]).ok()?,
            gamma_powers,
        })
    }

    pub(crate) fn round_poly_evals(
        &self,
        e_in: &DeviceFrVec,
        e_out: &DeviceFrVec,
    ) -> Option<[Fr; 4]> {
        let ctx = crate::cuda::shared_ctx()?;
        let chunk_refs: Vec<&DeviceFrVec> = self.chunks.iter().collect();
        ctx.ra_virtual_d4_round_poly(RaVirtualD4Inputs {
            chunks: &chunk_refs,
            gamma_powers: &self.gamma_powers,
            e_in,
            e_out,
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

pub(crate) enum CudaRaVirtualD4Sparse {
    Sparse {
        round: u32,
        tables: crate::cuda::CudaSlice<u64>,
        values: crate::cuda::CudaSlice<i16>,
        num_chunks: usize,
        chunk_domain: usize,
        source_rows: usize,
        gamma_powers: Vec<Fr>,
    },
    Dense(CudaRaVirtualD4State),
}

impl CudaRaVirtualD4Sparse {
    pub(crate) fn from_round1<F: Field>(
        tables: &[Vec<F>],
        indices: &[&[Option<u8>]],
        gamma_powers: &[F],
    ) -> Option<Self> {
        let ctx = crate::cuda::shared_ctx()?;
        let num_chunks = tables.len();
        if num_chunks == 0 || num_chunks != gamma_powers.len() * 4 || indices.len() != num_chunks {
            return None;
        }
        let chunk_domain = tables.first().map_or(0, Vec::len);
        if chunk_domain == 0 || tables.iter().any(|t| t.len() != chunk_domain) {
            return None;
        }
        let source_rows = indices.first().map_or(0, |c| c.len());
        if source_rows == 0 || indices.iter().any(|c| c.len() != source_rows) {
            return None;
        }

        let mut flat_tables: Vec<u64> = Vec::with_capacity(num_chunks * chunk_domain * 4);
        for table in tables {
            for v in crate::cuda::as_fr_slice(table)? {
                flat_tables.extend_from_slice(&v.inner_limbs().0);
            }
        }
        let mut values: Vec<i16> = Vec::with_capacity(num_chunks * source_rows);
        for chunk in indices {
            for entry in *chunk {
                values.push(entry.map_or(-1, i16::from));
            }
        }
        let gamma_powers = gamma_powers
            .iter()
            .map(|g| crate::cuda::into_fr(*g))
            .collect::<Option<Vec<Fr>>>()?;

        Some(Self::Sparse {
            round: 1,
            tables: ctx.upload_u64_slice(&flat_tables).ok()?,
            values: ctx.upload_i16_slice(&values).ok()?,
            num_chunks,
            chunk_domain,
            source_rows,
            gamma_powers,
        })
    }

    pub(crate) fn round_poly_evals(
        &self,
        e_in: &DeviceFrVec,
        e_out: &DeviceFrVec,
    ) -> Option<[Fr; 4]> {
        match self {
            Self::Sparse {
                round,
                tables,
                values,
                num_chunks,
                chunk_domain,
                source_rows,
                gamma_powers,
            } => {
                let ctx = crate::cuda::shared_ctx()?;
                ctx.ra_virtual_d4_sparse_round_poly(crate::cuda::RaVirtualD4SparseInputs {
                    tables,
                    values,
                    num_chunks: *num_chunks,
                    chunk_domain: *chunk_domain,
                    source_rows: *source_rows,
                    gamma_powers,
                    e_in,
                    e_out,
                    round: *round,
                })
                .ok()
            }
            Self::Dense(dense) => dense.round_poly_evals(e_in, e_out),
        }
    }

    pub(crate) fn bind(&mut self, challenge: Fr) -> Result<(), CudaError> {
        let ctx = crate::cuda::shared_ctx().ok_or(CudaError::Pool)?;
        match self {
            Self::Sparse {
                round,
                tables,
                values,
                num_chunks,
                chunk_domain,
                source_rows,
                gamma_powers,
            } => {
                let num_sets = 1usize << (*round - 1);
                let set_elems = *num_chunks * *chunk_domain;
                let bound =
                    ctx.ra_virtual_d4_sparse_bind(tables, num_sets, set_elems, challenge)?;
                if *round < 3 {
                    *tables = bound;
                    *round += 1;
                    Ok(())
                } else {
                    let out_len = *source_rows >> 3;
                    let chunks = ctx.ra_virtual_d4_sparse_collapse(
                        &bound,
                        values,
                        *num_chunks,
                        *chunk_domain,
                        *source_rows,
                        out_len,
                    )?;
                    let dense = CudaRaVirtualD4State::from_device_chunks(chunks, gamma_powers.clone())
                        .ok_or(CudaError::Pool)?;
                    *self = Self::Dense(dense);
                    Ok(())
                }
            }
            Self::Dense(dense) => dense.bind(challenge),
        }
    }

    pub(crate) fn chunk_first(&self, chunk: usize) -> Option<Result<Fr, CudaError>> {
        match self {
            Self::Sparse { .. } => None,
            Self::Dense(dense) => dense.chunk_first(chunk),
        }
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

    pub(crate) fn round_poly_q(&self, e_in: &DeviceFrVec, e_out: &DeviceFrVec) -> Option<[Fr; 2]> {
        let ctx = crate::cuda::shared_ctx()?;
        ctx.hamming_booleanity_round_poly(HammingBooleanityInputs {
            hamming_weight: &self.hamming_weight,
            e_in,
            e_out,
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
    pub(crate) fn from_device_h(h: Vec<DeviceFrVec>, rho: Vec<Fr>) -> Option<Self> {
        let ctx = crate::cuda::shared_ctx()?;
        Some(Self {
            h,
            scratch: ctx.upload(&[]).ok()?,
            rho,
        })
    }

    pub(crate) fn round_poly_q(&self, e_in: &DeviceFrVec, e_out: &DeviceFrVec) -> Option<[Fr; 2]> {
        let ctx = crate::cuda::shared_ctx()?;
        let h_refs: Vec<&DeviceFrVec> = self.h.iter().collect();
        ctx.core_booleanity_cycle_round_poly(CoreBooleanityCycleInputs {
            h_polys: &h_refs,
            rho: &self.rho,
            e_in,
            e_out,
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

pub(crate) enum CudaCoreBooleanitySparse {
    Sparse {
        round: u32,
        tables: crate::cuda::CudaSlice<u64>,
        present_mask: crate::cuda::CudaSlice<u64>,
        values: crate::cuda::CudaSlice<u8>,
        source_rows: usize,
        num_polys: usize,
        chunk_domain: usize,
        poly_stride: usize,
        rho: Vec<Fr>,
    },
    Dense(CudaCoreBooleanityState),
}

impl CudaCoreBooleanitySparse {
    pub(crate) fn from_round1<F: Field>(
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
        let source_rows = indices.len();
        let poly_stride = jolt_witness::STAGE6_BOOLEANITY_MAX_POLYS;

        let mut flat_tables: Vec<u64> = Vec::with_capacity(num_polys * chunk_domain * 4);
        for table in tables {
            for v in crate::cuda::as_fr_slice(table)? {
                flat_tables.extend_from_slice(&v.inner_limbs().0);
            }
        }
        let mut present_mask = Vec::with_capacity(source_rows);
        let mut values = Vec::with_capacity(source_rows * poly_stride);
        for row in indices {
            present_mask.push(row.present_mask());
            values.extend_from_slice(row.values());
        }
        let rho = rho
            .iter()
            .map(|r| crate::cuda::into_fr(*r))
            .collect::<Option<Vec<Fr>>>()?;

        Some(Self::Sparse {
            round: 1,
            tables: ctx.upload_u64_slice(&flat_tables).ok()?,
            present_mask: ctx.upload_u64_slice(&present_mask).ok()?,
            values: ctx.upload_u8_slice(&values).ok()?,
            source_rows,
            num_polys,
            chunk_domain,
            poly_stride,
            rho,
        })
    }

    pub(crate) fn round_poly_q(&self, e_in: &DeviceFrVec, e_out: &DeviceFrVec) -> Option<[Fr; 2]> {
        match self {
            Self::Sparse {
                round,
                tables,
                present_mask,
                values,
                source_rows,
                num_polys,
                chunk_domain,
                poly_stride,
                rho,
            } => {
                let ctx = crate::cuda::shared_ctx()?;
                ctx.core_booleanity_sparse_round_poly(crate::cuda::CoreBooleanitySparseInputs {
                    tables,
                    present_mask,
                    values,
                    source_rows: *source_rows,
                    rho,
                    e_in,
                    e_out,
                    num_polys: *num_polys,
                    chunk_domain: *chunk_domain,
                    poly_stride: *poly_stride,
                    round: *round,
                })
                .ok()
            }
            Self::Dense(dense) => dense.round_poly_q(e_in, e_out),
        }
    }

    pub(crate) fn bind(&mut self, challenge: Fr) -> Result<(), CudaError> {
        let ctx = crate::cuda::shared_ctx().ok_or(CudaError::Pool)?;
        match self {
            Self::Sparse {
                round,
                tables,
                present_mask,
                values,
                source_rows,
                num_polys,
                chunk_domain,
                poly_stride,
                rho,
            } => {
                let num_sets = 1usize << (*round - 1);
                let set_elems = *num_polys * *chunk_domain;
                let bound =
                    ctx.core_booleanity_sparse_bind(tables, num_sets, set_elems, challenge)?;
                if *round < 3 {
                    *tables = bound;
                    *round += 1;
                    Ok(())
                } else {
                    let out_len = *source_rows >> 3;
                    let h = ctx.core_booleanity_sparse_collapse8(
                        &bound,
                        present_mask,
                        values,
                        *num_polys,
                        *chunk_domain,
                        *poly_stride,
                        out_len,
                    )?;
                    let dense = CudaCoreBooleanityState::from_device_h(h, rho.clone())
                        .ok_or(CudaError::Pool)?;
                    *self = Self::Dense(dense);
                    Ok(())
                }
            }
            Self::Dense(dense) => dense.bind(challenge),
        }
    }

    pub(crate) fn poly_first(&self, index: usize) -> Option<Result<Fr, CudaError>> {
        match self {
            Self::Sparse { .. } => None,
            Self::Dense(dense) => dense.poly_first(index),
        }
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
        e_in: &DeviceFrVec,
        e_out: &DeviceFrVec,
        m: usize,
    ) -> Option<[Fr; 2]> {
        let ctx = crate::cuda::shared_ctx()?;
        let f_dev = crate::cuda::as_fr_slice(f_values)?;
        let g_refs: Vec<&DeviceFrVec> = self.g.iter().collect();
        ctx.core_booleanity_address_round_poly(CoreBooleanityAddressInputs {
            g: &g_refs,
            f_values: f_dev,
            gamma_squares: &self.gamma_squares,
            e_in,
            e_out,
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
    pub(crate) fn from_device<F: Field>(
        eq_ram: DeviceFrVec,
        ram_inc: &[F],
        eq_rd: DeviceFrVec,
        rd_inc: &[F],
        gamma2: F,
    ) -> Option<Self> {
        let ctx = crate::cuda::shared_ctx()?;
        Some(Self {
            eq_ram,
            ram_inc: ctx.upload(crate::cuda::as_fr_slice(ram_inc)?).ok()?,
            eq_rd,
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
