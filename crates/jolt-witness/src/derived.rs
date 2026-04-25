//! On-demand computation of derived polynomials.
//!
//! [`DerivedSource`] materializes [`PolySource::Derived`] polynomials from
//! the per-cycle R1CS witness vector. These are polynomials that require
//! non-trivial reshaping or combination of witness data — neither direct
//! witness columns nor R1CS matrix-vector products.
//!
//! Two sources of data:
//! - **Computed from witness**: ProductLeft/ProductRight — domain-indexed
//!   product factors extracted from the per-cycle R1CS witness.
//! - **RAM polynomials**: RamCombinedRa, RamVal, RamValFinal, RamRaIndicator —
//!   computed from R1CS witness columns + initial/final RAM state arrays.

use std::borrow::Cow;

use jolt_compiler::PolynomialId;
use jolt_field::Field;
use jolt_r1cs::constraints::rv64::*;

/// Maps a [`PolynomialId`] to its R1CS witness variable index, or `None`
/// if the polynomial is not a direct column of the per-cycle witness.
fn witness_var(poly_id: PolynomialId) -> Option<usize> {
    // First try the compiler's canonical mapping (covers R1CS inputs + OpFlags).
    if let Some(idx) = poly_id.r1cs_variable_index() {
        return Some(idx);
    }
    // Product factor variables live outside the R1CS input range.
    match poly_id {
        PolynomialId::BranchFlag => Some(V_BRANCH),
        PolynomialId::NextIsNoop => Some(V_NEXT_IS_NOOP),
        _ => None,
    }
}

/// Product constraint stride: next power of two of 3 product constraints.
const PRODUCT_STRIDE: usize = NUM_PRODUCT_CONSTRAINTS.next_power_of_two(); // 4

/// A-row variable indices for each product constraint (k = 0, 1, 2).
///
/// Maps directly to the R1CS product constraint definitions:
///   k=0: Product = LeftInstructionInput × RightInstructionInput
///   k=1: ShouldBranch = LookupOutput × Branch
///   k=2: ShouldJump = Jump × (1 − NextIsNoop)
const PRODUCT_A_VARS: [usize; NUM_PRODUCT_CONSTRAINTS] =
    [V_LEFT_INSTRUCTION_INPUT, V_LOOKUP_OUTPUT, V_FLAG_JUMP];

/// RAM configuration needed to build K×T polynomials from R1CS witness data.
pub struct RamConfig {
    /// Number of RAM addresses (K, power of two).
    pub ram_k: usize,
    /// Lowest byte address in memory layout, used to remap raw addresses:
    /// `k = (raw_addr - lowest_addr) / 8`.
    pub lowest_addr: u64,
    /// Initial RAM state: K elements (u64 words, address-indexed).
    pub initial_state: Vec<u64>,
    /// Final RAM state: K elements (u64 words, address-indexed).
    pub final_state: Vec<u64>,
}

/// Per-cycle register access data extracted from the execution trace.
///
/// Each vector has `trace_length` entries. `None` = no access that cycle.
pub struct RegisterAccessData {
    /// rd register index (0..127) when a write occurs.
    pub rd_indices: Vec<Option<usize>>,
    /// rs1 register index (0..127) when a read occurs.
    pub rs1_indices: Vec<Option<usize>>,
    /// rs2 register index (0..127) when a read occurs.
    pub rs2_indices: Vec<Option<usize>>,
}

/// Pre-computed instruction flag polynomials (from trace extraction).
pub struct InstructionFlags<F> {
    pub is_noop: Vec<F>,
    pub left_is_rs1: Vec<F>,
    pub left_is_pc: Vec<F>,
    pub right_is_rs2: Vec<F>,
    pub right_is_imm: Vec<F>,
}

/// Pre-computed per-cycle lookup table indices (in LookupTables enum order).
pub struct LookupFlagData {
    /// Per-cycle table index in `LookupTables<64>` ordering. None = no lookup.
    pub table_indices: Vec<Option<usize>>,
    /// Per-cycle RAF flag: true if instruction uses identity (non-interleaved) operands.
    pub is_raf: Vec<bool>,
}

/// Per-cycle BN254 Fr coprocessor access metadata — enough to drive
/// `replay_field_regs` and the FR Twist materializers.
#[derive(Clone, Debug, Default)]
pub struct FieldRegConfig {
    /// Per-cycle bytecode snapshot (frs1/frs2/reads_frs2). Length = T.
    pub bytecode: Vec<crate::field_reg::FrCycleBytecode>,
    /// FieldRegEvent stream emitted by the tracer. Sorted by cycle.
    pub events: Vec<crate::field_reg::FieldRegEvent>,
}

pub struct DerivedSource<'a, F> {
    witness: &'a [F],
    num_cycles: usize,
    vars_padded: usize,
    ram: Option<RamConfig>,
    iflags: Option<InstructionFlags<F>>,
    reg_access: Option<RegisterAccessData>,
    lookup_flags: Option<LookupFlagData>,
    field_reg: Option<FieldRegConfig>,
    _marker: std::marker::PhantomData<F>,
}

impl<'a, F: Field> DerivedSource<'a, F> {
    pub fn new(witness: &'a [F], num_cycles: usize, vars_padded: usize) -> Self {
        Self {
            witness,
            num_cycles,
            vars_padded,
            ram: None,
            iflags: None,
            reg_access: None,
            lookup_flags: None,
            field_reg: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Attach BN254 Fr coprocessor replay data. When absent, FR-derived
    /// polys materialize to all-zero (correct for traces with no FR cycles).
    pub fn with_field_reg(mut self, config: FieldRegConfig) -> Self {
        self.field_reg = Some(config);
        self
    }

    /// Attach RAM configuration for building RAM-derived polynomials.
    pub fn with_ram(mut self, config: RamConfig) -> Self {
        self.ram = Some(config);
        self
    }

    /// Attach pre-computed instruction flag polynomials.
    pub fn with_instruction_flags(mut self, flags: InstructionFlags<F>) -> Self {
        self.iflags = Some(flags);
        self
    }

    /// Attach register access data for building K_reg × T register polynomials.
    pub fn with_register_access(mut self, data: RegisterAccessData) -> Self {
        self.reg_access = Some(data);
        self
    }

    /// Attach pre-computed lookup flag data for LookupTableFlag/InstructionRafFlag polys.
    pub fn with_lookup_flags(mut self, data: LookupFlagData) -> Self {
        self.lookup_flags = Some(data);
        self
    }

    fn ram(&self) -> &RamConfig {
        self.ram.as_ref().expect(
            "DerivedSource: RAM polynomial requested but no RamConfig provided. \
             Call .with_ram() before proving.",
        )
    }

    /// Compute or retrieve a derived polynomial by ID.
    pub fn compute(&self, poly_id: PolynomialId) -> Cow<'_, [F]> {
        match poly_id {
            PolynomialId::ProductLeft => {
                let _s = tracing::info_span!("derived::product_left").entered();
                Cow::Owned(self.product_left())
            }
            PolynomialId::ProductRight => {
                let _s = tracing::info_span!("derived::product_right").entered();
                Cow::Owned(self.product_right())
            }
            PolynomialId::RamCombinedRa => {
                let _s = tracing::info_span!("derived::ram_combined_ra").entered();
                Cow::Owned(self.ram_combined_ra())
            }
            PolynomialId::RamVal => {
                let _s = tracing::info_span!("derived::ram_val").entered();
                Cow::Owned(self.ram_val())
            }
            PolynomialId::RamValFinal => {
                let _s = tracing::info_span!("derived::ram_val_final").entered();
                Cow::Owned(self.ram_val_final())
            }
            PolynomialId::RamRaIndicator => {
                let _s = tracing::info_span!("derived::ram_ra_indicator").entered();
                Cow::Owned(self.ram_ra_indicator())
            }
            PolynomialId::Rs1Ra => {
                let _s = tracing::info_span!("derived::reg_rs1_ra").entered();
                Cow::Owned(self.reg_rs1_ra())
            }
            PolynomialId::Rs2Ra => {
                let _s = tracing::info_span!("derived::reg_rs2_ra").entered();
                Cow::Owned(self.reg_rs2_ra())
            }
            PolynomialId::RdWa => {
                let _s = tracing::info_span!("derived::reg_rd_wa").entered();
                Cow::Owned(self.reg_rd_wa())
            }
            PolynomialId::RegistersVal => {
                let _s = tracing::info_span!("derived::reg_val").entered();
                Cow::Owned(self.reg_val())
            }
            // BN254 Fr coprocessor derived polys. When a FieldRegConfig is
            // attached, we replay the event stream once and cache per-cycle
            // snapshots; subsequent FR-poly requests materialize from those
            // snapshots. When no config is attached (traces with no FR
            // cycles: muldiv, sha2, etc.), all FR polys materialize as the
            // zero vector — self-consistent: inc=0 → val=0 → γ-batched
            // claim=0, reducing the FR Twist sumcheck to 0=0 at opening.
            PolynomialId::FieldRegRaRs1 => {
                let _s = tracing::info_span!("derived::field_reg_ra_rs1").entered();
                Cow::Owned(self.field_reg_ra_rs1())
            }
            PolynomialId::FieldRegRaRs2 => {
                let _s = tracing::info_span!("derived::field_reg_ra_rs2").entered();
                Cow::Owned(self.field_reg_ra_rs2())
            }
            PolynomialId::FieldRegWa => {
                let _s = tracing::info_span!("derived::field_reg_wa").entered();
                Cow::Owned(self.field_reg_wa())
            }
            PolynomialId::FieldRegVal => {
                let _s = tracing::info_span!("derived::field_reg_val").entered();
                Cow::Owned(self.field_reg_val())
            }
            PolynomialId::FrdGatherIndex => {
                let _s = tracing::info_span!("derived::frd_gather_index").entered();
                Cow::Owned(self.frd_gather_index())
            }
            PolynomialId::NoopFlag => {
                let _s = tracing::info_span!("derived::iflag_noop").entered();
                let f = self.iflags.as_ref().expect("InstructionFlags not attached");
                Cow::Borrowed(&f.is_noop)
            }
            PolynomialId::LeftIsRs1 => {
                let _s = tracing::info_span!("derived::iflag_left_is_rs1").entered();
                let f = self.iflags.as_ref().expect("InstructionFlags not attached");
                Cow::Borrowed(&f.left_is_rs1)
            }
            PolynomialId::LeftIsPc => {
                let _s = tracing::info_span!("derived::iflag_left_is_pc").entered();
                let f = self.iflags.as_ref().expect("InstructionFlags not attached");
                Cow::Borrowed(&f.left_is_pc)
            }
            PolynomialId::RightIsRs2 => {
                let _s = tracing::info_span!("derived::iflag_right_is_rs2").entered();
                let f = self.iflags.as_ref().expect("InstructionFlags not attached");
                Cow::Borrowed(&f.right_is_rs2)
            }
            PolynomialId::RightIsImm => {
                let _s = tracing::info_span!("derived::iflag_right_is_imm").entered();
                let f = self.iflags.as_ref().expect("InstructionFlags not attached");
                Cow::Borrowed(&f.right_is_imm)
            }
            PolynomialId::RdGatherIndex => {
                let _s = tracing::info_span!("derived::rd_gather_index").entered();
                Cow::Owned(self.rd_gather_index())
            }
            PolynomialId::RamGatherIndex => {
                let _s = tracing::info_span!("derived::ram_gather_index").entered();
                Cow::Owned(self.ram_gather_index())
            }
            PolynomialId::LookupTableFlag(i) => {
                let _s = tracing::info_span!("derived::lookup_table_flag").entered();
                let lf = self.lookup_flags.as_ref().expect(
                    "DerivedSource: LookupTableFlag requested but no LookupFlagData provided",
                );
                Cow::Owned(self.lookup_table_flag(&lf.table_indices, i))
            }
            PolynomialId::InstructionRafFlag => {
                let _s = tracing::info_span!("derived::instruction_raf_flag").entered();
                let lf = self.lookup_flags.as_ref().expect(
                    "DerivedSource: InstructionRafFlag requested but no LookupFlagData provided",
                );
                Cow::Owned(self.instruction_raf_flag(&lf.is_raf))
            }
            PolynomialId::HammingWeight => {
                let _s = tracing::info_span!("derived::hamming_weight").entered();
                Cow::Owned(self.hamming_weight())
            }
            other => {
                let _s = tracing::info_span!("derived::extract_column").entered();
                if let Some(var) = witness_var(other) {
                    Cow::Owned(self.extract_column(var))
                } else {
                    panic!(
                        "DerivedSource: {other:?} is PolySource::Derived but has no compute \
                         method — check that RamConfig is attached if RAM polys are needed"
                    )
                }
            }
        }
    }

    /// Extract a single R1CS variable column as a T-element vector.
    fn extract_column(&self, var: usize) -> Vec<F> {
        (0..self.num_cycles)
            .map(|c| self.witness[c * self.vars_padded + var])
            .collect()
    }

    /// RAM Hamming weight: H[j] = 1 if RAM address at cycle j is nonzero.
    fn hamming_weight(&self) -> Vec<F> {
        (0..self.num_cycles)
            .map(|c| {
                let addr = self.witness[c * self.vars_padded + V_RAM_ADDRESS];
                if addr == F::zero() {
                    F::zero()
                } else {
                    F::one()
                }
            })
            .collect()
    }

    /// Domain-indexed left (A-row) factors: `buf[c * 4 + k]` = A_k(c).
    fn product_left(&self) -> Vec<F> {
        let mut buf = vec![F::zero(); self.num_cycles * PRODUCT_STRIDE];
        for c in 0..self.num_cycles {
            let w = c * self.vars_padded;
            for (k, &var) in PRODUCT_A_VARS.iter().enumerate() {
                buf[c * PRODUCT_STRIDE + k] = self.witness[w + var];
            }
        }
        buf
    }

    /// Domain-indexed right (B-row) factors: `buf[c * 4 + k]` = B_k(c).
    ///
    /// k=2 (ShouldJump) has B = 1 − NextIsNoop, not a single variable.
    fn product_right(&self) -> Vec<F> {
        let mut buf = vec![F::zero(); self.num_cycles * PRODUCT_STRIDE];
        for c in 0..self.num_cycles {
            let w = c * self.vars_padded;
            buf[c * PRODUCT_STRIDE] = self.witness[w + V_RIGHT_INSTRUCTION_INPUT];
            buf[c * PRODUCT_STRIDE + 1] = self.witness[w + V_BRANCH];
            buf[c * PRODUCT_STRIDE + 2] =
                self.witness[w + V_CONST] - self.witness[w + V_NEXT_IS_NOOP];
        }
        buf
    }

    // Built from R1CS witness columns V_RAM_ADDRESS, V_RAM_READ_VALUE,
    // V_FLAG_LOAD, V_FLAG_STORE plus the initial/final RAM state arrays.
    //
    // Layout: address-major K*T -- index(k, t) = k * T + t.
    // This matches jolt-core's ReadWriteMatrixCycleMajor::materialize().

    /// Remap a raw byte address to a dense RAM index k.
    /// Returns `None` for address 0 (no-op cycles).
    fn remap(&self, raw_addr_field: F) -> Option<usize> {
        let raw = raw_addr_field
            .to_u64()
            .expect("RAM address must fit in u64");
        if raw == 0 {
            return None;
        }
        let ram = self.ram();
        Some(((raw - ram.lowest_addr) / 8) as usize)
    }

    /// K×T binary access indicator (address-major).
    ///
    /// `ra[k * T + t] = 1` if cycle t accesses address k, else 0.
    /// Used as both `RamCombinedRa` and `RamRaIndicator` (same data).
    fn ram_combined_ra(&self) -> Vec<F> {
        let ram = self.ram();
        let k = ram.ram_k;
        let t = self.num_cycles;
        let mut ra = vec![F::zero(); k * t];
        for c in 0..t {
            let w = c * self.vars_padded;
            if let Some(col) = self.remap(self.witness[w + V_RAM_ADDRESS]) {
                ra[col * t + c] = F::one();
            }
        }
        ra
    }

    /// K×T value polynomial (address-major).
    ///
    /// Initialized to `val_init[k]` replicated across all T positions per
    /// address k. At access points, overwritten with the pre-read value
    /// (`V_RAM_READ_VALUE` — which is `pre_value` for writes, `read_value`
    /// for reads, matching jolt-core's `val_coeff`).
    /// K×T value polynomial (address-major).
    ///
    /// Matches jolt-core's sparse matrix `val_coeff` semantics:
    /// - At access points: `V_RAM_READ_VALUE` (pre-access value)
    /// - At non-access positions: persisted value from the most recent access
    ///   (or `initial_state[k]` before any access)
    ///
    /// After the last access, remaining positions get the post-access value
    /// (`V_RAM_WRITE_VALUE`), matching jolt-core's `next_val` propagation.
    fn ram_val(&self) -> Vec<F> {
        let ram = self.ram();
        let k = ram.ram_k;
        let t = self.num_cycles;

        // Build per-address access event lists: (cycle, read_val, write_val).
        let mut access_events: Vec<Vec<(usize, F, F)>> = vec![Vec::new(); k];
        for c in 0..t {
            let w = c * self.vars_padded;
            if let Some(col) = self.remap(self.witness[w + V_RAM_ADDRESS]) {
                access_events[col].push((
                    c,
                    self.witness[w + V_RAM_READ_VALUE],
                    self.witness[w + V_RAM_WRITE_VALUE],
                ));
            }
        }

        let mut val = vec![F::zero(); k * t];
        for (addr, events) in access_events.iter().enumerate().take(k) {
            let base = addr * t;
            let mut current = F::from_u64(ram.initial_state[addr]);
            let mut ei = 0; // event index

            for c in 0..t {
                if ei < events.len() && events[ei].0 == c {
                    // Access point: use pre-access value (read_value)
                    val[base + c] = events[ei].1;
                    // Update persisted value to post-access value (write_value)
                    current = events[ei].2;
                    ei += 1;
                } else {
                    // Non-access: persisted value
                    val[base + c] = current;
                }
            }
        }
        val
    }

    /// T×K binary access indicator (cycle-major).
    ///
    /// `indicator[t * K + k] = 1` if cycle t accesses address k.
    /// Used by EqProject which reads `source[t * outer_size + k]`.
    fn ram_ra_indicator(&self) -> Vec<F> {
        let ram = self.ram();
        let k = ram.ram_k;
        let t = self.num_cycles;
        let mut indicator = vec![F::zero(); t * k];
        for c in 0..t {
            let w = c * self.vars_padded;
            if let Some(col) = self.remap(self.witness[w + V_RAM_ADDRESS]) {
                indicator[c * k + col] = F::one();
            }
        }
        indicator
    }

    /// K-element final RAM state as field elements.
    fn ram_val_final(&self) -> Vec<F> {
        let ram = self.ram();
        ram.final_state.iter().map(|&v| F::from_u64(v)).collect()
    }

    // K_reg x T address-major: index(k, t) = k * T + t.
    // K_reg = 128 (REGISTER_COUNT).

    const K_REG: usize = 128;

    fn reg_access(&self) -> &RegisterAccessData {
        self.reg_access.as_ref().expect(
            "DerivedSource: register polynomial requested but no RegisterAccessData provided. \
             Call .with_register_access() before proving.",
        )
    }

    /// K_reg × T binary indicator: rs1_ra(k, t) = 1 if rs1 at cycle t is register k.
    fn reg_rs1_ra(&self) -> Vec<F> {
        let ra = self.reg_access();
        let t = self.num_cycles;
        let mut out = vec![F::zero(); Self::K_REG * t];
        for c in 0..t {
            if let Some(k) = ra.rs1_indices[c] {
                out[k * t + c] = F::one();
            }
        }
        out
    }

    /// K_reg × T binary indicator: rs2_ra(k, t) = 1 if rs2 at cycle t is register k.
    fn reg_rs2_ra(&self) -> Vec<F> {
        let ra = self.reg_access();
        let t = self.num_cycles;
        let mut out = vec![F::zero(); Self::K_REG * t];
        for c in 0..t {
            if let Some(k) = ra.rs2_indices[c] {
                out[k * t + c] = F::one();
            }
        }
        out
    }

    /// K_reg × T binary indicator: rd_wa(k, t) = 1 if rd at cycle t is register k.
    fn reg_rd_wa(&self) -> Vec<F> {
        let ra = self.reg_access();
        let t = self.num_cycles;
        let mut out = vec![F::zero(); Self::K_REG * t];
        for c in 0..t {
            if let Some(k) = ra.rd_indices[c] {
                out[k * t + c] = F::one();
            }
        }
        out
    }

    /// Per-cycle register destination index for EqGather.
    ///
    /// Returns T field elements: `F::from_u64(rd[j])` if cycle j writes a
    /// register, or a sentinel (u64::MAX) if no write occurs. The sentinel
    /// is outside any valid eq table range, producing zero in the gather.
    fn rd_gather_index(&self) -> Vec<F> {
        let ra = self.reg_access();
        let sentinel = F::from_u64(u64::MAX);
        ra.rd_indices
            .iter()
            .map(|idx| match idx {
                Some(k) => F::from_u64(*k as u64),
                None => sentinel,
            })
            .collect()
    }

    /// Per-cycle RAM address index for EqGather.
    ///
    /// Returns T field elements: `F::from_u64(addr_index)` if cycle j
    /// accesses RAM, or a sentinel if no RAM access. The index is
    /// `(raw_addr - lowest_addr) / 8`.
    fn ram_gather_index(&self) -> Vec<F> {
        let sentinel = F::from_u64(u64::MAX);
        let t = self.num_cycles;
        let mut out = Vec::with_capacity(t);
        for c in 0..t {
            let w = c * self.vars_padded;
            match self.remap(self.witness[w + V_RAM_ADDRESS]) {
                Some(k) => out.push(F::from_u64(k as u64)),
                None => out.push(sentinel),
            }
        }
        out
    }

    /// T-element flag polynomial: 1 if cycle uses lookup table `table_idx`.
    fn lookup_table_flag(&self, table_indices: &[Option<usize>], table_idx: usize) -> Vec<F> {
        (0..self.num_cycles)
            .map(|j| {
                if j < table_indices.len() && table_indices[j] == Some(table_idx) {
                    F::one()
                } else {
                    F::zero()
                }
            })
            .collect()
    }

    /// T-element RAF flag polynomial: 1 if cycle uses identity (non-interleaved) operands.
    fn instruction_raf_flag(&self, is_raf: &[bool]) -> Vec<F> {
        (0..self.num_cycles)
            .map(|j| {
                if j < is_raf.len() && is_raf[j] {
                    F::one()
                } else {
                    F::zero()
                }
            })
            .collect()
    }

    /// BN254 Fr coprocessor slot count. See `specs/bn254-fr-coprocessor.md`
    /// §ISA: the low 4 bits of each 5-bit register field index
    /// `field_regs[0..=15]`.
    const K_FR: usize = 16;

    /// K_FR × T all-zero vector — used as the materializer fallback when no
    /// FieldRegConfig is attached.
    fn field_reg_zero_kxt(&self) -> Vec<F> {
        vec![F::zero(); Self::K_FR * self.num_cycles]
    }

    /// Lazily compute per-cycle FR snapshots if a config is attached.
    fn fr_snapshots(&self) -> Option<Vec<crate::field_reg::FrCycleData>> {
        let cfg = self.field_reg.as_ref()?;
        debug_assert_eq!(
            cfg.bytecode.len(),
            self.num_cycles,
            "FieldRegConfig.bytecode length must match num_cycles"
        );
        Some(crate::field_reg::replay_field_regs(
            self.num_cycles,
            &cfg.bytecode,
            &cfg.events,
        ))
    }

    /// K_FR × T one-hot indicator: 1 at `(frs1(t), t)` on cycles where an
    /// FR instruction reads `frs1`, 0 elsewhere.
    fn field_reg_ra_rs1(&self) -> Vec<F> {
        let Some(cfg) = self.field_reg.as_ref() else {
            return self.field_reg_zero_kxt();
        };
        let t = self.num_cycles;
        let mut out = vec![F::zero(); Self::K_FR * t];
        for (c, bc) in cfg.bytecode.iter().enumerate().take(t) {
            if bc.reads_frs1 {
                let slot = (bc.frs1 as usize) & 0xF;
                out[slot * t + c] = F::one();
            }
        }
        out
    }

    /// K_FR × T one-hot for `frs2` reads.
    fn field_reg_ra_rs2(&self) -> Vec<F> {
        let Some(cfg) = self.field_reg.as_ref() else {
            return self.field_reg_zero_kxt();
        };
        let t = self.num_cycles;
        let mut out = vec![F::zero(); Self::K_FR * t];
        for (c, bc) in cfg.bytecode.iter().enumerate().take(t) {
            if bc.reads_frs2 {
                let slot = (bc.frs2 as usize) & 0xF;
                out[slot * t + c] = F::one();
            }
        }
        out
    }

    /// K_FR × T one-hot write-address indicator. For writing FR cycles this
    /// is 1 at `(frd(t), t)` and 0 elsewhere. Sourced from the event
    /// stream so it matches what `FieldRegInc` commits (single write per
    /// cycle).
    fn field_reg_wa(&self) -> Vec<F> {
        let Some(snaps) = self.fr_snapshots() else {
            return self.field_reg_zero_kxt();
        };
        let t = self.num_cycles;
        let mut out = vec![F::zero(); Self::K_FR * t];
        for (c, snap) in snaps.iter().enumerate() {
            if let Some(slot) = snap.write_slot {
                out[(slot as usize) * t + c] = F::one();
            }
        }
        out
    }

    /// K_FR × T running FR register-file state — `val(k, t)` is the value
    /// of `field_regs[k]` at the START of cycle t (pre-execution). Mirrors
    /// integer `reg_val`. Values are 256-bit Fr elements encoded from the
    /// tracer's natural-form `[u64;4]` limbs.
    fn field_reg_val(&self) -> Vec<F> {
        let Some(cfg) = self.field_reg.as_ref() else {
            return self.field_reg_zero_kxt();
        };
        let t = self.num_cycles;
        let mut out = vec![F::zero(); Self::K_FR * t];

        // Running state: 16 slots, each an Fr element. Index by slot.
        let mut current: [F; 16] = [F::zero(); 16];
        let mut event_iter = cfg.events.iter().peekable();

        for c in 0..t {
            // Record pre-execution state of all slots at cycle c.
            for (k, val) in current.iter().enumerate().take(Self::K_FR) {
                out[k * t + c] = *val;
            }

            // Apply the event at cycle c (if any) to advance state.
            if let Some(ev) = event_iter.peek() {
                if ev.cycle == c {
                    let ev = *event_iter.next().unwrap();
                    let slot = (ev.slot as usize) & 0xF;
                    current[slot] = limbs_to_field::<F>(ev.new);
                }
            }
        }
        out
    }

    /// T-element per-cycle FR write-slot gather index. Non-writing cycles
    /// use sentinel `u64::MAX` (outside any valid eq range) so
    /// `eq_gather` returns 0; writing cycles encode the slot as a field
    /// element.
    fn frd_gather_index(&self) -> Vec<F> {
        let sentinel = F::from_u64(u64::MAX);
        let Some(snaps) = self.fr_snapshots() else {
            return vec![sentinel; self.num_cycles];
        };
        snaps
            .iter()
            .map(|s| match s.write_slot {
                Some(k) => F::from_u64(k as u64),
                None => sentinel,
            })
            .collect()
    }

    /// K_reg × T register value polynomial (address-major).
    ///
    /// val(k, t) = value of register k just before cycle t.
    /// Initialized to 0 (all registers start at 0). At write points,
    /// the value is updated to the post-write value for subsequent cycles.
    fn reg_val(&self) -> Vec<F> {
        let ra = self.reg_access();
        let t = self.num_cycles;
        let mut out = vec![F::zero(); Self::K_REG * t];

        // Track current register values. All registers start at 0.
        #[allow(clippy::useless_vec)]
        let mut current_vals = vec![F::zero(); Self::K_REG];

        for c in 0..t {
            let w = c * self.vars_padded;

            // Write the current (pre-access) value for ALL registers at cycle c.
            for k in 0..Self::K_REG {
                out[k * t + c] = current_vals[k];
            }

            // If rd is written this cycle, update the running value.
            if let Some(k) = ra.rd_indices[c] {
                // V_RD_WRITE_VALUE is the POST-write value.
                current_vals[k] = self.witness[w + V_RD_WRITE_VALUE];
            }
        }
        out
    }
}

/// Convert a natural-form `[u64; 4]` limb array to an Fr field element:
/// `a[0] + a[1]·2⁶⁴ + a[2]·2¹²⁸ + a[3]·2¹⁹²`. Used by `field_reg_val` to
/// encode the running FR register state.
fn limbs_to_field<F: Field>(limbs: [u64; 4]) -> F {
    let lo = F::from_u128((limbs[0] as u128) | ((limbs[1] as u128) << 64));
    let hi = F::from_u128((limbs[2] as u128) | ((limbs[3] as u128) << 64));
    lo + hi * F::one().mul_pow_2(128)
}
