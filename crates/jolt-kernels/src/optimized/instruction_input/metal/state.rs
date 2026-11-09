use super::*;

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(super) struct ResidentInput {
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub(super) gpu: Arc<Gpu>,
    pub(super) columns: Columns,
    pub(super) flags: Option<FlagTable<F>>,
    pub(super) prefix: usize,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub(super) tails: [F; NUM_TABLES],
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub(super) gamma: F,
}

impl InputDevice<F> for ResidentInput {
    fn bind_round(
        &mut self,
        challenge: F,
        eq: &GruenSplitEqPolynomial<F>,
    ) -> Result<InputStep<F>, SumcheckError<F>> {
        let span = tracing::info_span!(
            "metal_input_bind_round",
            prefix = self.prefix,
            bits = self.flags.as_ref().map_or(0, |flags| flags.bits),
            gpu_s = tracing::field::Empty
        );
        let _span = span.enter();
        let old_bits = self.flags.as_ref().map_or(0, |flags| flags.bits);
        let tail_word = self
            .flags
            .as_ref()
            .map_or([0; 4], |flags| words(flags.tail));
        let new_bits = if old_bits == 0 || old_bits == FlagTable::<F>::SATURATED_BITS {
            0
        } else {
            old_bits * 2
        };
        let next_prefix = self.prefix.div_ceil(2);
        let lut_update = tracing::info_span!("metal_input_flag_lut", bits = new_bits).entered();
        if new_bits != 0 {
            if let Some(flags) = &mut self.flags {
                flags.bind(challenge);
            }
        }
        let lut = self
            .flags
            .as_ref()
            .map(|flags| DeviceBuffer::from_slice(&self.gpu.device, &flags.values))
            .transpose()
            .map_err(compute_error)?;
        drop(lut_update);
        let equality = Equality::new(&self.gpu.device, eq).map_err(compute_error)?;
        let pairs = next_prefix.div_ceil(2);
        let groups = equality.groups(0, pairs).map_err(compute_error)?;
        let allocate = tracing::info_span!("metal_input_allocate", rows = next_prefix).entered();
        let pending = Pending::new(&self.gpu, next_prefix, new_bits != 0).map_err(compute_error)?;
        let partials = partials(groups, &self.gpu.device).map_err(compute_error)?;
        drop(allocate);
        let constants = Constants::new(self.tails, tail_word, challenge, self.gamma);
        let params = Params {
            prefix: word(self.prefix).map_err(compute_error)?,
            output_len: word(next_prefix).map_err(compute_error)?,
            first_pair: 0,
            pairs: word(pairs).map_err(compute_error)?,
            in_len: word(equality.inner_len).map_err(compute_error)?,
            old_bits,
            new_bits,
            need_one: equality.need_one,
        };
        let mut batch = Batch::new(&self.gpu.device).map_err(compute_error)?;
        // SAFETY: immutable old columns and distinct complete output storage.
        // Every input row is checked against prefix; missing rows use tails.
        // Patterns stay within80 bits. Dense modes never read the typed unused
        // word/LUT placeholders. All SIMD lanes participate in reductions.
        unsafe {
            batch
                .dispatch_unchecked(
                    self.gpu
                        .library
                        .pipeline(&host_name::<F>(BIND_ROUND))
                        .map_err(compute_error)?,
                    &[
                        Binding::buffer(&self.columns.values),
                        Binding::buffer(
                            self.columns
                                .words
                                .as_ref()
                                .unwrap_or(&self.gpu.unused_words),
                        ),
                        Binding::buffer(lut.as_ref().unwrap_or(&self.gpu.unused_lut)),
                        Binding::buffer(&equality.inner),
                        Binding::buffer(&equality.outer),
                        Binding::buffer(&pending.values),
                        pending.words_binding(&self.gpu),
                        Binding::buffer(&partials),
                        Binding::value(&constants),
                        Binding::value(&params),
                    ],
                    Grid::linear(groups * THREADS, THREADS),
                )
                .map_err(compute_error)?;
        }
        let gpu_time = batch.commit_and_wait().map_err(compute_error)?;
        let _ = span.record("gpu_s", gpu_time.as_secs_f64());
        // SAFETY: all output rows and partial lanes were written by the
        // completed dispatch, including canonical field representations.
        let (columns, terms) = unsafe { (pending.complete(), sum_partials(partials)) };
        self.columns = columns.map_err(compute_error)?;
        let terms = terms.map_err(compute_error)?;
        self.prefix = next_prefix;
        if new_bits == 0 {
            self.flags = None;
        }
        let dense = if self.flags.is_none() && next_prefix <= self.gpu.cpu_tail {
            let _restore = tracing::info_span!("metal_input_restore", rows = next_prefix).entered();
            let values = self.columns.values.read().map_err(compute_error)?;
            Some(
                values
                    .chunks_exact(next_prefix)
                    .map(<[F]>::to_vec)
                    .collect(),
            )
        } else {
            None
        };
        Ok(InputStep { terms, dense })
    }
}
