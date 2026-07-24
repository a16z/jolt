//! RAM virtual polynomials and memory-state reconstruction.

use super::*;

impl<T: TraceSource + Clone> TraceBackend<'_, T> {
    pub(crate) fn materialize_ram_read_write_virtual<F: Field>(
        &self,
        id: JoltVirtualPolynomial,
    ) -> Result<Vec<F>, WitnessError> {
        match id {
            JoltVirtualPolynomial::RamVal => self.materialize_ram_val(),
            JoltVirtualPolynomial::RamRa => self.materialize_ram_ra(),
            _ => Err(WitnessError::UnknownOracle {
                label: JOLT_VM_LABEL,
            }),
        }
    }

    pub(crate) fn materialize_ram_val<F: Field>(&self) -> Result<Vec<F>, WitnessError> {
        let cycles = checked_pow2(self.config.log_t)?;
        let addresses = self.config.ram_k;
        let mut state = self.initial_ram_state()?;
        let mut values = vec![F::zero(); addresses * cycles];
        let mut trace = self.trace.trace.clone();

        for cycle in 0..cycles {
            for (address, value) in state.iter().copied().enumerate() {
                values[address * cycles + cycle] = F::from_u64(value);
            }

            let Some(row) = trace.next_row() else {
                continue;
            };
            match row.ram_access {
                RamAccess::Read(read) => {
                    if let Some(address) = self.remapped_ram_address(read.address)? {
                        values[address * cycles + cycle] = F::from_u64(read.value);
                    }
                }
                RamAccess::Write(write) => {
                    if let Some(address) = self.remapped_ram_address(write.address)? {
                        values[address * cycles + cycle] = F::from_u64(write.pre_value);
                        state[address] = write.post_value;
                    }
                }
                RamAccess::NoOp => {}
            }
        }

        Ok(values)
    }

    pub(crate) fn materialize_ram_ra<F: Field>(&self) -> Result<Vec<F>, WitnessError> {
        let cycles = checked_pow2(self.config.log_t)?;
        let addresses = self.config.ram_k;
        let mut values = vec![F::zero(); addresses * cycles];
        let mut trace = self.trace.trace.clone();

        for cycle in 0..cycles {
            let Some(row) = trace.next_row() else {
                continue;
            };
            if let Some(raw_address) = ram_access_address(row.ram_access) {
                if let Some(address) = self.remapped_ram_address(raw_address)? {
                    values[address * cycles + cycle] = F::one();
                }
            }
        }

        Ok(values)
    }

    pub(crate) fn materialize_ram_val_final<F: Field>(&self) -> Result<Vec<F>, WitnessError> {
        self.final_ram_state()
            .map(|state| state.into_iter().map(F::from_u64).collect())
    }

    pub(crate) fn initial_ram_state(&self) -> Result<Vec<u64>, WitnessError> {
        if self.config.ram_k == 0 {
            return Err(WitnessError::InvalidDimensions {
                label: JOLT_VM_LABEL,
                reason: "ram_k must be nonzero".to_owned(),
            });
        }
        let mut state = vec![0; self.config.ram_k];
        if !self.preprocessing.ram.bytecode_words.is_empty() {
            let start = self.remapped_required_address(
                self.preprocessing.ram.min_bytecode_address,
                "bytecode",
            )?;
            populate_ram_words(
                &mut state,
                start,
                &self.preprocessing.ram.bytecode_words,
                "bytecode",
            )?;
        }
        if !self.trace.device.trusted_advice.is_empty() {
            let start = self.remapped_required_address(
                self.trace.device.memory_layout.trusted_advice_start,
                "trusted advice",
            )?;
            populate_ram_bytes(
                &mut state,
                start,
                &self.trace.device.trusted_advice,
                "trusted advice",
            )?;
        }
        if !self.trace.device.untrusted_advice.is_empty() {
            let start = self.remapped_required_address(
                self.trace.device.memory_layout.untrusted_advice_start,
                "untrusted advice",
            )?;
            populate_ram_bytes(
                &mut state,
                start,
                &self.trace.device.untrusted_advice,
                "untrusted advice",
            )?;
        }
        if !self.trace.device.inputs.is_empty() {
            let start = self
                .remapped_required_address(self.trace.device.memory_layout.input_start, "input")?;
            populate_ram_bytes(&mut state, start, &self.trace.device.inputs, "input")?;
        }
        Ok(state)
    }

    pub(crate) fn final_ram_state(&self) -> Result<Vec<u64>, WitnessError> {
        if self.config.ram_k == 0 {
            return Err(WitnessError::InvalidDimensions {
                label: JOLT_VM_LABEL,
                reason: "ram_k must be nonzero".to_owned(),
            });
        }
        let mut state = vec![0; self.config.ram_k];
        if let Some(final_memory) = &self.trace.final_memory {
            self.populate_final_memory_image(&mut state, &final_memory.bytes)?;
        }
        if !self.trace.device.trusted_advice.is_empty() {
            let start = self.remapped_required_address(
                self.trace.device.memory_layout.trusted_advice_start,
                "trusted advice",
            )?;
            populate_ram_bytes(
                &mut state,
                start,
                &self.trace.device.trusted_advice,
                "trusted advice",
            )?;
        }
        if !self.trace.device.untrusted_advice.is_empty() {
            let start = self.remapped_required_address(
                self.trace.device.memory_layout.untrusted_advice_start,
                "untrusted advice",
            )?;
            populate_ram_bytes(
                &mut state,
                start,
                &self.trace.device.untrusted_advice,
                "untrusted advice",
            )?;
        }
        if !self.trace.device.inputs.is_empty() {
            let start = self
                .remapped_required_address(self.trace.device.memory_layout.input_start, "input")?;
            populate_ram_bytes(&mut state, start, &self.trace.device.inputs, "input")?;
        }
        if !self.trace.device.outputs.is_empty() {
            let start = self.remapped_required_address(
                self.trace.device.memory_layout.output_start,
                "output",
            )?;
            populate_ram_bytes(&mut state, start, &self.trace.device.outputs, "output")?;
        }

        let panic_index =
            self.remapped_required_address(self.trace.device.memory_layout.panic, "panic")?;
        set_ram_word(
            &mut state,
            panic_index,
            self.trace.device.panic as u64,
            "panic",
        )?;
        if !self.trace.device.panic {
            let termination_index = self.remapped_required_address(
                self.trace.device.memory_layout.termination,
                "termination",
            )?;
            set_ram_word(&mut state, termination_index, 1, "termination")?;
        }
        Ok(state)
    }

    pub(crate) fn populate_final_memory_image(
        &self,
        state: &mut [u64],
        bytes: &[(u64, u8)],
    ) -> Result<(), WitnessError> {
        let dram_start = self.dram_start_address()?;
        for &(address, byte) in bytes {
            let absolute_address = if address >= dram_start {
                address
            } else {
                dram_start
                    .checked_add(address)
                    .ok_or_else(|| WitnessError::InvalidWitnessData {
                        label: JOLT_VM_LABEL,
                        reason: format!("final memory address offset {address:#x} overflows"),
                    })?
            };
            let Some(word_index) = self.remapped_ram_address(absolute_address)? else {
                continue;
            };
            if word_index >= state.len() {
                return Err(WitnessError::InvalidWitnessData {
                    label: JOLT_VM_LABEL,
                    reason: format!(
                        "final memory address {absolute_address:#x} remapped to {word_index}, beyond ram_k {}",
                        state.len()
                    ),
                });
            }
            let shift = ((absolute_address & 7) * 8) as usize;
            state[word_index] =
                (state[word_index] & !(0xff_u64 << shift)) | ((byte as u64) << shift);
        }
        Ok(())
    }

    pub(crate) fn dram_start_address(&self) -> Result<u64, WitnessError> {
        self.preprocessing
            .memory_layout
            .stack_end
            .checked_sub(self.preprocessing.memory_layout.program_size)
            .ok_or_else(|| WitnessError::InvalidWitnessData {
                label: JOLT_VM_LABEL,
                reason: "memory layout stack_end is below program_size".to_owned(),
            })
    }

    pub(crate) fn remapped_required_address(
        &self,
        address: u64,
        label: &'static str,
    ) -> Result<usize, WitnessError> {
        self.preprocessing
            .memory_layout
            .remapped_word_address(address)
            .map(|address| address as usize)
            .map_err(|error| WitnessError::InvalidWitnessData {
                label: JOLT_VM_LABEL,
                reason: format!("failed to remap {label} address {address:#x}: {error}"),
            })
    }

    pub(crate) fn remapped_ram_address(&self, address: u64) -> Result<Option<usize>, WitnessError> {
        let remapped = self
            .preprocessing
            .memory_layout
            .remap_word_address(address)
            .map_err(|error| WitnessError::InvalidWitnessData {
                label: JOLT_VM_LABEL,
                reason: format!("failed to remap RAM access address {address:#x}: {error}"),
            })?;
        let Some(address) = remapped else {
            return Ok(None);
        };
        let address = address as usize;
        if address >= self.config.ram_k {
            return Err(WitnessError::InvalidWitnessData {
                label: JOLT_VM_LABEL,
                reason: format!(
                    "RAM access address remapped to {address}, beyond ram_k {}",
                    self.config.ram_k
                ),
            });
        }
        Ok(Some(address))
    }
}

pub(crate) fn populate_ram_bytes(
    state: &mut [u64],
    start: usize,
    bytes: &[u8],
    label: &'static str,
) -> Result<(), WitnessError> {
    let words = bytes
        .chunks(8)
        .map(|chunk| {
            let mut word = [0; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            u64::from_le_bytes(word)
        })
        .collect::<Vec<_>>();
    populate_ram_words(state, start, &words, label)
}

pub(crate) fn set_ram_word(
    state: &mut [u64],
    index: usize,
    word: u64,
    label: &'static str,
) -> Result<(), WitnessError> {
    let Some(slot) = state.get_mut(index) else {
        return Err(WitnessError::InvalidWitnessData {
            label: JOLT_VM_LABEL,
            reason: format!("{label} memory index {index} exceeds ram_k {}", state.len()),
        });
    };
    *slot = word;
    Ok(())
}

pub(crate) fn populate_ram_words(
    state: &mut [u64],
    start: usize,
    words: &[u64],
    label: &'static str,
) -> Result<(), WitnessError> {
    let end = start
        .checked_add(words.len())
        .ok_or_else(|| WitnessError::InvalidWitnessData {
            label: JOLT_VM_LABEL,
            reason: format!("{label} memory range overflows"),
        })?;
    if end > state.len() {
        return Err(WitnessError::InvalidWitnessData {
            label: JOLT_VM_LABEL,
            reason: format!(
                "{label} memory range [{start}, {end}) exceeds ram_k {}",
                state.len()
            ),
        });
    }
    state[start..end].copy_from_slice(words);
    Ok(())
}
