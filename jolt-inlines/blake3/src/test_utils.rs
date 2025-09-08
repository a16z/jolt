use jolt_inlines_common::constants::{blake3, INLINE_OPCODE};
use tracer::emulator::mmu::DRAM_BASE;
use tracer::instruction::format::format_inline::FormatInline;
use tracer::instruction::inline::INLINE;
use tracer::utils::test_harness::CpuTestHarness;

pub type ChainingValue = [u32; crate::CHAINING_VALUE_LEN];
pub type MessageBlock = [u32; crate::MSG_BLOCK_LEN];

pub struct Blake3CpuHarness {
    pub harness: CpuTestHarness,
}

impl Blake3CpuHarness {
    /// Memory layout constants (all using 32-bit words)
    const CHAINING_VALUE_ADDR: u64 = DRAM_BASE;
    const MESSAGE_ADDR: u64 = DRAM_BASE + (crate::CHAINING_VALUE_LEN * 4) as u64; // 8 * 4 bytes
    const COUNTER_ADDR: u64 = Self::MESSAGE_ADDR + (crate::MSG_BLOCK_LEN * 4) as u64; // After message block
    const BLOCK_LEN_ADDR: u64 = Self::COUNTER_ADDR + 8; // Counter is 2 u32s = 8 bytes
    const FLAGS_ADDR: u64 = Self::BLOCK_LEN_ADDR + 4; // Block len is 1 u32 = 4 bytes

    pub const RS1: u8 = 10; // Points to chaining value
    pub const RS2: u8 = 11; // Points to message block + counter + block_len + flags

    pub fn new() -> Self {
        Self {
            harness: CpuTestHarness::new(),
        }
    }

    pub fn load_blake3_data(
        &mut self,
        chaining_value: &ChainingValue,
        message: &MessageBlock,
        counter: &[u32; 2],
        block_len: u32,
        flags: u32,
    ) {
        // Set up memory pointers in registers
        self.harness.cpu.x[Self::RS1 as usize] = Self::CHAINING_VALUE_ADDR as i64;
        self.harness.cpu.x[Self::RS2 as usize] = Self::MESSAGE_ADDR as i64;

        // Load chaining value into memory (as u32 words)
        for (i, &word) in chaining_value.iter().enumerate() {
            self.harness
                .cpu
                .mmu
                .store_word(Self::CHAINING_VALUE_ADDR.wrapping_add((i * 4) as u64), word)
                .expect("Failed to store chaining value to memory");
        }

        // Load message block into memory (as u32 words)
        for (i, &word) in message.iter().enumerate() {
            self.harness
                .cpu
                .mmu
                .store_word(Self::MESSAGE_ADDR.wrapping_add((i * 4) as u64), word)
                .expect("Failed to store message to memory");
        }

        // Load counter (2 u32 words)
        self.harness
            .cpu
            .mmu
            .store_word(Self::COUNTER_ADDR, counter[0])
            .expect("Failed to store counter[0] to memory");
        self.harness
            .cpu
            .mmu
            .store_word(Self::COUNTER_ADDR.wrapping_add(4), counter[1])
            .expect("Failed to store counter[1] to memory");

        // Load block length
        self.harness
            .cpu
            .mmu
            .store_word(Self::BLOCK_LEN_ADDR, block_len)
            .expect("Failed to store block_len to memory");

        // Load flags
        self.harness
            .cpu
            .mmu
            .store_word(Self::FLAGS_ADDR, flags)
            .expect("Failed to store flags to memory");
    }

    pub fn read_chaining_value(&mut self) -> ChainingValue {
        let mut chaining_value: ChainingValue = [0u32; crate::CHAINING_VALUE_LEN];
        for (i, word) in chaining_value.iter_mut().enumerate() {
            *word = self
                .harness
                .cpu
                .mmu
                .load_word(Self::CHAINING_VALUE_ADDR.wrapping_add((i * 4) as u64))
                .expect("Failed to load chaining value from memory")
                .0;
        }
        chaining_value
    }

    pub fn instruction() -> INLINE {
        INLINE {
            address: 0,
            operands: FormatInline {
                rs1: Self::RS1,
                rs2: Self::RS2,
                rs3: 0,
            },
            opcode: INLINE_OPCODE,
            funct3: blake3::FUNCT3,
            funct7: blake3::FUNCT7,
            inline_sequence_remaining: None,
            is_compressed: false,
        }
    }
}

impl Default for Blake3CpuHarness {
    fn default() -> Self {
        Self::new()
    }
}

pub mod helpers {
    pub fn generate_random_bytes(len: usize) -> Vec<u8> {
        let mut buf = vec![0u8; len];
        let mut rng = rand::thread_rng();
        use rand::RngCore;
        rng.fill_bytes(&mut buf);
        buf
    }

    pub fn bytes_to_u32_vec(bytes: &[u8]) -> Vec<u32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    pub fn compute_expected_result(input: &[u8]) -> [u8; crate::OUTPUT_SIZE_IN_BYTES] {
        blake3::hash(input).as_bytes()[0..crate::OUTPUT_SIZE_IN_BYTES]
            .try_into()
            .unwrap()
    }

    pub fn compute_keyed_expected_result(
        input: &[u8],
        key: [u32; crate::CHAINING_VALUE_LEN],
    ) -> [u8; crate::OUTPUT_SIZE_IN_BYTES] {
        let mut key_bytes = [0u8; 32];
        for (i, word) in key.iter().enumerate() {
            key_bytes[i * 4..(i + 1) * 4].copy_from_slice(&word.to_le_bytes());
        }
        blake3::keyed_hash(&key_bytes, input).as_bytes()[0..crate::OUTPUT_SIZE_IN_BYTES]
            .try_into()
            .unwrap()
    }
}
