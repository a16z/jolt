#[cfg(test)]
use crate::metal::field::FR_U32_LIMBS;

#[cfg(test)]
fn flat_word_offset(factor: usize, len: usize, element: usize) -> u64 {
    ((factor as u64) * (len as u64) + element as u64) * FR_U32_LIMBS as u64
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use super::*;
    use crate::metal::runtime::{KernelId, MetalContext};
    use crate::metal::testing::gpu_lock;

    #[test]
    fn bytecode_flat_offset_matches_device_at_2e27_shape() {
        let _lock = gpu_lock();
        let context = MetalContext::global().unwrap();
        let out = context.alloc_u32s(2).unwrap();
        let cases = [
            (0usize, 1usize << 27, 0usize),
            (3, 1usize << 27, (1usize << 27) - 1),
            (4, 1usize << 27, 0),
            (7, 1usize << 27, 17),
        ];

        for (factor, len, element) in cases {
            context
                .run_once(
                    KernelId::BytecodeOffsetProbe,
                    &[factor as u32, len as u32, element as u32],
                    &[&out],
                    1,
                )
                .unwrap();
            let mut got = [0u32; 2];
            out.copy_to_u32s(&mut got);
            let got = u64::from(got[0]) | (u64::from(got[1]) << 32);
            assert_eq!(got, flat_word_offset(factor, len, element));
        }

        assert_eq!(flat_word_offset(4, 1 << 27, 0), 1u64 << 32);
    }
}
