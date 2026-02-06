use crate::traits::WindowedScalar;

// 256-bit scalar as 4 limbs.
impl WindowedScalar for [u64; 4] {
    #[inline(always)]
    fn bit_len(&self) -> usize {
        256
    }

    #[inline(always)]
    fn window(&self, offset: usize, width: usize) -> u16 {
        debug_assert!(width <= 16);
        if width == 0 {
            return 0;
        }
        let limb_idx = offset >> 6;
        let bit_idx = offset & 63;
        if limb_idx >= 4 {
            return 0;
        }

        let mut val = self[limb_idx] >> bit_idx;
        if bit_idx + width > 64 && limb_idx + 1 < 4 {
            val |= self[limb_idx + 1] << (64 - bit_idx);
        }
        let mask = (1u64 << width) - 1;
        (val & mask) as u16
    }
}

// 128-bit scalar (for GLV half-scalars).
impl WindowedScalar for u128 {
    #[inline(always)]
    fn bit_len(&self) -> usize {
        128
    }

    #[inline(always)]
    fn window(&self, offset: usize, width: usize) -> u16 {
        debug_assert!(width <= 16);
        if width == 0 || offset >= 128 {
            return 0;
        }
        let val = *self >> offset;
        let mask = (1u128 << width) - 1;
        (val & mask) as u16
    }
}
