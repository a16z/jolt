use super::{F128, F192, F64, F8};
use crate::ExtField;

// The basis words are beta^0 through beta^7 for the roots fixed in binary-f8.md.
macro_rules! embedding_table {
    ($word:ty, $basis:expr) => {{
        const BASIS: [$word; 8] = $basis;
        let mut table = [0; 256];
        let mut i = 0;
        while i < 256 {
            let mut bit = 0;
            while bit < 8 {
                if i & (1 << bit) != 0 {
                    table[i] ^= BASIS[bit];
                }
                bit += 1;
            }
            i += 1;
        }
        table
    }};
}

const F64_IMAGES: [u64; 256] = embedding_table!(
    u64,
    [
        0x0000_0000_0000_0001,
        0x033c_e8be_ddc8_a656,
        0x5126_2037_5ed2_a108,
        0x0c9e_6360_90aa_fc01,
        0xba4f_3cd8_2801_769c,
        0xba26_e790_4adb_4a47,
        0x4676_9859_8926_dc01,
        0x4418_ae80_8b28_bdd0,
    ]
);

const F128_IMAGES: [u128; 256] = embedding_table!(
    u128,
    [
        0x0000_0000_0000_0000_0000_0000_0000_0001,
        0x053d_8555_a997_9a1c_a13f_e8ac_5560_ce0d,
        0x4cf4_b743_9cbf_bb84_ec77_59ca_3488_aee1,
        0x35ad_604f_7d51_d2c6_bfcf_02ae_3639_46a8,
        0x0dcb_3646_40a2_22fe_6b83_3048_3c2e_9849,
        0x5498_10e1_1a88_dea5_252b_4927_7b1b_82b4,
        0xd681_a568_6c0c_1f75_c72b_f2ef_2521_ff22,
        0x0950_311a_4fb7_8fe0_7a7a_8e94_e136_f9bc,
    ]
);

impl From<F8> for F64 {
    fn from(value: F8) -> Self {
        Self::from_raw(F64_IMAGES[usize::from(value.to_raw())])
    }
}

impl From<F8> for F128 {
    fn from(value: F8) -> Self {
        Self::from_raw(F128_IMAGES[usize::from(value.to_raw())])
    }
}

impl From<F8> for F192 {
    fn from(value: F8) -> Self {
        Self::lift_base(F64::from(value))
    }
}
