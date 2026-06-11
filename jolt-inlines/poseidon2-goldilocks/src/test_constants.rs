// SPDX-License-Identifier: MIT

use crate::Poseidon2GoldilocksState;

pub(crate) struct Poseidon2GoldilocksKat {
    pub(crate) input: Poseidon2GoldilocksState,
    pub(crate) output: Poseidon2GoldilocksState,
}

pub(crate) const POSEIDON2_GOLDILOCKS_KATS: &[Poseidon2GoldilocksKat] = &[
    Poseidon2GoldilocksKat {
        input: [
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
        ],
        output: [
            0x4411ec57c44145f5,
            0x5ff55b96baa2f47b,
            0xdee0e2ae35662802,
            0x023c96b32c07981d,
            0x777e4afeaaf2e6a1,
            0x606c248e5ef427da,
            0x862e82242b2c5001,
            0x61ea532cc4c908c7,
        ],
    },
    Poseidon2GoldilocksKat {
        input: [
            0x0000000000000001,
            0x0000000000000002,
            0x0000000000000003,
            0x0000000000000004,
            0x0000000000000005,
            0x0000000000000006,
            0x0000000000000007,
            0x0000000000000008,
        ],
        output: [
            0xd7314da15817d57e,
            0x298d56d49f1937a1,
            0x197376572d00355f,
            0xd302ce06a83b7f6e,
            0xcbfaa68735b06b4a,
            0x01a9337c49e10228,
            0x4a81976fb5dfc0ee,
            0xa98941ad4ca9232e,
        ],
    },
    Poseidon2GoldilocksKat {
        input: [
            0xffffffff00000000,
            0xfffffffeffffffff,
            0xfffffffefffffffe,
            0xfffffffefffffffd,
            0xfffffffefffffffc,
            0xfffffffefffffffb,
            0xfffffffefffffffa,
            0xfffffffefffffff9,
        ],
        output: [
            0xa785ac3b187380e2,
            0xbaf871af4702a41a,
            0xfdfa9a3998f6a535,
            0xd5ffa984d60dfc7c,
            0x847f180534bd6dc1,
            0xe07d1ab55263f732,
            0x0a84ee6f62e263ad,
            0x4068ba2f6dce11b2,
        ],
    },
    Poseidon2GoldilocksKat {
        input: [
            0x0123456789abcdef,
            0xfedcba9876543210,
            0x1111111122222222,
            0x3333333344444444,
            0x5555555566666666,
            0x7777777788888888,
            0x00099999999aaaaa,
            0xbbbbbbbbcccccccc,
        ],
        output: [
            0x4b3aa0d92edfae34,
            0x33dea26f790576cb,
            0xbd45605508de7e43,
            0x38259456bd796aae,
            0xad1b11a37b4ab584,
            0x7490a81972b0fa70,
            0x0a7e73b3531a193f,
            0x12ba121bc764e5c6,
        ],
    },
];
