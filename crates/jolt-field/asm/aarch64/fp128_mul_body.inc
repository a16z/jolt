// mul x5, x0, x2
.inst 0x9b027c05
// umulh x6, x0, x2
.inst 0x9bc27c06
// mul x7, x0, x3
.inst 0x9b037c07
// umulh x8, x0, x3
.inst 0x9bc37c08
// mul x9, x1, x2
.inst 0x9b027c29
// umulh x10, x1, x2
.inst 0x9bc27c2a
// mul x11, x1, x3
.inst 0x9b037c2b
// umulh x12, x1, x3
.inst 0x9bc37c2c

// adds x6, x6, x7
.inst 0xab0700c6
// cset w7, hs
.inst 0x1a9f37e7
// adds x8, x8, x10
.inst 0xab0a0108
// cset w10, hs
.inst 0x1a9f37ea
// adds x8, x8, x11
.inst 0xab0b0108
// cinc x10, x10, hs
.inst 0x9a8a354a
// adds x6, x6, x9
.inst 0xab0900c6
// adcs x8, x8, x7
.inst 0xba070108
// adc x12, x12, x10
.inst 0x9a0a018c

// mul x7, x8, x4
.inst 0x9b047d07
// umulh x9, x8, x4
.inst 0x9bc47d09
// mul x10, x12, x4
.inst 0x9b047d8a
// umulh x11, x12, x4
.inst 0x9bc47d8b

// adds x5, x5, x7
.inst 0xab0700a5
// adcs x6, x6, x9
.inst 0xba0900c6
// cset w8, hs
.inst 0x1a9f37e8
// adds x6, x6, x10
.inst 0xab0a00c6
// adc x12, x11, x8
.inst 0x9a08016c

// mul x7, x12, x4
.inst 0x9b047d87
// adds x5, x5, x7
.inst 0xab0700a5
// adcs x6, x6, xzr
.inst 0xba1f00c6
// cset w7, hs
.inst 0x1a9f37e7
// adds x9, x5, x4
.inst 0xab0400a9
// adcs x10, x6, xzr
.inst 0xba1f00ca
// ccmp w7, #0, #0, lo
.inst 0x7a4038e0
// csel x0, x9, x5, ne
.inst 0x9a851120
// csel x1, x10, x6, ne
.inst 0x9a861141
