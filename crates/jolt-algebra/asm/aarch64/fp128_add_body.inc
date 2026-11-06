// adds x5, x0, x2
.inst 0xab020005
// adcs x6, x1, x3
.inst 0xba030026
// cset w7, hs
.inst 0x1a9f37e7
// adds x8, x5, x4
.inst 0xab0400a8
// adcs x9, x6, xzr
.inst 0xba1f00c9
// ccmp w7, #0, #0, lo
.inst 0x7a4038e0
// csel x0, x8, x5, ne
.inst 0x9a851100
// csel x1, x9, x6, ne
.inst 0x9a861121
