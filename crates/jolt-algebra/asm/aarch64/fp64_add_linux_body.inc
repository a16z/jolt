// mov w8, #59
.inst 0x52800768
// adds x9, x1, x0
.inst 0xab000029
// csel x8, x8, xzr, hs
.inst 0x9a9f2108
// add x8, x8, x9
.inst 0x8b090108
// add x9, x8, #59
.inst 0x9100ed09
// cmp x9, x8
.inst 0xeb08013f
// csel x0, x9, x8, lo
.inst 0x9a883120
