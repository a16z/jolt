// adds x8, x1, x0
.inst 0xab000028
// mov w9, #59
.inst 0x52800769
// csel x9, x9, xzr, hs
.inst 0x9a9f2129
// add x8, x9, x8
.inst 0x8b080128
// add x9, x8, #59
.inst 0x9100ed09
// cmp x9, x8
.inst 0xeb08013f
// csel x0, x9, x8, lo
.inst 0x9a883120
