// mov x8, #-59
.inst 0x92800748
// subs x9, x0, x1
.inst 0xeb010009
// csel x8, x8, xzr, lo
.inst 0x9a9f3108
// add x0, x9, x8
.inst 0x8b080120
