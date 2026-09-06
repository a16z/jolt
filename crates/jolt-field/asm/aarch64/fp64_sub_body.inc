// subs x8, x0, x1
.inst 0xeb010008
// mov x9, #-59
.inst 0x92800749
// csel x9, x9, xzr, lo
.inst 0x9a9f3129
// add x0, x8, x9
.inst 0x8b090100
