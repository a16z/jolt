// umulh x9, x1, x0
.inst 0x9bc07c29
// mov w8, #59
.inst 0x52800768
// mul x10, x1, x0
.inst 0x9b007c2a
// umulh x11, x9, x8
.inst 0x9bc87d2b
// mul x12, x9, x8
.inst 0x9b087d2c
// sub x11, x11, x9
.inst 0xcb09016b
// adds x10, x12, x10
.inst 0xab0a018a
// adc x9, x11, x9
.inst 0x9a090169
// umulh x11, x9, x8
.inst 0x9bc87d2b
// mul x8, x9, x8
.inst 0x9b087d28
// sub x11, x11, x9
.inst 0xcb09016b
// adds x8, x8, x10
.inst 0xab0a0108
// adc x9, x11, x9
.inst 0x9a090169
// cmn x8, #59
.inst 0xb100ed1f
// add x10, x8, #59
.inst 0x9100ed0a
// sbcs xzr, x9, xzr
.inst 0xfa1f013f
// csel x0, x8, x10, lo
.inst 0x9a8a3100
