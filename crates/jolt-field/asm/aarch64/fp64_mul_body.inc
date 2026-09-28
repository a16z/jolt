// mul x8, x1, x0
.inst 0x9b007c28
// umulh x9, x1, x0
.inst 0x9bc07c29
// mov w10, #59
.inst 0x5280076a
// umulh x11, x9, x10
.inst 0x9bca7d2b
// sub x11, x11, x9
.inst 0xcb09016b
// mul x12, x9, x10
.inst 0x9b0a7d2c
// adds x8, x12, x8
.inst 0xab080188
// adc x9, x11, x9
.inst 0x9a090169
// umulh x11, x9, x10
.inst 0x9bca7d2b
// sub x11, x11, x9
.inst 0xcb09016b
// mul x10, x9, x10
.inst 0x9b0a7d2a
// adds x8, x10, x8
.inst 0xab080148
// adc x9, x11, x9
.inst 0x9a090169
// add x10, x8, #59
.inst 0x9100ed0a
// cmn x8, #59
.inst 0xb100ed1f
// sbcs xzr, x9, xzr
.inst 0xfa1f013f
// csel x0, x8, x10, lo
.inst 0x9a8a3100
