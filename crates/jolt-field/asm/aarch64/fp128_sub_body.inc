// subs x5, x0, x2
.inst 0xeb020005
// sbcs x6, x1, x3
.inst 0xfa030026
// csel x7, xzr, x4, hs
.inst 0x9a8423e7
// subs x0, x5, x7
.inst 0xeb0700a0
// sbc x1, x6, xzr
.inst 0xda1f00c1
