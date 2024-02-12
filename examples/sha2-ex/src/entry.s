.global _start
.extern _STACK_PTR

.section .text.boot

_start:	la sp, _STACK_PTR
	jal main
	j .
