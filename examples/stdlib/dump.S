
/tmp/jolt-guest-targets/stdlib-guest-int_to_string/riscv32im-jolt-zkvm-elf/release/stdlib-guest:     file format elf32-littleriscv


Disassembly of section .text.boot:

80000000 <.text.boot>:
80000000:	00002117          	auipc	sp,0x2
80000004:	36010113          	addi	sp,sp,864 # 0x80002360
80000008:	6d5000ef          	jal	0x80000edc
8000000c:	0000006f          	j	0x8000000c

Disassembly of section .text._ZN5alloc7raw_vec17capacity_overflow17h35314e8674496b65E.llvm.15770576173586076926:

80000010 <.text._ZN5alloc7raw_vec17capacity_overflow17h35314e8674496b65E.llvm.15770576173586076926>:
80000010:	fe010113          	addi	sp,sp,-32
80000014:	00112e23          	sw	ra,28(sp)
80000018:	00812c23          	sw	s0,24(sp)
8000001c:	02010413          	addi	s0,sp,32
80000020:	00050593          	mv	a1,a0
80000024:	80001537          	lui	a0,0x80001
80000028:	37050513          	addi	a0,a0,880 # 0x80001370
8000002c:	fea42023          	sw	a0,-32(s0)
80000030:	00100513          	li	a0,1
80000034:	fea42223          	sw	a0,-28(s0)
80000038:	fe042823          	sw	zero,-16(s0)
8000003c:	00400513          	li	a0,4
80000040:	fea42423          	sw	a0,-24(s0)
80000044:	fe042623          	sw	zero,-20(s0)
80000048:	fe040513          	addi	a0,s0,-32
8000004c:	00001097          	auipc	ra,0x1
80000050:	994080e7          	jalr	-1644(ra) # 0x800009e0

Disassembly of section .text._ZN5alloc7raw_vec19RawVec$LT$T$C$A$GT$8grow_one17h1c0bd3b852fc30bbE:

80000054 <.text._ZN5alloc7raw_vec19RawVec$LT$T$C$A$GT$8grow_one17h1c0bd3b852fc30bbE>:
80000054:	fd010113          	addi	sp,sp,-48
80000058:	02112623          	sw	ra,44(sp)
8000005c:	02812423          	sw	s0,40(sp)
80000060:	02912223          	sw	s1,36(sp)
80000064:	03212023          	sw	s2,32(sp)
80000068:	01312e23          	sw	s3,28(sp)
8000006c:	03010413          	addi	s0,sp,48
80000070:	00050913          	mv	s2,a0
80000074:	00052503          	lw	a0,0(a0)
80000078:	00150993          	addi	s3,a0,1
8000007c:	00151613          	slli	a2,a0,0x1
80000080:	00058493          	mv	s1,a1
80000084:	07367663          	bgeu	a2,s3,0x800000f0
80000088:	00800593          	li	a1,8
8000008c:	0735f863          	bgeu	a1,s3,0x800000fc
80000090:	0609ca63          	bltz	s3,0x80000104
80000094:	00050a63          	beqz	a0,0x800000a8
80000098:	00492583          	lw	a1,4(s2)
8000009c:	feb42023          	sw	a1,-32(s0)
800000a0:	fea42423          	sw	a0,-24(s0)
800000a4:	00100513          	li	a0,1
800000a8:	fea42223          	sw	a0,-28(s0)
800000ac:	fd440513          	addi	a0,s0,-44
800000b0:	fe040613          	addi	a2,s0,-32
800000b4:	00098593          	mv	a1,s3
800000b8:	00000097          	auipc	ra,0x0
800000bc:	06c080e7          	jalr	108(ra) # 0x80000124
800000c0:	fd442503          	lw	a0,-44(s0)
800000c4:	04051863          	bnez	a0,0x80000114
800000c8:	fd842503          	lw	a0,-40(s0)
800000cc:	00a92223          	sw	a0,4(s2)
800000d0:	01392023          	sw	s3,0(s2)
800000d4:	02c12083          	lw	ra,44(sp)
800000d8:	02812403          	lw	s0,40(sp)
800000dc:	02412483          	lw	s1,36(sp)
800000e0:	02012903          	lw	s2,32(sp)
800000e4:	01c12983          	lw	s3,28(sp)
800000e8:	03010113          	addi	sp,sp,48
800000ec:	00008067          	ret
800000f0:	00060993          	mv	s3,a2
800000f4:	00800593          	li	a1,8
800000f8:	f8c5ece3          	bltu	a1,a2,0x80000090
800000fc:	00800993          	li	s3,8
80000100:	f809dae3          	bgez	s3,0x80000094
80000104:	00000513          	li	a0,0
80000108:	00048593          	mv	a1,s1
8000010c:	00000097          	auipc	ra,0x0
80000110:	0a4080e7          	jalr	164(ra) # 0x800001b0
80000114:	fd842503          	lw	a0,-40(s0)
80000118:	00048593          	mv	a1,s1
8000011c:	00000097          	auipc	ra,0x0
80000120:	094080e7          	jalr	148(ra) # 0x800001b0

Disassembly of section .text.unlikely._ZN5alloc7raw_vec11finish_grow17h37bd137dfa0a8593E.llvm.15770576173586076926:

80000124 <.text.unlikely._ZN5alloc7raw_vec11finish_grow17h37bd137dfa0a8593E.llvm.15770576173586076926>:
80000124:	ff010113          	addi	sp,sp,-16
80000128:	00112623          	sw	ra,12(sp)
8000012c:	00812423          	sw	s0,8(sp)
80000130:	00912223          	sw	s1,4(sp)
80000134:	01212023          	sw	s2,0(sp)
80000138:	01010413          	addi	s0,sp,16
8000013c:	00462683          	lw	a3,4(a2)
80000140:	00058913          	mv	s2,a1
80000144:	00050493          	mv	s1,a0
80000148:	02068063          	beqz	a3,0x80000168
8000014c:	00862583          	lw	a1,8(a2)
80000150:	00058c63          	beqz	a1,0x80000168
80000154:	00062503          	lw	a0,0(a2)
80000158:	00090613          	mv	a2,s2
8000015c:	00000097          	auipc	ra,0x0
80000160:	078080e7          	jalr	120(ra) # 0x800001d4
80000164:	0280006f          	j	0x8000018c
80000168:	80002537          	lui	a0,0x80002
8000016c:	b2854003          	lbu	zero,-1240(a0) # 0x80001b28
80000170:	800025b7          	lui	a1,0x80002
80000174:	b145a503          	lw	a0,-1260(a1) # 0x80001b14
80000178:	00051663          	bnez	a0,0x80000184
8000017c:	80002537          	lui	a0,0x80002
80000180:	36050513          	addi	a0,a0,864 # 0x80002360
80000184:	01250633          	add	a2,a0,s2
80000188:	b0c5aa23          	sw	a2,-1260(a1)
8000018c:	00a4a223          	sw	a0,4(s1)
80000190:	0124a423          	sw	s2,8(s1)
80000194:	0004a023          	sw	zero,0(s1)
80000198:	00c12083          	lw	ra,12(sp)
8000019c:	00812403          	lw	s0,8(sp)
800001a0:	00412483          	lw	s1,4(sp)
800001a4:	00012903          	lw	s2,0(sp)
800001a8:	01010113          	addi	sp,sp,16
800001ac:	00008067          	ret

Disassembly of section .text.unlikely._ZN5alloc7raw_vec12handle_error17hee628d084cd9322dE:

800001b0 <.text.unlikely._ZN5alloc7raw_vec12handle_error17hee628d084cd9322dE>:
800001b0:	02051063          	bnez	a0,0x800001d0
800001b4:	ff010113          	addi	sp,sp,-16
800001b8:	00112623          	sw	ra,12(sp)
800001bc:	00812423          	sw	s0,8(sp)
800001c0:	01010413          	addi	s0,sp,16
800001c4:	00058513          	mv	a0,a1
800001c8:	00000097          	auipc	ra,0x0
800001cc:	e48080e7          	jalr	-440(ra) # 0x80000010
800001d0:	c0001073          	unimp

Disassembly of section .text.__rust_realloc:

800001d4 <.text.__rust_realloc>:
800001d4:	800027b7          	lui	a5,0x80002
800001d8:	b147a683          	lw	a3,-1260(a5) # 0x80001b14
800001dc:	00050713          	mv	a4,a0
800001e0:	00069663          	bnez	a3,0x800001ec
800001e4:	800026b7          	lui	a3,0x80002
800001e8:	36068693          	addi	a3,a3,864 # 0x80002360
800001ec:	00c68533          	add	a0,a3,a2
800001f0:	b0a7aa23          	sw	a0,-1260(a5)
800001f4:	00b66463          	bltu	a2,a1,0x800001fc
800001f8:	00058613          	mv	a2,a1
800001fc:	00068513          	mv	a0,a3
80000200:	00070593          	mv	a1,a4
80000204:	00001317          	auipc	t1,0x1
80000208:	04830067          	jr	72(t1) # 0x8000124c

Disassembly of section .text._ZN4core3fmt5write17h97bdded77bf47fd8E:

8000020c <.text._ZN4core3fmt5write17h97bdded77bf47fd8E>:
8000020c:	fb010113          	addi	sp,sp,-80
80000210:	04112623          	sw	ra,76(sp)
80000214:	04812423          	sw	s0,72(sp)
80000218:	04912223          	sw	s1,68(sp)
8000021c:	05212023          	sw	s2,64(sp)
80000220:	03312e23          	sw	s3,60(sp)
80000224:	03412c23          	sw	s4,56(sp)
80000228:	03512a23          	sw	s5,52(sp)
8000022c:	03612823          	sw	s6,48(sp)
80000230:	03712623          	sw	s7,44(sp)
80000234:	03812423          	sw	s8,40(sp)
80000238:	05010413          	addi	s0,sp,80
8000023c:	00060493          	mv	s1,a2
80000240:	fa042a23          	sw	zero,-76(s0)
80000244:	fa042e23          	sw	zero,-68(s0)
80000248:	02000613          	li	a2,32
8000024c:	fcc42223          	sw	a2,-60(s0)
80000250:	fc042423          	sw	zero,-56(s0)
80000254:	0104ab03          	lw	s6,16(s1)
80000258:	00300613          	li	a2,3
8000025c:	fcc40623          	sb	a2,-52(s0)
80000260:	fca42823          	sw	a0,-48(s0)
80000264:	fcb42a23          	sw	a1,-44(s0)
80000268:	100b0a63          	beqz	s6,0x8000037c
8000026c:	0144aa83          	lw	s5,20(s1)
80000270:	180a8263          	beqz	s5,0x800003f4
80000274:	0004aa03          	lw	s4,0(s1)
80000278:	0084a983          	lw	s3,8(s1)
8000027c:	fffa8513          	addi	a0,s5,-1
80000280:	00551513          	slli	a0,a0,0x5
80000284:	00555513          	srli	a0,a0,0x5
80000288:	00150913          	addi	s2,a0,1
8000028c:	004a0a13          	addi	s4,s4,4
80000290:	005a9a93          	slli	s5,s5,0x5
80000294:	010b0b13          	addi	s6,s6,16
80000298:	00200b93          	li	s7,2
8000029c:	00100c13          	li	s8,1
800002a0:	000a2603          	lw	a2,0(s4)
800002a4:	00060e63          	beqz	a2,0x800002c0
800002a8:	fd442683          	lw	a3,-44(s0)
800002ac:	ffca2583          	lw	a1,-4(s4)
800002b0:	fd042503          	lw	a0,-48(s0)
800002b4:	00c6a683          	lw	a3,12(a3)
800002b8:	000680e7          	jalr	a3
800002bc:	16051663          	bnez	a0,0x80000428
800002c0:	000b2603          	lw	a2,0(s6)
800002c4:	00cb4683          	lbu	a3,12(s6)
800002c8:	008b2703          	lw	a4,8(s6)
800002cc:	ff8b2583          	lw	a1,-8(s6)
800002d0:	ffcb2503          	lw	a0,-4(s6)
800002d4:	fcc42223          	sw	a2,-60(s0)
800002d8:	fcd40623          	sb	a3,-52(s0)
800002dc:	fce42423          	sw	a4,-56(s0)
800002e0:	02058263          	beqz	a1,0x80000304
800002e4:	01859a63          	bne	a1,s8,0x800002f8
800002e8:	00351513          	slli	a0,a0,0x3
800002ec:	00a98533          	add	a0,s3,a0
800002f0:	00052583          	lw	a1,0(a0)
800002f4:	00058663          	beqz	a1,0x80000300
800002f8:	00000593          	li	a1,0
800002fc:	00c0006f          	j	0x80000308
80000300:	00452503          	lw	a0,4(a0)
80000304:	00100593          	li	a1,1
80000308:	ff0b2603          	lw	a2,-16(s6)
8000030c:	fab42a23          	sw	a1,-76(s0)
80000310:	faa42c23          	sw	a0,-72(s0)
80000314:	01760e63          	beq	a2,s7,0x80000330
80000318:	ff4b2583          	lw	a1,-12(s6)
8000031c:	03861063          	bne	a2,s8,0x8000033c
80000320:	00359513          	slli	a0,a1,0x3
80000324:	00a98533          	add	a0,s3,a0
80000328:	00052583          	lw	a1,0(a0)
8000032c:	00058663          	beqz	a1,0x80000338
80000330:	00000613          	li	a2,0
80000334:	00c0006f          	j	0x80000340
80000338:	00452583          	lw	a1,4(a0)
8000033c:	00100613          	li	a2,1
80000340:	004b2503          	lw	a0,4(s6)
80000344:	00351513          	slli	a0,a0,0x3
80000348:	00a986b3          	add	a3,s3,a0
8000034c:	0006a503          	lw	a0,0(a3)
80000350:	0046a683          	lw	a3,4(a3)
80000354:	fac42e23          	sw	a2,-68(s0)
80000358:	fcb42023          	sw	a1,-64(s0)
8000035c:	fb440593          	addi	a1,s0,-76
80000360:	000680e7          	jalr	a3
80000364:	0c051263          	bnez	a0,0x80000428
80000368:	008a0a13          	addi	s4,s4,8
8000036c:	fe0a8a93          	addi	s5,s5,-32
80000370:	020b0b13          	addi	s6,s6,32
80000374:	f20a96e3          	bnez	s5,0x800002a0
80000378:	0700006f          	j	0x800003e8
8000037c:	00c4a503          	lw	a0,12(s1)
80000380:	06050a63          	beqz	a0,0x800003f4
80000384:	0084a983          	lw	s3,8(s1)
80000388:	00351a13          	slli	s4,a0,0x3
8000038c:	01498a33          	add	s4,s3,s4
80000390:	0004aa83          	lw	s5,0(s1)
80000394:	fff50513          	addi	a0,a0,-1
80000398:	00351513          	slli	a0,a0,0x3
8000039c:	00355513          	srli	a0,a0,0x3
800003a0:	00150913          	addi	s2,a0,1
800003a4:	004a8a93          	addi	s5,s5,4
800003a8:	000aa603          	lw	a2,0(s5)
800003ac:	00060e63          	beqz	a2,0x800003c8
800003b0:	fd442683          	lw	a3,-44(s0)
800003b4:	ffcaa583          	lw	a1,-4(s5)
800003b8:	fd042503          	lw	a0,-48(s0)
800003bc:	00c6a683          	lw	a3,12(a3)
800003c0:	000680e7          	jalr	a3
800003c4:	06051263          	bnez	a0,0x80000428
800003c8:	0009a503          	lw	a0,0(s3)
800003cc:	0049a603          	lw	a2,4(s3)
800003d0:	fb440593          	addi	a1,s0,-76
800003d4:	000600e7          	jalr	a2
800003d8:	04051863          	bnez	a0,0x80000428
800003dc:	00898993          	addi	s3,s3,8
800003e0:	008a8a93          	addi	s5,s5,8
800003e4:	fd4992e3          	bne	s3,s4,0x800003a8
800003e8:	0044a503          	lw	a0,4(s1)
800003ec:	00a96a63          	bltu	s2,a0,0x80000400
800003f0:	0400006f          	j	0x80000430
800003f4:	00000913          	li	s2,0
800003f8:	0044a503          	lw	a0,4(s1)
800003fc:	02a07a63          	bgeu	zero,a0,0x80000430
80000400:	0004a503          	lw	a0,0(s1)
80000404:	00391913          	slli	s2,s2,0x3
80000408:	01250933          	add	s2,a0,s2
8000040c:	fd442683          	lw	a3,-44(s0)
80000410:	fd042503          	lw	a0,-48(s0)
80000414:	00092583          	lw	a1,0(s2)
80000418:	00492603          	lw	a2,4(s2)
8000041c:	00c6a683          	lw	a3,12(a3)
80000420:	000680e7          	jalr	a3
80000424:	00050663          	beqz	a0,0x80000430
80000428:	00100513          	li	a0,1
8000042c:	0080006f          	j	0x80000434
80000430:	00000513          	li	a0,0
80000434:	04c12083          	lw	ra,76(sp)
80000438:	04812403          	lw	s0,72(sp)
8000043c:	04412483          	lw	s1,68(sp)
80000440:	04012903          	lw	s2,64(sp)
80000444:	03c12983          	lw	s3,60(sp)
80000448:	03812a03          	lw	s4,56(sp)
8000044c:	03412a83          	lw	s5,52(sp)
80000450:	03012b03          	lw	s6,48(sp)
80000454:	02c12b83          	lw	s7,44(sp)
80000458:	02812c03          	lw	s8,40(sp)
8000045c:	05010113          	addi	sp,sp,80
80000460:	00008067          	ret

Disassembly of section .text._ZN4core3fmt9Formatter12pad_integral12write_prefix17h9e5f601c56857129E:

80000464 <.text._ZN4core3fmt9Formatter12pad_integral12write_prefix17h9e5f601c56857129E>:
80000464:	fe010113          	addi	sp,sp,-32
80000468:	00112e23          	sw	ra,28(sp)
8000046c:	00812c23          	sw	s0,24(sp)
80000470:	00912a23          	sw	s1,20(sp)
80000474:	01212823          	sw	s2,16(sp)
80000478:	01312623          	sw	s3,12(sp)
8000047c:	02010413          	addi	s0,sp,32
80000480:	00110737          	lui	a4,0x110
80000484:	00068493          	mv	s1,a3
80000488:	00058913          	mv	s2,a1
8000048c:	02e60263          	beq	a2,a4,0x800004b0
80000490:	01092683          	lw	a3,16(s2)
80000494:	00050993          	mv	s3,a0
80000498:	00060593          	mv	a1,a2
8000049c:	000680e7          	jalr	a3
800004a0:	00050613          	mv	a2,a0
800004a4:	00098513          	mv	a0,s3
800004a8:	00100593          	li	a1,1
800004ac:	02061a63          	bnez	a2,0x800004e0
800004b0:	02048663          	beqz	s1,0x800004dc
800004b4:	00c92303          	lw	t1,12(s2)
800004b8:	00048593          	mv	a1,s1
800004bc:	00000613          	li	a2,0
800004c0:	01c12083          	lw	ra,28(sp)
800004c4:	01812403          	lw	s0,24(sp)
800004c8:	01412483          	lw	s1,20(sp)
800004cc:	01012903          	lw	s2,16(sp)
800004d0:	00c12983          	lw	s3,12(sp)
800004d4:	02010113          	addi	sp,sp,32
800004d8:	00030067          	jr	t1
800004dc:	00000593          	li	a1,0
800004e0:	00058513          	mv	a0,a1
800004e4:	01c12083          	lw	ra,28(sp)
800004e8:	01812403          	lw	s0,24(sp)
800004ec:	01412483          	lw	s1,20(sp)
800004f0:	01012903          	lw	s2,16(sp)
800004f4:	00c12983          	lw	s3,12(sp)
800004f8:	02010113          	addi	sp,sp,32
800004fc:	00008067          	ret

Disassembly of section .text._ZN42_$LT$$RF$T$u20$as$u20$core..fmt..Debug$GT$3fmt17h7f202709febcc8f8E:

80000500 <.text._ZN42_$LT$$RF$T$u20$as$u20$core..fmt..Debug$GT$3fmt17h7f202709febcc8f8E>:
80000500:	ff010113          	addi	sp,sp,-16
80000504:	00112623          	sw	ra,12(sp)
80000508:	00812423          	sw	s0,8(sp)
8000050c:	01010413          	addi	s0,sp,16
80000510:	00452603          	lw	a2,4(a0)
80000514:	00052503          	lw	a0,0(a0)
80000518:	00c62303          	lw	t1,12(a2)
8000051c:	00c12083          	lw	ra,12(sp)
80000520:	00812403          	lw	s0,8(sp)
80000524:	01010113          	addi	sp,sp,16
80000528:	00030067          	jr	t1

Disassembly of section .text._ZN44_$LT$$RF$T$u20$as$u20$core..fmt..Display$GT$3fmt17he1166ffdaa86f971E:

8000052c <.text._ZN44_$LT$$RF$T$u20$as$u20$core..fmt..Display$GT$3fmt17he1166ffdaa86f971E>:
8000052c:	fd010113          	addi	sp,sp,-48
80000530:	02112623          	sw	ra,44(sp)
80000534:	02812423          	sw	s0,40(sp)
80000538:	02912223          	sw	s1,36(sp)
8000053c:	03212023          	sw	s2,32(sp)
80000540:	01312e23          	sw	s3,28(sp)
80000544:	01412c23          	sw	s4,24(sp)
80000548:	01512a23          	sw	s5,20(sp)
8000054c:	01612823          	sw	s6,16(sp)
80000550:	01712623          	sw	s7,12(sp)
80000554:	03010413          	addi	s0,sp,48
80000558:	0005a603          	lw	a2,0(a1)
8000055c:	0085a683          	lw	a3,8(a1)
80000560:	00052903          	lw	s2,0(a0)
80000564:	00452483          	lw	s1,4(a0)
80000568:	00d66533          	or	a0,a2,a3
8000056c:	2e050a63          	beqz	a0,0x80000860
80000570:	0016f693          	andi	a3,a3,1
80000574:	0c068063          	beqz	a3,0x80000634
80000578:	00c5a703          	lw	a4,12(a1)
8000057c:	009906b3          	add	a3,s2,s1
80000580:	00000513          	li	a0,0
80000584:	06070063          	beqz	a4,0x800005e4
80000588:	00000793          	li	a5,0
8000058c:	0e000813          	li	a6,224
80000590:	0f000893          	li	a7,240
80000594:	00090293          	mv	t0,s2
80000598:	01c0006f          	j	0x800005b4
8000059c:	00128313          	addi	t1,t0,1
800005a0:	00178793          	addi	a5,a5,1
800005a4:	40a28533          	sub	a0,t0,a0
800005a8:	40a30533          	sub	a0,t1,a0
800005ac:	00030293          	mv	t0,t1
800005b0:	02f70c63          	beq	a4,a5,0x800005e8
800005b4:	08d28063          	beq	t0,a3,0x80000634
800005b8:	00028303          	lb	t1,0(t0)
800005bc:	fe0350e3          	bgez	t1,0x8000059c
800005c0:	0ff37313          	zext.b	t1,t1
800005c4:	01036863          	bltu	t1,a6,0x800005d4
800005c8:	01136a63          	bltu	t1,a7,0x800005dc
800005cc:	00428313          	addi	t1,t0,4
800005d0:	fd1ff06f          	j	0x800005a0
800005d4:	00228313          	addi	t1,t0,2
800005d8:	fc9ff06f          	j	0x800005a0
800005dc:	00328313          	addi	t1,t0,3
800005e0:	fc1ff06f          	j	0x800005a0
800005e4:	00090313          	mv	t1,s2
800005e8:	04d30663          	beq	t1,a3,0x80000634
800005ec:	00030683          	lb	a3,0(t1)
800005f0:	0006d663          	bgez	a3,0x800005fc
800005f4:	0ff6f693          	zext.b	a3,a3
800005f8:	0e000713          	li	a4,224
800005fc:	02050463          	beqz	a0,0x80000624
80000600:	02957063          	bgeu	a0,s1,0x80000620
80000604:	00a906b3          	add	a3,s2,a0
80000608:	00068683          	lb	a3,0(a3)
8000060c:	fc000713          	li	a4,-64
80000610:	00e6da63          	bge	a3,a4,0x80000624
80000614:	00000693          	li	a3,0
80000618:	00001a63          	bnez	zero,0x8000062c
8000061c:	0180006f          	j	0x80000634
80000620:	fe951ae3          	bne	a0,s1,0x80000614
80000624:	00090693          	mv	a3,s2
80000628:	00090663          	beqz	s2,0x80000634
8000062c:	00050493          	mv	s1,a0
80000630:	00068913          	mv	s2,a3
80000634:	22060663          	beqz	a2,0x80000860
80000638:	0045a503          	lw	a0,4(a1)
8000063c:	01000613          	li	a2,16
80000640:	02c4f863          	bgeu	s1,a2,0x80000670
80000644:	00000613          	li	a2,0
80000648:	1e048263          	beqz	s1,0x8000082c
8000064c:	009906b3          	add	a3,s2,s1
80000650:	00090713          	mv	a4,s2
80000654:	00070783          	lb	a5,0(a4) # 0x110000
80000658:	fc07a793          	slti	a5,a5,-64
8000065c:	0017c793          	xori	a5,a5,1
80000660:	00170713          	addi	a4,a4,1
80000664:	00f60633          	add	a2,a2,a5
80000668:	fed716e3          	bne	a4,a3,0x80000654
8000066c:	1c00006f          	j	0x8000082c
80000670:	00390613          	addi	a2,s2,3
80000674:	ffc67293          	andi	t0,a2,-4
80000678:	01248633          	add	a2,s1,s2
8000067c:	405607b3          	sub	a5,a2,t0
80000680:	0037f693          	andi	a3,a5,3
80000684:	00000613          	li	a2,0
80000688:	03228063          	beq	t0,s2,0x800006a8
8000068c:	00090713          	mv	a4,s2
80000690:	00070803          	lb	a6,0(a4)
80000694:	fc082813          	slti	a6,a6,-64
80000698:	00184813          	xori	a6,a6,1
8000069c:	00170713          	addi	a4,a4,1
800006a0:	01060633          	add	a2,a2,a6
800006a4:	fe5716e3          	bne	a4,t0,0x80000690
800006a8:	00000813          	li	a6,0
800006ac:	02068463          	beqz	a3,0x800006d4
800006b0:	ffc7f713          	andi	a4,a5,-4
800006b4:	00e28733          	add	a4,t0,a4
800006b8:	00070883          	lb	a7,0(a4)
800006bc:	fc08a893          	slti	a7,a7,-64
800006c0:	0018c893          	xori	a7,a7,1
800006c4:	01180833          	add	a6,a6,a7
800006c8:	fff68693          	addi	a3,a3,-1
800006cc:	00170713          	addi	a4,a4,1
800006d0:	fe0694e3          	bnez	a3,0x800006b8
800006d4:	0027d793          	srli	a5,a5,0x2
800006d8:	010106b7          	lui	a3,0x1010
800006dc:	10168713          	addi	a4,a3,257 # 0x1010101
800006e0:	00ff06b7          	lui	a3,0xff0
800006e4:	0ff68693          	addi	a3,a3,255 # 0xff00ff
800006e8:	00c80633          	add	a2,a6,a2
800006ec:	00400893          	li	a7,4
800006f0:	0340006f          	j	0x80000724
800006f4:	005802b3          	add	t0,a6,t0
800006f8:	406787b3          	sub	a5,a5,t1
800006fc:	00337393          	andi	t2,t1,3
80000700:	00defe33          	and	t3,t4,a3
80000704:	008ede93          	srli	t4,t4,0x8
80000708:	00defeb3          	and	t4,t4,a3
8000070c:	01ce8e33          	add	t3,t4,t3
80000710:	010e1e93          	slli	t4,t3,0x10
80000714:	01ce8e33          	add	t3,t4,t3
80000718:	010e5e13          	srli	t3,t3,0x10
8000071c:	00ce0633          	add	a2,t3,a2
80000720:	0a039863          	bnez	t2,0x800007d0
80000724:	10078463          	beqz	a5,0x8000082c
80000728:	00028813          	mv	a6,t0
8000072c:	0c000293          	li	t0,192
80000730:	00078313          	mv	t1,a5
80000734:	0057e463          	bltu	a5,t0,0x8000073c
80000738:	0c000313          	li	t1,192
8000073c:	00231293          	slli	t0,t1,0x2
80000740:	00000e93          	li	t4,0
80000744:	fb17e8e3          	bltu	a5,a7,0x800006f4
80000748:	3f02f393          	andi	t2,t0,1008
8000074c:	007803b3          	add	t2,a6,t2
80000750:	00080e13          	mv	t3,a6
80000754:	000e2f03          	lw	t5,0(t3)
80000758:	ffff4f93          	not	t6,t5
8000075c:	007fdf93          	srli	t6,t6,0x7
80000760:	006f5f13          	srli	t5,t5,0x6
80000764:	004e2983          	lw	s3,4(t3)
80000768:	01efef33          	or	t5,t6,t5
8000076c:	00ef7f33          	and	t5,t5,a4
80000770:	01df0eb3          	add	t4,t5,t4
80000774:	fff9cf13          	not	t5,s3
80000778:	007f5f13          	srli	t5,t5,0x7
8000077c:	008e2f83          	lw	t6,8(t3)
80000780:	0069d993          	srli	s3,s3,0x6
80000784:	013f6f33          	or	t5,t5,s3
80000788:	00ef7f33          	and	t5,t5,a4
8000078c:	ffffc993          	not	s3,t6
80000790:	0079d993          	srli	s3,s3,0x7
80000794:	006fdf93          	srli	t6,t6,0x6
80000798:	01f9efb3          	or	t6,s3,t6
8000079c:	00ce2983          	lw	s3,12(t3)
800007a0:	00efffb3          	and	t6,t6,a4
800007a4:	01ff0f33          	add	t5,t5,t6
800007a8:	01ee8eb3          	add	t4,t4,t5
800007ac:	fff9cf13          	not	t5,s3
800007b0:	007f5f13          	srli	t5,t5,0x7
800007b4:	0069df93          	srli	t6,s3,0x6
800007b8:	01ff6f33          	or	t5,t5,t6
800007bc:	00ef7f33          	and	t5,t5,a4
800007c0:	010e0e13          	addi	t3,t3,16
800007c4:	01ee8eb3          	add	t4,t4,t5
800007c8:	f87e16e3          	bne	t3,t2,0x80000754
800007cc:	f29ff06f          	j	0x800006f4
800007d0:	00000793          	li	a5,0
800007d4:	0fc37893          	andi	a7,t1,252
800007d8:	00289893          	slli	a7,a7,0x2
800007dc:	01180833          	add	a6,a6,a7
800007e0:	00239393          	slli	t2,t2,0x2
800007e4:	00082883          	lw	a7,0(a6)
800007e8:	00480813          	addi	a6,a6,4
800007ec:	fff8c293          	not	t0,a7
800007f0:	0072d293          	srli	t0,t0,0x7
800007f4:	0068d893          	srli	a7,a7,0x6
800007f8:	0112e8b3          	or	a7,t0,a7
800007fc:	00e8f8b3          	and	a7,a7,a4
80000800:	ffc38393          	addi	t2,t2,-4
80000804:	00f887b3          	add	a5,a7,a5
80000808:	fc039ee3          	bnez	t2,0x800007e4
8000080c:	00d7f733          	and	a4,a5,a3
80000810:	0087d793          	srli	a5,a5,0x8
80000814:	00d7f6b3          	and	a3,a5,a3
80000818:	00e686b3          	add	a3,a3,a4
8000081c:	01069713          	slli	a4,a3,0x10
80000820:	00d706b3          	add	a3,a4,a3
80000824:	0106d693          	srli	a3,a3,0x10
80000828:	00c68633          	add	a2,a3,a2
8000082c:	02a67a63          	bgeu	a2,a0,0x80000860
80000830:	0185c683          	lbu	a3,24(a1)
80000834:	ffd68713          	addi	a4,a3,-3
80000838:	00173713          	seqz	a4,a4
8000083c:	fff70713          	addi	a4,a4,-1
80000840:	00d776b3          	and	a3,a4,a3
80000844:	40c50ab3          	sub	s5,a0,a2
80000848:	06068263          	beqz	a3,0x800008ac
8000084c:	00100513          	li	a0,1
80000850:	04a69863          	bne	a3,a0,0x800008a0
80000854:	000a8693          	mv	a3,s5
80000858:	00000a93          	li	s5,0
8000085c:	0500006f          	j	0x800008ac
80000860:	0205a603          	lw	a2,32(a1)
80000864:	01c5a503          	lw	a0,28(a1)
80000868:	00c62303          	lw	t1,12(a2)
8000086c:	00090593          	mv	a1,s2
80000870:	00048613          	mv	a2,s1
80000874:	02c12083          	lw	ra,44(sp)
80000878:	02812403          	lw	s0,40(sp)
8000087c:	02412483          	lw	s1,36(sp)
80000880:	02012903          	lw	s2,32(sp)
80000884:	01c12983          	lw	s3,28(sp)
80000888:	01812a03          	lw	s4,24(sp)
8000088c:	01412a83          	lw	s5,20(sp)
80000890:	01012b03          	lw	s6,16(sp)
80000894:	00c12b83          	lw	s7,12(sp)
80000898:	03010113          	addi	sp,sp,48
8000089c:	00030067          	jr	t1
800008a0:	001ad693          	srli	a3,s5,0x1
800008a4:	001a8a93          	addi	s5,s5,1
800008a8:	001ada93          	srli	s5,s5,0x1
800008ac:	01c5a983          	lw	s3,28(a1)
800008b0:	0205ab03          	lw	s6,32(a1)
800008b4:	0105aa03          	lw	s4,16(a1)
800008b8:	00168b93          	addi	s7,a3,1
800008bc:	fffb8b93          	addi	s7,s7,-1
800008c0:	020b8063          	beqz	s7,0x800008e0
800008c4:	010b2603          	lw	a2,16(s6)
800008c8:	00098513          	mv	a0,s3
800008cc:	000a0593          	mv	a1,s4
800008d0:	000600e7          	jalr	a2
800008d4:	fe0504e3          	beqz	a0,0x800008bc
800008d8:	00100513          	li	a0,1
800008dc:	05c0006f          	j	0x80000938
800008e0:	00cb2683          	lw	a3,12(s6)
800008e4:	00098513          	mv	a0,s3
800008e8:	00090593          	mv	a1,s2
800008ec:	00048613          	mv	a2,s1
800008f0:	000680e7          	jalr	a3
800008f4:	00050593          	mv	a1,a0
800008f8:	00100513          	li	a0,1
800008fc:	02059e63          	bnez	a1,0x80000938
80000900:	41500933          	neg	s2,s5
80000904:	fff00b93          	li	s7,-1
80000908:	fff00493          	li	s1,-1
8000090c:	00990533          	add	a0,s2,s1
80000910:	03750063          	beq	a0,s7,0x80000930
80000914:	010b2603          	lw	a2,16(s6)
80000918:	00098513          	mv	a0,s3
8000091c:	000a0593          	mv	a1,s4
80000920:	000600e7          	jalr	a2
80000924:	00148493          	addi	s1,s1,1
80000928:	fe0502e3          	beqz	a0,0x8000090c
8000092c:	0080006f          	j	0x80000934
80000930:	000a8493          	mv	s1,s5
80000934:	0154b533          	sltu	a0,s1,s5
80000938:	02c12083          	lw	ra,44(sp)
8000093c:	02812403          	lw	s0,40(sp)
80000940:	02412483          	lw	s1,36(sp)
80000944:	02012903          	lw	s2,32(sp)
80000948:	01c12983          	lw	s3,28(sp)
8000094c:	01812a03          	lw	s4,24(sp)
80000950:	01412a83          	lw	s5,20(sp)
80000954:	01012b03          	lw	s6,16(sp)
80000958:	00c12b83          	lw	s7,12(sp)
8000095c:	03010113          	addi	sp,sp,48
80000960:	00008067          	ret

Disassembly of section .text.unlikely._ZN4core6result13unwrap_failed17h688273743937427dE:

80000964 <.text.unlikely._ZN4core6result13unwrap_failed17h688273743937427dE>:
80000964:	fc010113          	addi	sp,sp,-64
80000968:	02112e23          	sw	ra,60(sp)
8000096c:	02812c23          	sw	s0,56(sp)
80000970:	04010413          	addi	s0,sp,64
80000974:	fca42023          	sw	a0,-64(s0)
80000978:	fcb42223          	sw	a1,-60(s0)
8000097c:	fcc42423          	sw	a2,-56(s0)
80000980:	fcd42623          	sw	a3,-52(s0)
80000984:	fc040513          	addi	a0,s0,-64
80000988:	fea42423          	sw	a0,-24(s0)
8000098c:	80000537          	lui	a0,0x80000
80000990:	52c50513          	addi	a0,a0,1324 # 0x8000052c
80000994:	fea42623          	sw	a0,-20(s0)
80000998:	fc840513          	addi	a0,s0,-56
8000099c:	fea42823          	sw	a0,-16(s0)
800009a0:	80000537          	lui	a0,0x80000
800009a4:	50050513          	addi	a0,a0,1280 # 0x80000500
800009a8:	fea42a23          	sw	a0,-12(s0)
800009ac:	80001537          	lui	a0,0x80001
800009b0:	37c50513          	addi	a0,a0,892 # 0x8000137c
800009b4:	fca42823          	sw	a0,-48(s0)
800009b8:	00200513          	li	a0,2
800009bc:	fca42a23          	sw	a0,-44(s0)
800009c0:	fe042023          	sw	zero,-32(s0)
800009c4:	fe840593          	addi	a1,s0,-24
800009c8:	fcb42c23          	sw	a1,-40(s0)
800009cc:	fca42e23          	sw	a0,-36(s0)
800009d0:	fd040513          	addi	a0,s0,-48
800009d4:	00070593          	mv	a1,a4
800009d8:	00000097          	auipc	ra,0x0
800009dc:	008080e7          	jalr	8(ra) # 0x800009e0

Disassembly of section .text.unlikely._ZN4core9panicking9panic_fmt17h99d36ef349840c23E:

800009e0 <.text.unlikely._ZN4core9panicking9panic_fmt17h99d36ef349840c23E>:
800009e0:	fe010113          	addi	sp,sp,-32
800009e4:	00112e23          	sw	ra,28(sp)
800009e8:	00812c23          	sw	s0,24(sp)
800009ec:	02010413          	addi	s0,sp,32
800009f0:	fea42623          	sw	a0,-20(s0)
800009f4:	feb42823          	sw	a1,-16(s0)
800009f8:	00100513          	li	a0,1
800009fc:	fea41a23          	sh	a0,-12(s0)
80000a00:	fec40513          	addi	a0,s0,-20
80000a04:	00000097          	auipc	ra,0x0
80000a08:	10c080e7          	jalr	268(ra) # 0x80000b10

Disassembly of section .text._ZN4core5panic12PanicPayload6as_str17h0bcfc2a5a3ac02d7E:

80000a0c <.text._ZN4core5panic12PanicPayload6as_str17h0bcfc2a5a3ac02d7E>:
80000a0c:	ff010113          	addi	sp,sp,-16
80000a10:	00112623          	sw	ra,12(sp)
80000a14:	00812423          	sw	s0,8(sp)
80000a18:	01010413          	addi	s0,sp,16
80000a1c:	00000513          	li	a0,0
80000a20:	00c12083          	lw	ra,12(sp)
80000a24:	00812403          	lw	s0,8(sp)
80000a28:	01010113          	addi	sp,sp,16
80000a2c:	00008067          	ret

Disassembly of section .text._ZN3std3sys9backtrace26__rust_end_short_backtrace17h7cf26424a35e20deE:

80000a30 <.text._ZN3std3sys9backtrace26__rust_end_short_backtrace17h7cf26424a35e20deE>:
80000a30:	ff010113          	addi	sp,sp,-16
80000a34:	00112623          	sw	ra,12(sp)
80000a38:	00812423          	sw	s0,8(sp)
80000a3c:	01010413          	addi	s0,sp,16
80000a40:	00000097          	auipc	ra,0x0
80000a44:	008080e7          	jalr	8(ra) # 0x80000a48

Disassembly of section .text._ZN3std9panicking19begin_panic_handler28_$u7b$$u7b$closure$u7d$$u7d$17hba790dc4d8d6265dE:

80000a48 <.text._ZN3std9panicking19begin_panic_handler28_$u7b$$u7b$closure$u7d$$u7d$17hba790dc4d8d6265dE>:
80000a48:	fe010113          	addi	sp,sp,-32
80000a4c:	00112e23          	sw	ra,28(sp)
80000a50:	00812c23          	sw	s0,24(sp)
80000a54:	02010413          	addi	s0,sp,32
80000a58:	00052583          	lw	a1,0(a0)
80000a5c:	0045a683          	lw	a3,4(a1)
80000a60:	00c5a603          	lw	a2,12(a1)
80000a64:	00100713          	li	a4,1
80000a68:	00e68c63          	beq	a3,a4,0x80000a80
80000a6c:	04069663          	bnez	a3,0x80000ab8
80000a70:	04061463          	bnez	a2,0x80000ab8
80000a74:	00000593          	li	a1,0
80000a78:	00100713          	li	a4,1
80000a7c:	0140006f          	j	0x80000a90
80000a80:	02061c63          	bnez	a2,0x80000ab8
80000a84:	0005a583          	lw	a1,0(a1)
80000a88:	0005a703          	lw	a4,0(a1)
80000a8c:	0045a583          	lw	a1,4(a1)
80000a90:	00852503          	lw	a0,8(a0)
80000a94:	00854603          	lbu	a2,8(a0)
80000a98:	00954683          	lbu	a3,9(a0)
80000a9c:	fee42423          	sw	a4,-24(s0)
80000aa0:	feb42623          	sw	a1,-20(s0)
80000aa4:	800015b7          	lui	a1,0x80001
80000aa8:	ae458593          	addi	a1,a1,-1308 # 0x80000ae4
80000aac:	fe840513          	addi	a0,s0,-24
80000ab0:	00000097          	auipc	ra,0x0
80000ab4:	090080e7          	jalr	144(ra) # 0x80000b40
80000ab8:	00852583          	lw	a1,8(a0)
80000abc:	fea42a23          	sw	a0,-12(s0)
80000ac0:	0085c603          	lbu	a2,8(a1)
80000ac4:	0095c683          	lbu	a3,9(a1)
80000ac8:	80000537          	lui	a0,0x80000
80000acc:	fea42423          	sw	a0,-24(s0)
80000ad0:	800015b7          	lui	a1,0x80001
80000ad4:	a0c58593          	addi	a1,a1,-1524 # 0x80000a0c
80000ad8:	fe840513          	addi	a0,s0,-24
80000adc:	00000097          	auipc	ra,0x0
80000ae0:	064080e7          	jalr	100(ra) # 0x80000b40

Disassembly of section .text._ZN99_$LT$std..panicking..begin_panic_handler..StaticStrPayload$u20$as$u20$core..panic..PanicPayload$GT$6as_str17h47b2e1128c3906c8E:

80000ae4 <.text._ZN99_$LT$std..panicking..begin_panic_handler..StaticStrPayload$u20$as$u20$core..panic..PanicPayload$GT$6as_str17h47b2e1128c3906c8E>:
80000ae4:	ff010113          	addi	sp,sp,-16
80000ae8:	00112623          	sw	ra,12(sp)
80000aec:	00812423          	sw	s0,8(sp)
80000af0:	01010413          	addi	s0,sp,16
80000af4:	00052603          	lw	a2,0(a0) # 0x80000000
80000af8:	00452583          	lw	a1,4(a0)
80000afc:	00060513          	mv	a0,a2
80000b00:	00c12083          	lw	ra,12(sp)
80000b04:	00812403          	lw	s0,8(sp)
80000b08:	01010113          	addi	sp,sp,16
80000b0c:	00008067          	ret

Disassembly of section .text.rust_begin_unwind:

80000b10 <.text.rust_begin_unwind>:
80000b10:	fe010113          	addi	sp,sp,-32
80000b14:	00112e23          	sw	ra,28(sp)
80000b18:	00812c23          	sw	s0,24(sp)
80000b1c:	02010413          	addi	s0,sp,32
80000b20:	00052583          	lw	a1,0(a0)
80000b24:	00452603          	lw	a2,4(a0)
80000b28:	feb42623          	sw	a1,-20(s0)
80000b2c:	fec42823          	sw	a2,-16(s0)
80000b30:	fea42a23          	sw	a0,-12(s0)
80000b34:	fec40513          	addi	a0,s0,-20
80000b38:	00000097          	auipc	ra,0x0
80000b3c:	ef8080e7          	jalr	-264(ra) # 0x80000a30

Disassembly of section .text._ZN3std9panicking20rust_panic_with_hook17h9daad1b0e5d179c2E:

80000b40 <.text._ZN3std9panicking20rust_panic_with_hook17h9daad1b0e5d179c2E>:
80000b40:	ff010113          	addi	sp,sp,-16
80000b44:	00112623          	sw	ra,12(sp)
80000b48:	00812423          	sw	s0,8(sp)
80000b4c:	01010413          	addi	s0,sp,16
80000b50:	800026b7          	lui	a3,0x80002
80000b54:	b246a703          	lw	a4,-1244(a3) # 0x80001b24
80000b58:	00170793          	addi	a5,a4,1
80000b5c:	b2f6a223          	sw	a5,-1244(a3)
80000b60:	04074463          	bltz	a4,0x80000ba8
80000b64:	800026b7          	lui	a3,0x80002
80000b68:	b206c683          	lbu	a3,-1248(a3) # 0x80001b20
80000b6c:	00068663          	beqz	a3,0x80000b78
80000b70:	000580e7          	jalr	a1
80000b74:	c0001073          	unimp
80000b78:	80002537          	lui	a0,0x80002
80000b7c:	b1c52583          	lw	a1,-1252(a0) # 0x80001b1c
80000b80:	800026b7          	lui	a3,0x80002
80000b84:	b186a683          	lw	a3,-1256(a3) # 0x80001b18
80000b88:	00158593          	addi	a1,a1,1
80000b8c:	b0b52e23          	sw	a1,-1252(a0)
80000b90:	0006cc63          	bltz	a3,0x80000ba8
80000b94:	80002537          	lui	a0,0x80002
80000b98:	b2050023          	sb	zero,-1248(a0) # 0x80001b20
80000b9c:	00060663          	beqz	a2,0x80000ba8
80000ba0:	00000097          	auipc	ra,0x0
80000ba4:	00c080e7          	jalr	12(ra) # 0x80000bac
80000ba8:	c0001073          	unimp

Disassembly of section .text.rust_panic:

80000bac <.text.rust_panic>:
80000bac:	ff010113          	addi	sp,sp,-16
80000bb0:	00112623          	sw	ra,12(sp)
80000bb4:	00812423          	sw	s0,8(sp)
80000bb8:	01010413          	addi	s0,sp,16
80000bbc:	7fffe537          	lui	a0,0x7fffe
80000bc0:	00100593          	li	a1,1
80000bc4:	10b50023          	sb	a1,256(a0) # 0x7fffe100
80000bc8:	0000006f          	j	0x80000bc8

Disassembly of section .text._ZN132_$LT$alloc..vec..Vec$LT$T$C$A$GT$$u20$as$u20$alloc..vec..spec_extend..SpecExtend$LT$$RF$T$C$core..slice..iter..Iter$LT$T$GT$$GT$$GT$11spec_extend17h0f58daffc28761c5E:

80000bcc <.text._ZN132_$LT$alloc..vec..Vec$LT$T$C$A$GT$$u20$as$u20$alloc..vec..spec_extend..SpecExtend$LT$$RF$T$C$core..slice..iter..Iter$LT$T$GT$$GT$$GT$11spec_extend17h0f58daffc28761c5E>:
80000bcc:	fd010113          	addi	sp,sp,-48
80000bd0:	02112623          	sw	ra,44(sp)
80000bd4:	02812423          	sw	s0,40(sp)
80000bd8:	02912223          	sw	s1,36(sp)
80000bdc:	03212023          	sw	s2,32(sp)
80000be0:	01312e23          	sw	s3,28(sp)
80000be4:	01412c23          	sw	s4,24(sp)
80000be8:	01512a23          	sw	s5,20(sp)
80000bec:	00050413          	mv	s0,a0
80000bf0:	00852a03          	lw	s4,8(a0)
80000bf4:	00052683          	lw	a3,0(a0)
80000bf8:	00058913          	mv	s2,a1
80000bfc:	40b604b3          	sub	s1,a2,a1
80000c00:	41468533          	sub	a0,a3,s4
80000c04:	04956463          	bltu	a0,s1,0x80000c4c
80000c08:	00442503          	lw	a0,4(s0)
80000c0c:	009a0ab3          	add	s5,s4,s1
80000c10:	01450533          	add	a0,a0,s4
80000c14:	00090593          	mv	a1,s2
80000c18:	00048613          	mv	a2,s1
80000c1c:	00000097          	auipc	ra,0x0
80000c20:	630080e7          	jalr	1584(ra) # 0x8000124c
80000c24:	01542423          	sw	s5,8(s0)
80000c28:	02c12083          	lw	ra,44(sp)
80000c2c:	02812403          	lw	s0,40(sp)
80000c30:	02412483          	lw	s1,36(sp)
80000c34:	02012903          	lw	s2,32(sp)
80000c38:	01c12983          	lw	s3,28(sp)
80000c3c:	01812a03          	lw	s4,24(sp)
80000c40:	01412a83          	lw	s5,20(sp)
80000c44:	03010113          	addi	sp,sp,48
80000c48:	00008067          	ret
80000c4c:	009a0ab3          	add	s5,s4,s1
80000c50:	094ae063          	bltu	s5,s4,0x80000cd0
80000c54:	00169513          	slli	a0,a3,0x1
80000c58:	000a8993          	mv	s3,s5
80000c5c:	01556463          	bltu	a0,s5,0x80000c64
80000c60:	00050993          	mv	s3,a0
80000c64:	00800513          	li	a0,8
80000c68:	01356463          	bltu	a0,s3,0x80000c70
80000c6c:	00800993          	li	s3,8
80000c70:	0609c063          	bltz	s3,0x80000cd0
80000c74:	00068c63          	beqz	a3,0x80000c8c
80000c78:	00442503          	lw	a0,4(s0)
80000c7c:	00100593          	li	a1,1
80000c80:	00b12823          	sw	a1,16(sp)
80000c84:	00c10593          	addi	a1,sp,12
80000c88:	0080006f          	j	0x80000c90
80000c8c:	01010593          	addi	a1,sp,16
80000c90:	00d5a023          	sw	a3,0(a1)
80000c94:	01012583          	lw	a1,16(sp)
80000c98:	00058e63          	beqz	a1,0x80000cb4
80000c9c:	00c12583          	lw	a1,12(sp)
80000ca0:	00058a63          	beqz	a1,0x80000cb4
80000ca4:	00098613          	mv	a2,s3
80000ca8:	fffff097          	auipc	ra,0xfffff
80000cac:	52c080e7          	jalr	1324(ra) # 0x800001d4
80000cb0:	0140006f          	j	0x80000cc4
80000cb4:	00098513          	mv	a0,s3
80000cb8:	00000097          	auipc	ra,0x0
80000cbc:	034080e7          	jalr	52(ra) # 0x80000cec
80000cc0:	02050263          	beqz	a0,0x80000ce4
80000cc4:	00a42223          	sw	a0,4(s0)
80000cc8:	01342023          	sw	s3,0(s0)
80000ccc:	f45ff06f          	j	0x80000c10
80000cd0:	00000513          	li	a0,0
80000cd4:	800015b7          	lui	a1,0x80001
80000cd8:	6b458593          	addi	a1,a1,1716 # 0x800016b4
80000cdc:	fffff097          	auipc	ra,0xfffff
80000ce0:	4d4080e7          	jalr	1236(ra) # 0x800001b0
80000ce4:	00100513          	li	a0,1
80000ce8:	fedff06f          	j	0x80000cd4

Disassembly of section .text._ZN5alloc5alloc6Global10alloc_impl17he3e4754c500cb23eE:

80000cec <.text._ZN5alloc5alloc6Global10alloc_impl17he3e4754c500cb23eE>:
80000cec:	800025b7          	lui	a1,0x80002
80000cf0:	b285c003          	lbu	zero,-1240(a1) # 0x80001b28
80000cf4:	800026b7          	lui	a3,0x80002
80000cf8:	b146a603          	lw	a2,-1260(a3) # 0x80001b14
80000cfc:	00050593          	mv	a1,a0
80000d00:	00061663          	bnez	a2,0x80000d0c
80000d04:	80002637          	lui	a2,0x80002
80000d08:	36060613          	addi	a2,a2,864 # 0x80002360
80000d0c:	00b60533          	add	a0,a2,a1
80000d10:	b0a6aa23          	sw	a0,-1260(a3)
80000d14:	00060513          	mv	a0,a2
80000d18:	00008067          	ret

Disassembly of section .text._ZN4core3fmt5Write9write_fmt17h928e5469f648fb22E:

80000d1c <.text._ZN4core3fmt5Write9write_fmt17h928e5469f648fb22E>:
80000d1c:	00058613          	mv	a2,a1
80000d20:	0045a583          	lw	a1,4(a1)
80000d24:	00100693          	li	a3,1
80000d28:	800015b7          	lui	a1,0x80001
80000d2c:	45458593          	addi	a1,a1,1108 # 0x80001454
80000d30:	fffff317          	auipc	t1,0xfffff
80000d34:	4dc30067          	jr	1244(t1) # 0x8000020c

Disassembly of section .text._ZN5alloc6string6String8push_str17h929a5503e77cd50bE:

80000d38 <.text._ZN5alloc6string6String8push_str17h929a5503e77cd50bE>:
80000d38:	00c58633          	add	a2,a1,a2
80000d3c:	00000317          	auipc	t1,0x0
80000d40:	e9030067          	jr	-368(t1) # 0x80000bcc

Disassembly of section .text._ZN4core3ptr42drop_in_place$LT$alloc..string..String$GT$17hadadf074a3321267E:

80000d44 <.text._ZN4core3ptr42drop_in_place$LT$alloc..string..String$GT$17hadadf074a3321267E>:
80000d44:	00052603          	lw	a2,0(a0)
80000d48:	00452583          	lw	a1,4(a0)
80000d4c:	00060513          	mv	a0,a2
80000d50:	00000317          	auipc	t1,0x0
80000d54:	14830067          	jr	328(t1) # 0x80000e98

Disassembly of section .text._ZN58_$LT$alloc..string..String$u20$as$u20$core..fmt..Write$GT$9write_str17h04061b1aec9f4eb9E.514:

80000d58 <.text._ZN58_$LT$alloc..string..String$u20$as$u20$core..fmt..Write$GT$9write_str17h04061b1aec9f4eb9E.514>:
80000d58:	ff010113          	addi	sp,sp,-16
80000d5c:	00112623          	sw	ra,12(sp)
80000d60:	00000097          	auipc	ra,0x0
80000d64:	fd8080e7          	jalr	-40(ra) # 0x80000d38
80000d68:	00000513          	li	a0,0
80000d6c:	00c12083          	lw	ra,12(sp)
80000d70:	01010113          	addi	sp,sp,16
80000d74:	00008067          	ret

Disassembly of section .text._ZN58_$LT$alloc..string..String$u20$as$u20$core..fmt..Write$GT$10write_char17h02f088f1a1a7ff05E.515:

80000d78 <.text._ZN58_$LT$alloc..string..String$u20$as$u20$core..fmt..Write$GT$10write_char17h02f088f1a1a7ff05E.515>:
80000d78:	fe010113          	addi	sp,sp,-32
80000d7c:	00112e23          	sw	ra,28(sp)
80000d80:	00812c23          	sw	s0,24(sp)
80000d84:	00912a23          	sw	s1,20(sp)
80000d88:	01212823          	sw	s2,16(sp)
80000d8c:	08000613          	li	a2,128
80000d90:	00058493          	mv	s1,a1
80000d94:	00050413          	mv	s0,a0
80000d98:	02c5fe63          	bgeu	a1,a2,0x80000dd4
80000d9c:	00842903          	lw	s2,8(s0)
80000da0:	00042503          	lw	a0,0(s0)
80000da4:	00a91c63          	bne	s2,a0,0x80000dbc
80000da8:	800015b7          	lui	a1,0x80001
80000dac:	66458593          	addi	a1,a1,1636 # 0x80001664
80000db0:	00040513          	mv	a0,s0
80000db4:	fffff097          	auipc	ra,0xfffff
80000db8:	2a0080e7          	jalr	672(ra) # 0x80000054
80000dbc:	00442503          	lw	a0,4(s0)
80000dc0:	01250533          	add	a0,a0,s2
80000dc4:	00950023          	sb	s1,0(a0)
80000dc8:	00190913          	addi	s2,s2,1
80000dcc:	01242423          	sw	s2,8(s0)
80000dd0:	0ac0006f          	j	0x80000e7c
80000dd4:	00e10593          	addi	a1,sp,14
80000dd8:	00b4d513          	srli	a0,s1,0xb
80000ddc:	00012623          	sw	zero,12(sp)
80000de0:	00051e63          	bnez	a0,0x80000dfc
80000de4:	00d10513          	addi	a0,sp,13
80000de8:	0064d613          	srli	a2,s1,0x6
80000dec:	0c066613          	ori	a2,a2,192
80000df0:	00c10623          	sb	a2,12(sp)
80000df4:	00058613          	mv	a2,a1
80000df8:	0680006f          	j	0x80000e60
80000dfc:	0104d513          	srli	a0,s1,0x10
80000e00:	00f10613          	addi	a2,sp,15
80000e04:	02051463          	bnez	a0,0x80000e2c
80000e08:	00c4d513          	srli	a0,s1,0xc
80000e0c:	0e056513          	ori	a0,a0,224
80000e10:	00a10623          	sb	a0,12(sp)
80000e14:	01449513          	slli	a0,s1,0x14
80000e18:	01a55513          	srli	a0,a0,0x1a
80000e1c:	08050513          	addi	a0,a0,128
80000e20:	00a106a3          	sb	a0,13(sp)
80000e24:	00058513          	mv	a0,a1
80000e28:	0380006f          	j	0x80000e60
80000e2c:	00060513          	mv	a0,a2
80000e30:	01010613          	addi	a2,sp,16
80000e34:	0124d593          	srli	a1,s1,0x12
80000e38:	0f05e593          	ori	a1,a1,240
80000e3c:	00b10623          	sb	a1,12(sp)
80000e40:	00e49593          	slli	a1,s1,0xe
80000e44:	01a5d593          	srli	a1,a1,0x1a
80000e48:	08058593          	addi	a1,a1,128
80000e4c:	00b106a3          	sb	a1,13(sp)
80000e50:	01449593          	slli	a1,s1,0x14
80000e54:	01a5d593          	srli	a1,a1,0x1a
80000e58:	08058593          	addi	a1,a1,128
80000e5c:	00b10723          	sb	a1,14(sp)
80000e60:	03f4f593          	andi	a1,s1,63
80000e64:	08058593          	addi	a1,a1,128
80000e68:	00b50023          	sb	a1,0(a0)
80000e6c:	00c10593          	addi	a1,sp,12
80000e70:	00040513          	mv	a0,s0
80000e74:	00000097          	auipc	ra,0x0
80000e78:	d58080e7          	jalr	-680(ra) # 0x80000bcc
80000e7c:	00000513          	li	a0,0
80000e80:	01c12083          	lw	ra,28(sp)
80000e84:	01812403          	lw	s0,24(sp)
80000e88:	01412483          	lw	s1,20(sp)
80000e8c:	01012903          	lw	s2,16(sp)
80000e90:	02010113          	addi	sp,sp,32
80000e94:	00008067          	ret

Disassembly of section .text._ZN77_$LT$alloc..raw_vec..RawVec$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17h29a44361cd0d83ccE:

80000e98 <.text._ZN77_$LT$alloc..raw_vec..RawVec$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17h29a44361cd0d83ccE>:
80000e98:	ff010113          	addi	sp,sp,-16
80000e9c:	00050a63          	beqz	a0,0x80000eb0
80000ea0:	00100593          	li	a1,1
80000ea4:	00b12623          	sw	a1,12(sp)
80000ea8:	00810593          	addi	a1,sp,8
80000eac:	0080006f          	j	0x80000eb4
80000eb0:	00c10593          	addi	a1,sp,12
80000eb4:	00a5a023          	sw	a0,0(a1)
80000eb8:	01010113          	addi	sp,sp,16
80000ebc:	00008067          	ret

Disassembly of section .text._ZN53_$LT$core..fmt..Error$u20$as$u20$core..fmt..Debug$GT$3fmt17h2c006b9898dc048eE.516:

80000ec0 <.text._ZN53_$LT$core..fmt..Error$u20$as$u20$core..fmt..Debug$GT$3fmt17h2c006b9898dc048eE.516>:
80000ec0:	0205a603          	lw	a2,32(a1)
80000ec4:	01c5a503          	lw	a0,28(a1)
80000ec8:	00c62303          	lw	t1,12(a2)
80000ecc:	800015b7          	lui	a1,0x80001
80000ed0:	53b58593          	addi	a1,a1,1339 # 0x8000153b
80000ed4:	00500613          	li	a2,5
80000ed8:	00030067          	jr	t1

Disassembly of section .text.main:

80000edc <.text.main>:
80000edc:	fc010113          	addi	sp,sp,-64
80000ee0:	02112e23          	sw	ra,60(sp)
80000ee4:	02812c23          	sw	s0,56(sp)
80000ee8:	02912a23          	sw	s1,52(sp)
80000eec:	03212823          	sw	s2,48(sp)
80000ef0:	03312623          	sw	s3,44(sp)
80000ef4:	03412423          	sw	s4,40(sp)
80000ef8:	03512223          	sw	s5,36(sp)
80000efc:	00000593          	li	a1,0
80000f00:	00000513          	li	a0,0
80000f04:	fff00613          	li	a2,-1
80000f08:	7fffc6b7          	lui	a3,0x7fffc
80000f0c:	10068693          	addi	a3,a3,256 # 0x7fffc100
80000f10:	00400713          	li	a4,4
80000f14:	28e60e63          	beq	a2,a4,0x800011b0
80000f18:	00068783          	lb	a5,0(a3)
80000f1c:	00168693          	addi	a3,a3,1
80000f20:	07f7f813          	andi	a6,a5,127
80000f24:	00b81833          	sll	a6,a6,a1
80000f28:	00a86533          	or	a0,a6,a0
80000f2c:	00160613          	addi	a2,a2,1
80000f30:	00758593          	addi	a1,a1,7
80000f34:	fe07c0e3          	bltz	a5,0x80000f14
80000f38:	00400593          	li	a1,4
80000f3c:	00b61863          	bne	a2,a1,0x80000f4c
80000f40:	0ff7f593          	zext.b	a1,a5
80000f44:	01000613          	li	a2,16
80000f48:	26c5f463          	bgeu	a1,a2,0x800011b0
80000f4c:	00155593          	srli	a1,a0,0x1
80000f50:	01f51513          	slli	a0,a0,0x1f
80000f54:	41f55513          	srai	a0,a0,0x1f
80000f58:	00a5c533          	xor	a0,a1,a0
80000f5c:	00012623          	sw	zero,12(sp)
80000f60:	00100593          	li	a1,1
80000f64:	00b12823          	sw	a1,16(sp)
80000f68:	00012a23          	sw	zero,20(sp)
80000f6c:	41f55593          	srai	a1,a0,0x1f
80000f70:	00b54633          	xor	a2,a0,a1
80000f74:	40b60633          	sub	a2,a2,a1
80000f78:	00a00413          	li	s0,10
80000f7c:	3e800693          	li	a3,1000
80000f80:	800015b7          	lui	a1,0x80001
80000f84:	38c58593          	addi	a1,a1,908 # 0x8000138c
80000f88:	08d66863          	bltu	a2,a3,0x80001018
80000f8c:	00a00413          	li	s0,10
80000f90:	02310693          	addi	a3,sp,35
80000f94:	00002737          	lui	a4,0x2
80000f98:	71070713          	addi	a4,a4,1808 # 0x2710
80000f9c:	0000e7b7          	lui	a5,0xe
80000fa0:	8f078793          	addi	a5,a5,-1808 # 0xd8f0
80000fa4:	06400813          	li	a6,100
80000fa8:	f9c00893          	li	a7,-100
80000fac:	009892b7          	lui	t0,0x989
80000fb0:	67f28293          	addi	t0,t0,1663 # 0x98967f
80000fb4:	00060313          	mv	t1,a2
80000fb8:	02e65633          	divu	a2,a2,a4
80000fbc:	02f603b3          	mul	t2,a2,a5
80000fc0:	006383b3          	add	t2,t2,t1
80000fc4:	01039e13          	slli	t3,t2,0x10
80000fc8:	010e5e13          	srli	t3,t3,0x10
80000fcc:	030e5e33          	divu	t3,t3,a6
80000fd0:	001e1e93          	slli	t4,t3,0x1
80000fd4:	01d58eb3          	add	t4,a1,t4
80000fd8:	000ecf03          	lbu	t5,0(t4)
80000fdc:	031e0e33          	mul	t3,t3,a7
80000fe0:	007e03b3          	add	t2,t3,t2
80000fe4:	ffe68ea3          	sb	t5,-3(a3)
80000fe8:	001ece03          	lbu	t3,1(t4)
80000fec:	01139393          	slli	t2,t2,0x11
80000ff0:	0103d393          	srli	t2,t2,0x10
80000ff4:	007583b3          	add	t2,a1,t2
80000ff8:	0003ce83          	lbu	t4,0(t2)
80000ffc:	0013c383          	lbu	t2,1(t2)
80001000:	ffc40413          	addi	s0,s0,-4
80001004:	ffc68f23          	sb	t3,-2(a3)
80001008:	ffd68fa3          	sb	t4,-1(a3)
8000100c:	00768023          	sb	t2,0(a3)
80001010:	ffc68693          	addi	a3,a3,-4
80001014:	fa62e0e3          	bltu	t0,t1,0x80000fb4
80001018:	00900693          	li	a3,9
8000101c:	04c6f863          	bgeu	a3,a2,0x8000106c
80001020:	01061693          	slli	a3,a2,0x10
80001024:	0106d693          	srli	a3,a3,0x10
80001028:	06400713          	li	a4,100
8000102c:	02e6d6b3          	divu	a3,a3,a4
80001030:	f9c00713          	li	a4,-100
80001034:	02e68733          	mul	a4,a3,a4
80001038:	00c70633          	add	a2,a4,a2
8000103c:	01161613          	slli	a2,a2,0x11
80001040:	01065613          	srli	a2,a2,0x10
80001044:	01a10713          	addi	a4,sp,26
80001048:	00c58633          	add	a2,a1,a2
8000104c:	00064783          	lbu	a5,0(a2)
80001050:	00870833          	add	a6,a4,s0
80001054:	00164603          	lbu	a2,1(a2)
80001058:	ffe40413          	addi	s0,s0,-2
8000105c:	00870733          	add	a4,a4,s0
80001060:	00f70023          	sb	a5,0(a4)
80001064:	fec80fa3          	sb	a2,-1(a6)
80001068:	00068613          	mv	a2,a3
8000106c:	00050463          	beqz	a0,0x80001074
80001070:	02060263          	beqz	a2,0x80001094
80001074:	00161613          	slli	a2,a2,0x1
80001078:	01e67613          	andi	a2,a2,30
8000107c:	00c585b3          	add	a1,a1,a2
80001080:	0015c583          	lbu	a1,1(a1)
80001084:	fff40413          	addi	s0,s0,-1
80001088:	01a10613          	addi	a2,sp,26
8000108c:	00860633          	add	a2,a2,s0
80001090:	00b60023          	sb	a1,0(a2)
80001094:	00110637          	lui	a2,0x110
80001098:	00055463          	bgez	a0,0x800010a0
8000109c:	02d00613          	li	a2,45
800010a0:	800015b7          	lui	a1,0x80001
800010a4:	45458593          	addi	a1,a1,1108 # 0x80001454
800010a8:	00c10513          	addi	a0,sp,12
800010ac:	00000693          	li	a3,0
800010b0:	fffff097          	auipc	ra,0xfffff
800010b4:	3b4080e7          	jalr	948(ra) # 0x80000464
800010b8:	12051463          	bnez	a0,0x800011e0
800010bc:	01a10593          	addi	a1,sp,26
800010c0:	008585b3          	add	a1,a1,s0
800010c4:	00a00613          	li	a2,10
800010c8:	40860633          	sub	a2,a2,s0
800010cc:	00c10513          	addi	a0,sp,12
800010d0:	00c10a13          	addi	s4,sp,12
800010d4:	00000097          	auipc	ra,0x0
800010d8:	c84080e7          	jalr	-892(ra) # 0x80000d58
800010dc:	10051263          	bnez	a0,0x800011e0
800010e0:	00c12403          	lw	s0,12(sp)
800010e4:	01412903          	lw	s2,20(sp)
800010e8:	01012483          	lw	s1,16(sp)
800010ec:	00010823          	sb	zero,16(sp)
800010f0:	00012623          	sw	zero,12(sp)
800010f4:	00090593          	mv	a1,s2
800010f8:	00500993          	li	s3,5
800010fc:	03350c63          	beq	a0,s3,0x80001134
80001100:	00aa06b3          	add	a3,s4,a0
80001104:	0075d613          	srli	a2,a1,0x7
80001108:	00b68023          	sb	a1,0(a3)
8000110c:	00060e63          	beqz	a2,0x80001128
80001110:	0805e593          	ori	a1,a1,128
80001114:	00b68023          	sb	a1,0(a3)
80001118:	00150513          	addi	a0,a0,1
8000111c:	00060593          	mv	a1,a2
80001120:	ff3510e3          	bne	a0,s3,0x80001100
80001124:	0100006f          	j	0x80001134
80001128:	00150993          	addi	s3,a0,1
8000112c:	00001537          	lui	a0,0x1
80001130:	0d356c63          	bltu	a0,s3,0x80001208
80001134:	7fffe537          	lui	a0,0x7fffe
80001138:	10050a93          	addi	s5,a0,256 # 0x7fffe100
8000113c:	7fffd537          	lui	a0,0x7fffd
80001140:	10050a13          	addi	s4,a0,256 # 0x7fffd100
80001144:	00c10593          	addi	a1,sp,12
80001148:	000a0513          	mv	a0,s4
8000114c:	00098613          	mv	a2,s3
80001150:	00000097          	auipc	ra,0x0
80001154:	0fc080e7          	jalr	252(ra) # 0x8000124c
80001158:	01498533          	add	a0,s3,s4
8000115c:	40aa85b3          	sub	a1,s5,a0
80001160:	0b25e463          	bltu	a1,s2,0x80001208
80001164:	00048593          	mv	a1,s1
80001168:	00090613          	mv	a2,s2
8000116c:	00000097          	auipc	ra,0x0
80001170:	0e0080e7          	jalr	224(ra) # 0x8000124c
80001174:	00100513          	li	a0,1
80001178:	00aa8223          	sb	a0,4(s5)
8000117c:	00040513          	mv	a0,s0
80001180:	00048593          	mv	a1,s1
80001184:	00000097          	auipc	ra,0x0
80001188:	d14080e7          	jalr	-748(ra) # 0x80000e98
8000118c:	03c12083          	lw	ra,60(sp)
80001190:	03812403          	lw	s0,56(sp)
80001194:	03412483          	lw	s1,52(sp)
80001198:	03012903          	lw	s2,48(sp)
8000119c:	02c12983          	lw	s3,44(sp)
800011a0:	02812a03          	lw	s4,40(sp)
800011a4:	02412a83          	lw	s5,36(sp)
800011a8:	04010113          	addi	sp,sp,64
800011ac:	00008067          	ret
800011b0:	00500513          	li	a0,5
800011b4:	00a10623          	sb	a0,12(sp)
800011b8:	80001537          	lui	a0,0x80001
800011bc:	51050513          	addi	a0,a0,1296 # 0x80001510
800011c0:	800016b7          	lui	a3,0x80001
800011c4:	50068693          	addi	a3,a3,1280 # 0x80001500
800011c8:	80001737          	lui	a4,0x80001
800011cc:	6e470713          	addi	a4,a4,1764 # 0x800016e4
800011d0:	02b00593          	li	a1,43
800011d4:	00c10613          	addi	a2,sp,12
800011d8:	fffff097          	auipc	ra,0xfffff
800011dc:	78c080e7          	jalr	1932(ra) # 0x80000964
800011e0:	80001537          	lui	a0,0x80001
800011e4:	47c50513          	addi	a0,a0,1148 # 0x8000147c
800011e8:	800016b7          	lui	a3,0x80001
800011ec:	46c68693          	addi	a3,a3,1132 # 0x8000146c
800011f0:	80001737          	lui	a4,0x80001
800011f4:	4f070713          	addi	a4,a4,1264 # 0x800014f0
800011f8:	03700593          	li	a1,55
800011fc:	01a10613          	addi	a2,sp,26
80001200:	fffff097          	auipc	ra,0xfffff
80001204:	764080e7          	jalr	1892(ra) # 0x80000964
80001208:	00200513          	li	a0,2
8000120c:	fa9ff06f          	j	0x800011b4

Disassembly of section .text._ZN59_$LT$postcard..error..Error$u20$as$u20$core..fmt..Debug$GT$3fmt17hb2eb621a0f9f9641E:

80001210 <.text._ZN59_$LT$postcard..error..Error$u20$as$u20$core..fmt..Debug$GT$3fmt17hb2eb621a0f9f9641E>:
80001210:	00054503          	lbu	a0,0(a0)
80001214:	00251513          	slli	a0,a0,0x2
80001218:	80001637          	lui	a2,0x80001
8000121c:	6f460613          	addi	a2,a2,1780 # 0x800016f4
80001220:	00a60633          	add	a2,a2,a0
80001224:	800016b7          	lui	a3,0x80001
80001228:	73468693          	addi	a3,a3,1844 # 0x80001734
8000122c:	00a68533          	add	a0,a3,a0
80001230:	0205a703          	lw	a4,32(a1)
80001234:	00062603          	lw	a2,0(a2)
80001238:	00052683          	lw	a3,0(a0)
8000123c:	01c5a503          	lw	a0,28(a1)
80001240:	00c72303          	lw	t1,12(a4)
80001244:	00068593          	mv	a1,a3
80001248:	00030067          	jr	t1

Disassembly of section .text.memcpy:

8000124c <.text.memcpy>:
8000124c:	ff010113          	addi	sp,sp,-16
80001250:	00112623          	sw	ra,12(sp)
80001254:	00812423          	sw	s0,8(sp)
80001258:	01010413          	addi	s0,sp,16
8000125c:	01000693          	li	a3,16
80001260:	08d66063          	bltu	a2,a3,0x800012e0
80001264:	40a006b3          	neg	a3,a0
80001268:	0036f693          	andi	a3,a3,3
8000126c:	00d50733          	add	a4,a0,a3
80001270:	02e57463          	bgeu	a0,a4,0x80001298
80001274:	00068793          	mv	a5,a3
80001278:	00050813          	mv	a6,a0
8000127c:	00058893          	mv	a7,a1
80001280:	0008c283          	lbu	t0,0(a7)
80001284:	00580023          	sb	t0,0(a6)
80001288:	00180813          	addi	a6,a6,1
8000128c:	fff78793          	addi	a5,a5,-1
80001290:	00188893          	addi	a7,a7,1
80001294:	fe0796e3          	bnez	a5,0x80001280
80001298:	00d585b3          	add	a1,a1,a3
8000129c:	40d60633          	sub	a2,a2,a3
800012a0:	ffc67793          	andi	a5,a2,-4
800012a4:	0035f813          	andi	a6,a1,3
800012a8:	00f706b3          	add	a3,a4,a5
800012ac:	06081463          	bnez	a6,0x80001314
800012b0:	00d77e63          	bgeu	a4,a3,0x800012cc
800012b4:	00058813          	mv	a6,a1
800012b8:	00082883          	lw	a7,0(a6)
800012bc:	01172023          	sw	a7,0(a4)
800012c0:	00470713          	addi	a4,a4,4
800012c4:	00480813          	addi	a6,a6,4
800012c8:	fed768e3          	bltu	a4,a3,0x800012b8
800012cc:	00f585b3          	add	a1,a1,a5
800012d0:	00367613          	andi	a2,a2,3
800012d4:	00c68733          	add	a4,a3,a2
800012d8:	00e6ea63          	bltu	a3,a4,0x800012ec
800012dc:	0280006f          	j	0x80001304
800012e0:	00050693          	mv	a3,a0
800012e4:	00c50733          	add	a4,a0,a2
800012e8:	00e57e63          	bgeu	a0,a4,0x80001304
800012ec:	0005c703          	lbu	a4,0(a1)
800012f0:	00e68023          	sb	a4,0(a3)
800012f4:	00168693          	addi	a3,a3,1
800012f8:	fff60613          	addi	a2,a2,-1
800012fc:	00158593          	addi	a1,a1,1
80001300:	fe0616e3          	bnez	a2,0x800012ec
80001304:	00c12083          	lw	ra,12(sp)
80001308:	00812403          	lw	s0,8(sp)
8000130c:	01010113          	addi	sp,sp,16
80001310:	00008067          	ret
80001314:	fad77ce3          	bgeu	a4,a3,0x800012cc
80001318:	00359893          	slli	a7,a1,0x3
8000131c:	0188f813          	andi	a6,a7,24
80001320:	ffc5f293          	andi	t0,a1,-4
80001324:	0002a303          	lw	t1,0(t0)
80001328:	411008b3          	neg	a7,a7
8000132c:	0188f893          	andi	a7,a7,24
80001330:	00428293          	addi	t0,t0,4
80001334:	0002a383          	lw	t2,0(t0)
80001338:	01035333          	srl	t1,t1,a6
8000133c:	01139e33          	sll	t3,t2,a7
80001340:	006e6333          	or	t1,t3,t1
80001344:	00672023          	sw	t1,0(a4)
80001348:	00470713          	addi	a4,a4,4
8000134c:	00428293          	addi	t0,t0,4
80001350:	00038313          	mv	t1,t2
80001354:	fed760e3          	bltu	a4,a3,0x80001334
80001358:	f75ff06f          	j	0x800012cc
